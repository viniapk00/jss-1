"""Deterministic Greedy scheduling, multi-strategy dispatching, and targeted repairs."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.preprocessing import col_end, col_job, col_pri, col_start, tardiness_seconds


@dataclass
class BaseScheduler:
    data: object
    state: object
    machine_preference: dict = None

    def lot_order(self, weights=None, family_batch=None):
        """Rank lots with global product clustering and urgency lookahead.

        Steps:
          1. Score calculation: Combines normalized priority scores (U_p) and due-date proximity scores.
          2. Clustering decision: If actual sequence setup is active, lots are clustered by Product ID
             to minimize expensive changeovers on the shop floor.
          3. Urgency bucketing: Groups product families into temporal horizon buckets (e.g. 48h) to prevent
             urgent lots of other product families from being delayed behind non-urgent lots of the current family.
          4. Within-cluster sorting: Sorts lots within the same product cluster by highest combined score.
        """
        data, state = self.data, self.state

        # Blend priority and due date weights (default 50/50 if not explicitly specified)
        if weights is None: combined_scores = state.priority_scores + state.due_scores
        else:
            priority_weight, due_date_weight = weights.get('priority', weights.get('p', 0.5)), weights.get('due_date', weights.get('d', 0.5))
            combined_scores = priority_weight * state.priority_scores + due_date_weight * state.due_scores

        use_clustering = data.actual_setup if family_batch is None else family_batch

        # If sequence setups are not modeled, pure greedy score sorting suffices
        if not use_clustering: return [data.P[idx] for idx in (-combined_scores).argsort(kind='mergesort')]

        # Step 2: Global Product Clustering: Group lots by Product ID
        product_to_lots = {}
        for idx, lot in enumerate(data.P): product_to_lots.setdefault(data.product.get(lot, ''), []).append((idx, lot))

        # Step 3: Compute product family urgency tuple to rank clusters
        product_urgency, horizon = {}, (172800 if weights is None else weights.get('horizon', 172800))
        for product_id, items in product_to_lots.items():
            minimum_due_date, maximum_score = min(data.Dp.get(lot, 0) for _, lot in items), max(combined_scores[idx] for idx, _ in items)
            total_prod_move = sum(min(data.route_move_lb.get((lot, opt), 0.0) for opt in data.Op.get(lot, [1])) for _, lot in items)
            max_ops = max(len(data.Ipo.get((lot, data.fp.get(lot, 1)), [])) for _, lot in items)

            # Assign product family to a discrete time bucket (horizon)
            urgency_bucket = minimum_due_date // horizon
            product_urgency[product_id] = (
                urgency_bucket,     # Primary: Time bucket (e.g. 48-hour window)
                -total_prod_move,   # Secondary: High-transport product groups prioritized
                -max_ops,           # Tertiary: Multi-operation routes scheduled early
                -maximum_score,     # Quaternary: Highest urgency score inside cluster
                minimum_due_date    # Quinary: Earliest individual due date
            )

        # Sort product families by their multi-attribute urgency profile
        sorted_products = sorted(product_to_lots.keys(), key=lambda prod: product_urgency[prod])

        # Step 4: Assemble ordered lot list, sorting internally within each product group
        ordered_lots = [lot for product_id in sorted_products for _, lot in sorted(product_to_lots[product_id], key=lambda item: (-combined_scores[item[0]], data.Dp.get(item[1], 0)))]

        return ordered_lots

    def option_mapping(self, lot, availability, last_operation, context):
        """Select the best feasible process option (o in O_p) for one lot.

        Evaluates each available process route option by simulating machine assignment
        and measuring the marginal increase in the composite objective function.
        """
        data, state, fixed_option = self.data, self.state, self.data.fp.get(lot, 0)

        # Enforce fixed route option if specified by input parameter
        if fixed_option > 0 and (lot, fixed_option) not in state.option_meta: raise ValueError(f'Lot {lot} fixed_option={fixed_option} is infeasible')

        available_options = [fixed_option] if fixed_option > 0 else [opt for opt in data.Op.get(lot, []) if (lot, opt) in state.option_meta]
        if not available_options: return float('inf'), None, None

        due_date, priority, best_candidate = data.Dp.get(lot, float('inf')), data.Up.get(lot, 1.0), None

        # Evaluate candidate options and select the one yielding minimal composite objective
        for option in available_options:
            completion_time, operations, moving_time, setup_time, processing_time = self.machine_assign(lot, option, availability, last_operation, priority, due_date, context)
            tardiness_sec = tardiness_seconds(completion_time, due_date)
            objective_score = state.objective(context['weighted_tardiness'] + priority * tardiness_sec, context['movement_seconds'] + moving_time, context['setup_seconds'] + setup_time, max(context['makespan_seconds'], completion_time), context['processing_seconds'] + processing_time,)
            candidate = ((objective_score,), objective_score, option, operations)
            if best_candidate is None or candidate[0] < best_candidate[0]: best_candidate = candidate

        return best_candidate[1], best_candidate[2], best_candidate[3]

    def machine_assign(self, lot, option, availability, last_operation, priority, due_date, context):
        """Select a machine route for every operation of one lot via bounded beam search.

        For each step i in {1, ..., K_po}:
          1. Retrieves candidate eligible machines m in M_poi.
          2. Calculates setup duration: S_a,b,m if predecessor exists on machine m, else S0_a,m.
          3. Calculates transit duration: E_m',m from previous machine m' to current machine m.
          4. Computes start time: max(machine_ready, lot_ready + transit) + setup.
          5. Adds lookahead: suffix processing lower bounds and future movement shortest paths.
          6. Ranks candidate paths by multi-criteria tie-breaker and prunes to beam width.
        """
        data, state = self.data, self.state
        meta_entry = state.option_meta[lot, option]
        gateway_demand, beam_limit = getattr(state, 'gateway_demand', {}), max(64, getattr(state, 'route_limit', 64))
        jobs, suffix_remaining, touched_machines, local_machine_index = meta_entry[:4]
        future_move_maps = meta_entry[4] if len(meta_entry) > 4 else [{} for discard_job in jobs]
        current_product, touched_machine_list = data.product.get(lot, ''), list(touched_machines)
        local_availability, local_predecessor = availability[touched_machine_list], last_operation[touched_machine_list]

        initial_release_time = data.Rp.get(lot, 0)
        # Initial beam state: (lot_current_time, current_machine, move_sec, setup_sec, proc_sec, path, avail, preds, ops)
        routes = [(initial_release_time, None, 0.0, 0.0, 0.0, (), local_availability, local_predecessor, ())]

        # Iterate sequentially through process steps of this route option
        for job_position, (job_sequence, eligible_machines) in enumerate(jobs):
            operation_key, remaining_suffix_time = (lot, option, job_sequence), suffix_remaining[job_position]
            future_move_map = future_move_maps[job_position]
            candidate_machines = tuple((machine, local_machine_index[state.machine_index[machine]], data.Tam[operation_key + (machine,)]) for machine in eligible_machines)
            expanded_routes, has_next_job = [], job_position + 1 < len(jobs)

            for (current_end_time, previous_machine, total_moving_time, total_setup_time, total_processing_time, machine_path, current_availability, current_predecessor, current_operations) in routes:

                for machine, local_index, process_duration in candidate_machines:
                    preceding_operation = current_predecessor[local_index]
                    # Compute sequence-dependent setup or initial depot setup
                    setup_duration = (data.S.get((preceding_operation, operation_key, machine), 0.0) if preceding_operation else data.S0.get((operation_key, machine), 0.0)) if data.actual_setup else 0.0
                    # Compute inter-machine transit duration (E_mn)
                    transport_duration = data.Emn.get((previous_machine, machine), 0) if previous_machine and previous_machine != machine else 0
                    expected_future_move = future_move_map.get(machine, 0.0)

                    # Operation start time respecting machine ready time, setup, and transit arrival
                    start_time = max(current_availability[local_index] + setup_duration, current_end_time + transport_duration)
                    end_time = start_time + process_duration

                    # Clone and update machine tracking states
                    if has_next_job:
                        next_availability, next_predecessor = current_availability.copy(), current_predecessor.copy()
                        next_availability[local_index], next_predecessor[local_index] = end_time, operation_key
                    else: next_availability, next_predecessor = current_availability, current_predecessor
                    next_operations = current_operations + ((machine, start_time, end_time, setup_duration, transport_duration, process_duration, job_sequence),)
                    next_path = machine_path + (machine,)

                    # Lookahead: estimate completion with suffix processing and future transit lower bound
                    tardiness_sec = tardiness_seconds(end_time + remaining_suffix_time + expected_future_move, due_date)
                    estimated_objective = state.objective(context['weighted_tardiness'] + priority * tardiness_sec, context['movement_seconds'] + total_moving_time + transport_duration + expected_future_move, context['setup_seconds'] + total_setup_time + setup_duration, max(context['makespan_seconds'], end_time + remaining_suffix_time + expected_future_move), context['processing_seconds'] + total_processing_time + process_duration,)

                    # Setup continuity tie-breaker: prefer continuing the same product (zero changeover)
                    same_product = (data.product.get(preceding_operation[0]) == current_product) if preceding_operation else False

                    # Gateway reservation penalty: reserve multi-operation transit hubs for multi-op jobs
                    res_penalty = gateway_demand.get(machine, 0) if expected_future_move == 0 else 0

                    # Lexicographic sorting key for beam selection
                    sorting_key = (round(estimated_objective, 4),not same_product,            # 1. Prefer machines currently set up for the same product
                        setup_duration > 0,          # 2. Prefer 0 setup over setup switch
                        total_moving_time + transport_duration + expected_future_move, # 3. Shortest cumulative transport
                        res_penalty,                 # 4. Preserve multi-op gateway machines
                        end_time,                    # 5. Earlier completion
                        total_setup_time + setup_duration,
                        next_path)
                    expanded_routes.append((sorting_key, end_time, machine, total_moving_time + transport_duration, total_setup_time + setup_duration, total_processing_time + process_duration, next_path, next_availability, next_predecessor, next_operations))

            # Prune expanded candidates to top beam_limit paths
            expanded_routes.sort(key=lambda route_item: route_item[0])
            routes = [route_item[1:] for route_item in expanded_routes[:beam_limit]]

        # Select the winning path from the beam search
        (final_end_time, _discarded_machine, final_move_time, final_setup_time, final_proc_time, _discarded_path, _discarded_availability, _discarded_predecessor, scheduled_operations) = routes[0]
        return final_end_time, scheduled_operations, final_move_time, final_setup_time, final_proc_time

    def schedule(self, lot_order, output=False):
        """Decode one lot permutation into schedule metrics or tabular DataFrame.

        Iterates sequentially over the given lot sequence:
          1. Calls option_mapping to select optimal route option and machine path.
          2. Updates machine availability and predecessor lot tracking.
          3. Accumulates objective components (tardiness, transport, setup, makespan, processing).
          4. Formats operation timing records if output=True.
        """
        data, state = self.data, self.state
        order = lot_order if isinstance(lot_order, list) else list(lot_order)

        if (output or len(order) == len(data.P)) and (len(order) != len(data.P) or set(order) != set(data.P)): raise ValueError('lot_order must be a complete permutation of data.P')

        # Initialize machine availability with B_m (machine ready times)
        availability, last_operation = state.initial_availability.copy(), np.full(len(state.machines), None, dtype=object)

        # Objective tracking context
        context = {'weighted_tardiness': 0.0, 'movement_seconds': 0.0, 'setup_seconds': 0.0, 'makespan_seconds': 0.0, 'processing_seconds': 0.0}
        schedule_rows = []

        for lot in order:
            _option_score, chosen_option, operations = self.option_mapping(lot, availability, last_operation, context)
            if chosen_option is None: raise RuntimeError(f'Greedy decoder found no feasible option for lot {lot}')

            lot_row, previous_machine = data.lot_index[lot], None

            for machine, begin_time, end_time, setup_duration, transport_duration, process_duration, job_sequence in operations:
                machine_index = state.machine_index[machine]
                if output: schedule_rows.append({'lot ID': lot, 'Product ID': lot_row['Product_ID'], 'Option': chosen_option, 'Qty': lot_row['Qty'], 'Due date': lot_row['Due_date'], col_pri: lot_row['Priority'], col_job: job_sequence, 'Machine ID': machine, 'Prev Machine': previous_machine, col_start: begin_time, col_end: end_time, 'Setup Time': setup_duration, 'Moving Time': transport_duration, 'Processing Time': process_duration})
                availability[machine_index], last_operation[machine_index], previous_machine = end_time, (lot, chosen_option, job_sequence), machine

            lot_completion_time, lot_due_date, lot_priority = operations[-1][2], data.Dp.get(lot, 0), data.Up.get(lot, 1.0)

            context['weighted_tardiness'] += lot_priority * tardiness_seconds(lot_completion_time, lot_due_date)
            context['movement_seconds'] += sum(float(op[4]) for op in operations)
            context['setup_seconds'] += sum(float(op[3]) for op in operations)
            context['processing_seconds'] += sum(float(op[5]) for op in operations)
            context['makespan_seconds'] = max(context['makespan_seconds'], lot_completion_time)

        total_objective = state.objective(context['weighted_tardiness'], context['movement_seconds'], context['setup_seconds'], context['makespan_seconds'], context['processing_seconds'],)

        return (pd.DataFrame(schedule_rows), total_objective, (availability, last_operation)) if output else (total_objective,)
