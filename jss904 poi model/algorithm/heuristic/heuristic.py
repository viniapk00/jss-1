"""Iterative metaheuristics: Enhanced Greedy portfolio, Roulette sampling, and Targeted LNS."""
from dataclasses import dataclass
import random
import time
import numpy as np

from algorithm.heuristic.greedy import BaseScheduler


@dataclass
class IterativeScheduler:
    config: dict
    data: object

    def run_greedy(self):
        """Execute deterministic multi-strategy Greedy portfolio and select the best candidate.

        Strategies evaluated:
          1. greedy_default: Balanced priority and due-date scoring with product clustering.
          2. greedy_pri: Priority-heavy blend (70% priority score, 30% due score).
          3. greedy_due: Due-date heavy blend (30% priority score, 70% due score).
          4. greedy_pri_only: Pure priority-based ordering (100% priority).
          5. greedy_due_only: Pure Earliest Due Date (EDD) ordering (100% due date).
          6. greedy_atc: Apparent Tardiness Cost with exponential slack discounting.
        """
        data = self.data
        greedy_state = data.greedy_state
        base_scheduler = BaseScheduler(data, greedy_state)

        # Define the portfolio of deterministic dispatching strategies
        strategies = [
            ('greedy_default',  lambda: (base_scheduler.lot_order(None), base_scheduler.schedule(base_scheduler.lot_order(None), output=True))),
            ('greedy_pri',      lambda: (base_scheduler.lot_order({'priority': 0.7, 'due_date': 0.3}), base_scheduler.schedule(base_scheduler.lot_order({'priority': 0.7, 'due_date': 0.3}), output=True))),
            ('greedy_due',      lambda: (base_scheduler.lot_order({'priority': 0.3, 'due_date': 0.7}), base_scheduler.schedule(base_scheduler.lot_order({'priority': 0.3, 'due_date': 0.7}), output=True))),
            ('greedy_pri_only', lambda: (base_scheduler.lot_order({'priority': 1.0, 'due_date': 0.0}), base_scheduler.schedule(base_scheduler.lot_order({'priority': 1.0, 'due_date': 0.0}), output=True))),
            ('greedy_due_only', lambda: (base_scheduler.lot_order({'priority': 0.0, 'due_date': 1.0}), base_scheduler.schedule(base_scheduler.lot_order({'priority': 0.0, 'due_date': 1.0}), output=True))),
            ('greedy_atc',      lambda: (greedy_state.atc_order, base_scheduler.schedule(greedy_state.atc_order, output=True))),
        ]
        results = {}
        metadata = {}
        self._machine_preference = {}
        started_greedy = time.perf_counter()
        best_candidate = None

        print(f"  [Greedy - one flow, {len(strategies)} strategies]")
        # Evaluate each dispatching strategy sequentially
        for strategy_name, strategy_function in strategies:
            started = time.perf_counter()
            lot_order, (result_frame, objective_value, _discarded_machine_state) = strategy_function()
            solution_quality = (float(objective_value),)
            elapsed_time = time.perf_counter() - started

            # Store schedule DataFrame and objective for comparison reporting
            results[strategy_name] = (result_frame, solution_quality[0], elapsed_time)
            metadata[strategy_name] = {'number_iterations': 1, 'best_iteration': 1, 'number_updates': 0}

            # Retain the best candidate across all deterministic strategies
            if best_candidate is None or solution_quality < best_candidate[0]:
                best_candidate = solution_quality, result_frame, list(lot_order), strategy_name

            print(f"    {strategy_name:<20} obj={solution_quality[0]:.4f} ({elapsed_time:.3f}s)")

        best_quality, best_frame, best_lot_order, winning_strategy = best_candidate
        total_greedy_elapsed = time.perf_counter() - started_greedy

        # Register the winning greedy solution as 'best_greedy' for downstream metaheuristics
        results['best_greedy'] = (best_frame, best_quality[0], total_greedy_elapsed)
        metadata['best_greedy'] = {
            'number_iterations': 1,
            'best_iteration': 1,
            'number_updates': 0,
        }
        print(f"    best_greedy={best_quality[0]:.4f} from {winning_strategy} | total={total_greedy_elapsed:.3f}s")
        return results, (best_frame, best_quality[0], best_lot_order, total_greedy_elapsed), metadata

    def run_roulette(self, seed_solution):
        """Urgency-weighted permutation sampling using independent exponential races.

        Instead of O(N^2) roulette selection, this uses the continuous-time Gumbel/exponential race trick:
        Generating X_p ~ Exp(lambda_p) and sorting by X_p yields exact sampling without replacement
        proportional to lambda_p in O(N log N) time.
        """
        data = self.data
        base_scheduler = BaseScheduler(
            data, data.greedy_state,
            getattr(self, '_machine_preference', None) or None
        )
        _discarded_seed_frame, seed_objective, seed_lot_order, greedy_elapsed_time = seed_solution
        best_lot_order, best_quality = list(seed_lot_order), (float(seed_objective),)

        iterations = max(1, int(float(self.config['iterations'])))
        random_generator = random.Random(int(self.config.get('solver_seed', 42)))
        priority_weight = float(self.config.get('roulette_w_priority', 1.0))
        due_date_weight = float(self.config.get('roulette_w_due', 1.0))

        # Precompute sampling weights (intensity parameters) per lot
        sampling_weights = {
            lot: max(1e-12, priority_weight * float(base_scheduler.state.priority_scores[idx]) + due_date_weight * float(base_scheduler.state.due_scores[idx]))
            for idx, lot in enumerate(data.P)
        }
        search_start_time = time.perf_counter()
        update_count = best_iteration = 0

        print(f"\n  [Roulette - {iterations} iterations]")
        for iteration_index in range(1, iterations + 1):
            # Sample lot sequence by independent exponential random variables
            sampled_order = sorted(data.P, key=lambda lot: (random_generator.expovariate(sampling_weights[lot]), str(lot)))
            candidate_quality = base_scheduler.schedule(sampled_order)
            improved = candidate_quality < best_quality

            # Accept strictly better solutions (greedy hill-climbing over roulette proposals)
            if improved:
                best_quality, best_lot_order = candidate_quality, sampled_order
                update_count += 1
                best_iteration = iteration_index

            print(f"  [Roulette {iteration_index:>4}/{iterations}] candidate={candidate_quality[0]:.4f} best={best_quality[0]:.4f}{' <- NEW BEST' if improved else ''}")

        # Construct final schedule DataFrame for the best found sequence
        final_frame, final_objective, _final_state = base_scheduler.schedule(best_lot_order, output=True)
        if float(final_objective) > float(seed_objective) + 1e-9:
            raise RuntimeError('Roulette degraded its Greedy seed')

        search_duration = time.perf_counter() - search_start_time
        total_time = greedy_elapsed_time + search_duration

        print(f"  Roulette done: best={final_objective:.4f} | updates={update_count} | search={search_duration:.3f}s | total+Greedy={total_time:.3f}s")
        return final_frame, float(final_objective), best_lot_order, total_time, {
            'number_iterations': iterations,
            'best_iteration': best_iteration,
            'number_updates': update_count,
        }

    def run_lns(self, seed_solution):
        """Large Neighborhood Search (LNS) with formulation-aware destroy and 5 targeted repair operators.

        The search cycle:
          1. DESTROY: Evaluates current schedule to identify critical lots (tardy lots and lots triggering setup).
             Samples a destroy subset of size determined by lns_destroy_pct.
          2. REPAIR: Applies 5 distinct repair heuristics:
             - Repair 1 (Urgency Insertion): Re-inserts destroyed lots prioritized by weight U_p and due date D_p.
             - Repair 2 (Modified Due Date Insertion): Uses MDD = max(D_p, R_p) for tight scheduling.
             - Repair 3 (Product Family Clustering): Minimizes setup by clustering lots of identical products.
             - Repair 4 (Targeted Swap): Directly swaps a tardy lot with an earlier non-tardy lot.
             - Repair 5 (Best Position Family Insertion): Inserts destroyed lots directly adjacent to same-family lots.
          3. SELECT: Evaluates all 5 candidates against the composite objective and accepts the best.
        """
        data = self.data
        base_scheduler = BaseScheduler(
            data, data.greedy_state,
            getattr(self, '_machine_preference', None) or None
        )
        _discarded_seed_frame, seed_objective, seed_lot_order, greedy_elapsed_time = seed_solution
        best_lot_order, best_quality = list(seed_lot_order), (float(seed_objective),)

        iterations = max(1, int(float(self.config['iterations'])))
        destroy_percentage = min(100.0, max(0.1, float(self.config.get('lns_destroy_pct', 10.0))))
        random_generator = random.Random(int(self.config.get('solver_seed', 42)) + 100003)
        numpy_random = np.random.RandomState(int(self.config.get('solver_seed', 42)) + 100003)
        lot_products = data.product if data.actual_setup else {}

        search_start_time = time.perf_counter()
        update_count = best_iteration = 0

        print(f"\n  [LNS - {iterations} iterations, destroy={destroy_percentage:g}%, multi-repair/iteration]")

        for iteration_index in range(1, iterations + 1):
            # Step 1: Compute dynamic adaptive destroy weights targeting tardy lots and high-setup lots
            current_frame, _current_objective, _current_state = base_scheduler.schedule(best_lot_order, output=True)
            last_operations = current_frame.sort_values('Operation Sequence').groupby('lot ID').last().reset_index()

            tardiness_destroy_weights = {}
            for _row_index, operation_row in last_operations.iterrows():
                lot = operation_row['lot ID']
                lot_due_date = data.Dp.get(lot, float('inf'))
                lot_end_time = float(operation_row['End Time (sec)'])
                lot_priority = data.Up.get(lot, 1.0)
                tardiness_amount = max(0.0, lot_end_time - lot_due_date)

                # Heavily weight lots that actually incurred tardiness in the schedule
                tardiness_destroy_weights[lot] = (
                    1.0 + 10.0 * (lot_priority * tardiness_amount > 0) + (lot_priority * tardiness_amount) / 10000.0
                )

            # In setup-sensitive objectives, also penalize lots that incurred sequence setup
            if data.actual_setup:
                setup_lots = set(current_frame[current_frame['Setup Time'] > 0]['lot ID'])
                for lot in setup_lots:
                    tardiness_destroy_weights[lot] = tardiness_destroy_weights.get(lot, 1.0) + 25.0

            lots_list = list(best_lot_order)
            weights_array = np.array([tardiness_destroy_weights[lot] for lot in lots_list], dtype=float)
            weights_array /= weights_array.sum()

            # Determine number of lots to destroy
            destroy_count = min(
                len(best_lot_order),
                max(2 if len(best_lot_order) > 2 else 1, round(len(best_lot_order) * destroy_percentage / 100.0))
            )
            # Sample lots to remove without replacement
            destroyed_lots = list(numpy_random.choice(lots_list, size=destroy_count, replace=False, p=weights_array))
            removed_lot_set = set(destroyed_lots)
            remaining_lots = [lot for lot in best_lot_order if lot not in removed_lot_set]
            original_positions = {lot: idx for idx, lot in enumerate(best_lot_order)}

            # Sorted order of destroyed lots by descending priority and ascending due date
            repair_sequence = sorted(
                destroyed_lots,
                key=lambda lot: (-data.Up.get(lot, 1.0), data.Dp.get(lot, 0), str(lot))
            )

            # -----------------------------------------------------------------
            # Repair 1: Urgency insertion (insert by -Up, Dp into remaining lots)
            # -----------------------------------------------------------------
            urgency_order = list(remaining_lots)
            for lot in repair_sequence:
                sort_criteria = (-data.Up.get(lot, 1.0), data.Dp.get(lot, 0), str(lot))
                insert_position = next(
                    (i for i, item in enumerate(urgency_order)
                     if sort_criteria < (-data.Up.get(item, 1.0), data.Dp.get(item, 0), str(item))),
                    len(urgency_order)
                )
                urgency_order.insert(insert_position, lot)

            # -----------------------------------------------------------------
            # Repair 2: Modified Due Date (MDD) insertion (MDD = max(Dp, Rp))
            # -----------------------------------------------------------------
            repair_sequence_mdd = sorted(
                destroyed_lots,
                key=lambda lot: (max(data.Dp.get(lot, 0), data.Rp.get(lot, 0)), -data.Up.get(lot, 1.0), str(lot))
            )
            mdd_order = list(remaining_lots)
            for lot in repair_sequence_mdd:
                sort_criteria = (max(data.Dp.get(lot, 0), data.Rp.get(lot, 0)), -data.Up.get(lot, 1.0), str(lot))
                insert_position = next(
                    (i for i, item in enumerate(mdd_order)
                     if sort_criteria < (max(data.Dp.get(item, 0), data.Rp.get(item, 0)), -data.Up.get(item, 1.0), str(item))),
                    len(mdd_order)
                )
                mdd_order.insert(insert_position, lot)

            # -----------------------------------------------------------------
            # Repair 3: Product clustering / family affinity insertion
            # -----------------------------------------------------------------
            family_cluster_order = list(remaining_lots)
            if data.actual_setup:
                for lot in repair_sequence:
                    related_positions = [
                        i for i, item in enumerate(family_cluster_order)
                        if lot_products.get(item) == lot_products.get(lot)
                    ]
                    if related_positions:
                        # Randomly place at start or end of the same-product cluster
                        insert_position = related_positions[0] if random_generator.random() < 0.5 else related_positions[-1] + 1
                    else:
                        insert_position = min(len(family_cluster_order), original_positions[lot])
                    family_cluster_order.insert(insert_position, lot)
            else:
                due_sequence = sorted(destroyed_lots, key=lambda lot: (data.Dp.get(lot, 0), -data.Up.get(lot, 1.0), str(lot)))
                for lot in due_sequence:
                    sort_criteria = (data.Dp.get(lot, 0), -data.Up.get(lot, 1.0), str(lot))
                    insert_position = next(
                        (i for i, item in enumerate(family_cluster_order)
                         if sort_criteria < (data.Dp.get(item, 0), -data.Up.get(item, 1.0), str(item))),
                        len(family_cluster_order)
                    )
                    family_cluster_order.insert(insert_position, lot)

            # -----------------------------------------------------------------
            # Repair 4: Targeted Swap of Tardy lot with earlier non-tardy lot
            # -----------------------------------------------------------------
            targeted_swap_order = list(best_lot_order)
            if len(targeted_swap_order) >= 2:
                tardy_lot_candidates = [lot for lot in best_lot_order if tardiness_destroy_weights[lot] > 1.0]
                if tardy_lot_candidates:
                    chosen_tardy_lot = random_generator.choice(tardy_lot_candidates)
                    tardy_index = best_lot_order.index(chosen_tardy_lot)
                    if tardy_index > 0:
                        # Advance the tardy lot by swapping with an earlier random index
                        swap_target_index = random_generator.randint(0, tardy_index - 1)
                        targeted_swap_order[tardy_index], targeted_swap_order[swap_target_index] = (
                            targeted_swap_order[swap_target_index], targeted_swap_order[tardy_index]
                        )
                else:
                    # If no tardy lots exist, perform random 2-opt swap for exploration
                    first_idx, second_idx = random_generator.sample(range(len(targeted_swap_order)), 2)
                    targeted_swap_order[first_idx], targeted_swap_order[second_idx] = (
                        targeted_swap_order[second_idx], targeted_swap_order[first_idx]
                    )

            # -----------------------------------------------------------------
            # Repair 5: Best Position Family Insertion (directly append to family)
            # -----------------------------------------------------------------
            greedy_insertion_order = list(remaining_lots)
            for lot in destroyed_lots:
                current_product = lot_products.get(lot)
                related_positions = [
                    i for i, item in enumerate(greedy_insertion_order)
                    if lot_products.get(item) == current_product
                ]
                original_pos = original_positions.get(lot, len(greedy_insertion_order))
                if related_positions:
                    best_pos = related_positions[-1] + 1
                else:
                    best_pos = min(len(greedy_insertion_order), original_pos)
                greedy_insertion_order.insert(best_pos, lot)

            # Evaluate all 5 repaired permutation candidates
            repair_candidates = [
                (base_scheduler.schedule(urgency_order), urgency_order),
                (base_scheduler.schedule(mdd_order), mdd_order),
                (base_scheduler.schedule(family_cluster_order), family_cluster_order),
                (base_scheduler.schedule(targeted_swap_order), targeted_swap_order),
                (base_scheduler.schedule(greedy_insertion_order), greedy_insertion_order),
            ]
            candidate_quality, candidate_order = min(repair_candidates, key=lambda item: item[0])
            improved = candidate_quality < best_quality

            # Update best incumbent if candidate strictly improves objective
            if improved:
                best_quality, best_lot_order = candidate_quality, list(candidate_order)
                update_count += 1
                best_iteration = iteration_index

            print(f"  [LNS {iteration_index:>4}/{iterations}] candidate={candidate_quality[0]:.4f} best={best_quality[0]:.4f}{' <- NEW BEST' if improved else ''}")

        # Final full schedule reconstruction with output DataFrame
        final_frame, final_objective, _final_state = base_scheduler.schedule(best_lot_order, output=True)
        if float(final_objective) > float(seed_objective) + 1e-9:
            raise RuntimeError('LNS degraded its Greedy seed')

        search_duration = time.perf_counter() - search_start_time
        total_time = greedy_elapsed_time + search_duration

        print(f"  LNS done: best={final_objective:.4f} | updates={update_count} | search={search_duration:.3f}s | total+Greedy={total_time:.3f}s")
        return final_frame, float(final_objective), best_lot_order, total_time, {
            'number_iterations': iterations,
            'best_iteration': best_iteration,
            'number_updates': update_count,
        }

