"""Iterative metaheuristics: Enhanced Greedy portfolio, Roulette sampling, and Targeted LNS."""
from dataclasses import dataclass
from functools import cached_property
import math, random, time

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
        data, greedy_state = self.data, self.data.greedy_state; base_scheduler = BaseScheduler(data, greedy_state)

        processing = dict(zip(data.P, greedy_state.Pp)); mean_processing = max(1.0, greedy_state.mean_p)
        standard_atc = sorted(data.P, key=lambda lot: (-data.Up.get(lot, 1.0) / max(1.0, processing[lot]) * math.exp(-max(0.0, data.Dp.get(lot, 0.0) - data.Rp.get(lot, 0.0) - processing[lot]) / (2.0 * mean_processing)), str(lot)))

        # Define the portfolio of deterministic dispatching strategies (strictly original 6 strategies)
        strategies = [('greedy_default',  lambda family_batch=None: base_scheduler.lot_order(None, family_batch)), ('greedy_pri',      lambda family_batch=None: base_scheduler.lot_order({'priority': 0.7, 'due_date': 0.3}, family_batch)), ('greedy_due',      lambda family_batch=None: base_scheduler.lot_order({'priority': 0.3, 'due_date': 0.7}, family_batch)), ('greedy_pri_only', lambda family_batch=None: base_scheduler.lot_order({'priority': 1.0, 'due_date': 0.0}, family_batch)), ('greedy_due_only', lambda family_batch=None: base_scheduler.lot_order({'priority': 0.0, 'due_date': 1.0}, family_batch)), ('greedy_atc',      lambda family_batch=None: greedy_state.atc_order if family_batch is None else standard_atc),]
        results, metadata, evaluated_orders, started_greedy, best_candidate = {}, {}, {}, time.perf_counter(), None

        print(f"  [Greedy - one flow, {len(strategies)} strategies]")
        # Evaluate each dispatching strategy sequentially
        for strategy_name, strategy_function in strategies:
            started = time.perf_counter()
            orders = [strategy_function(), strategy_function(False)]
            for order in orders:
                order_key = tuple(order)
                if order_key not in evaluated_orders: evaluated_orders[order_key] = base_scheduler.schedule(order, output=True)[:2]
            lot_order = min(orders, key=lambda order: evaluated_orders[tuple(order)][1])
            cached_frame, objective_value = evaluated_orders[tuple(lot_order)]; result_frame = cached_frame.copy()
            solution_quality, elapsed_time = (float(objective_value),), time.perf_counter() - started

            # Store schedule DataFrame and objective for comparison reporting
            results[strategy_name], metadata[strategy_name] = (result_frame, solution_quality[0], elapsed_time), {'number_iterations': 1, 'best_iteration': 1, 'number_updates': 0}

            # Retain the best candidate across all deterministic strategies
            if best_candidate is None or solution_quality < best_candidate[0]: best_candidate = solution_quality, result_frame, list(lot_order), strategy_name

            print(f"    {strategy_name:<20} obj={solution_quality[0]:.4f} ({elapsed_time:.3f}s)")

        best_quality, best_frame, best_lot_order, winning_strategy = best_candidate
        best_frame, polished_objective, best_lot_order, polish_updates, polish_sweeps = self._polish(base_scheduler, best_frame, best_quality[0], best_lot_order)
        best_frame, polished_objective, route_updates = self._refine_routes(base_scheduler, best_frame, polished_objective, best_lot_order)
        polish_updates += route_updates
        best_quality, total_greedy_elapsed = (polished_objective,), time.perf_counter() - started_greedy

        # Register the winning greedy solution as 'best_greedy' for downstream metaheuristics
        results['best_greedy'], metadata['best_greedy'] = (best_frame, best_quality[0], total_greedy_elapsed), {'number_iterations': 1 + polish_sweeps, 'best_iteration': 1 + polish_sweeps if polish_updates else 1, 'number_updates': polish_updates}
        print(f"    best_greedy={best_quality[0]:.4f} from {winning_strategy} | total={total_greedy_elapsed:.3f}s")
        return results, (best_frame, best_quality[0], best_lot_order, total_greedy_elapsed), metadata

    def run_roulette(self, seed_solution):
        """Roulette Wheel Metaheuristic with Block-Level Tardy Selection and Intra-Block Dispatching.

        Enhancements:
          1. Block-Level Roulette: Clusters lot sequence into contiguous product blocks. Calculates
             weighted tardiness per block. Spins roulette wheel to select critical tardy blocks for forward shift/swap.
          2. Intra-Block Dispatching (EDD/SPT): Sorts lots inside each product family without incurring setup penalty.
          3. Caching of operation end-times: Evaluates full DataFrame only when solution improves.
        """
        data, delay_due = self.data, self.delay_due
        _discarded_seed_frame, seed_objective, seed_lot_order, greedy_elapsed_time = seed_solution
        base_scheduler = BaseScheduler(data, data.greedy_state, cache_prefix=True, route_choices=dict(_discarded_seed_frame.attrs.get('route_choices', {})))
        current_lot_order, current_quality = list(seed_lot_order), (float(seed_objective),)
        best_lot_order, best_quality = list(current_lot_order), current_quality

        iterations, random_generator = max(1, int(float(self.config['iterations']))), random.Random(int(self.config.get('solver_seed', 42)))

        # Precompute lot processing & weights for fast sorting
        Up_arr, Pp_arr, lots_all = base_scheduler.state.Up, getattr(base_scheduler.state, 'Pp', None), list(data.P)
        dispatch_ratio = {l: Pp_arr[i] / max(0.1, Up_arr[i]) for i, l in enumerate(lots_all)} if Pp_arr is not None and len(Pp_arr) == len(lots_all) else {l: 1.0 / max(0.1, data.Up.get(l, 1.0)) for l in lots_all}

        search_start_time, update_count, best_iteration = time.perf_counter(), 0, 0

        # Initial schedule evaluation to get starting lot completion times
        current_frame, _current_obj, _current_state = base_scheduler.schedule(current_lot_order, output=True); lot_end_times = current_frame.groupby('lot ID')['End Time (sec)'].max().to_dict()
        evaluation_cache = {tuple(current_lot_order): (float(_current_obj),)}; accepted_orders = {tuple(current_lot_order)}
        best_frame, current_quality = current_frame, (float(_current_obj),)
        best_quality = current_quality
        temperature = max(10.0, abs(seed_objective) * 0.0001)

        print(f"\n  [Roulette - {iterations} iterations, Block-Level Roulette + Intra-Block Dispatching]")
        for iteration_index in range(1, iterations + 1):
            if iteration_index - best_iteration > 15 and best_lot_order != current_lot_order: current_lot_order, current_quality, current_frame = list(best_lot_order), best_quality, best_frame; lot_end_times = current_frame.groupby('lot ID')['End Time (sec)'].max().to_dict()
            candidates_to_test = []

            if data.actual_setup:
                # 1. Partition sequence into contiguous product blocks
                blocks, curr_prod, curr_block = [], None, []
                for l in current_lot_order:
                    p = data.product.get(l)
                    if p != curr_prod:
                        if curr_block: blocks.append((curr_prod, curr_block))
                        curr_prod, curr_block = p, [l]
                    else: curr_block.append(l)
                if curr_block: blocks.append((curr_prod, curr_block))

                # 2. Compute block weights combining weighted tardiness, changeover setup duration, and product fragmentation
                prod_counts = {p: sum(1 for bp, _ in blocks if bp == p) for p, _ in blocks}
                lot_setups = current_frame.groupby('lot ID')['Setup Time'].sum().to_dict() if 'Setup Time' in current_frame.columns else {}
                block_weights = [sum(data.Up.get(l, 1.0) * max(0.0, lot_end_times.get(l, 0.0) - delay_due[l]) + lot_setups.get(l, 0.0) for l in blk_lots) + 1800.0 * (prod_counts.get(_p, 1) - 1) + 1.0 for _p, blk_lots in blocks]

                total_bw = sum(block_weights)
                if len(blocks) > 1:
                    chosen_idx = random_generator.choices(range(len(blocks)), weights=[bw / total_bw for bw in block_weights], k=1)[0] if total_bw > 0 else random_generator.randint(0, len(blocks) - 1)

                    # Proposal A: Block Shift (advance chosen tardy block earlier)
                    if chosen_idx > 0:
                        target_idx, shifted = random_generator.randint(0, chosen_idx - 1), list(blocks)
                        shifted.insert(target_idx, shifted.pop(chosen_idx))
                        candidates_to_test.append([l for _, b in shifted for l in b])

                    # Proposal B: Block Swap (swap chosen block with another block)
                    swap_target, swapped = (random_generator.randint(0, chosen_idx - 1) if chosen_idx > 0 else random_generator.randint(1, len(blocks) - 1)), list(blocks)
                    swapped[chosen_idx], swapped[swap_target] = swapped[swap_target], swapped[chosen_idx]
                    candidates_to_test.append([l for _, b in swapped for l in b])

                    # Proposal C: Product Block Consolidation (merge fragmented blocks of same product to avoid setup changeovers)
                    frag_prods = [p for p, c in prod_counts.items() if c > 1]
                    for p_sel in random_generator.sample(frag_prods, min(len(frag_prods), 2)):
                        p_idxs = [i for i, (p, _) in enumerate(blocks) if p == p_sel]
                        if len(p_idxs) >= 2:
                            s_idx, t_idx = random_generator.sample(p_idxs, 2)
                            for adj_pos in ([t_idx if t_idx < s_idx else t_idx - 1, (t_idx if t_idx < s_idx else t_idx - 1) + 1]):
                                c_blks = list(blocks); blk = c_blks.pop(s_idx); c_blks.insert(max(0, min(len(c_blks), adj_pos)), blk)
                                candidates_to_test.append([l for _, b in c_blks for l in b])

                # Proposal D: Intra-Block Dispatching (EDD & SPT within each family block - 0 setup penalty!)
                edd_blocks = [(p_id, sorted(blk_lots, key=lambda l: (data.Dp.get(l, 0), dispatch_ratio[l], str(l)))) for p_id, blk_lots in blocks]
                candidates_to_test.append([l for _, b in edd_blocks for l in b])

            else:
                # No sequence setups: Pure lot-level Roulette selection based on tardiness
                tardy_lots = [l for l in current_lot_order if lot_end_times.get(l, 0.0) > delay_due[l]]
                if tardy_lots:
                    tw = [data.Up.get(l, 1.0) * (lot_end_times.get(l, 0.0) - delay_due[l]) for l in tardy_lots]
                    total_tw = sum(tw); chosen_lot = random_generator.choices(tardy_lots, weights=[w / total_tw for w in tw], k=1)[0]
                    curr_pos = current_lot_order.index(chosen_lot)
                    if curr_pos > 0:
                        step, cand = random_generator.randint(1, min(20, curr_pos)), list(current_lot_order)
                        cand.pop(curr_pos); cand.insert(curr_pos - step, chosen_lot); candidates_to_test.append(cand)

            if len(current_lot_order) > 1:
                weights = [data.Up.get(lot, 1.0) * max(0.0, lot_end_times.get(lot, 0.0) - delay_due[lot]) for lot in current_lot_order]
                floor = max(1.0, sum(weights) / len(weights)) * 0.1
                source = random_generator.choices(range(len(current_lot_order)), weights=[weight + floor for weight in weights], k=1)[0]
                target = random_generator.randrange(len(current_lot_order) - 1); target += target >= source
                shifted, swapped = list(current_lot_order), list(current_lot_order)
                shifted.insert(target, shifted.pop(source)); swapped[source], swapped[target] = swapped[target], swapped[source]
                candidates_to_test.extend((shifted, swapped))

            # Fallback if no candidate generated
            if not candidates_to_test:
                cand = list(current_lot_order)
                if len(cand) >= 2:
                    i1, i2 = random_generator.sample(range(len(cand)), 2); cand[i1], cand[i2] = cand[i2], cand[i1]
                candidates_to_test.append(cand)

            # Evaluate candidate proposals (fast: output=False)
            best_iter_quality, best_iter_order = None, None
            for cand_order in candidates_to_test:
                if cand_order == current_lot_order: continue
                order_key = tuple(cand_order); cand_quality = evaluation_cache.get(order_key)
                if cand_quality is None: cand_quality = evaluation_cache[order_key] = base_scheduler.schedule(cand_order)
                if best_iter_quality is None or (cand_quality, order_key in accepted_orders) < (best_iter_quality, tuple(best_iter_order) in accepted_orders): best_iter_quality, best_iter_order = cand_quality, cand_order

            improved = best_iter_quality is not None and best_iter_quality < best_quality
            cooling = temperature * (0.01 ** (iteration_index / iterations))
            accepted = best_iter_quality is not None and (best_iter_quality <= current_quality or random_generator.random() < math.exp(min(0.0, (current_quality[0] - best_iter_quality[0]) / cooling)))
            if accepted and best_iter_order != current_lot_order:
                current_quality, current_lot_order = best_iter_quality, best_iter_order; accepted_orders.add(tuple(current_lot_order))
                # Refresh lot completion times for next iteration's roulette
                current_frame, _current_obj, _current_state = base_scheduler.schedule(current_lot_order, output=True)
                lot_end_times = current_frame.groupby('lot ID')['End Time (sec)'].max().to_dict()
                if improved: best_quality, best_lot_order, best_frame, update_count, best_iteration = current_quality, list(current_lot_order), current_frame, update_count + 1, iteration_index

            current_frame, current_quality, route_changed = self._route_step(base_scheduler, current_frame, current_lot_order, current_quality, random_generator, cooling)
            if route_changed:
                lot_end_times = current_frame.groupby('lot ID')['End Time (sec)'].max().to_dict()
                evaluation_cache, accepted_orders = {tuple(current_lot_order): current_quality}, {tuple(current_lot_order)}
                if current_quality < best_quality:
                    best_quality, best_lot_order, best_frame, update_count, best_iteration = current_quality, list(current_lot_order), current_frame, update_count + 1, iteration_index
                    improved = True

            eval_score = best_iter_quality[0] if best_iter_quality is not None else best_quality[0]
            print(f"  [Roulette {iteration_index:>4}/{iterations}] candidate={eval_score:.4f} best={best_quality[0]:.4f}{' <- NEW BEST' if improved else ''}")

        # Construct final schedule DataFrame for the best found sequence
        final_frame, final_objective, route_updates = self._refine_routes(base_scheduler, best_frame, best_quality[0], best_lot_order)
        if route_updates: update_count, best_iteration = update_count + route_updates, iterations
        if float(final_objective) > float(seed_objective) + 1e-9: raise RuntimeError('Roulette degraded its Greedy seed')

        search_duration = time.perf_counter() - search_start_time; total_time = greedy_elapsed_time + search_duration

        print(f"  Roulette done: best={final_objective:.4f} | updates={update_count} | search={search_duration:.3f}s | total+Greedy={total_time:.3f}s")
        return final_frame, float(final_objective), best_lot_order, total_time, {'number_iterations': iterations, 'best_iteration': best_iteration, 'number_updates': update_count}

    def run_lns(self, seed_solution):
        """Large Neighborhood Search (LNS) with deep block ruin-and-recreate and zero-overhead evaluation.

        Enhancements:
          1. Targeted Block Forward Shift: Advances tardy product blocks earlier while preserving lot sequences.
          2. Product Family Consolidation: Consolidates scattered product blocks to eliminate setup overhead.
          3. Strategic Block Swap: Swaps critical tardy blocks across bottlenecks preserving internal orders.
          4. Targeted Leapfrog: Explores deep fractional jumps across bottlenecks.
          5. Zero redundant evaluations: Deduplicates candidate sequences to evaluate only novel permutations.
        """
        data, delay_due = self.data, self.delay_due
        _discarded_seed_frame, seed_objective, seed_lot_order, greedy_elapsed_time = seed_solution
        base_scheduler = BaseScheduler(data, data.greedy_state, cache_prefix=True, route_choices=dict(_discarded_seed_frame.attrs.get('route_choices', {})))
        current_lot_order, current_quality = list(seed_lot_order), (float(seed_objective),)
        best_lot_order, best_quality = list(current_lot_order), current_quality

        iterations, random_generator = max(1, int(float(self.config['iterations']))), random.Random(int(self.config.get('solver_seed', 42)) + 100003)

        # Precompute lot processing & weights for fast sorting
        Up_arr, Pp_arr, lots_all = base_scheduler.state.Up, getattr(base_scheduler.state, 'Pp', None), list(data.P)
        dispatch_ratio = {l: Pp_arr[i] / max(0.1, Up_arr[i]) for i, l in enumerate(lots_all)} if Pp_arr is not None and len(Pp_arr) == len(lots_all) else {l: 1.0 / max(0.1, data.Up.get(l, 1.0)) for l in lots_all}

        search_start_time, update_count, best_iteration = time.perf_counter(), 0, 0

        # Initial schedule evaluation to get starting lot completion times
        current_frame, _current_obj, _current_state = base_scheduler.schedule(current_lot_order, output=True); lot_end_times = current_frame.groupby('lot ID')['End Time (sec)'].max().to_dict()
        evaluation_cache = {tuple(current_lot_order): (float(_current_obj),)}; accepted_orders = {tuple(current_lot_order)}
        best_frame, current_quality = current_frame, (float(_current_obj),)
        best_quality = current_quality
        temperature = max(10.0, abs(seed_objective) * 0.0001)

        print(f"\n  [LNS - {iterations} iterations, deep block ruin-and-recreate]")

        for iteration_index in range(1, iterations + 1):
            if iteration_index - best_iteration > 15 and best_lot_order != current_lot_order: current_lot_order, current_quality, current_frame = list(best_lot_order), best_quality, best_frame; lot_end_times = current_frame.groupby('lot ID')['End Time (sec)'].max().to_dict()
            repair_candidates = []

            if data.actual_setup:
                # 1. Partition into contiguous product blocks
                blocks, curr_p, curr_b = [], None, []
                for l in current_lot_order:
                    p = data.product.get(l)
                    if p != curr_p:
                        if curr_b: blocks.append((curr_p, curr_b))
                        curr_p, curr_b = p, [l]
                    else: curr_b.append(l)
                if curr_b: blocks.append((curr_p, curr_b))

                # 2. Compute block weights combining weighted tardiness, changeover setup duration, and product fragmentation
                prod_counts = {p: sum(1 for bp, _ in blocks if bp == p) for p, _ in blocks}
                lot_setups = current_frame.groupby('lot ID')['Setup Time'].sum().to_dict() if 'Setup Time' in current_frame.columns else {}
                block_weights = [sum(data.Up.get(l, 1.0) * max(0.0, lot_end_times.get(l, 0.0) - delay_due[l]) + lot_setups.get(l, 0.0) for l in blk_lots) + 1800.0 * (prod_counts.get(_p, 1) - 1) + 1.0 for _p, blk_lots in blocks]

                total_bw = sum(block_weights)
                if len(blocks) > 1:
                    chosen_tardy_idx = random_generator.choices(range(len(blocks)), weights=[bw / total_bw for bw in block_weights], k=1)[0] if total_bw > 0 else random_generator.randint(0, len(blocks) - 1)

                    p_chosen, blk_chosen = blocks[chosen_tardy_idx]

                    # --- CANDIDATE 1: Targeted Block Forward Shift (Preserving internal lot sequence) ---
                    if chosen_tardy_idx > 0:
                        target_idx, shifted_blocks = random_generator.randint(0, chosen_tardy_idx - 1), list(blocks)
                        shifted_blocks.insert(target_idx, shifted_blocks.pop(chosen_tardy_idx))
                        repair_candidates.append([l for _, b in shifted_blocks for l in b])

                    # --- CANDIDATE 2: Strategic Block Swap (Swap chosen block with an earlier block) ---
                    swap_target, swapped_blocks = (random_generator.randint(0, chosen_tardy_idx - 1) if chosen_tardy_idx > 0 else random_generator.randint(1, len(blocks) - 1)), list(blocks)
                    swapped_blocks[chosen_tardy_idx], swapped_blocks[swap_target] = swapped_blocks[swap_target], swapped_blocks[chosen_tardy_idx]
                    repair_candidates.append([l for _, b in swapped_blocks for l in b])

                    # --- CANDIDATE 3: Product Family Consolidation (Setup & Idle Time Reduction) ---
                    frag_prods = [p for p, c in prod_counts.items() if c > 1]
                    p_target = p_chosen if prod_counts.get(p_chosen, 0) > 1 else (random_generator.choice(frag_prods) if frag_prods else None)
                    if p_target:
                        p_indices = [i for i, (p, _) in enumerate(blocks) if p == p_target]
                        if len(p_indices) >= 2:
                            s_idx, t_idx = random_generator.sample(p_indices, 2)
                            for adj_target in ([t_idx if t_idx < s_idx else t_idx - 1, (t_idx if t_idx < s_idx else t_idx - 1) + 1]):
                                consol_blocks = list(blocks); blk = consol_blocks.pop(s_idx)
                                consol_blocks.insert(max(0, min(len(consol_blocks), adj_target)), blk)
                                repair_candidates.append([l for _, b in consol_blocks for l in b])
                    elif chosen_tardy_idx > 1:
                        # Deep Leapfrog to mid-schedule
                        deep_idx, deep_blocks = chosen_tardy_idx // 2, list(blocks)
                        deep_blocks.insert(deep_idx, deep_blocks.pop(chosen_tardy_idx))
                        repair_candidates.append([l for _, b in deep_blocks for l in b])

                    # --- CANDIDATE 4: Targeted Intra-Block Tuning (Only for the active tardy block) ---
                    if len(blk_chosen) > 1:
                        tuned_blocks, tuned_lots = list(blocks), sorted(blk_chosen, key=lambda l: (data.Dp.get(l, 0), dispatch_ratio[l], str(l)))
                        tuned_blocks[chosen_tardy_idx] = (p_chosen, tuned_lots); repair_candidates.append([l for _, b in tuned_blocks for l in b])

            else:
                # No sequence setups: classic lot-level destroy and repair
                tardy_lots = [l for l in current_lot_order if lot_end_times.get(l, 0.0) > delay_due[l]]

                # Repair 1: Targeted forward advance of top tardy lot
                c1 = list(current_lot_order)
                if tardy_lots:
                    top_tardy = max(tardy_lots, key=lambda l: data.Up.get(l, 1.0) * (lot_end_times.get(l, 0.0) - delay_due[l]))
                    curr_idx = c1.index(top_tardy)
                    if curr_idx > 0:
                        c1.pop(curr_idx); c1.insert(random_generator.randint(0, curr_idx - 1), top_tardy)
                repair_candidates.append(c1)

                # Repair 2: Ruin 10% lots and reinsert by WSPT / Due Date
                destroy_cnt = min(len(current_lot_order), max(2, round(len(current_lot_order) * 0.1)))
                d_lots = list(random_generator.sample(current_lot_order, destroy_cnt))
                d_set = set(d_lots); rem_lots = [l for l in current_lot_order if l not in d_set]

                c2 = list(rem_lots)
                for l in sorted(d_lots, key=lambda x: (dispatch_ratio[x], data.Dp.get(x, 0))):
                    pos = next((i for i, item in enumerate(c2) if (dispatch_ratio[l]) < (dispatch_ratio[item])), len(c2))
                    c2.insert(pos, l)
                repair_candidates.append(c2)

            repair_candidates.append(self._ruin_repair(base_scheduler, current_lot_order, lot_end_times, random_generator, iteration_index - best_iteration, evaluation_cache))

            # Filter candidates: Only evaluate genuinely novel permutations!
            unique_candidates, seen_permutations = [], {tuple(current_lot_order)}
            for cand in repair_candidates:
                t_cand = tuple(cand)
                if t_cand not in seen_permutations:
                    seen_permutations.add(t_cand); unique_candidates.append(cand)

            if not unique_candidates:
                cand = list(current_lot_order)
                if len(cand) >= 2:
                    i1, i2 = random_generator.sample(range(len(cand)), 2); cand[i1], cand[i2] = cand[i2], cand[i1]
                unique_candidates.append(cand)

            # Evaluate only novel candidate proposals (fast: output=False)
            evaluated = []
            for cand in unique_candidates:
                order_key = tuple(cand); cand_quality = evaluation_cache.get(order_key)
                if cand_quality is None: cand_quality = evaluation_cache[order_key] = base_scheduler.schedule(cand)
                evaluated.append((cand_quality, cand))
            candidate_quality, candidate_order = min(evaluated, key=lambda item: (item[0], tuple(item[1]) in accepted_orders))
            improved = candidate_quality < best_quality

            cooling = temperature * (0.01 ** (iteration_index / iterations))
            accepted = candidate_quality <= current_quality or random_generator.random() < math.exp(min(0.0, (current_quality[0] - candidate_quality[0]) / cooling))
            if accepted and candidate_order != current_lot_order:
                current_quality, current_lot_order = candidate_quality, list(candidate_order); accepted_orders.add(tuple(current_lot_order))
                # Refresh lot completion times only when improved
                current_frame, _current_obj, _current_state = base_scheduler.schedule(current_lot_order, output=True)
                lot_end_times = current_frame.groupby('lot ID')['End Time (sec)'].max().to_dict()
                if improved: best_quality, best_lot_order, best_frame, update_count, best_iteration = current_quality, list(current_lot_order), current_frame, update_count + 1, iteration_index

            current_frame, current_quality, route_changed = self._route_step(base_scheduler, current_frame, current_lot_order, current_quality, random_generator, cooling)
            if route_changed:
                lot_end_times = current_frame.groupby('lot ID')['End Time (sec)'].max().to_dict()
                evaluation_cache, accepted_orders = {tuple(current_lot_order): current_quality}, {tuple(current_lot_order)}
                if current_quality < best_quality:
                    best_quality, best_lot_order, best_frame, update_count, best_iteration = current_quality, list(current_lot_order), current_frame, update_count + 1, iteration_index
                    improved = True

            print(f"  [LNS {iteration_index:>4}/{iterations}] candidate={candidate_quality[0]:.4f} best={best_quality[0]:.4f}{' <- NEW BEST' if improved else ''}")

        # Final schedule reconstruction
        final_frame, final_objective, route_updates = self._refine_routes(base_scheduler, best_frame, best_quality[0], best_lot_order)
        if route_updates: update_count, best_iteration = update_count + route_updates, iterations
        if float(final_objective) > float(seed_objective) + 1e-9: raise RuntimeError('LNS degraded its Greedy seed')

        search_duration = time.perf_counter() - search_start_time; total_time = greedy_elapsed_time + search_duration

        print(f"  LNS done: best={final_objective:.4f} | updates={update_count} | search={search_duration:.3f}s | total+Greedy={total_time:.3f}s")
        return final_frame, float(final_objective), best_lot_order, total_time, {'number_iterations': iterations, 'best_iteration': best_iteration, 'number_updates': update_count}


    def _ruin_repair(self, scheduler, order, end_times, rng, stagnation, cache):
        if len(order) < 2: return list(order)
        data, units, delay_due = self.data, [], self.delay_due
        group_products = data.actual_setup and rng.random() < 0.75
        for lot in order:
            if group_products and units and data.product.get(units[-1][0]) == data.product.get(lot): units[-1].append(lot)
            else: units.append([lot])
        count = min(len(units), 2 + min(2, stagnation // 10))
        weights = {index: sum(data.Up.get(lot, 1.0) * max(0.0, end_times.get(lot, 0.0) - delay_due[lot]) for lot in unit) for index, unit in enumerate(units)}
        floor = max(1.0, sum(weights.values()) / len(units)) * 0.1
        pool, removed = list(range(len(units))), []
        for _ in range(count):
            index = rng.choices(pool, weights=[weights[item] + floor for item in pool], k=1)[0]; pool.remove(index); removed.append(index)
        remaining = list(pool); rng.shuffle(removed)
        for index in removed:
            original = sum(item < index for item in remaining)
            positions = sorted({original, 0, len(remaining), rng.randrange(len(remaining) + 1), rng.randrange(len(remaining) + 1)})
            candidates = []
            for position in positions:
                candidate = remaining[:position] + [index] + remaining[position:]; key = tuple(lot for item in candidate for lot in units[item])
                if key not in cache: cache[key] = scheduler.schedule(key)
                candidates.append((cache[key], candidate))
            remaining = min(candidates, key=lambda item: item[0])[1]
        return [lot for index in remaining for lot in units[index]]

    @cached_property
    def delay_due(self):
        data, targets = self.data, {}
        for lot in data.P:
            earliest = float('inf')
            for option in ([data.fp[lot]] if data.fp.get(lot, 0) > 0 else data.Op[lot]):
                ready = {None: data.Rp.get(lot, 0.0)}
                for job, machines in data.greedy_state.option_meta[lot, option][0]:
                    ready = {machine: max(data.Bm.get(machine, 0.0), min(end + data.Emn.get((previous, machine), 0.0) for previous, end in ready.items())) + data.Tam[lot, option, job, machine] for machine in machines}
                earliest = min(earliest, min(ready.values()))
            targets[lot] = max(data.Dp.get(lot, 0.0), earliest)
        return targets

    def _polish(self, scheduler, frame, objective, order):
        data, order, updates, sweeps = self.data, list(order), 0, 0
        budget = max(32, min(512, int(float(self.config.get('iterations', 100))) * 4))
        cache, scheduler.cache_prefix = {tuple(order): (float(objective),)}, True
        scheduler.schedule(order, output=True)
        for sweep in range(2):
            ends = frame.groupby('lot ID')['End Time (sec)'].max().to_dict()
            setups = frame.groupby('lot ID')['Setup Time'].sum().to_dict() if data.actual_setup and 'Setup Time' in frame.columns else {}
            tardy_crit = sorted(order, key=lambda lot: (-data.Up.get(lot, 1.0) * max(0.0, ends[lot] - self.delay_due[lot]), -data.Up.get(lot, 1.0) * max(0.0, ends[lot] - data.Dp.get(lot, 0.0))))[:8]
            setup_crit = sorted(order, key=lambda lot: -setups.get(lot, 0.0))[:8] if data.actual_setup else []
            critical = list(dict.fromkeys(tardy_crit + setup_crit))
            changed, sweeps = False, sweep + 1
            for lot in critical:
                index = order.index(lot); remaining = order[:index] + order[index + 1:]
                machines = set(frame.loc[frame['lot ID'] == lot, 'Machine ID']); related = set(frame.loc[frame['Machine ID'].isin(machines), 'lot ID'])
                lot_prod = data.product.get(lot)
                same_prod_pos = {i for i, item in enumerate(remaining) if data.product.get(item) == lot_prod} if data.actual_setup else set()
                same_prod_pos |= {i + 1 for i in same_prod_pos}
                positions = sorted({0, len(remaining), max(0, index - 1), min(len(remaining), index + 1)} | {i for i, item in enumerate(remaining) if item in related} | same_prod_pos | {i * (len(order) - 1) // 15 for i in range(16)})
                best_quality, best_order = (float(objective),), order
                for position in positions:
                    candidate = remaining[:position] + [lot] + remaining[position:]; key = tuple(candidate)
                    if key not in cache:
                        if budget <= 0: break
                        cache[key] = scheduler.schedule(candidate); budget -= 1
                    if cache[key] < best_quality: best_quality, best_order = cache[key], candidate
                if best_quality[0] < objective:
                    order = best_order; frame, objective, _ = scheduler.schedule(order, output=True); updates += 1; changed = True
                if budget <= 0: break
            if not changed or budget <= 0: break
        return frame, float(objective), order, updates, sweeps

    def _refine_routes(self, scheduler, frame, objective, order):
        data, updates = self.data, 0
        scheduler.route_choices = dict(frame.attrs.get('route_choices', {}))
        ends = frame.groupby('lot ID')['End Time (sec)'].max().to_dict()
        critical = sorted(order, key=lambda lot: (-data.Up.get(lot, 1.0) * max(0.0, ends[lot] - self.delay_due[lot]), -data.Up.get(lot, 1.0) * max(0.0, ends[lot] - data.Dp.get(lot, 0.0))))[:16]
        tardy_machines = set(frame.loc[frame['lot ID'].isin(critical[:8]), 'Machine ID'])
        congesting_lots = list(frame.loc[frame['Machine ID'].isin(tardy_machines), 'lot ID'].unique())
        setup_lots = list(frame.loc[frame['Setup Time'] > 0, 'lot ID'].unique()) if data.actual_setup and 'Setup Time' in frame.columns else []
        search_lots = list(dict.fromkeys(critical[:8] + setup_lots + critical[8:] + congesting_lots))
        budget = max(96, len(search_lots) * 2)
        for lot in search_lots:
            candidates = self._route_candidates(frame, lot)
            best_choice, best_value = scheduler.route_choices.get(lot), objective
            for candidate in candidates:
                if budget <= 0: break
                scheduler.route_choices[lot] = candidate; value = scheduler.schedule(order)[0]; budget -= 1
                if value < best_value: best_value, best_choice = value, candidate
            if best_choice is None: scheduler.route_choices.pop(lot, None)
            else: scheduler.route_choices[lot] = best_choice
            if best_value < objective: frame, objective, _ = scheduler.schedule(order, output=True); updates += 1
            if budget <= 0: break
        return frame, float(objective), updates

    def _route_step(self, scheduler, frame, order, quality, rng, temperature):
        data = self.data
        ends = frame.groupby('lot ID')['End Time (sec)'].max().to_dict()
        lot_setups = frame.groupby('lot ID')['Setup Time'].sum().to_dict() if data.actual_setup and 'Setup Time' in frame.columns else {}
        delays = {lot: data.Up.get(lot, 1.0) * max(0.0, ends[lot] - self.delay_due[lot]) + lot_setups.get(lot, 0.0) for lot in order}
        machine_pressure = {}
        for lot, machine in frame[['lot ID', 'Machine ID']].itertuples(index=False, name=None): machine_pressure[machine] = machine_pressure.get(machine, 0.0) + delays[lot]
        assigned = frame.groupby('lot ID')['Machine ID'].agg(set).to_dict()
        floor = max(1.0, sum(delays.values()) / max(1, len(order))) * 0.1
        weights = [delays[lot] + 0.25 * sum(machine_pressure[machine] for machine in sorted(assigned[lot])) + floor for lot in order]
        lot = rng.choices(order, weights=weights, k=1)[0]
        candidates = self._route_candidates(frame, lot)
        previous = scheduler.route_choices.get(lot)
        if previous is not None: candidates.append(None)
        if not candidates: return frame, quality, False
        chosen = rng.choice(candidates)
        if chosen is None: scheduler.route_choices.pop(lot, None)
        else: scheduler.route_choices[lot] = chosen
        candidate_quality = scheduler.schedule(order)
        if candidate_quality <= quality or rng.random() < math.exp(min(0.0, (quality[0] - candidate_quality[0]) / temperature)):
            candidate_frame, value, _ = scheduler.schedule(order, output=True)
            return candidate_frame, (float(value),), True
        if previous is None: scheduler.route_choices.pop(lot, None)
        else: scheduler.route_choices[lot] = previous
        return frame, quality, False

    def _route_candidates(self, frame, lot):
        data = self.data
        rows = frame.loc[frame['lot ID'] == lot].sort_values('Operation Sequence'); option, path = rows['Option'].iloc[0], tuple(rows['Machine ID'])
        candidates = [(option, path[:position] + (machine,) + path[position + 1:]) for position, (_, machines) in enumerate(data.greedy_state.option_meta[lot, option][0]) for machine in machines if machine != path[position]]
        if not data.fp.get(lot, 0): candidates.extend((alternative, None) for alternative in data.Op[lot] if alternative != option and (lot, alternative) in data.greedy_state.option_meta)
        return candidates
