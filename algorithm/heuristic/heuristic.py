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

        # Define the portfolio of deterministic dispatching strategies (strictly original 6 strategies)
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
        """Roulette Wheel Metaheuristic with Block-Level Tardy Selection and Intra-Block Dispatching.

        Enhancements:
          1. Block-Level Roulette: Clusters lot sequence into contiguous product blocks. Calculates
             weighted tardiness per block. Spins roulette wheel to select critical tardy blocks for forward shift/swap.
          2. Intra-Block Dispatching (EDD/SPT): Sorts lots inside each product family without incurring setup penalty.
          3. Caching of operation end-times: Evaluates full DataFrame only when solution improves.
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

        # Precompute lot processing & weights for fast sorting
        Dp_arr = base_scheduler.state.Dp
        Rp_arr = base_scheduler.state.Rp
        Up_arr = base_scheduler.state.Up
        Pp_arr = getattr(base_scheduler.state, 'Pp', None)
        lots_all = list(data.P)
        if Pp_arr is not None and len(Pp_arr) == len(lots_all):
            p_map = {l: (Dp_arr[i], Rp_arr[i], Up_arr[i], Pp_arr[i]) for i, l in enumerate(lots_all)}
        else:
            p_map = {l: (data.Dp.get(l, 0), data.Rp.get(l, 0), data.Up.get(l, 1.0), 1.0) for l in lots_all}

        search_start_time = time.perf_counter()
        update_count = best_iteration = 0

        # Initial schedule evaluation to get starting lot completion times
        current_frame, _current_obj, _current_state = base_scheduler.schedule(best_lot_order, output=True)
        last_ops = current_frame.sort_values('Operation Sequence').groupby('lot ID').last().reset_index()
        lot_end_times = dict(zip(last_ops['lot ID'], last_ops['End Time (sec)'].astype(float)))

        print(f"\n  [Roulette - {iterations} iterations, Block-Level Roulette + Intra-Block Dispatching]")
        for iteration_index in range(1, iterations + 1):
            candidates_to_test = []

            if data.actual_setup:
                # 1. Partition sequence into contiguous product blocks
                blocks = []
                curr_prod = None
                curr_block = []
                for l in best_lot_order:
                    p = data.product.get(l)
                    if p != curr_prod:
                        if curr_block:
                            blocks.append((curr_prod, curr_block))
                        curr_prod = p
                        curr_block = [l]
                    else:
                        curr_block.append(l)
                if curr_block:
                    blocks.append((curr_prod, curr_block))

                # 2. Compute weighted tardiness for each product block
                block_weights = []
                for _p, blk_lots in blocks:
                    bw = sum(
                        data.Up.get(l, 1.0) * max(0.0, lot_end_times.get(l, 0.0) - data.Dp.get(l, 0.0))
                        for l in blk_lots
                    )
                    block_weights.append(bw)

                total_bw = sum(block_weights)
                if len(blocks) > 2:
                    if total_bw > 0:
                        probs = [bw / total_bw for bw in block_weights]
                        chosen_idx = random_generator.choices(range(len(blocks)), weights=probs, k=1)[0]
                    else:
                        chosen_idx = random_generator.randint(0, len(blocks) - 1)

                    # Proposal A: Block Shift (advance chosen tardy block earlier)
                    if chosen_idx > 0:
                        target_idx = random_generator.randint(0, chosen_idx - 1)
                        shifted = list(blocks)
                        blk = shifted.pop(chosen_idx)
                        shifted.insert(target_idx, blk)
                        candidates_to_test.append([l for _, b in shifted for l in b])

                    # Proposal B: Block Swap (swap chosen block with an earlier block)
                    swap_target = random_generator.randint(0, chosen_idx - 1) if chosen_idx > 0 else random_generator.randint(1, len(blocks) - 1)
                    swapped = list(blocks)
                    swapped[chosen_idx], swapped[swap_target] = swapped[swap_target], swapped[chosen_idx]
                    candidates_to_test.append([l for _, b in swapped for l in b])

                # Proposal C: Intra-Block Dispatching (EDD & SPT within each family block - 0 setup penalty!)
                edd_blocks = []
                for p_id, blk_lots in blocks:
                    sorted_lots = sorted(blk_lots, key=lambda l: (data.Dp.get(l, 0), p_map[l][3] / max(0.1, p_map[l][2]), str(l)))
                    edd_blocks.append((p_id, sorted_lots))
                candidates_to_test.append([l for _, b in edd_blocks for l in b])

            else:
                # No sequence setups: Pure lot-level Roulette selection based on tardiness
                tardy_lots = [l for l in best_lot_order if lot_end_times.get(l, 0.0) > data.Dp.get(l, 0.0)]
                if tardy_lots:
                    tw = [data.Up.get(l, 1.0) * (lot_end_times.get(l, 0.0) - data.Dp.get(l, 0.0)) for l in tardy_lots]
                    tw_sum = sum(tw)
                    t_probs = [w / tw_sum for w in tw]
                    chosen_lot = random_generator.choices(tardy_lots, weights=t_probs, k=1)[0]
                    curr_pos = best_lot_order.index(chosen_lot)
                    if curr_pos > 0:
                        step = random_generator.randint(1, min(20, curr_pos))
                        cand = list(best_lot_order)
                        cand.pop(curr_pos)
                        cand.insert(curr_pos - step, chosen_lot)
                        candidates_to_test.append(cand)

            # Fallback if no candidate generated
            if not candidates_to_test:
                cand = list(best_lot_order)
                if len(cand) >= 2:
                    i1, i2 = random_generator.sample(range(len(cand)), 2)
                    cand[i1], cand[i2] = cand[i2], cand[i1]
                candidates_to_test.append(cand)

            # Evaluate candidate proposals (fast: output=False)
            best_iter_quality = None
            best_iter_order = None
            for cand_order in candidates_to_test:
                cand_quality = base_scheduler.schedule(cand_order)
                if best_iter_quality is None or cand_quality < best_iter_quality:
                    best_iter_quality = cand_quality
                    best_iter_order = cand_order

            improved = best_iter_quality is not None and best_iter_quality < best_quality
            if improved:
                best_quality, best_lot_order = best_iter_quality, best_iter_order
                update_count += 1
                best_iteration = iteration_index
                # Refresh lot completion times for next iteration's roulette
                current_frame, _current_obj, _current_state = base_scheduler.schedule(best_lot_order, output=True)
                last_ops = current_frame.sort_values('Operation Sequence').groupby('lot ID').last().reset_index()
                lot_end_times = dict(zip(last_ops['lot ID'], last_ops['End Time (sec)'].astype(float)))

            eval_score = best_iter_quality[0] if best_iter_quality is not None else best_quality[0]
            print(f"  [Roulette {iteration_index:>4}/{iterations}] candidate={eval_score:.4f} best={best_quality[0]:.4f}{' <- NEW BEST' if improved else ''}")

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
        """Large Neighborhood Search (LNS) with deep block ruin-and-recreate and zero-overhead evaluation.

        Enhancements:
          1. Targeted Block Forward Shift: Advances tardy product blocks earlier while preserving lot sequences.
          2. Product Family Consolidation: Consolidates scattered product blocks to eliminate setup overhead.
          3. Strategic Block Swap: Swaps critical tardy blocks across bottlenecks preserving internal orders.
          4. Targeted Leapfrog: Explores deep fractional jumps across bottlenecks.
          5. Zero redundant evaluations: Deduplicates candidate sequences to evaluate only novel permutations.
        """
        data = self.data
        base_scheduler = BaseScheduler(
            data, data.greedy_state,
            getattr(self, '_machine_preference', None) or None
        )
        _discarded_seed_frame, seed_objective, seed_lot_order, greedy_elapsed_time = seed_solution
        best_lot_order, best_quality = list(seed_lot_order), (float(seed_objective),)

        iterations = max(1, int(float(self.config['iterations'])))
        random_generator = random.Random(int(self.config.get('solver_seed', 42)) + 100003)

        # Precompute lot processing & weights for fast sorting
        Dp_arr = base_scheduler.state.Dp
        Rp_arr = base_scheduler.state.Rp
        Up_arr = base_scheduler.state.Up
        Pp_arr = getattr(base_scheduler.state, 'Pp', None)
        lots_all = list(data.P)
        if Pp_arr is not None and len(Pp_arr) == len(lots_all):
            p_map = {l: (Dp_arr[i], Rp_arr[i], Up_arr[i], Pp_arr[i]) for i, l in enumerate(lots_all)}
        else:
            p_map = {l: (data.Dp.get(l, 0), data.Rp.get(l, 0), data.Up.get(l, 1.0), 1.0) for l in lots_all}

        search_start_time = time.perf_counter()
        update_count = best_iteration = 0
        stagnant_iterations = 0

        # Initial schedule evaluation to get starting lot completion times
        current_frame, _current_obj, _current_state = base_scheduler.schedule(best_lot_order, output=True)
        last_ops = current_frame.sort_values('Operation Sequence').groupby('lot ID').last().reset_index()
        lot_end_times = dict(zip(last_ops['lot ID'], last_ops['End Time (sec)'].astype(float)))

        print(f"\n  [LNS - {iterations} iterations, deep block ruin-and-recreate]")

        for iteration_index in range(1, iterations + 1):
            repair_candidates = []

            if data.actual_setup:
                # 1. Partition into contiguous product blocks
                blocks = []
                curr_p = None
                curr_b = []
                for l in best_lot_order:
                    p = data.product.get(l)
                    if p != curr_p:
                        if curr_b:
                            blocks.append((curr_p, curr_b))
                        curr_p = p
                        curr_b = [l]
                    else:
                        curr_b.append(l)
                if curr_b:
                    blocks.append((curr_p, curr_b))

                # 2. Compute block tardiness weights
                block_weights = []
                for _p, blk_lots in blocks:
                    bw = sum(
                        data.Up.get(l, 1.0) * max(0.0, lot_end_times.get(l, 0.0) - data.Dp.get(l, 0.0))
                        for l in blk_lots
                    )
                    block_weights.append(bw)

                total_bw = sum(block_weights)
                if len(blocks) > 2:
                    if total_bw > 0:
                        probs = [bw / total_bw for bw in block_weights]
                        chosen_tardy_idx = random_generator.choices(range(len(blocks)), weights=probs, k=1)[0]
                    else:
                        chosen_tardy_idx = random_generator.randint(0, len(blocks) - 1)

                    p_chosen, blk_chosen = blocks[chosen_tardy_idx]

                    # --- CANDIDATE 1: Targeted Block Forward Shift (Preserving internal lot sequence) ---
                    if chosen_tardy_idx > 0:
                        target_idx = random_generator.randint(0, chosen_tardy_idx - 1)
                        shifted_blocks = list(blocks)
                        blk = shifted_blocks.pop(chosen_tardy_idx)
                        shifted_blocks.insert(target_idx, blk)
                        repair_candidates.append([l for _, b in shifted_blocks for l in b])

                    # --- CANDIDATE 2: Strategic Block Swap (Swap chosen block with an earlier block) ---
                    swap_target = random_generator.randint(0, chosen_tardy_idx - 1) if chosen_tardy_idx > 0 else random_generator.randint(1, len(blocks) - 1)
                    swapped_blocks = list(blocks)
                    swapped_blocks[chosen_tardy_idx], swapped_blocks[swap_target] = swapped_blocks[swap_target], swapped_blocks[chosen_tardy_idx]
                    repair_candidates.append([l for _, b in swapped_blocks for l in b])

                    # --- CANDIDATE 3: Product Family Consolidation (Setup & Idle Time Reduction) ---
                    same_prod_indices = [i for i, (p, _) in enumerate(blocks) if p == p_chosen and i != chosen_tardy_idx]
                    if same_prod_indices:
                        target_same = random_generator.choice(same_prod_indices)
                        consol_blocks = list(blocks)
                        blk = consol_blocks.pop(chosen_tardy_idx)
                        adj_target = target_same if target_same < chosen_tardy_idx else target_same - 1
                        ins_pos = adj_target if random_generator.random() < 0.5 else adj_target + 1
                        ins_pos = max(0, min(len(consol_blocks), ins_pos))
                        consol_blocks.insert(ins_pos, blk)
                        repair_candidates.append([l for _, b in consol_blocks for l in b])
                    elif chosen_tardy_idx > 1:
                        # Deep Leapfrog to mid-schedule
                        deep_idx = chosen_tardy_idx // 2
                        deep_blocks = list(blocks)
                        blk = deep_blocks.pop(chosen_tardy_idx)
                        deep_blocks.insert(deep_idx, blk)
                        repair_candidates.append([l for _, b in deep_blocks for l in b])

                    # --- CANDIDATE 4: Targeted Intra-Block Tuning (Only for the active tardy block) ---
                    if len(blk_chosen) > 1:
                        tuned_blocks = list(blocks)
                        tuned_lots = sorted(blk_chosen, key=lambda l: (data.Dp.get(l, 0), p_map[l][3] / max(0.1, p_map[l][2]), str(l)))
                        tuned_blocks[chosen_tardy_idx] = (p_chosen, tuned_lots)
                        repair_candidates.append([l for _, b in tuned_blocks for l in b])

            else:
                # No sequence setups: classic lot-level destroy and repair
                tardy_lots = [l for l in best_lot_order if lot_end_times.get(l, 0.0) > data.Dp.get(l, 0.0)]
                t_weights = [1.0 + data.Up.get(l, 1.0) * max(0.0, lot_end_times.get(l, 0.0) - data.Dp.get(l, 0.0)) for l in best_lot_order]
                t_probs = np.array(t_weights, dtype=float)
                t_probs /= t_probs.sum()

                # Repair 1: Targeted forward advance of top tardy lot
                c1 = list(best_lot_order)
                if tardy_lots:
                    top_tardy = max(tardy_lots, key=lambda l: data.Up.get(l, 1.0) * (lot_end_times.get(l, 0.0) - data.Dp.get(l, 0.0)))
                    curr_idx = c1.index(top_tardy)
                    if curr_idx > 0:
                        c1.pop(curr_idx)
                        c1.insert(random_generator.randint(0, curr_idx - 1), top_tardy)
                repair_candidates.append(c1)

                # Repair 2: Ruin 10% lots and reinsert by WSPT / Due Date
                destroy_cnt = min(len(best_lot_order), max(2, round(len(best_lot_order) * 0.1)))
                d_lots = list(random_generator.sample(best_lot_order, destroy_cnt))
                d_set = set(d_lots)
                rem_lots = [l for l in best_lot_order if l not in d_set]

                c2 = list(rem_lots)
                for l in sorted(d_lots, key=lambda x: (p_map[x][3] / max(0.1, p_map[x][2]), data.Dp.get(x, 0))):
                    pos = next((i for i, item in enumerate(c2) if (p_map[l][3] / max(0.1, p_map[l][2])) < (p_map[item][3] / max(0.1, p_map[item][2]))), len(c2))
                    c2.insert(pos, l)
                repair_candidates.append(c2)

            # Filter candidates: Only evaluate genuinely novel permutations!
            unique_candidates = []
            seen_permutations = {tuple(best_lot_order)}
            for cand in repair_candidates:
                t_cand = tuple(cand)
                if t_cand not in seen_permutations:
                    seen_permutations.add(t_cand)
                    unique_candidates.append(cand)

            if not unique_candidates:
                cand = list(best_lot_order)
                if len(cand) >= 2:
                    i1, i2 = random_generator.sample(range(len(cand)), 2)
                    cand[i1], cand[i2] = cand[i2], cand[i1]
                unique_candidates.append(cand)

            # Evaluate only novel candidate proposals (fast: output=False)
            evaluated = [(base_scheduler.schedule(cand), cand) for cand in unique_candidates]
            candidate_quality, candidate_order = min(evaluated, key=lambda item: item[0])
            improved = candidate_quality < best_quality

            if improved:
                best_quality, best_lot_order = candidate_quality, list(candidate_order)
                update_count += 1
                best_iteration = iteration_index
                stagnant_iterations = 0
                # Refresh lot completion times only when improved
                current_frame, _current_obj, _current_state = base_scheduler.schedule(best_lot_order, output=True)
                last_ops = current_frame.sort_values('Operation Sequence').groupby('lot ID').last().reset_index()
                lot_end_times = dict(zip(last_ops['lot ID'], last_ops['End Time (sec)'].astype(float)))
            else:
                stagnant_iterations += 1

            print(f"  [LNS {iteration_index:>4}/{iterations}] candidate={candidate_quality[0]:.4f} best={best_quality[0]:.4f}{' <- NEW BEST' if improved else ''}")

        # Final schedule reconstruction
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

