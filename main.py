"""Root executor shared by all five scheduling objectives."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import time
import traceback
from typing import Optional, Sequence

import numpy as np

from utils.preprocessing import (
    ConfigLoader, DataPreprocessor, OBJECTIVES, load_objective_class,
)
from utils.result_saver import (
    assert_schedule_feasible,
    compare_results,
    save_heuristic_results,
    save_mip_result,
    schedule_metrics,
    write_run_manifest,
)


MODE_MAP = {'1': 'mip', '2': 'heuristic', '3': 'both', 'mip': 'mip', 'heuristic': 'heuristic', 'both': 'both'}
DATASET_CHOICES = {'1': '1', '2': '2', '3': '3', 'small': '1', 'medium': '2', 'large': '3'}
PROJECT_ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run one scheduling objective')
    parser.add_argument(
        '--objective', '--package', dest='objective', required=True,
        choices=list(OBJECTIVES), help='objective model to execute'
    )
    parser.add_argument('--solver', choices=['cplex', 'gurobi'], default='cplex')
    parser.add_argument('--dataset', choices=sorted(DATASET_CHOICES.keys()), help='1=small, 2=medium, 3=large')
    parser.add_argument('--mode', choices=['mip', 'heuristic', 'both'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output')
    parser.add_argument('--input-root')
    parser.add_argument('--parameter-file')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--lns-destroy-pct', type=float)
    parser.add_argument('--route-limit', type=int)
    parser.add_argument('--time-limit', type=float)
    parser.add_argument('--threads', type=int)
    return parser


def _interactive_choice(value, prompt, choices):
    if value and (value in choices or value in choices.values()):
        return choices.get(value, value)
    while not value:
        user_input = input(prompt).strip().lower()
        value = choices.get(user_input)
        if not value:
            print('  Invalid choice.')
    return value


def _configure(args):
    parameter_path = Path(args.parameter_file).resolve() if args.parameter_file else PROJECT_ROOT / 'parameter.csv'
    config = ConfigLoader.load(parameter_path, args.objective)
    if not config:
        return None, None

    if args.output:
        config['output_dir'] = os.path.abspath(args.output)
    if args.input_root:
        config['input_dir'] = os.path.abspath(args.input_root)
    config['_project_root'] = str(PROJECT_ROOT)
    config['solver_seed'] = args.seed

    if args.iterations is not None:
        config['iterations'] = max(1, args.iterations)
    if args.lns_destroy_pct is not None:
        config['lns_destroy_pct'] = max(0.1, min(100.0, args.lns_destroy_pct))
    if args.route_limit is not None:
        config['greedy_route_limit'] = max(1, args.route_limit)
    if args.time_limit is not None:
        if args.time_limit <= 0:
            raise ValueError('--time-limit must be positive')
        config['time_limit_seconds'] = float(args.time_limit)
    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError('--threads must be positive')
        config[f'{args.solver}_threads'] = int(args.threads)
    return config, args.objective


def _solver_model(objective_key: str, solver: str):
    mip_class = load_objective_class(objective_key, 'mip')
    if solver == 'gurobi':
        from algorithm.mip.gurobi_solver import model_for
        return model_for(mip_class)
    return mip_class


def _run_mip(data, output_dir: str, solver: str, objective_key: str):
    started = time.perf_counter()
    try:
        model = _solver_model(objective_key, solver)(data)
        model.build()
        if not model.solve():
            print(f'  {solver.upper()} finished without solution.')
            return None
        model.extract_results()
        model.total_pipeline_time = time.perf_counter() - started
        return save_mip_result(data.time, data, model, output_dir, solver_name=solver)
    except Exception as exc:
        traceback.print_exc()
        print(f'\n  {solver.upper()} failed - {exc}')
        return None


def _execute(args) -> int:
    args.mode = _interactive_choice(args.mode, '\nSELECT MODE: 1=MIP  2=Heuristic  3=Both: ', MODE_MAP)
    args.dataset = _interactive_choice(
        args.dataset, '\nSELECT DATASET: 1=Small  2=Medium  3=Large: ', {key: key for key in DATASET_CHOICES}
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        config, objective_key = _configure(args)
    except (OSError, ValueError) as exc:
        print(f'  Configuration failed - {exc}')
        return 2

    if not config or objective_key is None:
        return 2

    print(
        f"  Effective config | objective={objective_key} | solver={args.solver} | "
        f"mode={args.mode} | iters={config['iterations']} | seed={args.seed}"
    )

    data = DataPreprocessor(config)
    if not data.run_pipeline(args.dataset):
        return 2

    dataset_dir = os.path.join(config['output_dir'], data.dataset_size)
    os.makedirs(dataset_dir, exist_ok=True)
    write_run_manifest(dataset_dir, config, data, args.seed, args.mode, args.solver)

    results, metas, metrics_cache = {}, {}, {}
    mip_gap = None

    # 1. MIP execution
    if args.mode in ('mip', 'both'):
        print(f"\n{'=' * 60}\n  MIP ({args.solver.upper()})\n{'=' * 60}")
        mip_result = _run_mip(
            data, os.path.join(dataset_dir, 'mip'), args.solver, objective_key
        )
        if mip_result:
            frame, mip_objective, elapsed, mip_gap = mip_result
            results['mip'] = (frame, mip_objective, elapsed)

    # 2. Heuristic execution (Greedy, Roulette, LNS)
    if args.mode in ('heuristic', 'both'):
        print(f"\n{'=' * 60}\n  HEURISTIC SCHEDULING\n{'=' * 60}")
        scheduler = load_objective_class(objective_key, 'heuristic')(config, data)

        # Best Greedy (reuse if cached from MIP warm start)
        if hasattr(data, 'greedy_seed') and data.greedy_seed is not None:
            heuristic_results = getattr(data, 'heuristic_results', {})
            greedy_seed = data.greedy_seed
            heuristic_metas = getattr(data, 'heuristic_metas', {})
            frame, reported_obj, order, elapsed = greedy_seed
        else:
            heuristic_results, greedy_seed, heuristic_metas = scheduler.run_greedy()
            frame, reported_obj, order, elapsed = greedy_seed
            assert_schedule_feasible(frame, data)
            metrics = schedule_metrics(frame, data)
            greedy_seed = (frame, float(metrics['objective']), order, elapsed)
            heuristic_results['best_greedy'] = (frame, float(metrics['objective']), elapsed)

        # Roulette Wheel
        roulette_frame, roulette_obj, _, roulette_elapsed, roulette_meta = scheduler.run_roulette(greedy_seed)
        heuristic_results['roulette'] = (roulette_frame, roulette_obj, roulette_elapsed)
        heuristic_metas['roulette'] = roulette_meta

        # LNS
        lns_frame, lns_obj, _, lns_elapsed, lns_meta = scheduler.run_lns(greedy_seed)
        heuristic_results['lns'] = (lns_frame, lns_obj, lns_elapsed)
        heuristic_metas['lns'] = lns_meta

        # Save heuristic results
        saved, saved_metas = save_heuristic_results(
            data.time, data, heuristic_results, heuristic_metas, dataset_dir, metrics_cache=metrics_cache
        )
        results.update(saved)
        metas.update(saved_metas)

    # 3. Comparison & summary table
    if results:
        compare_results(
            data.time, data, results, os.path.join(dataset_dir, 'compare'),
            mip_gap=mip_gap, metas=metas, metrics_cache=metrics_cache,
            mip_best_bound=getattr(data, 'mip_best_bound', None)
        )

    print(f'  Output: {dataset_dir}')
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _execute(_parser().parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
