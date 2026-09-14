"""Launch every objective with one solver and shared runtime."""
import argparse
import os
import subprocess
import sys
import time

DATASET_NAME = {'1': 'small', '2': 'medium', '3': 'large'}
OBJECTIVE_TYPES = (
    'tardy_only',
    'tardy_move',
    'tardy_move_setup',
    'tardy_total_time',
    'tardy_move_makespan',
)


def _get_python_executable(custom=None, solver='cplex'):
    if custom and os.path.isfile(custom):
        return custom
    if solver == 'gurobi':
        candidates = [
            r"C:\Users\hkuser\AppData\Local\Programs\Python\Python312\python.exe",
            sys.executable,
            r"C:\Users\hkuser\anaconda3\python.exe",
            os.path.expanduser(r"~\anaconda3\python.exe"),
            r"C:\ProgramData\anaconda3\python.exe",
        ]
    else:
        candidates = [
            sys.executable,
            r"C:\Users\hkuser\anaconda3\python.exe",
            os.path.expanduser(r"~\anaconda3\python.exe"),
            r"C:\ProgramData\anaconda3\python.exe",
        ]
    required = 'numpy, pandas, gurobipy, docplex' if solver == 'gurobi' else 'numpy, pandas, docplex, cplex'
    for c in candidates:
        if c and os.path.isfile(c):
            try:
                res = subprocess.run([c, '-c', f'import {required}'], capture_output=True, timeout=5)
                if res.returncode == 0:
                    return c
            except Exception:
                pass
    return sys.executable


def _build_parser(default_solver='cplex', allow_solver_override=True):
    parser = argparse.ArgumentParser(description='Run every objective type through the shared scheduler.')
    parser.add_argument('--dataset', choices=list(DATASET_NAME), help='1=small, 2=medium, 3=large')
    parser.add_argument('--mode', choices=['mip', 'heuristic', 'both'], default='both')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-root')
    parser.add_argument('--stop-on-error', action='store_true')
    if allow_solver_override:
        parser.add_argument('--solver', choices=['cplex', 'gurobi'], default=default_solver)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--lns-destroy-pct', type=float)
    parser.add_argument('--route-limit', type=int)
    parser.add_argument('--input-root')
    parser.add_argument('--python-executable')
    return parser


def main(default_solver='cplex', allow_solver_override=True):
    parser = _build_parser(default_solver, allow_solver_override)
    args = parser.parse_args()
    solver = args.solver if allow_solver_override else default_solver
    root = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.abspath(args.input_root or os.path.join(root, 'input'))

    while not args.dataset:
        pick = input('\nSELECT DATASET: 1=Small  2=Medium  3=Large: ').strip()
        args.dataset = pick if pick in DATASET_NAME else None
        if not args.dataset:
            print('  Invalid choice.')

    python_exe = _get_python_executable(args.python_executable, solver=solver)
    child_env = os.environ.copy()
    child_env['PYTHONPATH'] = root

    dataset = DATASET_NAME[args.dataset]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    output_group = 'all_types' if solver == 'cplex' else 'all_types_gurobi'
    out_root = os.path.abspath(args.out_root or os.path.join(root, 'output', output_group, f'{dataset}_{args.mode}_{stamp}'))
    os.makedirs(out_root, exist_ok=True)

    main_file = os.path.join(root, 'main.py')
    parameter_file = os.path.join(root, 'parameter.csv')

    runs = []
    for type_key in OBJECTIVE_TYPES:
        type_out = os.path.join(out_root, type_key)
        print(f"\n{'=' * 80}\nRUN TYPE: {type_key} | dataset={dataset} | mode={args.mode} | solver={solver.upper()}\n{'=' * 80}", flush=True)
        started = time.perf_counter()
        command = [
            python_exe, main_file,
            '--objective', type_key,
            '--solver', solver,
            '--dataset', args.dataset,
            '--mode', args.mode,
            '--seed', str(args.seed),
            '--output', type_out,
            '--input-root', input_root,
            '--parameter-file', parameter_file,
        ]
        for flag, value in (('--iterations', args.iterations), ('--lns-destroy-pct', args.lns_destroy_pct), ('--route-limit', args.route_limit)):
            if value is not None:
                command.extend([flag, str(value)])

        code = int(subprocess.run(command, cwd=root, env=child_env).returncode)
        runs.append({'type': type_key, 'solver': solver, 'returncode': code, 'elapsed_sec': round(time.perf_counter() - started, 2)})
        if code != 0 and args.stop_on_error:
            break

    print(f"\n{'=' * 80}\nALL TYPES RUN COMPLETE: {len(runs)} objectives executed\nOutput: {out_root}\n{'=' * 80}\n")
    return max((r['returncode'] for r in runs), default=0)


if __name__ == '__main__':
    raise SystemExit(main())
