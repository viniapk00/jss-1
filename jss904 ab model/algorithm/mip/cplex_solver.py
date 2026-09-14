"""CPLEX solve policy shared by all MIP formulations."""
import time


def solve(mip):
    print("\n" + "=" * 50 + "\nSOLVING - MIP (CPLEX)\n" + "=" * 50)
    config = mip.data.config
    limit = float(config['time_limit_seconds'])
    mip.solve_time = mip.node_count = 0.0
    mip.best_bound = mip.obj_value = mip.mip_gap = None
    mip.solver_status = None

    mip.model.parameters.timelimit.set(limit)
    mip.model.parameters.mip.tolerances.mipgap.set(float(config['mip_gap']))
    mip.model.parameters.threads.set(int(config['cplex_threads']))
    mip.model.parameters.randomseed.set(int(config.get('solver_seed', 42)))

    t0 = time.monotonic()
    sol = mip.model.solve(log_output=True)
    solve_duration = time.monotonic() - t0

    details = getattr(mip.model, 'solve_details', None)
    status_str = getattr(details, 'status', None)
    nodes = getattr(details, 'nb_nodes_processed', 0) or 0

    if sol is None:
        mip.solution = None
        mip.solve_time = solve_duration
        mip.node_count = nodes
        mip.mip_gap = None
        mip.best_bound = getattr(details, 'best_bound', None)
        mip.solver_status = status_str
        print(f'  No solution found. Solver status: {status_str}')
        return False

    mip.solution = sol
    mip.solve_time = solve_duration
    mip.node_count = nodes
    mip.best_bound = getattr(details, 'best_bound', None)
    mip.obj_value = getattr(sol, 'objective_value', getattr(details, 'best_bound', None))
    mip.mip_gap = getattr(details, 'mip_relative_gap', None)
    mip.solver_status = status_str
    gap_pct = (mip.mip_gap * 100.0) if mip.mip_gap is not None else 0.0
    print(f"  CPLEX solved in {solve_duration:.2f}s | Nodes: {int(nodes)} | Gap: {gap_pct:.2f}%")
    return True
