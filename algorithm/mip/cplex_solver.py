"""CPLEX solve policy shared by all MIP formulations."""
import time


def solve(mip):
    print("\n" + "=" * 50 + "\nSOLVING - MIP (CPLEX)\n" + "=" * 50)
    config, limit = mip.data.config, float(mip.data.config['time_limit_seconds'])
    mip.solve_time, mip.node_count = 0.0, 0.0; mip.best_bound, mip.obj_value, mip.mip_gap, mip.solver_status = None, None, None, None

    mip.model.parameters.timelimit.set(limit); mip.model.parameters.mip.tolerances.mipgap.set(float(config['mip_gap'])); mip.model.parameters.threads.set(int(config['cplex_threads'])); mip.model.parameters.randomseed.set(int(config.get('solver_seed', 42)))

    t0 = time.monotonic(); sol = mip.model.solve(log_output=True); solve_duration = time.monotonic() - t0

    details = getattr(mip.model, 'solve_details', None); status_str, nodes = getattr(details, 'status', None), (getattr(details, 'nb_nodes_processed', 0) or 0)

    if sol is None:
        mip.solution, mip.solve_time, mip.node_count, mip.mip_gap, mip.best_bound, mip.solver_status = None, solve_duration, nodes, None, getattr(details, 'best_bound', None), status_str
        print(f'  No solution found. Solver status: {status_str}')
        return False

    mip.solution, mip.solve_time, mip.node_count, mip.best_bound, mip.obj_value, mip.mip_gap, mip.solver_status = sol, solve_duration, nodes, getattr(details, 'best_bound', None), getattr(sol, 'objective_value', getattr(details, 'best_bound', None)), getattr(details, 'mip_relative_gap', None), status_str
    gap_pct = (mip.mip_gap * 100.0) if mip.mip_gap is not None else 0.0
    print(f"  CPLEX solved in {solve_duration:.2f}s | Nodes: {int(nodes)} | Gap: {gap_pct:.2f}%")
    return True

