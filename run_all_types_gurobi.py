"""Launch every objective with Gurobi optimizer."""
from run_all_types import main

if __name__ == '__main__': raise SystemExit(main(default_solver='gurobi', allow_solver_override=False))