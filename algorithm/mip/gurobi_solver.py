"""Gurobi solve policy shared by all MIP formulations."""
import os
import time


class GurobiMixin:
    def solve(self):
        try:
            import gurobipy as gp
        except ImportError as exc:
            raise RuntimeError('gurobipy is not installed or its license is unavailable.') from exc

        config = self.data.config
        output = os.path.join(config['output_dir'], self.data.dataset_size, 'mip')
        os.makedirs(output, exist_ok=True)
        lp_path = os.path.join(output, f'model_{self.data.dataset_size}_gurobi.lp')

        print("\n" + "=" * 50 + "\nSOLVING - MIP (GUROBI)\n" + "=" * 50)
        self.model.export_as_lp(lp_path)
        try:
            model = gp.read(lp_path)
        finally:
            if os.path.exists(lp_path):
                try:
                    os.remove(lp_path)
                except Exception:
                    pass

        try:
            model.Params.TimeLimit = float(config['time_limit_seconds'])
            model.Params.MIPGap = 0.0
            model.Params.Threads = int(config['gurobi_threads'])
            model.Params.Seed = int(config.get('solver_seed', 42))
            model.Params.OutputFlag = 1

            t0 = time.monotonic()
            model.optimize()
            solve_duration = time.monotonic() - t0

            self.solve_time = solve_duration
            self.node_count = float(model.NodeCount)
            self.solver_status = int(model.Status)
            self.solver_status_name = (
                'OPTIMAL' if self.solver_status == gp.GRB.OPTIMAL
                else 'TIME_LIMIT' if self.solver_status == gp.GRB.TIME_LIMIT
                else str(self.solver_status)
            )
            self.best_bound = float(model.ObjBound) if model.NumVars > 0 else None

            if int(model.SolCount) == 0:
                self.solution = self.obj_value = self.mip_gap = None
                print(f'\n  No feasible Gurobi solution found. Status={self.solver_status_name}')
                return False

            self.obj_value = float(model.ObjVal)
            self.mip_gap = float(model.MIPGap) if model.IsMIP else 0.0
            values = {v.VarName: v.X for v in model.getVars()}

            solution = self.model.new_solution()
            for variable in self.model.iter_variables():
                variable_name = variable.name
                variable_value = values.get(variable_name)
                if variable_value is None and variable_name:
                    variable_value = values.get(variable_name.replace('-', 'm'))
                if variable_value is None and variable_name:
                    variable_value = values.get(variable_name.replace('-', '_'))
                if variable_value is None:
                    variable_value = values.get(getattr(variable, 'lp_name', None))
                if variable_value is not None:
                    solution.add_var_value(variable, variable_value)
            self.model._set_solution(solution)
            self.solution = solution

            gap_pct = self.mip_gap * 100.0
            print(f"\n  Gurobi solved in {solve_duration:.2f}s | Nodes: {int(self.node_count)} | Gap: {gap_pct:.2f}%")
            return True
        finally:
            model.dispose()


def model_for(mip_model_class):
    return type(f'Gurobi{mip_model_class.__name__}', (GurobiMixin, mip_model_class), {'__module__': __name__})
