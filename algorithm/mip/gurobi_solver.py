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
            is_cloud = bool(
                os.environ.get('STREAMLIT_SERVER_PORT')
                or os.path.exists('/mount/src')
                or os.environ.get('IS_STREAMLIT_CLOUD')
            )
            threads = int(config.get('gurobi_threads', 8))
            if is_cloud:
                # Streamlit Community Cloud has a hard 3 GB RAM limit.
                # Dual Simplex (Method=1) and capping threads at 2 avoids the 4-8 GB Barrier Cholesky OOM kill.
                threads = min(threads, 2)
                model.Params.Method = 1
                model.Params.NodefileStart = 0.5

            model.Params.TimeLimit = float(config['time_limit_seconds'])
            model.Params.MIPGap = 0.0
            model.Params.Threads = threads
            model.Params.Seed = int(config.get('solver_seed', 42))
            model.Params.OutputFlag = 1
            model.Params.MIPFocus = 1

            # Inject warm start from Greedy ATC
            self._apply_warm_start(model)

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

    def _apply_warm_start(self, gmodel):
        """Inject a high-quality initial feasible solution from Greedy ATC into Gurobi as a MIP start."""
        frame = getattr(self.data, 'greedy_schedule_frame', None)
        if frame is None:
            try:
                from utils.preprocessing import load_objective_class
                scheduler_cls = load_objective_class(self.objective_type, 'heuristic')
                scheduler = scheduler_cls(self.data.config, self.data)
                heuristic_results, greedy_seed, heuristic_metas = scheduler.run_greedy()
                frame, reported_obj, order, elapsed = greedy_seed
                self.data.greedy_schedule_frame = frame
                self.data.greedy_seed = greedy_seed
                self.data.heuristic_results = heuristic_results
                self.data.heuristic_metas = heuristic_metas
            except Exception as exc:
                print(f"  [Warm Start] Note: Could not generate greedy warm start: {exc}")
                return

        if frame is None or frame.empty:
            return

        try:
            start_map = {}
            # Route options
            selected_opts = {row['lot ID']: row['Option'] for _, row in frame.iterrows()}
            for p in self.data.P:
                sel_o = selected_opts.get(p)
                for o in self.data.Op[p]:
                    start_map[f'w_{p}_{o}'] = 1.0 if o == sel_o else 0.0

            # Operations and machine assignments
            m_jobs = {m: [] for m in self.data.M}
            for _, row in frame.iterrows():
                p = row['lot ID']
                o = row['Option']
                i = row['Operation Sequence']
                m = row['Machine ID']
                st = float(row['Start Time (sec)'])
                mtag = str(m).replace('-', '_')
                start_map[f'x_{p}_{o}_{i}_{mtag}'] = 1.0
                start_map[f't_{p}_{o}_{i}'] = st
                m_jobs[m].append((st, (p, o, i), mtag))

            # Machine sequencing
            for m, jobs in m_jobs.items():
                if not jobs:
                    continue
                jobs.sort(key=lambda x: x[0])
                # Immediate depot arcs
                first_p, first_o, first_i = jobs[0][1]
                start_map[f'd_plus_{first_p}_{first_o}_{first_i}_{jobs[0][2]}'] = 1.0
                last_p, last_o, last_i = jobs[-1][1]
                start_map[f'd_minus_{last_p}_{last_o}_{last_i}_{jobs[-1][2]}'] = 1.0

                # Immediate y arcs
                for idx in range(len(jobs) - 1):
                    (p1, o1, i1), mtag1 = jobs[idx][1], jobs[idx][2]
                    (p2, o2, i2) = jobs[idx + 1][1]
                    start_map[f'y_{p1}_{o1}_{i1}_{p2}_{o2}_{i2}_{mtag1}'] = 1.0

                # Pairwise y arcs
                for u in range(len(jobs)):
                    for v in range(u + 1, len(jobs)):
                        (p1, o1, i1), mtag1 = jobs[u][1], jobs[u][2]
                        (p2, o2, i2) = jobs[v][1]
                        start_map[f'y_{p1}_{o1}_{i1}_{p2}_{o2}_{i2}_{mtag1}'] = 1.0
                        start_map[f'y_{p2}_{o2}_{i2}_{p1}_{o1}_{i1}_{mtag1}'] = 0.0

            matched = 0
            for v in gmodel.getVars():
                if v.VarName in start_map:
                    v.Start = start_map[v.VarName]
                    matched += 1
            if matched > 0:
                print(f"  MIP warm start injected ({matched} variables initialized from Greedy ATC).")
        except Exception as exc:
            print(f"  [Warm Start] Note: Failed to inject warm start into Gurobi: {exc}")


def model_for(mip_model_class):
    return type(f'Gurobi{mip_model_class.__name__}', (GurobiMixin, mip_model_class), {'__module__': __name__})
