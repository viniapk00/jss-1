"""Weighted tardiness, movement, and actual setup FJSP MIP with immediate machine sequencing."""
from docplex.mp.model import Model
from algorithm.mip.cplex_solver import solve as solve_cplex
from utils.result_saver import extract_mip_schedule
from utils.preprocessing import seconds_per_day


class TardyMoveSetupModel:
    objective_type = 'tardy_move_setup'

    def __init__(self, data):
        if data.objective_type != self.objective_type: raise ValueError(f'{type(self).__name__} requires objective_type={self.objective_type!r}')
        self.data, self.model, self.solution, self.solve_time = data, None, None, 0.0
        self.best_bound, self.node_count, self.mip_gap, self.obj_value = None, None, None, None
        self.proven_optimal, self.machines_schedule = False, {}
        self.w, self.x, self.t, self.y = {}, {}, {}, {}
        self.d_plus, self.d_minus = {}, {}
        self.q, self.C, self.z = {}, {}, {}

    def add_variables(self):
        D, H, m = self.data, self.data.H, self.model

        # w_p,o in {0, 1} : 1 if lot p selects route option o
        for p in D.P:
            for o in D.Op[p]: self.w[p, o] = m.binary_var(name=f'w_{p}_{o}')

        for a in D.A:
            p, o, i = a
            # x_a,m in {0, 1} : 1 if operation a is processed on machine m
            for mach in D.Ma[a]:
                mtag = str(mach).replace('-', '_')
                self.x[a, mach] = m.binary_var(name=f'x_{p}_{o}_{i}_{mtag}')
                # d^+_a,m in [0, 1] : 1 if operation a has an incoming arc from the depot on machine m
                # d^-_a,m in [0, 1] : 1 if operation a has an outgoing arc to the depot on machine m
                self.d_plus[a, mach] = m.continuous_var(lb=0.0, ub=1.0, name=f'd_plus_{p}_{o}_{i}_{mtag}')
                self.d_minus[a, mach] = m.continuous_var(lb=0.0, ub=1.0, name=f'd_minus_{p}_{o}_{i}_{mtag}')
            # t_a >= 0 : start time of operation a
            self.t[a] = m.continuous_var(lb=0.0, ub=H, name=f't_{p}_{o}_{i}')

        # q_a,m,n in [0, 1] : 1 if operation a uses machine m and successor a' uses machine n
        for a, a_prime in D.Ar:
            p, o, i = a
            for mach in D.Ma[a]:
                mtag = str(mach).replace('-', '_')
                for n in D.Ma[a_prime]: self.q[a, mach, n] = m.continuous_var(lb=0.0, ub=1.0, name=f'q_{p}_{o}_{i}_{mtag}_{str(n).replace("-", "_")}')

        # y_a,b,m in {0, 1} : 1 if operation a immediately precedes operation b on machine m
        for a, b, mach in D.immediate_candidates:
            self.y[a, b, mach] = m.binary_var(name=f'y_{a[0]}_{a[1]}_{a[2]}_{b[0]}_{b[1]}_{b[2]}_{str(mach).replace("-", "_")}')

        for p in D.P:
            # c_p >= 0 : completion time of the last operation of lot p
            self.C[p] = m.continuous_var(lb=0.0, ub=H, name=f'C_{p}')
            # z_p >= 0 : tardiness of lot p
            self.z[p] = m.continuous_var(lb=0.0, ub=max(0.0, H - D.Dp[p]), name=f'z_{p}')

    def add_constraints(self):
        D, H, m = self.data, self.data.H, self.model

        # (1) Route option selection: sum_{o in O_p} w_p,o = 1, forall p in P
        for p in D.P: m.add_constraint(m.sum(self.w[p, o] for o in D.Op[p]) == 1)

        # (2) Machine assignment: sum_{m in M_a} x_a,m = w_p,o, forall a=(p,o,i) in A
        # (3) Machine ready time: t_a >= B_m - H(1 - x_a,m), forall a in A, m in M_a
        # (4) Lot release time (first operation): t_a >= R_p - H(1 - x_a,m), forall a=(p,o,1) in A, m in M_a
        for a in D.A:
            p, o, i = a
            w_po = self.w[p, o]
            m.add_constraint(m.sum(self.x[a, mach] for mach in D.Ma[a]) == w_po)
            for mach in D.Ma[a]:
                if D.Bm[mach] > 0: m.add_constraint(self.t[a] >= D.Bm[mach] - H * (1 - self.x[a, mach]))
                if i == D.Ipo[p, o][0] and D.Rp[p] > 0: m.add_constraint(self.t[a] >= D.Rp[p] - H * (1 - self.x[a, mach]))

        # (5) Process time sequence: t_a' >= t_a + T_a,m + E_m,n - H(2 - x_a,m - x_a',n), forall (a, a') in A_r, m in M_a, n in M_a'
        for a, a_prime in D.Ar:
            for mach in D.Ma[a]:
                for n in D.Ma[a_prime]: m.add_constraint(self.t[a_prime] >= self.t[a] + D.Tam[a, mach] + D.Emn.get((mach, n), 0.0) - H * (2 - self.x[a, mach] - self.x[a_prime, n]))

        # (6) Machine predecessor: d^+_a,m + sum_{b in A_m, b != a} y_b,a,m = x_a,m, forall m in M, a in A_m
        # (7) Machine successor: sum_{b in A_m, b != a} y_a,b,m + d^-_a,m = x_a,m, forall m in M, a in A_m
        # (8) One start and one end per machine: sum_{a in A_m} d^+_a,m = sum_{a in A_m} d^-_a,m <= 1, forall m in M
        for mach in D.M:
            operations = D.Am[mach]
            if not operations: continue

            for a in operations:
                x = self.x[a, mach]
                incoming_y = [self.y[b, a, mach] for b in operations if (b, a, mach) in self.y]
                m.add_constraint(self.d_plus[a, mach] + m.sum(incoming_y) == x)

                outgoing_y = [self.y[a, b, mach] for b in operations if (a, b, mach) in self.y]
                m.add_constraint(m.sum(outgoing_y) + self.d_minus[a, mach] == x)

            departure = m.sum(self.d_plus[a, mach] for a in operations)
            return_arc = m.sum(self.d_minus[a, mach] for a in operations)
            m.add_constraint(departure == return_arc); m.add_constraint(departure <= 1)

        # (9) Initial setup at machine: t_a >= B_m + S_0,a,m - (H + S_0,a,m)(1 - d^+_a,m), forall a in A, m in M_a
        for (a, mach), depot_arc in self.d_plus.items():
            s0 = D.S0[a, mach]
            m.add_constraint(self.t[a] >= D.Bm[mach] + s0 - (H + s0) * (1 - depot_arc))

        # (10) Machine Time: t_b >= t_a + T_a,m + S_a,b,m - H(1 - y_a,b,m), forall (a, b, m) in immediate_candidates
        for (a, b, mach), y in self.y.items():
            m.add_constraint(self.t[b] >= self.t[a] + D.Tam[a, mach] + D.S[a, b, mach] - H * (1 - y))

        for p in D.P:
            # (11) Completion time: c_p >= t_a + T_a,m - H(1 - x_a,m), forall p in P, o in O_p, a=(p,o,K_p,o), m in M_a
            for o in D.Op[p]:
                last_a = (p, o, D.Ipo[p, o][-1])
                for mach in D.Ma[last_a]: m.add_constraint(self.C[p] >= self.t[last_a] + D.Tam[last_a, mach] - H * (1 - self.x[last_a, mach]))
            # (12) Tardiness definition: z_p >= c_p - D_p, forall p in P
            # (13) Tardiness non-negativity: z_p >= 0, forall p in P (enforced by variable lower bound)
            m.add_constraint(self.z[p] >= self.C[p] - D.Dp[p])

        # (14) Movement flow out of m: sum_{n in M_a'} q_a,m,n = x_a,m, forall (a, a') in A_r, m in M_a
        # (15) Movement flow into n: sum_{m in M_a} q_a,m,n = x_a',n, forall (a, a') in A_r, n in M_a'
        for a, a_prime in D.Ar:
            for mach in D.Ma[a]: m.add_constraint(m.sum(self.q[a, mach, n] for n in D.Ma[a_prime]) == self.x[a, mach])
            for n in D.Ma[a_prime]: m.add_constraint(m.sum(self.q[a, mach, n] for mach in D.Ma[a]) == self.x[a_prime, n])

        # (V1) Direct product entry cut:
        # sum_{a in A_gm} d^+_a,m + sum_{(b,a,m) : a in A_gm, b not in P_g} y_b,a,m >= x_a,m, forall g in G, m in M, a in A_gm
        changeovers_by_gm = {}
        for g in D.G:
            for mach in D.M:
                operations = D.Agm[g, mach]
                if not operations: continue

                depot_entries = [self.d_plus[a, mach] for a in operations]
                changeover_entries = [self.y[b, a, mach] for (b, a, m_idx) in self.y if m_idx == mach and b[0] not in D.Pg[g] and a[0] in D.Pg[g]]
                changeovers_by_gm[g, mach] = changeover_entries
                if depot_entries or changeover_entries:
                    entry_sum = m.sum(depot_entries + changeover_entries)
                    for a in operations: m.add_constraint(entry_sum >= self.x[a, mach])

        # (V2) Plant-wide changeover bound:
        # sum_{(a,b,m) in Delta} y_a,b,m >= sum_{l in L} max{0, |G_l| - |M_l|}
        plant_wide_bound = sum(max(0, len(D.Gl[l]) - len(D.Ml[l])) for l in D.L)
        if plant_wide_bound > 0 and self.y:
            all_changeovers = [y for entries in changeovers_by_gm.values() for y in entries]
            if all_changeovers: m.add_constraint(m.sum(all_changeovers) >= plant_wide_bound)

        # (V3) Route-based tardiness bound:
        # z_p >= sum_{o in O_p} max{0, LB_p,o - D_p} * w_p,o, forall p in P
        for p in D.P:
            option_bounds = {o: max(0.0, D.route_lb[p, o] - D.Dp[p]) for o in D.Op[p]}
            if any(val > 0 for val in option_bounds.values()): m.add_constraint(self.z[p] >= m.sum(option_bounds[o] * self.w[p, o] for o in D.Op[p]))

    def set_objective(self):
        D, m = self.data, self.model
        # min W_z * (sum_{p in P} U_p * z_p) + W_e * (sum_{(a,a') in A_r, m, n} E_m,n * q_a,m,n) + W_s * (sum_{(a,b,m)} S_a,b,m * y_a,b,m + sum_{a in A, m in M_a} S_0,a,m * d^+_a,m)
        tardiness = m.sum(D.Up[p] * self.z[p] for p in D.P)
        movement = m.sum(D.Emn.get((mach, n), 0.0) * q for (a, mach, n), q in self.q.items())
        setup = m.sum(D.S[a, b, mach] * y for (a, b, mach), y in self.y.items()) + m.sum(D.S0[a, mach] * depot_arc for (a, mach), depot_arc in self.d_plus.items())
        m.minimize(D.weight['tardy'] * tardiness + D.weight['move'] * movement + D.weight['setup'] * setup)

    def build(self):
        self.model = Model(f'FJSP_{self.data.dataset_size}', ignore_names=False)
        self.add_variables(); self.add_constraints(); self.set_objective(); return self

    def solve(self): return solve_cplex(self)

    def extract_results(self): self.machines_schedule = extract_mip_schedule(self)
