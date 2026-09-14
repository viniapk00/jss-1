"""Weighted tardiness plus total time FJSP MIP with immediate machine sequencing."""
from docplex.mp.model import Model
from algorithm.mip.cplex_solver import solve as solve_cplex
from utils.result_saver import extract_mip_schedule
from utils.preprocessing import seconds_per_day


class TardyTotalTimeModel:
    objective_type = 'tardy_total_time'

    def __init__(self, data):
        if data.objective_type != self.objective_type:
            raise ValueError(f'{type(self).__name__} requires objective_type={self.objective_type!r}')
        self.data = data
        self.model = None
        self.solution = None
        self.solve_time = 0.0
        self.best_bound = None
        self.node_count = None
        self.mip_gap = None
        self.obj_value = None
        self.proven_optimal = False
        self.machines_schedule = {}

        self.w, self.x, self.t, self.y = {}, {}, {}, {}
        self.d_plus, self.d_minus = {}, {}
        self.q, self.C, self.z = {}, {}, {}

    def add_variables(self):
        D, H, m = self.data, self.data.H, self.model

        # w_p,o in {0, 1} : 1 if lot p selects route option o
        for p in D.P:
            for o in D.Op[p]:
                self.w[p, o] = m.binary_var(name=f'w_{p}_{o}')

        for p in D.P:
            for o in D.Op[p]:
                for i in D.Ipo[p, o]:
                    poi = (p, o, i)
                    # x_(p,o,i),m in {0, 1} : 1 if operation (p, o, i) is processed on machine m
                    for mach in D.Mpoi[poi]:
                        mtag = str(mach).replace('-', '_')
                        self.x[poi, mach] = m.binary_var(name=f'x_{p}_{o}_{i}_{mtag}')
                        # d^+_(p,o,i),m in [0, 1] : 1 if operation (p, o, i) has an incoming arc from the depot on machine m
                        # d^-_(p,o,i),m in [0, 1] : 1 if operation (p, o, i) has an outgoing arc to the depot on machine m
                        self.d_plus[poi, mach] = m.continuous_var(lb=0.0, ub=1.0, name=f'd_plus_{p}_{o}_{i}_{mtag}')
                        self.d_minus[poi, mach] = m.continuous_var(lb=0.0, ub=1.0, name=f'd_minus_{p}_{o}_{i}_{mtag}')
                    # t_(p,o,i) >= 0 : start time of operation (p, o, i)
                    self.t[poi] = m.continuous_var(lb=0.0, ub=H, name=f't_{p}_{o}_{i}')

        # q_(p,o,i),m,n in [0, 1] : 1 if operation (p, o, i) uses machine m and successor uses machine n
        for p in D.P:
            for o in D.Op[p]:
                seq = D.Ipo[p, o]
                for i, j in zip(seq, seq[1:]):
                    poi, poj = (p, o, i), (p, o, j)
                    for mach in D.Mpoi[poi]:
                        mtag = str(mach).replace('-', '_')
                        for n in D.Mpoi[poj]:
                            ntag = str(n).replace('-', '_')
                            self.q[poi, mach, n] = m.continuous_var(lb=0.0, ub=1.0, name=f'q_{p}_{o}_{i}_{mtag}_{ntag}')

        # y_(p,o,i),(p',o',i'),m in {0, 1} : 1 if (p, o, i) immediately precedes (p', o', i') on machine m
        for poi1, poi2, mach in D.immediate_candidates:
            p1, o1, i1 = poi1
            p2, o2, i2 = poi2
            mtag = str(mach).replace('-', '_')
            self.y[poi1, poi2, mach] = m.binary_var(name=f'y_{p1}_{o1}_{i1}_{p2}_{o2}_{i2}_{mtag}')

        for p in D.P:
            # c_p >= 0 : completion time of the last operation of lot p
            self.C[p] = m.continuous_var(lb=0.0, ub=H, name=f'C_{p}')
            # z_p >= 0 : tardiness of lot p
            self.z[p] = m.continuous_var(lb=0.0, ub=max(0.0, H - D.Dp[p]), name=f'z_{p}')

    def add_constraints(self):
        D, H, m = self.data, self.data.H, self.model

        # (1) Route option selection: sum_{o in O_p} w_p,o = 1, forall p in P
        for p in D.P:
            m.add_constraint(m.sum(self.w[p, o] for o in D.Op[p]) == 1)

        # (2) Machine assignment: sum_{m in M_(p,o,i)} x_(p,o,i),m = w_p,o, forall p in P, o in O_p, i in I_p,o
        # (3) Machine ready time: t_(p,o,i) >= B_m - H(1 - x_(p,o,i),m), forall p in P, o in O_p, i in I_p,o, m in M_(p,o,i)
        # (4) Lot release time (first operation): t_(p,o,i) >= R_p - H(1 - x_(p,o,i),m), forall p in P, o in O_p, i = 1, m in M_(p,o,i)
        for p in D.P:
            for o in D.Op[p]:
                w_po = self.w[p, o]
                seq = D.Ipo[p, o]
                for i in seq:
                    poi = (p, o, i)
                    m.add_constraint(m.sum(self.x[poi, mach] for mach in D.Mpoi[poi]) == w_po)
                    for mach in D.Mpoi[poi]:
                        if D.Bm[mach] > 0:
                            m.add_constraint(self.t[poi] >= D.Bm[mach] - H * (1 - self.x[poi, mach]))
                    if i == seq[0]:
                        for mach in D.Mpoi[poi]:
                            if D.Rp[p] > 0:
                                m.add_constraint(self.t[poi] >= D.Rp[p] - H * (1 - self.x[poi, mach]))

        # (5) Process time sequence: t_(p,o,i+1) >= t_(p,o,i) + T_(p,o,i),m + E_m,n - H(2 - x_(p,o,i),m - x_(p,o,i+1),n), forall p in P, o in O_p, i < K_p,o, m in M_(p,o,i), n in M_(p,o,i+1)
        for p in D.P:
            for o in D.Op[p]:
                seq = D.Ipo[p, o]
                for i, j in zip(seq, seq[1:]):
                    poi, poi_succ = (p, o, i), (p, o, j)
                    for mach in D.Mpoi[poi]:
                        for n in D.Mpoi[poi_succ]:
                            m.add_constraint(
                                self.t[poi_succ] >= self.t[poi] + D.Tam[p, o, i, mach] + D.Emn.get((mach, n), 0.0) - H * (2 - self.x[poi, mach] - self.x[poi_succ, n])
                            )

        # (6) Machine predecessor: d^+_(p,o,i),m + sum_{(p',o',i') | m in M_(p',o',i')} y_(p',o',i'),(p,o,i),m = x_(p,o,i),m, forall m in M_(p,o,i)
        # (7) Machine successor: sum_{(p',o',i') | m in M_(p',o',i')} y_(p,o,i),(p',o',i'),m + d^-_{(p,o,i),m} = x_(p,o,i),m, forall m in M_(p,o,i)
        # (8) One start and one end per machine: sum_{(p,o,i) | m in M_(p,o,i)} d^+_(p,o,i),m = sum_{(p,o,i) | m in M_(p,o,i)} d^-_{(p,o,i),m} <= 1, forall m in M
        for mach in D.M:
            operations = [
                (p, o, i)
                for p in D.P
                for o in D.Op[p]
                for i in D.Ipo[p, o]
                if mach in D.Mpoi.get((p, o, i), ())
            ]
            if not operations:
                continue

            for poi in operations:
                x = self.x[poi, mach]
                incoming_y = [self.y[poi_prev, poi, mach] for poi_prev in operations if (poi_prev, poi, mach) in self.y]
                m.add_constraint(self.d_plus[poi, mach] + m.sum(incoming_y) == x)

                outgoing_y = [self.y[poi, poi_succ, mach] for poi_succ in operations if (poi, poi_succ, mach) in self.y]
                m.add_constraint(m.sum(outgoing_y) + self.d_minus[poi, mach] == x)

            departure = m.sum(self.d_plus[poi, mach] for poi in operations)
            return_arc = m.sum(self.d_minus[poi, mach] for poi in operations)
            m.add_constraint(departure == return_arc)
            m.add_constraint(departure <= 1)

        # (9) Initial setup at machine: t_(p,o,i) >= B_m + S_0,(p,o,i),m - (H + S_0,(p,o,i),m)(1 - d^+_(p,o,i),m), forall m in M_(p,o,i)
        for (poi, mach), depot_arc in self.d_plus.items():
            s0 = D.S0[poi, mach]
            m.add_constraint(self.t[poi] >= D.Bm[mach] + s0 - (H + s0) * (1 - depot_arc))

        # (10) Machine Time: t_(p',o',i') >= t_(p,o,i) + T_(p,o,i),m + S_(p,o,i),(p',o',i'),m - H(1 - y_(p,o,i),(p',o',i'),m), forall m in M_(p,o,i) \cap M_(p',o',i')
        for (poi1, poi2, mach), y in self.y.items():
            m.add_constraint(self.t[poi2] >= self.t[poi1] + D.Tam[poi1 + (mach,)] + D.S[poi1, poi2, mach] - H * (1 - y))

        for p in D.P:
            # (11) Completion time: c_p >= t_(p,o,i) + T_(p,o,i),m - H(1 - x_(p,o,i),m), forall p in P, o in O_p, i = K_p,o, m in M_(p,o,i)
            for o in D.Op[p]:
                last_poi = (p, o, D.Ipo[p, o][-1])
                for mach in D.Mpoi[last_poi]:
                    m.add_constraint(self.C[p] >= self.t[last_poi] + D.Tam[last_poi + (mach,)] - H * (1 - self.x[last_poi, mach]))
            # (12) Tardiness definition: z_p >= c_p - D_p, forall p in P
            # (13) Tardiness non-negativity: z_p >= 0, forall p in P (enforced by variable lower bound)
            m.add_constraint(self.z[p] >= self.C[p] - D.Dp[p])

        # (14) Movement flow out of m: sum_{n in M_(p,o,i+1)} q_(p,o,i),m,n = x_(p,o,i),m, forall p in P, o in O_p, i < K_p,o, m in M_(p,o,i)
        # (15) Movement flow into n: sum_{m in M_(p,o,i)} q_(p,o,i),m,n = x_(p,o,i+1),n, forall p in P, o in O_p, i < K_p,o, n in M_(p,o,i+1)
        for p in D.P:
            for o in D.Op[p]:
                seq = D.Ipo[p, o]
                for i, j in zip(seq, seq[1:]):
                    poi, poj = (p, o, i), (p, o, j)
                    for mach in D.Mpoi[poi]:
                        m.add_constraint(m.sum(self.q[poi, mach, n] for n in D.Mpoi[poj]) == self.x[poi, mach])
                    for n in D.Mpoi[poj]:
                        m.add_constraint(m.sum(self.q[poi, mach, n] for mach in D.Mpoi[poi]) == self.x[poj, n])

        # (V1) Direct product entry cut:
        # sum_{(p,o,i): p in P_g} d^+_(p,o,i),m + sum_{(p',o',i'),(p,o,i): p in P_g, p' not in P_g} y_(p',o',i'),(p,o,i),m >= x_(p,o,i),m, forall g in G, m in M
        changeovers_by_gm = {}
        for g in D.G:
            for mach in D.M:
                operations = [
                    (p, o, i)
                    for p in D.Pg[g]
                    for o in D.Op[p]
                    for i in D.Ipo[p, o]
                    if mach in D.Mpoi.get((p, o, i), ())
                ]
                if not operations:
                    continue

                depot_entries = [self.d_plus[poi, mach] for poi in operations]
                changeover_entries = [
                    self.y[poi1, poi2, mach]
                    for (poi1, poi2, m_idx) in self.y
                    if m_idx == mach and poi1[0] not in D.Pg[g] and poi2[0] in D.Pg[g]
                ]
                changeovers_by_gm[g, mach] = changeover_entries
                if depot_entries or changeover_entries:
                    entry_sum = m.sum(depot_entries + changeover_entries)
                    for poi in operations:
                        m.add_constraint(entry_sum >= self.x[poi, mach])

        # (V2) Plant-wide changeover bound:
        # sum_{m in M, (p,o,i),(p',o',i'): m in M_(p,o,i) \cap M_(p',o',i'), p in P_g, p' not in P_g} y_(p,o,i),(p',o',i'),m >= sum_{l in L} max{0, |G_l| - |M_l|}
        plant_wide_bound = sum(max(0, len(D.Gl[l]) - len(D.Ml[l])) for l in D.L)
        if plant_wide_bound > 0 and self.y:
            all_changeovers = [y for entries in changeovers_by_gm.values() for y in entries]
            if all_changeovers:
                m.add_constraint(m.sum(all_changeovers) >= plant_wide_bound)

        # (V3) Route-based tardiness bound:
        # z_p >= sum_{o in O_p} max{0, LB_p,o - D_p} * w_p,o, forall p in P
        for p in D.P:
            option_bounds = {o: max(0.0, D.route_lb[p, o] - D.Dp[p]) for o in D.Op[p]}
            if any(val > 0 for val in option_bounds.values()):
                m.add_constraint(self.z[p] >= m.sum(option_bounds[o] * self.w[p, o] for o in D.Op[p]))

    def set_objective(self):
        D, m = self.data, self.model
        # min W_z * (sum_{p in P} U_p * z_p) + W_t * (sum_{m,n} E_m,n * q_(p,o,i),m,n + sum_{(p,o,i),m} S_(p,o,i),(p',o',i'),m * y_(p,o,i),(p',o',i'),m + sum_{(p,o,i),m} T_(p,o,i),m * x_(p,o,i),m + sum_{(p,o,i),m} S_0,(p,o,i),m * d^+_(p,o,i),m)
        tardiness = m.sum(D.Up[p] * self.z[p] for p in D.P)
        processing = m.sum(
            D.Tam[p, o, i, mach] * self.x[(p, o, i), mach]
            for p in D.P for o in D.Op[p] for i in D.Ipo[p, o] for mach in D.Mpoi.get((p, o, i), ())
        )
        movement = m.sum(
            D.Emn.get((mach, n), 0.0) * q for (poi, mach, n), q in self.q.items()
        )
        setup = m.sum(D.S[poi1, poi2, mach] * y for (poi1, poi2, mach), y in self.y.items())
        setup += m.sum(
            D.S0[poi, mach] * depot_arc for (poi, mach), depot_arc in self.d_plus.items()
        )
        m.minimize(
            D.weight['tardy'] * tardiness + D.weight['total_time'] * (processing + movement + setup)
        )

    def build(self):
        self.model = Model(f'FJSP_{self.data.dataset_size}', ignore_names=False)
        self.add_variables()
        self.add_constraints()
        self.set_objective()
        return self

    def solve(self):
        return solve_cplex(self)

    def extract_results(self):
        self.machines_schedule = extract_mip_schedule(self)
