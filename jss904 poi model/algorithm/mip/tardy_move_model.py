"""Weighted tardiness plus movement FJSP MIP with pairwise machine sequencing."""
from docplex.mp.model import Model
from algorithm.mip.cplex_solver import solve as solve_cplex
from utils.result_saver import extract_mip_schedule
from utils.preprocessing import seconds_per_day


class TardyMoveModel:
    objective_type = 'tardy_move'

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

        self.w, self.x, self.t = {}, {}, {}
        self.q, self.y, self.C, self.z = {}, {}, {}, {}

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

        # y_(p,o,i),(p',o',i'),m in {0, 1} : 1 if (p, o, i) precedes (p', o', i') on machine m
        for poi1, poi2, mach in D.pairwise_candidates:
            p1, o1, i1 = poi1
            p2, o2, i2 = poi2
            mtag = str(mach).replace('-', '_')
            self.y[poi1, poi2, mach] = m.binary_var(name=f'y_{p1}_{o1}_{i1}_{p2}_{o2}_{i2}_{mtag}')
            self.y[poi2, poi1, mach] = m.binary_var(name=f'y_{p2}_{o2}_{i2}_{p1}_{o1}_{i1}_{mtag}')

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

        # (6) Sequencing link on (p,o,i): y_(p,o,i),(p',o',i'),m <= x_(p,o,i),m, forall m in M_(p,o,i) \cap M_(p',o',i')
        # (7) Sequencing link on (p',o',i'): y_(p,o,i),(p',o',i'),m <= x_(p',o',i'),m, forall m in M_(p,o,i) \cap M_(p',o',i')
        # (8) Machine precedence: y_(p,o,i),(p',o',i'),m + y_(p',o',i'),(p,o,i),m >= x_(p,o,i),m + x_(p',o',i'),m - 1, forall m in M_(p,o,i) \cap M_(p',o',i')
        # (9) Machine Time: t_(p',o',i') >= t_(p,o,i) + T_(p,o,i),m + S_(p,o,i),(p',o',i'),m - H(1 - y_(p,o,i),(p',o',i'),m), forall m in M_(p,o,i) \cap M_(p',o',i')
        for poi1, poi2, mach in D.pairwise_candidates:
            y12 = self.y[poi1, poi2, mach]
            y21 = self.y[poi2, poi1, mach]
            x1 = self.x[poi1, mach]
            x2 = self.x[poi2, mach]
            s12 = D.S[poi1, poi2, mach]
            s21 = D.S[poi2, poi1, mach]
            m.add_constraint(y12 <= x1)
            m.add_constraint(y12 <= x2)
            m.add_constraint(y21 <= x1)
            m.add_constraint(y21 <= x2)
            m.add_constraint(y12 + y21 >= x1 + x2 - 1)
            m.add_constraint(self.t[poi2] >= self.t[poi1] + D.Tam[poi1 + (mach,)] + s12 - H * (1 - y12))
            m.add_constraint(self.t[poi1] >= self.t[poi2] + D.Tam[poi2 + (mach,)] + s21 - H * (1 - y21))

        for p in D.P:
            # (10) Completion time: c_p >= t_(p,o,i) + T_(p,o,i),m - H(1 - x_(p,o,i),m), forall p in P, o in O_p, i = K_p,o, m in M_(p,o,i)
            for o in D.Op[p]:
                last_poi = (p, o, D.Ipo[p, o][-1])
                for mach in D.Mpoi[last_poi]:
                    m.add_constraint(self.C[p] >= self.t[last_poi] + D.Tam[last_poi + (mach,)] - H * (1 - self.x[last_poi, mach]))
            # (11) Tardiness definition: z_p >= c_p - D_p, forall p in P
            # (12) Tardiness non-negativity: z_p >= 0, forall p in P (enforced by variable lower bound)
            m.add_constraint(self.z[p] >= self.C[p] - D.Dp[p])

        # (13) Movement flow out of m: sum_{n in M_(p,o,i+1)} q_(p,o,i),m,n = x_(p,o,i),m, forall p in P, o in O_p, i < K_p,o, m in M_(p,o,i)
        # (14) Movement flow into n: sum_{m in M_(p,o,i)} q_(p,o,i),m,n = x_(p,o,i+1),n, forall p in P, o in O_p, i < K_p,o, n in M_(p,o,i+1)
        for p in D.P:
            for o in D.Op[p]:
                seq = D.Ipo[p, o]
                for i, j in zip(seq, seq[1:]):
                    poi, poj = (p, o, i), (p, o, j)
                    for mach in D.Mpoi[poi]:
                        m.add_constraint(m.sum(self.q[poi, mach, n] for n in D.Mpoi[poj]) == self.x[poi, mach])
                    for n in D.Mpoi[poj]:
                        m.add_constraint(m.sum(self.q[poi, mach, n] for mach in D.Mpoi[poi]) == self.x[poj, n])

        # (V3) Route-based tardiness bound: z_p >= sum_{o in O_p} max{0, LB_p,o - D_p} * w_p,o, forall p in P
        for p in D.P:
            option_bounds = {o: max(0.0, D.route_lb[p, o] - D.Dp[p]) for o in D.Op[p]}
            if any(val > 0 for val in option_bounds.values()):
                m.add_constraint(self.z[p] >= m.sum(option_bounds[o] * self.w[p, o] for o in D.Op[p]))

    def set_objective(self):
        D, m = self.data, self.model
        # min W_z * (sum_{p in P} U_p * z_p) + W_e * (sum_{m,n} E_m,n * q_(p,o,i),m,n)
        tardiness = m.sum(D.Up[p] * self.z[p] for p in D.P)
        movement = m.sum(
            D.Emn.get((mach, n), 0.0) * q for (poi, mach, n), q in self.q.items()
        )
        m.minimize(D.weight['tardy'] * tardiness + D.weight['move'] * movement)

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
