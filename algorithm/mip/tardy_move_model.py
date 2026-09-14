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

        for a in D.A:
            p, o, i = a
            # x_a,m in {0, 1} : 1 if operation a is processed on machine m
            for mach in D.Ma[a]:
                mtag = str(mach).replace('-', '_')
                self.x[a, mach] = m.binary_var(name=f'x_{p}_{o}_{i}_{mtag}')
            # t_a >= 0 : start time of operation a
            self.t[a] = m.continuous_var(lb=0.0, ub=H, name=f't_{p}_{o}_{i}')

        # q_a,m,n in [0, 1] : 1 if operation a uses machine m and successor a' uses machine n
        for a, a_prime in D.Ar:
            p, o, i = a
            for mach in D.Ma[a]:
                mtag = str(mach).replace('-', '_')
                for n in D.Ma[a_prime]:
                    ntag = str(n).replace('-', '_')
                    self.q[a, mach, n] = m.continuous_var(lb=0.0, ub=1.0, name=f'q_{p}_{o}_{i}_{mtag}_{ntag}')

        # y_a,b,m in {0, 1} : 1 if operation a precedes operation b on machine m
        for a, b, mach in D.pairwise_candidates:
            p1, o1, i1 = a
            p2, o2, i2 = b
            mtag = str(mach).replace('-', '_')
            self.y[a, b, mach] = m.binary_var(name=f'y_{p1}_{o1}_{i1}_{p2}_{o2}_{i2}_{mtag}')
            self.y[b, a, mach] = m.binary_var(name=f'y_{p2}_{o2}_{i2}_{p1}_{o1}_{i1}_{mtag}')

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

        # (2) Machine assignment: sum_{m in M_a} x_a,m = w_p,o, forall a=(p,o,i) in A
        # (3) Machine ready time: t_a >= B_m - H(1 - x_a,m), forall a in A, m in M_a
        # (4) Lot release time (first operation): t_a >= R_p - H(1 - x_a,m), forall a=(p,o,1) in A, m in M_a
        for a in D.A:
            p, o, i = a
            w_po = self.w[p, o]
            m.add_constraint(m.sum(self.x[a, mach] for mach in D.Ma[a]) == w_po)
            for mach in D.Ma[a]:
                if D.Bm[mach] > 0:
                    m.add_constraint(self.t[a] >= D.Bm[mach] - H * (1 - self.x[a, mach]))
                if i == D.Ipo[p, o][0] and D.Rp[p] > 0:
                    m.add_constraint(self.t[a] >= D.Rp[p] - H * (1 - self.x[a, mach]))

        # (5) Process time sequence: t_a' >= t_a + T_a,m + E_m,n - H(2 - x_a,m - x_a',n), forall (a, a') in A_r, m in M_a, n in M_a'
        for a, a_prime in D.Ar:
            for mach in D.Ma[a]:
                for n in D.Ma[a_prime]:
                    m.add_constraint(
                        self.t[a_prime] >= self.t[a] + D.Tam[a, mach] + D.Emn.get((mach, n), 0.0) - H * (2 - self.x[a, mach] - self.x[a_prime, n])
                    )

        # (6) Sequencing link on a: y_a,b,m <= x_a,m, forall m in M, a, b in A_m
        # (7) Sequencing link on b: y_a,b,m <= x_b,m, forall m in M, a, b in A_m
        # (8) Machine precedence: y_a,b,m + y_b,a,m >= x_a,m + x_b,m - 1, forall m in M, a, b in A_m
        # (9) Machine Time:
        #     t_b >= t_a + T_a,m + S_a,b,m - H(1 - y_a,b,m)
        #     t_a >= t_b + T_b,m + S_b,a,m - H(1 - y_b,a,m)
        for a, b, mach in D.pairwise_candidates:
            y12 = self.y[a, b, mach]
            y21 = self.y[b, a, mach]
            x1 = self.x[a, mach]
            x2 = self.x[b, mach]
            s12 = D.S[a, b, mach]
            s21 = D.S[b, a, mach]
            m.add_constraint(y12 <= x1)
            m.add_constraint(y12 <= x2)
            m.add_constraint(y21 <= x1)
            m.add_constraint(y21 <= x2)
            m.add_constraint(y12 + y21 >= x1 + x2 - 1)
            m.add_constraint(self.t[b] >= self.t[a] + D.Tam[a, mach] + s12 - H * (1 - y12))
            m.add_constraint(self.t[a] >= self.t[b] + D.Tam[b, mach] + s21 - H * (1 - y21))

        for p in D.P:
            # (10) Completion time: c_p >= t_a + T_a,m - H(1 - x_a,m), forall p in P, o in O_p, a=(p,o,K_p,o), m in M_a
            for o in D.Op[p]:
                last_a = (p, o, D.Ipo[p, o][-1])
                for mach in D.Ma[last_a]:
                    m.add_constraint(self.C[p] >= self.t[last_a] + D.Tam[last_a, mach] - H * (1 - self.x[last_a, mach]))
            # (11) Tardiness definition: z_p >= c_p - D_p, forall p in P
            # (12) Tardiness non-negativity: z_p >= 0, forall p in P (enforced by variable lower bound)
            m.add_constraint(self.z[p] >= self.C[p] - D.Dp[p])

        # (13) Movement flow out of m: sum_{n in M_a'} q_a,m,n = x_a,m, forall (a, a') in A_r, m in M_a
        # (14) Movement flow into n: sum_{m in M_a} q_a,m,n = x_a',n, forall (a, a') in A_r, n in M_a'
        for a, a_prime in D.Ar:
            for mach in D.Ma[a]:
                m.add_constraint(m.sum(self.q[a, mach, n] for n in D.Ma[a_prime]) == self.x[a, mach])
            for n in D.Ma[a_prime]:
                m.add_constraint(m.sum(self.q[a, mach, n] for mach in D.Ma[a]) == self.x[a_prime, n])

        # (V3) Route-based tardiness bound: z_p >= sum_{o in O_p} max{0, LB_p,o - D_p} * w_p,o, forall p in P
        for p in D.P:
            option_bounds = {o: max(0.0, D.route_lb[p, o] - D.Dp[p]) for o in D.Op[p]}
            if any(val > 0 for val in option_bounds.values()):
                m.add_constraint(self.z[p] >= m.sum(option_bounds[o] * self.w[p, o] for o in D.Op[p]))

    def set_objective(self):
        D, m = self.data, self.model
        # min W_z * (sum_{p in P} U_p * z_p) + W_e * (sum_{(a,a') in A_r, m, n} E_m,n * q_a,m,n)
        tardiness = m.sum(D.Up[p] * self.z[p] for p in D.P)
        movement = m.sum(
            D.Emn.get((mach, n), 0.0) * q for (a, mach, n), q in self.q.items()
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
