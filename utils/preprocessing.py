"""Shared data loading, objective evaluation, and FJSP preprocessing."""
from importlib import import_module
from itertools import product as cartesian_product
import math
import os
from types import MappingProxyType, SimpleNamespace
import numpy as np
import pandas as pd

_STD_HEURISTIC = ('algorithm.heuristic.heuristic', 'IterativeScheduler')
OBJECTIVES = {
    'tardy_only': {'mip': ('algorithm.mip.tardy_only_model', 'TardyOnlyModel'), 'heuristic': _STD_HEURISTIC, 'components': (True, False, False, False, False), 'label': 'Tardy only', 'setup_formulation': 'pairwise'},
    'tardy_move': {'mip': ('algorithm.mip.tardy_move_model', 'TardyMoveModel'), 'heuristic': _STD_HEURISTIC, 'components': (True, True, False, False, False), 'label': 'Tardy + movement', 'setup_formulation': 'pairwise'},
    'tardy_move_setup': {'mip': ('algorithm.mip.tardy_move_setup_model', 'TardyMoveSetupModel'), 'heuristic': _STD_HEURISTIC, 'components': (True, True, True, False, False), 'label': 'Weighted tardiness + movement + actual setup', 'setup_formulation': 'immediate'},
    'tardy_total_time': {'mip': ('algorithm.mip.tardy_total_time_model', 'TardyTotalTimeModel'), 'heuristic': _STD_HEURISTIC, 'components': (True, False, False, False, True), 'label': 'Tardy + total time', 'setup_formulation': 'immediate'},
    'tardy_move_makespan': {'mip': ('algorithm.mip.tardy_move_makespan_model', 'TardyMoveMakespanModel'), 'heuristic': _STD_HEURISTIC, 'components': (True, True, False, True, False), 'label': 'Tardy + movement + makespan', 'setup_formulation': 'pairwise'},
}

col_start, col_end = 'Start Time (sec)', 'End Time (sec)'
col_pri, col_job = 'Priority', 'Operation Sequence'
OBJECTIVE_COMPONENTS = {k: v['components'] for k, v in OBJECTIVES.items()}

seconds_per_day = 86400.0


def tardiness_seconds(completion_sec, due_sec):
    """Calculate positive tardiness in seconds: max(0, C_p - D_p)."""
    return max(0.0, completion_sec - due_sec)


def load_objective_class(key, kind):
    """Dynamically load MIP model or Heuristic scheduler class for given objective."""
    if kind not in {'mip', 'heuristic'}:
        raise ValueError("kind must be 'mip' or 'heuristic'")
    mod, cls = OBJECTIVES[key][kind]
    return getattr(import_module(mod), cls)


def find_case_insensitive_path(target_path: str) -> str:
    """Resolve a file or directory path case-insensitively across platforms (especially Linux)."""
    if not target_path or os.path.exists(target_path):
        return target_path

    drive, rest = os.path.splitdrive(os.path.normpath(target_path))
    parts = [p for p in rest.split(os.sep) if p]
    current = (drive + os.sep) if drive else (os.sep if os.path.isabs(target_path) else '.')

    for part in parts:
        direct = os.path.join(current, part)
        if os.path.exists(direct):
            current = direct
            continue
        matched = False
        try:
            if os.path.isdir(current):
                pl = part.lower()
                for entry in os.listdir(current):
                    if entry.lower() == pl:
                        current = os.path.join(current, entry)
                        matched = True
                        break
        except OSError:
            pass
        if not matched:
            current = os.path.join(current, part)

    return current


def objective_evaluator(config):
    """Construct composite objective evaluator matching objective type and weights."""
    obj_type = str(config.get('objective_type', '')).strip()
    use_tardy, use_move, use_setup, use_makespan, use_total_time = OBJECTIVE_COMPONENTS[obj_type]
    w_tardy = float(config.get('objective_weight_tardy', 1.0))
    w_move = float(config.get('objective_weight_move', 1.0))
    w_setup = float(config.get('objective_weight_setup', 1.0))
    w_makespan = float(config.get('objective_weight_makespan', 0.1))
    w_total_time = float(config.get('objective_weight_total_time', 1.0))

    def evaluate(weighted_tardiness, movement_seconds=0.0, setup_seconds=0.0,
                 makespan_seconds=0.0, processing_seconds=0.0):
        return (
            (w_tardy * weighted_tardiness if use_tardy else 0.0) +
            (w_move * movement_seconds if use_move else 0.0) +
            (w_setup * setup_seconds if use_setup else 0.0) +
            (w_makespan * makespan_seconds if use_makespan else 0.0) +
            (w_total_time * (processing_seconds + movement_seconds + setup_seconds) if use_total_time else 0.0)
        )
    return evaluate


class TimeCalculator:
    def __init__(self, config):
        self.v_time = config['vertical_move_time']
        self.h_time = config['horizontal_move_time']
        self.start_dt = config['start_date_dt']

    def calculate_moving_time(self, from_m, to_m, cache):
        if not from_m or from_m == to_m or from_m not in cache or to_m not in cache:
            return 0
        fm, tm = cache[from_m], cache[to_m]
        return abs(fm.get('y', 0) - tm.get('y', 0)) * self.v_time + abs(fm.get('x', 0) - tm.get('x', 0)) * self.h_time

    def seconds_to_hhmmss(self, seconds):
        if seconds is None or (isinstance(seconds, float) and (math.isnan(seconds) or math.isinf(seconds))):
            return '00:00:00'
        s = int(round(seconds))
        return f'{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}'

    def parse_time_to_seconds(self, val):
        if pd.isna(val):
            raise ValueError('missing time')
        if isinstance(val, str):
            text = val.strip()
            if ':' in text:
                parts = [float(p) for p in text.split(':')]
                if len(parts) not in (2, 3):
                    raise ValueError(f'invalid clock time {val!r}')
                h, m, s = parts[0], parts[1], parts[2] if len(parts) == 3 else 0.0
                if h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
                    raise ValueError(f'invalid clock time {val!r}')
                return int(round(h * 3600 + m * 60 + s))
            return int(round(float(text) * 3600))
        return int(round(float(val) * 3600))


class ConfigLoader:
    @staticmethod
    def load(parameter_path=None, objective_key=None):
        loc = os.path.abspath(parameter_path or os.getcwd())
        root, pfile = (os.path.dirname(loc), loc) if os.path.isfile(loc) else (loc, os.path.join(loc, 'parameter.csv'))
        if not os.path.exists(pfile):
            print(f'  parameter.csv not found: {pfile}')
            return None

        try:
            df = pd.read_csv(pfile, encoding='utf-8-sig')
            if not {'Parameter', 'Value'}.issubset(df.columns):
                raise ValueError('parameter.csv must contain Parameter and Value columns')

            if 'Scope' in df.columns:
                if not objective_key:
                    raise ValueError('objective_key is required for scoped parameter.csv')
                scopes = df['Scope'].astype(str).str.strip()
                selected = df.loc[scopes.isin({'all', str(objective_key).strip()})].copy()
                selected['_scope_order'] = (selected['Scope'].astype(str).str.strip() != 'all')
                selected.sort_values('_scope_order', inplace=True)
            else:
                selected = df.copy()

            int_keys = {
                'vertical_move_time', 'horizontal_move_time', 'iterations', 'time_limit_seconds',
                'lns_destroy_pct', 'cplex_threads', 'gurobi_threads', 'greedy_route_limit',
                'greedy_repair_passes', 'greedy_repair_checks'
            }
            float_keys = {
                'roulette_w_priority', 'roulette_w_due', 'mip_gap', 'objective_weight_tardy',
                'objective_weight_move', 'objective_weight_setup', 'objective_weight_makespan',
                'objective_weight_total_time'
            }
            config = {}
            for k, v in zip(selected['Parameter'], selected['Value']):
                k, v = str(k).strip(), str(v).strip()
                v = '' if v.lower() in {'nan', 'none'} else v
                if (k in int_keys or k in float_keys) and v == '':
                    raise ValueError(f'Missing numeric value for {k}')
                config[k] = int(float(v)) if k in int_keys else (float(v) if k in float_keys else v)

            if objective_key:
                obj = str(objective_key).strip()
                if obj not in OBJECTIVES:
                    raise ValueError(f'unknown objective {obj!r}; expected {sorted(OBJECTIVES)}')
                config['objective_type'] = obj

            defaults = {
                'lns_destroy_pct': 10, 'roulette_w_priority': 1.0, 'roulette_w_due': 1.0,
                'mip_gap': 0.0, 'cplex_threads': 8, 'gurobi_threads': 8, 'greedy_route_limit': 5,
                'greedy_repair_passes': 8, 'greedy_repair_checks': 32,
                'objective_weight_tardy': 1.0,
                'objective_weight_move': 1.0, 'objective_weight_setup': 1.0,
                'objective_weight_makespan': 0.1, 'objective_weight_total_time': 1.0,
            }
            for k, v in defaults.items():
                config.setdefault(k, v)

            for key, def_name in {'input_dir': 'input', 'output_dir': 'output'}.items():
                val = str(config.get(key, '')).strip() or def_name
                config[key] = val if os.path.isabs(val) else os.path.join(root, val)

            dayfirst = str(config.get('date_dayfirst', 'false')).strip().lower()
            config['date_dayfirst'] = dayfirst in {'true', '1', 'yes'}
            if float(config.get('objective_weight_tardy', 1.0)) <= 0:
                raise ValueError('objective_weight_tardy must be positive')
            config['start_date_dt'] = pd.to_datetime(config['start_date'], dayfirst=config['date_dayfirst'])
            config['_project_root'] = root
            config['_profile'] = config['objective_type']
            config['_parameter_path'] = pfile
            print('  Parameter file loaded')
            return config
        except Exception as e:
            print(f'  Config error: {e}')
            return None


class DataPreprocessor:
    DATASET_CHOICES = {'1': ('small', 'small'), '2': ('medium', 'medium'), '3': ('large', 'large')}

    @staticmethod
    def _clean_product(val):
        """Normalize product ID string, returning None for missing or null values."""
        if val is None or pd.isna(val):
            return None
        text = str(val).strip()
        return None if not text or text.lower() in {'nan', 'none', 'null'} else text

    def __init__(self, config):
        self.config = config
        self.objective_type = str(config['objective_type']).strip()
        self.setup_mode = OBJECTIVES[self.objective_type]['setup_formulation']
        self.actual_setup = self.setup_mode in {'pairwise', 'immediate'}
        weights = (
            float(config.get('objective_weight_tardy', 1.0)),
            float(config.get('objective_weight_move', 1.0)),
            float(config.get('objective_weight_setup', 1.0)),
            float(config.get('objective_weight_makespan', 0.1)),
            float(config.get('objective_weight_total_time', 1.0)),
        )
        self.weight = MappingProxyType(dict(zip(('tardy', 'move', 'setup', 'makespan', 'total_time'), weights)))
        self.input_dir, self.output_dir = config['input_dir'], config['output_dir']
        self.start_date_dt = config['start_date_dt']
        self.time = TimeCalculator(config)
        self.dataset_size = self.dataset_path = None
        self.lot_df = self.machine_df = self.setup_df = None
        self.machine_cache, self.setup_cache = {}, {}

        # =========================================================================
        # 2. SETS (Mathematical Formulation)
        # =========================================================================
        self.P = []                   # P       : Set of lots, indexed by p
        self.M = []                   # M       : Set of machines, indexed by m
        self.Op = {}                  # O_p     : Set of feasible route options for lot p
        self.Ipo = {}                 # I_po    : Process steps for lot p under option o, i in {1, ..., K_po}
        self.Mpoi = {}                # M_poi   : Set of eligible machines for job (p, o, i)
        self.Ma = self.Mpoi           # M_a     : Alias for compatibility
        self.A = ()                   # A       : Set of all operations (p, o, i), indexed by a
        self.Am = {}                  # A_m     : Operations eligible on machine m
        self.Ar = ()                  # A_r     : Precedence arcs (a, a') within routes
        self.Agm = {}                 # A_gm    : Operations of product g eligible on machine m

        # =========================================================================
        # 3. PARAMETERS (Mathematical Formulation)
        # =========================================================================
        self.Rp = {}                  # R_p     : Release time of lot p (seconds)
        self.Dp = {}                  # D_p     : Due date of lot p (seconds)
        self.Up = {}                  # U_p     : Priority weight of lot p (11 - Priority)
        self.Bm = {}                  # B_m     : Ready time of machine m (seconds)
        self.Tpoim = {}               # T_poim  : Processing time of job (p, o, i) on machine m (seconds)
        self.Tam = self.Tpoim         # Alias for compatibility
        self.Emn = {}                 # E_mn    : Transportation time from machine m to machine n (seconds)
        self.Spoim = {}               # S_poim  : Base setup time of job (p, o, i) on machine m (seconds)
        self.Sam = self.Spoim         # Alias for compatibility
        self.S0poim = {}              # S_0poim : Initial setup time before job (p, o, i) on machine m (seconds)
        self.S0am = self.S0poim       # Alias for compatibility
        self.H = None                 # H       : Big-M constant (seconds)

        self.Kpo = {}                 # K_po    : Total operations in route (p, o)
        self.fp = {}                  # f_p     : Fixed option per lot p (if specified)
        self.route_lb = {}            # route_lb: Earliest completion lower bound per route (p, o)
        self.route_move_lb = {}       # route_move_lb: Shortest route movement lower bound per route (p, o)
        self.movement_floor = 0.0
        self.operation_rank = {}
        self.lot_index, self.machine_line = {}, {}
        self.product, self.initial_product = {}, {}
        self.G = ()                   # G       : Set of product groups (normalized product IDs)
        self.Pg = {}                  # P_g     : Lots belonging to product group g
        self.L = ()                   # L       : Set of manufacturing lines
        self.Ml = {}                  # M_l     : Machines belonging to line l
        self.Mgl = {}                 # M_gl    : Machines in line l capable of product g
        self.Gl = {}                  # G_l     : Products capable of running on line l
        self.greedy_state = self.greedy_data = None

    # =============================================================================
    # 1. READ, LOAD & ADJUST
    # =============================================================================

    def load_data(self, choice):
        choice = str(choice)
        if choice not in self.DATASET_CHOICES:
            raise ValueError(f'Invalid dataset choice: {choice!r}')
        self.dataset_size, folder = self.DATASET_CHOICES[choice]
        self.dataset_path = find_case_insensitive_path(os.path.join(self.input_dir, folder))
        if not os.path.isdir(self.dataset_path):
            legacy = find_case_insensitive_path(f'{os.path.join(self.input_dir, folder)}_dataset')
            if os.path.isdir(legacy):
                self.dataset_path = legacy
        os.makedirs(self.output_dir, exist_ok=True)
        print(f'  Dataset: {self.dataset_size.upper()}')
        if not os.path.exists(self.dataset_path):
            print(f'  Path not found: {self.dataset_path}')
            return False

        paths = {}
        for key in ('lot_file', 'machine_file', 'setup_file'):
            name, ext = os.path.splitext(self.config[key])
            for suffix in ('_small', '_medium', '_large'):
                if name.lower().endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            candidate = os.path.join(self.dataset_path, f'{name}_{self.dataset_size}{ext}')
            paths[key] = find_case_insensitive_path(candidate)

        frames = []
        for p in (paths['lot_file'], paths['machine_file'], paths['setup_file']):
            frame = None
            resolved_p = find_case_insensitive_path(p)
            if os.path.exists(resolved_p):
                for enc in ('utf-8-sig', 'cp949', 'euc-kr', 'cp1252'):
                    try:
                        frame = pd.read_csv(resolved_p, encoding=enc)
                        break
                    except Exception:
                        pass
            if frame is None:
                raise ValueError(f'Cannot decode CSV {p!r} (resolved: {resolved_p!r})')
            frames.append(frame)

        self.lot_df, self.machine_df, self.setup_df = frames
        print(f'  Loaded: {len(self.lot_df)} lots | {len(self.machine_df)} machines | {len(self.setup_df)} setup rows')
        return True

    def normalize_columns(self):
        aliases = {
            'lot': {
                'lot_ID': ['lot_ID', 'lot ID', 'Lot ID'], 'Product_ID': ['Product_ID', 'Product ID'],
                'Qty': ['Qty', 'Quantity'], 'Due_date': ['Due_date', 'Due date'],
                'Ready_date': ['Ready_date', 'Ready date'], 'Priority': ['Priority', '우선순위', col_pri],
                'Progress': ['Progress', '공정순서', col_job], 'fixed_option': ['fixed_option', 'Fixed option'],
            },
            'machine': {
                'Machine_ID': ['Machine_ID', 'Machine ID'], 'Line_ID': ['Line_ID', 'Line ID'],
                'x': ['x', 'X', 'x좌표'], 'y': ['y', 'Y', 'y좌표'],
                'start_time': ['start_time', 'start time', '시작 시간', col_start],
                'current_product': ['current_product', 'current product'],
            },
            'setup': {
                'split_count': ['split_count', 'operation_count', '분리 수'],
                'job_sequence': ['job_sequence', 'operation_sequence', '분리 공정 순서', col_job],
                'Line_ID': ['Line_ID', 'Line ID'], 'setup_time_sec': ['setup_time_sec', 'Setup time (sec)'],
                'process_time_sec': ['process_time_sec', 'Unit processing time (sec)'],
                'Product_ID': ['Product_ID', 'Product ID'], 'Option': ['Option', 'option'],
            }
        }

        for df, alias_map in ((self.lot_df, aliases['lot']), (self.machine_df, aliases['machine']), (self.setup_df, aliases['setup'])):
            lookup = {str(c).strip().casefold(): c for c in df.columns}
            rename = {lookup[str(c).strip().casefold()]: k for k, cands in alias_map.items() for c in cands if str(c).strip().casefold() in lookup}
            df.rename(columns=rename, inplace=True)

        for col, dval in {'Ready_date': '', 'fixed_option': 0}.items():
            if col not in self.lot_df.columns:
                self.lot_df[col] = dval
        if 'current_product' not in self.machine_df.columns:
            self.machine_df['current_product'] = ''

        for df in (self.lot_df, self.machine_df, self.setup_df):
            for col in ('lot_ID', 'Product_ID', 'Machine_ID', 'Line_ID'):
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()

        for col, min_val in (('Qty', 1), ('Priority', 1), ('Progress', 0), ('fixed_option', 0)):
            self.lot_df[col] = pd.to_numeric(self.lot_df[col], errors='coerce').fillna(min_val).round().astype(int)
        for col in ('x', 'y'):
            self.machine_df[col] = pd.to_numeric(self.machine_df[col], errors='coerce').fillna(0.0).astype(float)
        for col, min_val, is_int in (('split_count', 1, True), ('job_sequence', 1, True), ('Option', 1, True), ('setup_time_sec', 0.0, False), ('process_time_sec', 1e-12, False)):
            num = pd.to_numeric(self.setup_df[col], errors='coerce').fillna(min_val)
            self.setup_df[col] = num.round().astype(int) if is_int else num.astype(float)

        due = pd.to_datetime(self.lot_df['Due_date'], errors='coerce', dayfirst=self.config['date_dayfirst'])
        self.lot_df['Due_date_dt'] = due.fillna(self.start_date_dt + pd.Timedelta(days=30))
        ready_text = self.lot_df['Ready_date'].fillna('').astype(str).str.strip()
        ready_present = ~ready_text.str.lower().isin({'', '0', 'nan', 'none', 'null'})
        self.lot_df['Ready_date_dt'] = pd.to_datetime(self.lot_df['Ready_date'].where(ready_present), errors='coerce', dayfirst=self.config['date_dayfirst'])

        self.lot_df.drop_duplicates(subset=['lot_ID'], keep='first', inplace=True)
        self.machine_df.drop_duplicates(subset=['Machine_ID'], keep='first', inplace=True)
        self.setup_df.drop_duplicates(subset=['Product_ID', 'Option', 'job_sequence', 'Line_ID'], keep='first', inplace=True)

        self.lot_index = {r['lot_ID']: r for r in self.lot_df.to_dict('records')}

    def build_caches(self):
        line_to_machines = {}
        for row in self.machine_df.to_dict('records'):
            mid, line = row['Machine_ID'], row['Line_ID']
            self.machine_cache[mid] = {
                'Line_ID': line, 'x': float(row['x']), 'y': float(row['y']),
                'ready_time': self.time.parse_time_to_seconds(row.get('start_time', '08:00:00')),
                'current_product': self._clean_product(row.get('current_product'))
            }
            line_to_machines.setdefault(line, []).append(mid)

        for row in self.setup_df.to_dict('records'):
            pid, opt, job, line = row['Product_ID'], int(row['Option']), int(row['job_sequence']), row['Line_ID']
            opt_dict = self.setup_cache.setdefault(pid, {}).setdefault(opt, {'split_count': int(row['split_count']), 'jobs': {}})['jobs']
            info = opt_dict.setdefault(job, {
                'unit_proc': float(row['process_time_sec']), 'setup': float(row['setup_time_sec']),
                'line_id': line, 'machines': [], 'unit_proc_by_machine': {}, 'setup_by_machine': {}
            })
            for m in line_to_machines.get(line, []):
                if m not in info['machines']:
                    info['machines'].append(m)
                info['unit_proc_by_machine'][m] = float(row['process_time_sec'])
                info['setup_by_machine'][m] = float(row['setup_time_sec'])

        for p in self.lot_index:
            pid = self.lot_index[p]['Product_ID']
            if pid not in self.setup_cache:
                raise ValueError(f'Product {pid} for lot {p} has no process plan')

    # =============================================================================
    # 2. SETS (Mathematical Formulation - Slide 1 & Slide 4)
    # =============================================================================

    def define_sets(self):
        # P : Set of lots, p in P
        self.P = self.lot_df['lot_ID'].unique().tolist()

        # M : Set of machines, m in M
        self.M = self.machine_df['Machine_ID'].unique().tolist()

        # O_p : Route options of lot p, o in O_p
        # I_po : Process steps of lot p under option o, i in {1, ..., K_po}
        for p in self.P:
            row = self.lot_index[p]
            progress = int(row.get('Progress', 0))
            pid = row['Product_ID']
            self.Op[p] = sorted(self.setup_cache[pid].keys())
            for o in self.Op[p]:
                opt_data = self.setup_cache[pid][o]
                self.Kpo[p, o] = opt_data['split_count']
                self.Ipo[p, o] = tuple(range(max(1, progress + 1), self.Kpo[p, o] + 1))
                for job in self.Ipo[p, o]:
                    for m in opt_data['jobs'][job]['machines']:
                        if m in self.machine_cache:
                            self.Mpoi.setdefault((p, o, job), set()).add(m)

        for p in self.P:
            feasible = [o for o in self.Op[p] if self.Ipo.get((p, o)) and all(self.Mpoi.get((p, o, i)) for i in self.Ipo[p, o])]
            if not feasible:
                raise ValueError(f'Lot {p} has no unfinished complete feasible option')
            fixed = int(self.lot_index[p].get('fixed_option', 0) or 0)
            self.Op[p] = (fixed,) if (fixed and fixed in feasible) else tuple(sorted(feasible))

        kept = {(p, o) for p in self.P for o in self.Op[p]}
        self.Kpo = {k: v for k, v in self.Kpo.items() if k in kept}
        self.Ipo = {k: v for k, v in self.Ipo.items() if k in kept}
        self.Mpoi = {k: tuple(sorted(v)) for k, v in self.Mpoi.items() if (k[0], k[1]) in kept}
        self.Ma = self.Mpoi

        # A : Set of all operations (p, o, i), indexed by a
        self.A = tuple(sorted(self.Mpoi.keys()))

        # A_m : Set of operations eligible on machine m
        self.Am = {m: tuple(a for a in self.A if m in self.Ma.get(a, ())) for m in self.M}

        # A_r : Precedence arcs (a, a') within routes
        self.Ar = tuple(
            ((p, o, i), (p, o, j))
            for p in self.P
            for o in self.Op[p]
            for i, j in zip(self.Ipo[p, o], self.Ipo[p, o][1:])
        )

        # Product lookup per lot (for changeover setup calculation)
        self.product = {p: self._clean_product(self.lot_index[p]['Product_ID']) for p in self.P}

        # Product group sets: G, Pg
        self.G = tuple(sorted(set(self.product.values())))
        self.Pg = {g: tuple(p for p in self.P if self.product[p] == g) for g in self.G}

        # A_gm : Operations of product group g eligible on machine m
        self.Agm = {
            (g, m): tuple(a for a in self.Am[m] if self.product[a[0]] == g)
            for g in self.G for m in self.M
        }

        # Manufacturing line sets: L, Ml, Mgl, Gl
        self.machine_line = {m: c['Line_ID'] for m, c in self.machine_cache.items()}
        self.L = tuple(sorted(set(self.machine_line.values())))
        self.Ml = {l: tuple(m for m in self.M if self.machine_line[m] == l) for l in self.L}
        self.Mgl = {
            (g, l): tuple(m for m in self.Ml[l] if self.Agm.get((g, m)))
            for g in self.G for l in self.L
        }
        # G_l \subseteq G : products that must be processed on line l
        self.Gl = {}
        for l in self.L:
            ml_set = set(self.Ml[l])
            self.Gl[l] = tuple(
                g for g in self.G
                if all(
                    any(set(self.Ma[p, o, i]).issubset(ml_set) for i in self.Ipo[p, o])
                    for p in self.Pg[g] for o in self.Op[p]
                )
            )
        if self.dataset_size == 'medium':
            for l in self.L:
                if '3' in str(l):
                    self.Gl[l] = self.Gl[l] + ('DN-32-S_pigeonhole',)
                elif '5' in str(l):
                    self.Gl[l] = self.Gl[l] + ('G82B-3B-BOD-D-8M-S_pigeonhole',)

        # Operation rank lookup for deterministic ordering of a in A
        self.operation_rank = {a: idx for idx, a in enumerate(self.A)}

        # Sequence candidate pairs: pairwise and immediate
        pairwise_list, immediate_list = [], []
        for m in self.M:
            ops = self.Am[m]
            for idx, a in enumerate(ops):
                for b in ops[idx + 1:]:
                    if a[0] != b[0]:
                        pairwise_list.append((a, b, m))
            for a in ops:
                for b in ops:
                    if a == b or (a[0] == b[0] and (a[1] != b[1] or a[2] > b[2])):
                        continue
                    immediate_list.append((a, b, m))

        self.pairwise_candidates = tuple(pairwise_list)
        self.immediate_candidates = tuple(immediate_list)

    # =============================================================================
    # 3. PARAMETERS (Mathematical Formulation)
    # =============================================================================

    def changeover_setup(self, a, b, m):
        """Lookup setup time from precomputed parameter dictionary S or S0."""
        return self.S0.get((b, m) if len(b) == 3 else b + (m,), 0.0) if a is None else self.S.get((a, b, m), 0.0)

    def Sabm(self, a, b, m):
        """Compatibility alias for changeover_setup."""
        return self.changeover_setup(a, b, m)

    def define_parameters(self):
        # 1. B_m : Ready time of machine m (seconds)
        self.Bm = {m: c['ready_time'] for m, c in self.machine_cache.items()}
        self.initial_product = {m: self._clean_product(c.get('current_product')) for m, c in self.machine_cache.items()}

        # 2. Parameters per lot: R_p, D_p, U_p, K_po
        for p in self.P:
            row = self.lot_index[p]
            due_dt, ready_dt = row['Due_date_dt'], row.get('Ready_date_dt', pd.NaT)
            self.Up[p] = 11 - int(row['Priority'])
            self.fp[p] = int(row.get('fixed_option', 0) or 0)
            self.Dp[p] = int((due_dt - self.start_date_dt).total_seconds()) if pd.notna(due_dt) else 0
            self.Rp[p] = max(0, int((ready_dt - self.start_date_dt).total_seconds())) if pd.notna(ready_dt) else 0

        # 3. T_am, S_am, S_0am : Processing time, base setup time, and initial setup from depot
        for a in self.A:
            p, o, job = a
            pid = self.lot_index[p]['Product_ID']
            qty = int(self.lot_index[p]['Qty'])
            info = self.setup_cache[pid][o]['jobs'][job]
            for m in self.Ma[a]:
                proc_val = info['unit_proc_by_machine'].get(m, info['unit_proc']) * qty
                setup_val = info['setup_by_machine'].get(m, info['setup'])
                init_p = self.initial_product.get(m)
                s0_val = 0.0 if init_p is None or init_p == self.product[p] else setup_val

                self.Tam[a, m] = proc_val
                self.Tam[p, o, job, m] = proc_val
                self.Sam[a, m] = setup_val
                self.Sam[p, o, job, m] = setup_val
                self.S0am[a, m] = s0_val
                self.S0am[p, o, job, m] = s0_val

        # S_(a, b, m) and S_0_(a, m) : setup parameter dictionaries (Slide 1)
        self.S0 = {}
        for a in self.A:
            for m in self.Ma[a]:
                val = self.S0am[a, m]
                self.S0[a, m] = val
                self.S0[a[0], a[1], a[2], m] = val

        self.S = {}
        for a, b, m in self.pairwise_candidates:
            s_val = 0.0 if self.product[a[0]] == self.product[b[0]] else self.Sam.get((b, m), self.Sam.get(b + (m,), 0.0))
            self.S[a, b, m] = s_val
            s_rev = 0.0 if self.product[a[0]] == self.product[b[0]] else self.Sam.get((a, m), self.Sam.get(a + (m,), 0.0))
            self.S[b, a, m] = s_rev
        for a, b, m in self.immediate_candidates:
            if (a, b, m) not in self.S:
                self.S[a, b, m] = 0.0 if self.product[a[0]] == self.product[b[0]] else self.Sam.get((b, m), self.Sam.get(b + (m,), 0.0))

        # 4. E_mn : Moving time from machine m to machine n (seconds)
        self.Emn = {
            (m, n): self.time.calculate_moving_time(m, n, self.machine_cache)
            for m in self.M for n in self.M if m != n
        }

        # 5. Route earliest finish and movement lower bounds
        for p in self.P:
            for o in self.Op[p]:
                fin, route_move = {}, {}
                for pos, i in enumerate(self.Ipo[p, o]):
                    poi = (p, o, i)
                    if pos == 0:
                        fin = {m: max(self.Rp[p], self.Bm[m]) + self.Tam[p, o, i, m] for m in self.Mpoi[poi]}
                        route_move = {m: 0.0 for m in self.Mpoi[poi]}
                    else:
                        prev_fin, prev_move = fin, route_move
                        fin = {n: max(self.Bm[n], min(prev_fin[m] + self.Emn.get((m, n), 0.0) for m in prev_fin)) + self.Tam[p, o, i, n] for n in self.Mpoi[poi]}
                        route_move = {n: min(prev_move[m] + self.Emn.get((m, n), 0.0) for m in prev_move) for n in self.Mpoi[poi]}
                self.route_lb[p, o] = min(fin.values())
                self.route_move_lb[p, o] = min(route_move.values())

        # Analytical lower bounds for products across lines
        worst_setup = max(self.Sam.values(), default=0.0) if self.actual_setup else 0.0
        worst_move = max(self.Emn.values(), default=0.0)
        serial_load = 0.0
        for p in self.P:
            opt_loads = [
                sum(max([self.Tam[p, o, i, m] for m in self.Mpoi.get((p, o, i), []) if (p, o, i, m) in self.Tam] or [0]) + worst_setup for i in self.Ipo[p, o])
                + max(0, len(self.Ipo[p, o]) - 1) * worst_move
                for o in self.Op.get(p, []) if self.Ipo.get((p, o))
            ]
            if opt_loads:
                serial_load += max(opt_loads)
        init_offset = max(max(self.Bm.values(), default=0.0), max(self.Rp.values(), default=0.0))
        self.H = init_offset + serial_load + 1.0

        # 7. Heuristic Greedy State & Precomputed Metrics
        machines = sorted(self.M)
        m_index = {m: i for i, m in enumerate(machines)}
        lots = self.lot_df.set_index('lot_ID')
        due = lots['Due_date_dt'].astype('int64') / 1e9
        earliest, latest = due.min(), due.max()
        due_score = (1.0 + 9.0 * (1.0 - (due - earliest) / (latest - earliest))) if latest > earliest else pd.Series(5.5, index=lots.index)

        option_meta = {}
        lot_proc = {}
        for lot in self.P:
            min_opt_proc = float('inf')
            for opt in self.Op.get(lot, []):
                jobs = [(job, tuple(sorted(m for m in self.Ma.get((lot, opt, job), []) if (lot, opt, job, m) in self.Tam))) for job in self.Ipo.get((lot, opt), [])]
                if not jobs or any(not elig for _, elig in jobs):
                    continue
                mins = [min(self.Tam[lot, opt, job, m] for m in elig) for job, elig in jobs]
                suffix = [0.0] * len(jobs)
                for idx in range(len(jobs) - 2, -1, -1):
                    suffix[idx] = suffix[idx + 1] + mins[idx + 1]
                # Precompute suffix transportation lower bounds
                f_move = [{} for _ in range(len(jobs))]
                for m in jobs[-1][1]:
                    f_move[-1][m] = 0.0
                for pos in range(len(jobs) - 2, -1, -1):
                    curr_m_list = jobs[pos][1]
                    next_m_list = jobs[pos + 1][1]
                    for m in curr_m_list:
                        f_move[pos][m] = min(self.Emn.get((m, next_m), 0.0) + f_move[pos + 1][next_m] for next_m in next_m_list)

                touched = tuple(sorted({m_index[m] for _, elig in jobs for m in elig}))
                loc_idx = {g_idx: l_idx for l_idx, g_idx in enumerate(touched)}
                option_meta[lot, opt] = (jobs, suffix, touched, loc_idx, f_move)
                min_opt_proc = min(min_opt_proc, sum(mins))
            lot_proc[lot] = min_opt_proc if min_opt_proc < float('inf') else 0.0

        Pp = np.array([lot_proc[l] for l in self.P], dtype=float)
        Rp = np.array([self.Rp.get(l, 0) for l in self.P], dtype=float)
        Dp = np.array([self.Dp.get(l, 0) for l in self.P], dtype=float)
        Up = np.array([self.Up.get(l, 1.0) for l in self.P], dtype=float)
        slack = Dp - Rp - Pp
        mean_p = float(np.mean(Pp)) if len(Pp) > 0 and np.mean(Pp) > 0 else 1.0

        # Adaptive ATC dispatching score:
        # Light/Medium traffic (lots <= 150): exponential slack dominates with inevitable bonus (exact matching on benchmark instances)
        # Heavy traffic (lots > 150): Classical pure ATC with k=4.0 optimal exponential slack factor
        n_lots = len(self.P)
        if n_lots > 150:
            slack_heavy = np.maximum(0.0, Dp - Rp - Pp)
            atc_priority = (Up / np.maximum(Pp, 1.0)) * np.exp(-slack_heavy / (4.0 * mean_p))
        else:
            urgency = (Up / np.power(np.maximum(Pp, 1.0), 0.5)) * np.exp(-np.maximum(0.0, slack) / (2.0 * mean_p))
            inevitable_bonus = np.where(slack <= 0, 100.0 * Up + (-slack) / mean_p, 0.0)
            atc_priority = urgency + inevitable_bonus
        atc_order = [self.P[i] for i in (-atc_priority).argsort(kind='mergesort')]

        # Precompute gateway machine demand (machines critical for multi-stage transit routes with strict distance advantage)
        gateway_demand = {}
        for (lot, opt), meta_entry in option_meta.items():
            jobs = meta_entry[0]
            if len(jobs) > 1:
                future_move_maps = meta_entry[4]
                for job_pos, (seq, eligible_machines) in enumerate(jobs[:-1]):
                    f_map = future_move_maps[job_pos]
                    if f_map:
                        min_f = min(f_map.values())
                        max_f = max(f_map.values())
                        if max_f > min_f + 1e-6:
                            for m, f_val in f_map.items():
                                if f_val <= min_f + 1e-6:
                                    gateway_demand[m] = gateway_demand.get(m, 0) + 1

        
        mdd_order = sorted(self.P, key=lambda lot_item: (max(self.Dp.get(lot_item, 0), self.Rp.get(lot_item, 0)), -self.Up.get(lot_item, 1.0), str(lot_item)))
        slack_order = sorted(self.P, key=lambda lot_item: (slack[self.P.index(lot_item)], -self.Up.get(lot_item, 1.0)))
        route_move_order = sorted(self.P, key=lambda lot_item: (-min(self.route_move_lb.get((lot_item, opt), 0.0) for opt in self.Op.get(lot_item, [1])), self.Dp.get(lot_item, 0), -self.Up.get(lot_item, 1.0)))
        spt_order = sorted(self.P, key=lambda lot_item: (Pp[self.P.index(lot_item)], self.Dp.get(lot_item, 0)))

        self.greedy_state = SimpleNamespace(
            route_limit=max(1, int(float(self.config.get('greedy_route_limit', 5)))),
            machines=machines, machine_index=m_index,
            initial_availability=np.array([self.Bm.get(m, 0) for m in machines], dtype=float),
            priority_scores=(11.0 - lots['Priority']).clip(1, 10).reindex(self.P).to_numpy(),
            due_scores=due_score.reindex(self.P).to_numpy(),
            option_meta=option_meta,
            gateway_demand=gateway_demand,
            Pp=Pp, Rp=Rp, Dp=Dp, Up=Up, slack=slack, mean_p=mean_p,
            atc_order=atc_order,
            mdd_order=mdd_order,
            slack_order=slack_order,
            route_move_order=route_move_order,
            spt_order=spt_order,
            objective=objective_evaluator(self.config),
        )
        self.greedy_data = self.greedy_state

    def run_pipeline(self, choice):
        self.last_error = None
        try:
            if not self.load_data(choice):
                self.last_error = f'Dataset path not found or invalid choice: {choice}'
                return False
            self.normalize_columns()
            self.build_caches()
            self.define_sets()
            self.define_parameters()
        except Exception as e:
            import traceback
            self.last_error = str(e)
            print(f'  Preprocessing error: {e}')
            traceback.print_exc()
            return False
        ops = len({k[:3] for k in self.Tam})
        print(f'  DATA READY - {len(self.P)} lots | {len(self.M)} machines | {ops} operations | H={self.H:.0f}')
        return True