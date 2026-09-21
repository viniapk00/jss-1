"""Shared result validation, reporting, persistence, and comparisons."""
from datetime import datetime, timezone
import json
import math
import os
import platform
import re
import sys
import pandas as pd

from utils.preprocessing import (col_end, col_job, col_pri, col_start, objective_evaluator, tardiness_seconds,)

TARDY_EPS = 1e-6


def sha256_file(path):
    """Compute SHA-256 hex digest of a file for run manifest recording."""
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def _nat_key(val): return ''.join(f'{int(p):010d}' if p.isdigit() else p.lower() for p in re.split(r'(\d+)', str(val)))


def _format_numeric(value, precision=None):
    if value is None: return '-'
    num = float(value)
    return repr(num) if precision is None else format(num, f'.{precision}f')


_fmt = _format_numeric


def safe_to_csv(df, path, **kwargs):
    kwargs.setdefault('encoding', 'utf-8-sig')
    kwargs.setdefault('index', False)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, **kwargs)
    return True


# =============================================================================
# 1. VALIDATION & METRICS
# =============================================================================

def validate_schedule(df, data, tol=1e-5):
    if df is None or df.empty: return ['schedule is empty']
    req = {'lot ID', 'Product ID', 'Option', 'Machine ID', col_job, col_start, col_end, 'Setup Time', 'Moving Time', 'Processing Time'}
    if not req.issubset(df.columns): return [f'missing columns: {sorted(req - set(df.columns))}']

    errs = []
    for c in (col_start, col_end, 'Setup Time', 'Moving Time', 'Processing Time'):
        v = pd.to_numeric(df[c], errors='coerce')
        if not bool(v.map(lambda x: pd.notna(x) and math.isfinite(float(x))).all()): errs.append(f'{c}: non-finite')
        if bool((v < -tol).any()): errs.append(f'{c}: negative')
    if errs: return errs

    if set(df['lot ID'].dropna()) != set(data.P): errs.append(f'lot coverage mismatch: missing={sorted(set(data.P) - set(df["lot ID"].dropna()))}')

    for lot, grp in df.groupby('lot ID', sort=False):
        opts = grp['Option'].dropna().unique().tolist()
        if len(opts) != 1:
            errs.append(f'{lot}: multiple options {opts}')
            continue
        opt = opts[0]
        fixed = data.fp.get(lot, 0)
        if fixed and opt != fixed: errs.append(f'{lot}: selected option {opt} != fixed {fixed}')

        ordered = grp.assign(_j=pd.to_numeric(grp[col_job], errors='coerce')).sort_values(['_j', col_job], kind='mergesort')
        prev = None
        for _, r in ordered.iterrows():
            j, m, s, e = r[col_job], r['Machine ID'], float(r[col_start]), float(r[col_end])
            exp_p = float(data.Tam.get((lot, opt, j, m), float('nan')))
            if abs(e - s - exp_p) > tol or abs(float(r['Processing Time']) - exp_p) > tol: errs.append(f'{lot}/{opt}/{j}: processing mismatch')
            if s + tol < float(data.Rp.get(lot, 0)): errs.append(f'{lot}/{opt}/{j}: starts before ready time')
            if prev is not None:
                exp_m = float(data.Emn.get((prev['Machine ID'], m), 0)) if prev['Machine ID'] != m else 0.0
                if abs(float(r.get('Moving Time', 0) or 0) - exp_m) > tol: errs.append(f'{lot}/{opt}/{j}: move mismatch')
                if s + tol < float(prev[col_end]) + exp_m: errs.append(f'{lot}/{opt}/{j}: precedence violation')
            prev = r

    for m, grp in df.groupby('Machine ID', sort=False):
        ordered = grp.assign(_occ=pd.to_numeric(grp[col_start], errors='coerce').fillna(0) - pd.to_numeric(grp['Setup Time'], errors='coerce').fillna(0)).sort_values(['_occ', col_end], kind='mergesort')
        prev_end = prev_poi = None
        for _, r in ordered.iterrows():
            occ = float(r['_occ'])
            poi = (r['lot ID'], r['Option'], r[col_job])
            exp_s = data.changeover_setup(prev_poi, poi, m) if data.actual_setup and (prev_poi is not None or data.setup_mode == 'immediate') else 0.0
            if abs(float(r.get('Setup Time', 0) or 0) - exp_s) > tol: errs.append(f'{m}/{r["lot ID"]}: setup mismatch')
            if occ + tol < float(data.Bm.get(m, 0)): errs.append(f'{m}/{r["lot ID"]}: occupies before machine ready')
            if prev_end is not None and occ + tol < prev_end: errs.append(f'{m}: overlap at {r["lot ID"]}')
            prev_end = max(float(r[col_end]), prev_end or float('-inf'))
            prev_poi = poi
    return errs


def assert_schedule_feasible(df, data):
    errs = validate_schedule(df, data)
    if errs: raise RuntimeError(f'infeasible schedule: {"; ".join(errs[:5])}')


def validate_pairwise_schedule(df, data, tol=1e-5):
    if df is None or df.empty: return []
    errs = []
    for m, grp in df.groupby('Machine ID', sort=False):
        rows = list(grp.to_dict('records'))
        for i, l in enumerate(rows):
            poi1, sa, ea = (l['lot ID'], l['Option'], l[col_job]), float(l[col_start]), float(l[col_end])
            for r in rows[i + 1:]:
                if l['lot ID'] == r['lot ID']: continue
                poi2, sb, eb = (r['lot ID'], r['Option'], r[col_job]), float(r[col_start]), float(r[col_end])
                if not ((sb + tol >= ea + float(data.changeover_setup(poi1, poi2, m))) or (sa + tol >= eb + float(data.changeover_setup(poi2, poi1, m)))): errs.append(f'{m}: pairwise overlap between {l["lot ID"]} and {r["lot ID"]}')
    return errs


def assert_pairwise_schedule_feasible(df, data):
    errs = validate_pairwise_schedule(df, data)
    if errs: raise RuntimeError(f'infeasible pairwise schedule: {"; ".join(errs[:5])}')


def schedule_metrics(df, data):
    if df is None or df.empty: return {'objective': 0.0, 'objective_per_lot': 0.0, 'weighted_tardiness': 0.0, 'weighted_tardiness_days': 0.0, 'moving_seconds': 0.0, 'moving_days': 0.0, 'setup_seconds': 0.0, 'setup_days': 0.0, 'transition_seconds': 0.0, 'transition_days': 0.0, 'processing_seconds': 0.0, 'processing_days': 0.0, 'total_time_seconds': 0.0, 'total_time_days': 0.0, 'total_time_component': 0.0, 'average_flowtime_seconds': 0.0, 'average_flowtime_days': 0.0, 'tardiness_seconds': 0.0, 'tardy_lots': 0, 'tardy_lot_pct': 0.0, 'makespan_seconds': 0.0, 'makespan_days': 0.0, 'makespan_component': 0.0, 'schedule_span_seconds': 0.0, 'product_machine_pairs': 0, 'lot_items': [], 'lot_details': []}

    evaluate_obj = objective_evaluator(data.config)
    cmax, crit_lot = float(df[col_end].max()), df.loc[df[col_end].idxmax(), 'lot ID']
    lot_items, lot_details, w_tardy, tardy_sec, flow_sum = [], [], 0.0, 0.0, 0.0

    for lot_id, grp in df.groupby('lot ID'):
        last = grp.loc[grp[col_end].idxmax()]
        comp = float(last[col_end])
        delay = tardiness_seconds(comp, data.Dp[lot_id])
        flow = tardiness_seconds(comp, float(data.Rp.get(lot_id, 0.0)))
        weight = float(data.Up.get(lot_id, 1.0))
        m_s = float(grp['Moving Time'].fillna(0).sum())
        s_s = float(grp['Setup Time'].fillna(0).sum())
        p_s = float(grp['Processing Time'].fillna(0).sum())
        pen = weight * delay
        w_tardy += pen
        tardy_sec += delay
        flow_sum += flow
        lot_items.append((lot_id, weight, delay, pen))
        lot_details.append({'lot_ID': lot_id, 'product': last.get('Product ID', ''), 'option': last.get('Option', ''), 'priority': last.get(col_pri, ''), 'priority_weight': weight, 'due_date': str(last.get('Due date', '')), 'completion_sec': comp, 'delay_sec': delay, 'flowtime_sec': flow, 'weighted_penalty': pen, 'total_move_sec': m_s, 'total_setup_sec': s_s, 'makespan_sec': cmax if lot_id == crit_lot else 0.0, 'total_processing_sec': p_s, 'total_time_sec': p_s + m_s + s_s, 'lot_objective': evaluate_obj(pen, m_s, s_s, cmax if lot_id == crit_lot else 0.0, p_s), 'is_tardy': delay > TARDY_EPS})

    m_tot = float(df['Moving Time'].fillna(0).sum())
    s_tot = float(df['Setup Time'].fillna(0).sum())
    p_tot = float(df['Processing Time'].fillna(0).sum())
    tot_time = p_tot + m_tot + s_tot
    obj = evaluate_obj(w_tardy, m_tot, s_tot, cmax, p_tot)
    n_lots = len(lot_items)
    n_tardy = sum(1 for _, _, d, _ in lot_items if d > TARDY_EPS)

    return {'objective': obj, 'objective_per_lot': obj / n_lots if n_lots else 0.0, 'weighted_tardiness': w_tardy, 'weighted_tardiness_days': w_tardy, 'moving_seconds': m_tot, 'moving_days': m_tot, 'setup_seconds': s_tot, 'setup_days': s_tot, 'transition_seconds': m_tot + s_tot, 'transition_days': m_tot + s_tot, 'processing_seconds': p_tot, 'processing_days': p_tot, 'total_time_seconds': tot_time, 'total_time_days': tot_time, 'total_time_component': float(data.config.get('objective_weight_total_time', 1.0)) * tot_time if data.config.get('objective_type') == 'tardy_total_time' else 0.0, 'average_flowtime_seconds': flow_sum / max(1, n_lots), 'average_flowtime_days': flow_sum / max(1, n_lots), 'tardiness_seconds': tardy_sec, 'tardy_lots': n_tardy, 'tardy_lot_pct': 100.0 * n_tardy / n_lots if n_lots else 0.0, 'makespan_seconds': cmax, 'makespan_days': cmax, 'makespan_component': evaluate_obj(0.0, 0.0, 0.0, cmax), 'schedule_span_seconds': cmax - float(df[col_start].min()), 'schedule_span_days': cmax - float(df[col_start].min()), 'product_machine_pairs': int(df[['Machine ID', 'Product ID']].drop_duplicates().shape[0]), 'critical_lot': crit_lot, 'lot_items': lot_items, 'lot_details': lot_details}


def format_objective_breakdown(obj, metrics, objective_type=''):
    pfx = f"  OBJECTIVE [MIP/{objective_type}]" if objective_type else '  OBJECTIVE'
    g = lambda *keys: next((metrics[k] for k in keys if k in metrics and metrics[k] is not None), 0.0)
    return (f"{pfx}: {_fmt(obj, 15)} " f"(delay={_fmt(g('weighted_tardiness_days', 'weighted_tardiness'), 6)}, " f"processing={_fmt(g('processing_days', 'processing_seconds'), 6)}, " f"total_time={_fmt(g('total_time_days', 'total_time_seconds'), 6)}, " f"total_time_term={_fmt(g('total_time_component'), 6)}, " f"avg_flow={_fmt(g('average_flowtime_days', 'average_flowtime_seconds'), 6)}, " f"makespan={_fmt(g('makespan_days', 'makespan_seconds'), 6)}, " f"makespan_term={_fmt(g('makespan_component'), 6)}, " f"move={_fmt(g('moving_days', 'moving_seconds'), 6)}, " f"setup={_fmt(g('setup_days', 'setup_seconds'), 6)}, " f"transition={_fmt(g('transition_days', 'transition_seconds'), 6)}, " f"tardy={metrics.get('tardy_lots', 0)}/{len(metrics.get('lot_items', []))})")


# =============================================================================
# 2. OUTPUT & PERSISTENCE
# =============================================================================

def build_gantt_figure(tc, schedule_df, machine_line=None, tag='', data=None):
    """Generate interactive Plotly Gantt chart figure with red hatching ('arsir merah') on tardy portions."""
    if schedule_df is None or schedule_df.empty: return None
    try: import plotly.graph_objects as go
    except ImportError: return None

    m_col = 'Machine ID' if 'Machine ID' in schedule_df.columns else ('Machine' if 'Machine' in schedule_df.columns else None)
    if not m_col: return None

    ml = machine_line if isinstance(machine_line, dict) else (getattr(machine_line, '_line_map', {}) or {})
    machines = sorted(schedule_df[m_col].unique(), key=lambda m: (_nat_key(ml.get(m, '')), _nat_key(m)))
    y_pos = {m: i for i, m in enumerate(machines)}

    # Distinct non-red palette for lots so red (#DC2626) is reserved exclusively for tardy hatching
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#0ea5e9', '#10b981', '#6366f1', '#f59e0b', '#8b5cf6', '#14b8a6', '#393b79', '#637939', '#8c6d31', '#7b4173']
    lots = sorted(schedule_df['lot ID'].unique(), key=_nat_key)
    lot_color = {lot: colors[i % len(colors)] for i, lot in enumerate(lots)}

    # 1. Resolve start_dt reference
    ref_dt = tc.start_dt if (tc is not None and hasattr(tc, 'start_dt')) else None
    if ref_dt is None:
        try: ref_dt = pd.to_datetime('2026-01-01 00:00:00')
        except Exception: pass

    # 2. Extract due date per lot in seconds
    lot_due_sec = {}
    if data is not None and hasattr(data, 'Dp') and data.Dp: lot_due_sec = {str(k): float(v) for k, v in data.Dp.items()}
    else:
        for r in schedule_df.to_dict('records'):
            lot = r.get('lot ID')
            if lot and lot not in lot_due_sec:
                due_val = r.get('Due date')
                if due_val is not None and str(due_val).strip() not in ('', 'nan', 'None', '0'):
                    try:
                        due_dt = pd.to_datetime(due_val)
                        if ref_dt is not None: lot_due_sec[lot] = float((due_dt - ref_dt).total_seconds())
                        else: lot_due_sec[lot] = float('inf')
                    except Exception: lot_due_sec[lot] = float('inf')
                else: lot_due_sec[lot] = float('inf')

    # 3. Calculate max completion time per lot to identify tardy lots
    lot_max_end = {}
    parsed_records = []
    for r in schedule_df.to_dict('records'):
        m = r.get(m_col)
        if m not in y_pos: continue
        lot = r.get('lot ID')

        # Parse start time (seconds)
        raw_start = r.get('Processing Start (raw sec)')
        if raw_start is not None and not (isinstance(raw_start, float) and math.isnan(raw_start)):
            try: start_sec = float(raw_start)
            except (ValueError, TypeError): start_sec = None
        else: start_sec = None

        if start_sec is None:
            start_val = r.get(col_start)
            if isinstance(start_val, str):
                try: start_sec = float(start_val)
                except ValueError:
                    if ref_dt is not None:
                        dt_val = pd.to_datetime(start_val)
                        start_sec = (dt_val - ref_dt).total_seconds()
                    else: start_sec = 0.0
            else: start_sec = float(start_val or 0.0)

        setup_sec = float(r.get('Setup Time', 0.0) or 0.0)
        proc_sec = float(r.get('Processing Time', 0.0) or 0.0)
        end_sec = start_sec + proc_sec

        lot_max_end[lot] = max(lot_max_end.get(lot, 0.0), end_sec)
        parsed_records.append((r, m, lot, start_sec, setup_sec, proc_sec, end_sec))

    # A lot is tardy if its final completion exceeds its due date
    tardy_lots = {lot for lot, comp in lot_max_end.items() if comp > lot_due_sec.get(lot, float('inf')) + 1e-6}

    max_end_sec = max((item[6] for item in parsed_records), default=0.0)
    max_horizon_days = max_end_sec / 86400.0
    num_days = int(math.ceil(max_horizon_days)) + 1

    # Vertical divider lines for every single day across the scheduling horizon ("garis tiap hari")
    day_shapes = [dict(type='line', x0=d, x1=d, y0=0, y1=1, yref='paper', line=dict(color='#64748B' if d == 0 else 'rgba(148, 163, 184, 0.55)', width=2 if d == 0 else 1.5, dash='solid' if d == 0 else 'dash'), layer='below') for d in range(num_days + 1)]

    fig = go.Figure()
    shown_lots = set()
    tardy_legend_shown = False

    for r, m, lot, start_sec, setup_sec, proc_sec, end_sec in parsed_records:
        color = lot_color.get(lot, '#808080')
        product_id = r.get('Product ID', '')
        option_id = r.get('Option', '')
        job_seq = r.get(col_job, r.get('Operation Sequence', ''))
        due_sec = lot_due_sec.get(lot, float('inf'))
        is_tardy = (lot in tardy_lots)

        common = dict(y=[y_pos[m]], orientation='h', legendgroup=str(lot))

        # Setup bar (in days, stippled dots pattern)
        if setup_sec > 0:
            setup_start_sec = max(0.0, start_sec - setup_sec) if start_sec >= setup_sec else start_sec
            setup_start_day = setup_start_sec / 86400.0
            setup_dur_day = setup_sec / 86400.0
            fig.add_trace(go.Bar(x=[setup_dur_day], base=[setup_start_day], marker=dict(color=color, pattern=dict(shape='.', size=6, solidity=0.3)), showlegend=False, name=f'Lot {lot} Setup', hovertemplate=(f"<b>[SETUP] Lot {lot}</b><br>" f"Product: {product_id}<br>" f"Machine: {m}<br>" f"Setup Time: Day {setup_start_day:.3f} → Day {(setup_start_sec + setup_sec)/86400.0:.3f}<br>" f"Duration: {setup_dur_day:.3f} days ({setup_sec/3600.0:.2f}h | {setup_sec:.0f}s)<extra></extra>"), **common))

        start_day = start_sec / 86400.0
        end_day = end_sec / 86400.0
        proc_day = proc_sec / 86400.0
        due_day = due_sec / 86400.0 if math.isfinite(due_sec) else float('inf')
        due_day_str = f"Day {due_day:.3f} ({due_sec/3600.0:.2f}h)" if math.isfinite(due_sec) else "None"

        if not is_tardy:
            # Entire operation is on-time (unit: days)
            fig.add_trace(go.Bar(x=[proc_day], base=[start_day], marker_color=color, name=f'Lot {lot}', showlegend=(lot not in shown_lots), hovertemplate=(f"<b>Lot {lot}</b> (Job {job_seq})<br>" f"Product: {product_id}<br>" f"Option: {option_id}<br>" f"Machine: {m}<br>" f"Start: Day {start_day:.3f} ({start_sec/3600.0:.2f}h)<br>" f"Process: {proc_day:.3f} days ({proc_sec/3600.0:.2f}h)<br>" f"End: Day {end_day:.3f} ({end_sec/3600.0:.2f}h)<br>" f"Due Date: {due_day_str}<extra></extra>"), **common))
            shown_lots.add(lot)
        else:
            # Tardy lot: partition into on-time portion (before due_sec) and tardy portion (from due_sec onwards)
            ontime_end_sec = min(end_sec, due_sec)
            if ontime_end_sec > start_sec:
                ontime_dur_sec = ontime_end_sec - start_sec
                ontime_dur_day = ontime_dur_sec / 86400.0
                fig.add_trace(go.Bar(x=[ontime_dur_day], base=[start_day], marker_color=color, name=f'Lot {lot}', showlegend=(lot not in shown_lots), hovertemplate=(f"<b>Lot {lot}</b> (Job {job_seq}) [On-Time Portion]<br>" f"Product: {product_id}<br>" f"Option: {option_id}<br>" f"Machine: {m}<br>" f"Start: Day {start_day:.3f} ({start_sec/3600.0:.2f}h)<br>" f"On-Time Duration: {ontime_dur_day:.3f} days ({ontime_dur_sec/3600.0:.2f}h)<br>" f"Cutoff (Due Date): {due_day_str}<br>" f"Total Operation: {proc_day:.3f} days ({proc_sec/3600.0:.2f}h)<extra></extra>"), **common))
                shown_lots.add(lot)

            # Tardy portion: from due_sec onwards until completion, marked with red hatching ('arsir merah')
            if end_sec > due_sec:
                tardy_start_sec = max(start_sec, due_sec)
                tardy_start_day = tardy_start_sec / 86400.0
                tardy_dur_sec = end_sec - tardy_start_sec
                tardy_dur_day = tardy_dur_sec / 86400.0

                fig.add_trace(go.Bar(x=[tardy_dur_day], base=[tardy_start_day], marker=dict(color=color, pattern=dict(shape='/', size=8, solidity=0.5, fgcolor='#DC2626', bgcolor=color), line=dict(color='#DC2626', width=2)), name='⚠️ Tardy / Overdue' if not tardy_legend_shown else f'Lot {lot} (Tardy)', showlegend=not tardy_legend_shown, legendgroup='tardy_legend' if not tardy_legend_shown else str(lot), hovertemplate=(f"<b>⚠️ [TARDY] Lot {lot}</b> (Job {job_seq}) [Overdue Portion]<br>" f"Product: {product_id}<br>" f"Option: {option_id}<br>" f"Machine: {m}<br>" f"Due Date: {due_day_str}<br>" f"Tardy Portion: Day {tardy_start_day:.3f} → Day {end_day:.3f}<br>" f"Overdue Duration: +{tardy_dur_day:.3f} days (+{tardy_dur_sec/3600.0:.2f}h)<br>" f"Total Operation: {proc_day:.3f} days ({proc_sec/3600.0:.2f}h)<extra></extra>"), y=[y_pos[m]], orientation='h'))
                tardy_legend_shown = True

    # Ensure every lot appears in the legend even if entirely tardy
    for lot in lots:
        if lot not in shown_lots and lot in lot_color: fig.add_trace(go.Bar(x=[0], base=[0], y=[0], orientation='h', marker_color=lot_color[lot], name=f'Lot {lot}', showlegend=True, legendgroup=str(lot), visible=True))

    date_title = f" - {ref_dt:%m/%d/%Y}" if ref_dt is not None else ""
    fig.update_layout(title=f'Gantt Schedule [{tag}] (Days){date_title}' if tag else f'Gantt Schedule (Days){date_title}', xaxis=dict(title=dict(text='Timeline (Days)', font=dict(size=14, color='#1E293B')), tickmode='linear', tick0=0, dtick=1, tickprefix='Day ', showgrid=True, gridwidth=1.5, gridcolor='rgba(148, 163, 184, 0.45)', zeroline=True, zerolinewidth=2, zerolinecolor='#475569', range=[-0.1, max(1.0, max_horizon_days + 0.5)], rangeslider=dict(visible=True)), shapes=day_shapes, barmode='overlay', width=1200, height=max(500, 35 * len(machines)), margin=dict(l=200, r=50, t=60, b=50), yaxis=dict(tickmode='array', tickvals=list(y_pos.values()), ticktext=[f'{ml.get(m, "-")} - {m}' for m in machines], autorange='reversed'), hovermode='closest', legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.02))
    return fig


def save_gantt_chart(tc, results_dir, schedule_df, machine_line, tag, data=None):
    """Generate and save interactive Plotly Gantt chart HTML."""
    path = os.path.join(results_dir, f'gantt_{tag}.html')
    if schedule_df is None or schedule_df.empty: return None
    fig = build_gantt_figure(tc, schedule_df, machine_line=machine_line, tag=tag, data=data)
    if fig is None: return None
    os.makedirs(results_dir, exist_ok=True)
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"  Gantt chart saved: {path}")
    return path


def save_result(tc, results_dir, ds_size, data, schedule_df, solve_time, method='heuristic', metrics=None):
    if schedule_df is None or schedule_df.empty: return None
    assert_schedule_feasible(schedule_df, data)
    if metrics is None: metrics = schedule_metrics(schedule_df, data)

    cols = ['lot ID', 'Product ID', 'Option', 'Qty', 'Due date', col_pri, 'Prev Machine', 'Machine ID', 'Moving Time', 'Setup Time', 'Processing Time', col_job, col_start, col_end]
    out = schedule_df.reindex(columns=cols).copy()
    proc_start = out[col_start].fillna(0)
    out['Processing Start (raw sec)'] = proc_start
    out['End Time (raw sec)'] = out[col_end].fillna(0)
    if tc is not None:
        out['Processing Start'] = (tc.start_dt + pd.to_timedelta(proc_start, unit='s')).dt.strftime('%m/%d/%Y %H:%M:%S')
        out[col_start] = (proc_start - out['Setup Time'].fillna(0)).clip(lower=0)
        out.sort_values(['lot ID', 'Option', col_job, col_start, 'Machine ID'], key=lambda c: c.map(_nat_key) if c.name == 'lot ID' else c, kind='mergesort', inplace=True)
        for c in (col_start, col_end): out[c] = (tc.start_dt + pd.to_timedelta(out[c].fillna(0), unit='s')).dt.strftime('%m/%d/%Y %H:%M:%S')

    ordered = ['lot ID', 'Product ID', 'Option', 'Qty', 'Due date', col_pri, col_job, 'Prev Machine', 'Machine ID', 'Moving Time', 'Setup Time', 'Processing Time', col_start, 'Processing Start', col_end, 'Processing Start (raw sec)', 'End Time (raw sec)']
    safe_to_csv(out[[c for c in ordered if c in out.columns]], os.path.join(results_dir, f'schedule_{ds_size}_{method}.csv'))

    summary = {'method': method, **{k: metrics[k] for k in ('objective', 'objective_per_lot', 'weighted_tardiness', 'moving_seconds', 'setup_seconds', 'product_machine_pairs', 'transition_seconds', 'processing_seconds', 'total_time_seconds', 'average_flowtime_seconds', 'total_time_component', 'makespan_component', 'tardy_lots', 'tardy_lot_pct', 'makespan_seconds', 'schedule_span_seconds')}, 'solve_time': solve_time, 'lots': schedule_df['lot ID'].nunique(), 'jobs': len(schedule_df), 'dataset': ds_size, 'objective_type': data.config.get('objective_type', ''), 'start_date': tc.start_dt.strftime('%m/%d/%Y') if tc is not None else ''}
    safe_to_csv(pd.DataFrame([summary]), os.path.join(results_dir, f'summary_{ds_size}_{method}.csv'))
    return metrics


def write_run_manifest(output_dir, config, data, seed, mode, solver):
    root = os.path.abspath(config.get('_project_root') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    code_hashes = {rel: sha256_file(os.path.join(root, rel)) for rel in ('main.py', 'parameter.csv', 'run_all_types.py') if os.path.isfile(os.path.join(root, rel))}
    for folder in ('algorithm', 'utils'):
        for cur, _, files in os.walk(os.path.join(root, folder)):
            for f in sorted(files):
                if f.endswith('.py'):
                    p = os.path.join(cur, f)
                    code_hashes[os.path.relpath(p, root).replace('\\', '/')] = sha256_file(p)

    p_path = os.path.abspath(config.get('_parameter_path') or os.path.join(root, 'parameter.csv'))
    manifest = {'schema_version': 2, 'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset': data.dataset_size, 'objective_type': config.get('objective_type'), 'seed': int(seed), 'mode': mode, 'solver': solver, 'configuration': {k: v.isoformat() if hasattr(v, 'isoformat') else v for k, v in sorted(config.items())}, 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'logical_cpu_count': os.cpu_count()}, 'loaded_parameter': {'path': p_path, 'sha256': sha256_file(p_path) if os.path.isfile(p_path) else None}, 'code_sha256': code_hashes,}
    path = os.path.join(output_dir, 'run_manifest.json')
    with open(path, 'w', encoding='utf-8') as h: json.dump(manifest, h, indent=2, ensure_ascii=False, default=str)
    return path


# =============================================================================
# 3. MIP & HEURISTIC POST-PROCESSING & COMPARISON
# =============================================================================

def extract_mip_schedule(mip):
    """Extract machine schedules directly from MIP decision variables."""
    if not mip.solution: return {}
    data = mip.data
    selected = {lot: opt for (lot, opt), var in mip.w.items() if var.solution_value > 0.5}
    by_key, by_operation = {}, {}
    by_machine = {m: [] for m in data.M}
    for (poi, m), var in mip.x.items():
        if var.solution_value > 0.5:
            lot, opt, op = poi
            proc = data.Tam[lot, opt, op, m]
            s = float(mip.t[poi].solution_value)
            row = data.lot_index[lot]
            j = {'lot': lot, 'product': row['Product_ID'], 'option': opt, 'job': op, 'machine': m, 'start': s, 'finish': s + proc, 'proc': proc, 'setup': 0.0, 'move': 0.0, 'prev_machine': '', 'qty': row['Qty'], 'due': row['Due_date'], 'pri': row['Priority']}
            k = (lot, opt, op, m)
            by_key[k] = j
            by_operation.setdefault((lot, opt, op), k)
            by_machine[m].append(k)

    # Operation precedence transfer / movement time
    for lot, opt in selected.items():
        seq = sorted(data.Ipo.get((lot, opt), []))
        for op1, op2 in zip(seq, seq[1:]):
            k1 = by_operation.get((lot, opt, op1))
            k2 = by_operation.get((lot, opt, op2))
            if k1 is not None and k2 is not None:
                by_key[k2]['prev_machine'] = k1[3]
                by_key[k2]['move'] = data.Emn.get((k1[3], k2[3]), 0.0) if k1[3] != k2[3] else 0.0

    # Machine setup time
    if data.setup_mode == 'immediate':
        depot_starts = {poi + (m,) for (poi, m), var in getattr(mip, 'd_plus', {}).items() if var.solution_value > 0.5}
        selected_arcs = [k for k, var in getattr(mip, 'y', {}).items() if var.solution_value > 0.5]
        successors = {(poi2[0], poi2[1], poi2[2], m) for poi1, poi2, m in selected_arcs}
        for poi1, poi2, m in selected_arcs:
            k2 = poi2 + (m,)
            if k2 in by_key: by_key[k2]['setup'] = data.changeover_setup(poi1, poi2, m)
        for k, j in by_key.items():
            if (k in depot_starts) or (not depot_starts and k not in successors and data.initial_product.get(k[3]) is not None): j['setup'] = data.changeover_setup(None, k[:3], k[3])
    else:
        for m in data.M:
            m_jobs = sorted(by_machine[m], key=lambda k: by_key[k]['start'])
            if m_jobs:
                by_key[m_jobs[0]]['setup'] = data.changeover_setup(None, m_jobs[0][:3], m)
                for k1, k2 in zip(m_jobs, m_jobs[1:]): by_key[k2]['setup'] = data.changeover_setup(k1[:3], k2[:3], m)

    schedules = {m: [by_key[k] for k in keys] for m, keys in by_machine.items()}
    for m in schedules: schedules[m].sort(key=lambda j: (j['start'], str(j['lot']), j['job']))
    return schedules


def _make_summary_row(method, solver_name, run_status, obj, metrics, mip_time, total_time, gap_pct, data, solver_obj=None, proven=False, claim='', model=None, best_bound=None, nodes=None, status=None): return {'method': method, 'solver': solver_name, 'run_status': run_status, 'objective': obj, 'objective_per_lot': metrics.get('objective_per_lot', 0.0), 'weighted_tardiness': metrics.get('weighted_tardiness', metrics.get('weighted_tardiness_days', 0.0)), 'moving_seconds': metrics.get('moving_seconds', metrics.get('moving_days', 0.0)), 'setup_seconds': metrics.get('setup_seconds', metrics.get('setup_days', 0.0)), 'makespan_component': metrics.get('makespan_component', 0.0), 'processing_seconds': metrics.get('processing_seconds', metrics.get('processing_days', 0.0)), 'total_time_seconds': metrics.get('total_time_seconds', metrics.get('total_time_days', 0.0)), 'total_time_component': metrics.get('total_time_component', 0.0), 'average_flowtime_seconds': metrics.get('average_flowtime_seconds', metrics.get('average_flowtime_days', 0.0)), 'product_machine_pairs': metrics.get('product_machine_pairs', 0), 'transition_seconds': metrics.get('transition_seconds', metrics.get('transition_days', 0.0)), 'tardy_lots': metrics.get('tardy_lots', 0), 'tardy_lot_pct': metrics.get('tardy_lot_pct', 0.0), 'solver_objective': solver_obj, 'objective_delta': (obj - solver_obj) if solver_obj is not None else 0.0, 'solver_time': mip_time, 'solve_time': mip_time, 'total_pipeline_time': total_time, 'gap_pct': gap_pct, 'solver_gap_pct': gap_pct, 'reported_schedule_gap_pct': gap_pct, 'normalized_incumbent': (run_status == 'solved_normalized'), 'proven_optimal': proven, 'optimality_claim': claim, 'lots': len(data.P), 'machines': len(data.M), 'variables': getattr(model, 'number_of_variables', None) if model else None, 'constraints': getattr(model, 'number_of_constraints', None) if model else None, 'best_bound': best_bound, 'nodes': nodes, 'solver_status': status,}


def normalize_mip_schedule(df, data):
    """Normalize MIP schedule start times so that machine occupancy and job precedence are 100% strictly feasible."""
    if df is None or df.empty: return df
    df = df.copy()
    for _ in range(30):
        changed = False
        for m, grp in df.groupby('Machine ID', sort=False):
            ordered = grp.assign(_occ=pd.to_numeric(grp[col_start], errors='coerce').fillna(0) - pd.to_numeric(grp['Setup Time'], errors='coerce').fillna(0)).sort_values(['_occ', col_end], kind='mergesort')
            prev_end = None
            for idx in ordered.index:
                setup = float(df.at[idx, 'Setup Time'] or 0)
                proc = float(df.at[idx, 'Processing Time'] or 0)
                curr_s = float(df.at[idx, col_start])
                bm = float(data.Bm.get(m, 0.0))
                min_start = max(curr_s, bm + setup)
                if prev_end is not None: min_start = max(min_start, prev_end + setup)
                if min_start > curr_s + 1e-5:
                    df.at[idx, col_start] = min_start
                    df.at[idx, col_end] = min_start + proc
                    changed = True
                    prev_end = min_start + proc
                else: prev_end = max(prev_end or float('-inf'), float(df.at[idx, col_end]))

        for lot, grp in df.groupby('lot ID', sort=False):
            ordered = grp.assign(_j=pd.to_numeric(grp[col_job], errors='coerce')).sort_values(['_j', col_job], kind='mergesort')
            prev = None
            rp = float(data.Rp.get(lot, 0.0))
            for idx in ordered.index:
                curr_s = float(df.at[idx, col_start])
                proc = float(df.at[idx, 'Processing Time'] or 0)
                min_start = max(curr_s, rp)
                if prev is not None:
                    prev_m = df.at[prev, 'Machine ID']
                    curr_m = df.at[idx, 'Machine ID']
                    exp_m = float(data.Emn.get((prev_m, curr_m), 0.0)) if prev_m != curr_m else 0.0
                    min_start = max(min_start, float(df.at[prev, col_end]) + exp_m)
                if min_start > curr_s + 1e-5:
                    df.at[idx, col_start] = min_start
                    df.at[idx, col_end] = min_start + proc
                    changed = True
                prev = idx
        if not changed: break
    return df


def save_mip_result(tc, data, mip, out_dir, solver_name='cplex'):
    rows = [{'lot ID': j['lot'], 'Product ID': j['product'], 'Machine ID': j['machine'], 'Option': j['option'], col_job: j['job'], col_start: j['start'], col_end: j['finish'], 'Setup Time': j.get('setup', 0), 'Moving Time': j.get('move', 0), 'Processing Time': j['proc'], 'Qty': j['qty'], 'Due date': j['due'], col_pri: j['pri'], 'Prev Machine': j.get('prev_machine', '')} for jobs in mip.machines_schedule.values() for j in jobs]
    gdf = pd.DataFrame(rows) if rows else pd.DataFrame()
    gdf = normalize_mip_schedule(gdf, data)
    if getattr(data, 'setup_mode', None) == 'pairwise': assert_pairwise_schedule_feasible(gdf, data)

    metrics = schedule_metrics(gdf, data)
    obj = metrics['objective']
    solver_obj = float(getattr(mip, 'obj_value', obj))
    normalized = abs(obj - solver_obj) > (1e-6 + 1e-6 * max(1.0, abs(solver_obj)))
    if normalized: print(f'  NOTE: normalized solver incumbent {solver_obj:.9f} to validated schedule {obj:.9f}')

    print(format_objective_breakdown(obj, metrics, objective_type=data.config['objective_type']))
    total_time = getattr(mip, 'total_pipeline_time', None) or float(mip.solve_time)
    save_result(tc, out_dir, data.dataset_size, data, gdf, total_time, method=f'mip_{solver_name}', metrics=metrics)
    save_gantt_chart(tc, out_dir, gdf, data.machine_line, f'{data.dataset_size}_mip_{solver_name}', data=data)

    best_bound = getattr(mip, 'best_bound', None)
    status = getattr(mip, 'solver_status_name', None) or getattr(mip, 'solver_status', None) or getattr(getattr(getattr(mip, 'model', None), 'solve_details', None), 'status', None)
    status_str = str(status).strip().upper() if status is not None else ''
    gap_pct = (obj - best_bound) / max(1e-12, abs(obj)) * 100.0 if best_bound is not None and (obj - best_bound) > 1e-6 else (0.0 if best_bound is not None else (mip.mip_gap * 100 if mip.mip_gap is not None else None))
    proven = bool(not normalized and gap_pct is not None and gap_pct <= 1e-6 and ('OPTIMAL' in status_str))
    mip.proven_optimal, mip.reported_gap_pct = proven, gap_pct

    # Preserve best bound and MIP gap on data object for subsequent heuristic gap comparisons
    data.mip_best_bound = best_bound if best_bound is not None else (obj if (proven or (gap_pct is not None and gap_pct <= 1e-6)) else None)
    data.mip_gap = gap_pct

    row = _make_summary_row('mip', solver_name, 'solved_normalized' if normalized else 'solved', obj, metrics, mip.solve_time, total_time, gap_pct, data, solver_obj=solver_obj, proven=proven, claim='solver_tolerance_optimal' if proven else 'not_proven', model=mip.model, best_bound=best_bound, nodes=getattr(mip, 'node_count', None), status=status)
    safe_to_csv(pd.DataFrame([row]), os.path.join(out_dir, f'summary_{data.dataset_size}_mip_{solver_name}.csv'))
    return gdf, obj, total_time, gap_pct


def save_heuristic_results(tc, data, results, metas, ds_dir, metrics_cache=None):
    saved = {}
    cache = metrics_cache if metrics_cache is not None else {}
    for name, (df, obj, elapsed) in results.items():
        if name in {'best_greedy', 'roulette', 'lns'} and df is not None and not df.empty:
            out_dir = os.path.join(ds_dir, name)
            cache[name] = save_result(tc, out_dir, data.dataset_size, data, df, elapsed, method=name, metrics=cache.get(name))
            save_gantt_chart(tc, out_dir, df, data.machine_line, f'{data.dataset_size}_{name}', data=data)
            saved[name] = (df, obj, elapsed)
    return saved, {k: (metas or {}).get(k, {}) for k in saved}


def comparison_summary(tc, data, results, cmp_dir, mip_gap=None, metas=None, metrics_cache=None, mip_failure=None, mip_best_bound=None):
    labels = {'mip': 'MIP', 'best_greedy': 'B.Greedy', 'roulette': 'G+Roulette', 'lns': 'G+LNS'}
    metrics_rows = ['Lots', 'Products', 'Setup Time', 'Setup Count', 'Product-Machine Pairs', 'Moving Time', 'Moving Count', 'Proc Time', 'Processing (sec)', 'Total Time (sec)', 'Average Flow Time (sec)', 'Makespan', 'Schedule Span', 'Tardiness', 'Tardy Lots']
    extra_rows = [('Objective', 'objs'), ('Solve Time', 'times'), ('Number of Iterations', 'iters'), ('Best Iteration Found', 'best_iter'), ('Number of Updates', 'updates')]

    results = dict(results)
    if mip_failure and 'mip' not in results: results = {'mip': (pd.DataFrame(), None, mip_failure.get('elapsed')), **results}
    metas, cache = metas or {}, metrics_cache or {}
    stats, exact_objs = {}, {}

    for name, (df, _, elapsed) in results.items():
        if df is None or df.empty:
            stats[name] = {}
            continue
        m = cache.get(name) or schedule_metrics(df, data)
        exact_objs[name] = float(m['objective'])
        stats[name] = {'Lots': df['lot ID'].nunique(), 'Products': df['Product ID'].nunique(), 'Setup Time': tc.seconds_to_hhmmss(df['Setup Time'].sum()) if tc is not None else '00:00:00', 'Setup Count': int((df['Setup Time'] > 0).sum()), 'Product-Machine Pairs': m['product_machine_pairs'], 'Moving Time': tc.seconds_to_hhmmss(m['moving_seconds']) if tc is not None else '00:00:00', 'Moving Count': int((df['Moving Time'] > 0).sum()), 'Proc Time': tc.seconds_to_hhmmss(df['Processing Time'].sum()) if tc is not None else '00:00:00', 'Processing (sec)': f"{m['processing_seconds']:.2f}", 'Total Time (sec)': f"{m['total_time_seconds']:.2f}", 'Average Flow Time (sec)': f"{m['average_flowtime_seconds']:.2f}", 'Makespan': tc.seconds_to_hhmmss(m['makespan_seconds']) if tc is not None else '00:00:00', 'Schedule Span': tc.seconds_to_hhmmss(m['schedule_span_seconds']) if tc is not None else '00:00:00', 'Tardiness': tc.seconds_to_hhmmss(m['tardiness_seconds']) if tc is not None else '00:00:00', 'Tardy Lots': m['tardy_lots'],}

    meta_val = lambda n, k: str(metas.get(n, {}).get(k, '-'))
    ctx = {'stats': stats, 'objs': {n: exact_objs.get(n, 'N/A') for n in results}, 'times': {n: (f'{el:.2f}s' if el is not None else 'N/A') for n, (_, _, el) in results.items()}, 'iters': {n: meta_val(n, 'number_iterations') for n in results}, 'best_iter': {n: meta_val(n, 'best_iteration') for n in results}, 'updates': {n: meta_val(n, 'number_updates') for n in results},}

    best_val = min(exact_objs.values()) if exact_objs else None
    ctx['absolute_gap'] = {n: ('N/A' if n not in exact_objs or best_val is None else f'{exact_objs[n] - best_val:.6f}') for n in results}
    ctx['rpd'] = {n: ('N/A' if n not in exact_objs or best_val is None or abs(best_val) <= 1e-9 else f'{((exact_objs[n] - best_val) / abs(best_val) * 100):.4f}%') for n in results}
    extra_rows.extend([('Absolute Gap vs Best', 'absolute_gap'), ('RPD vs Best (%)', 'rpd')])

    # Determine MIP best bound for MIP Gap evaluation across all methods
    bound = mip_best_bound if mip_best_bound is not None else getattr(data, 'mip_best_bound', None)
    if bound is None and 'mip' in exact_objs:
        if mip_gap is not None and abs(mip_gap) <= 1e-6: bound = exact_objs['mip']
        elif mip_gap is not None and mip_gap < 100.0: bound = exact_objs['mip'] * (1.0 - mip_gap / 100.0)

    mip_benchmark = exact_objs.get('mip')
    # Fallback: check for previously saved MIP summary on disk if bound is not in memory
    if hasattr(data, 'dataset_size'):
        dataset_dir = os.path.dirname(cmp_dir)
        mip_dir = os.path.join(dataset_dir, 'mip')
        if os.path.isdir(mip_dir):
            for fname in sorted(os.listdir(mip_dir)):
                if fname.startswith(f'summary_{data.dataset_size}_mip_') and fname.endswith('.csv'):
                    try:
                        sdf = pd.read_csv(os.path.join(mip_dir, fname))
                        if mip_benchmark is None and 'objective' in sdf.columns and pd.notna(sdf['objective'].iloc[0]): mip_benchmark = float(sdf['objective'].iloc[0])
                        if bound is None and 'best_bound' in sdf.columns and pd.notna(sdf['best_bound'].iloc[0]): bound = float(sdf['best_bound'].iloc[0])
                        elif bound is None and 'objective' in sdf.columns and 'gap_pct' in sdf.columns:
                            m_obj = float(sdf['objective'].iloc[0]); m_gap = float(sdf['gap_pct'].iloc[0]) if pd.notna(sdf['gap_pct'].iloc[0]) else 0.0
                            bound = m_obj * (1.0 - m_gap / 100.0)
                        if bound is not None and mip_benchmark is not None: break
                    except Exception: pass

    # Calculate MIP Gap (%) for each method
    has_gap_info = (mip_gap is not None) or (bound is not None) or (mip_benchmark is not None)
    gap_values = {}
    if has_gap_info:
        for n in results:
            if n == 'mip':
                if mip_gap is not None: gap_values[n] = f'{mip_gap:.2f}%'
                elif bound is not None and n in exact_objs:
                    diff = exact_objs[n] - bound
                    val = 0.0 if diff <= 1e-6 else (diff / max(1e-12, abs(exact_objs[n]))) * 100.0
                    gap_values[n] = f'{val:.2f}%'
                else: gap_values[n] = '-'
            else:
                benchmark = mip_benchmark if mip_benchmark is not None else bound
                if benchmark is not None and n in exact_objs:
                    diff = exact_objs[n] - benchmark
                    val = 0.0 if diff <= 1e-6 else (diff / max(1e-12, abs(benchmark))) * 100.0
                    gap_values[n] = f'{val:.2f}%'
                else: gap_values[n] = '-'

    keys = [k for k in ['mip', 'best_greedy', 'roulette', 'lns'] if k in results]
    if keys:
        col_w = 14
        width = 28 + (col_w + 1) * len(keys)
        sep = '=' * width
        print(f"\n{sep}\n  SUMMARY\n{sep}")
        print(f"{'Metric':<28}" + ''.join(f" {labels.get(n, n):>{col_w}}" for n in keys) + f"\n{'-' * width}")
        for m in metrics_rows: print(f'{m:<28}' + ''.join(f" {str(ctx['stats'][n].get(m, 'N/A')):>{col_w}}" for n in keys))
        print('-' * width)
        for lbl, k in extra_rows:
            vals = {n: (f"{ctx[k][n]:.4f}" if lbl == 'Objective' and isinstance(ctx[k][n], (int, float)) else str(ctx[k][n])) for n in keys}
            print(f'{lbl:<28}' + ''.join(f" {vals[n]:>{col_w}}" for n in keys))
        if has_gap_info: print(f"{'MIP Gap':<28}" + ''.join(f" {gap_values.get(n, '-'):>{col_w}}" for n in keys))
        print(sep)

    rows = [{'Metric': m, **{labels.get(n, n): ctx['stats'][n].get(m, 'N/A') for n in keys}} for m in metrics_rows] + [{'Metric': lbl, **{labels.get(n, n): ctx[k][n] for n in keys}} for lbl, k in extra_rows]
    if has_gap_info: rows.append({'Metric': 'MIP Gap', **{labels.get(n, n): gap_values.get(n, '-') for n in keys}})
    safe_to_csv(pd.DataFrame(rows), os.path.join(cmp_dir, f'comparison_{data.dataset_size}_summary.csv'))
    print(f"\n  Saved to: {cmp_dir}\n")


compare_results = comparison_summary
