"""Job Shop Scheduling (JSS) - Interactive Streamlit Dashboard.

Run with:
    streamlit run app.py
"""
from __future__ import annotations
import os, sys, time, glob, json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocessing import ConfigLoader, DataPreprocessor, OBJECTIVES, load_objective_class
from utils.result_saver import (
    assert_schedule_feasible, compare_results, safe_to_csv,
    save_heuristic_results, save_mip_result, schedule_metrics,
    write_run_manifest, save_gantt_chart
)
from main import _run_mip

# Pass through any Gurobi WLS credentials configured in Streamlit Cloud Secrets
try:
    if hasattr(st, 'secrets') and 'gurobi' in st.secrets:
        for _k, _v in st.secrets['gurobi'].items():
            os.environ[f'GRB_{_k.upper()}'] = str(_v)
except Exception:
    pass

# Page Configuration
st.set_page_config(
    page_title="FJSS Optimization Suite",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


def get_available_datasets():
    return {
        "Small (10 Lots, 63 Machines)": "1",
        "Medium (80 Lots, 63 Machines)": "2",
        "Large (350 Lots, 63 Machines)": "3"
    }


def get_available_objectives():
    return {
        "tardy_only": "Minimize Weighted Tardiness",
        "tardy_move": "Minimize Tardiness + Moving Time",
        "tardy_move_setup": "Minimize Tardiness + Moving + Setup Time",
        "tardy_total_time": "Minimize Tardiness + Total Time",
        "tardy_move_makespan": "Minimize Tardiness + Moving + Makespan"
    }


def find_existing_runs():
    out_root = PROJECT_ROOT / "output"
    runs = []
    if out_root.exists():
        for d in sorted(out_root.iterdir()):
            if d.is_dir():
                cmp_file = d / "compare" / f"comparison_{d.name}_summary.csv"
                manifest = d / "run_manifest.json"
                if cmp_file.exists() or manifest.exists() or any((d / m).exists() for m in ('mip', 'best_greedy', 'lns', 'roulette')):
                    runs.append(d.name)
    return runs


def render_kpi_card(title: str, value: str, subtitle: str = ""):
    sub_html = f"<div class='kpi-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>{title}</div>
        <div class='kpi-value'>{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_comparison_charts(summary_df: pd.DataFrame):
    if summary_df.empty or 'Metric' not in summary_df.columns:
        return
    methods = [c for c in summary_df.columns if c != 'Metric']
    if not methods:
        return

    m_idx = summary_df.set_index('Metric')
    c1, c2 = st.columns(2)

    # Chart 1: Objective Comparison
    if 'Objective' in m_idx.index:
        obj_vals = []
        for m in methods:
            try: obj_vals.append(float(str(m_idx.loc['Objective', m]).replace(',', '')))
            except ValueError: obj_vals.append(None)
        
        fig_obj = go.Figure(go.Bar(
            x=methods, y=obj_vals,
            text=[f"{v:,.0f}" if v is not None else "N/A" for v in obj_vals],
            textposition='auto',
            marker=dict(color=['#3B82F6', '#10B981', '#F59E0B', '#6366F1'][:len(methods)])
        ))
        fig_obj.update_layout(
            title="<b>Objective Value Comparison (Lower is Better)</b>",
            yaxis_title="Objective Score",
            template="plotly_white", height=380, margin=dict(l=40, r=20, t=50, b=40)
        )
        c1.plotly_chart(fig_obj, use_container_width=True)

    # Chart 2: Makespan & Total Time
    if 'Makespan' in m_idx.index:
        def time_str_to_hours(val):
            try:
                parts = str(val).split(':')
                if len(parts) == 3: return float(parts[0]) + float(parts[1])/60.0 + float(parts[2])/3600.0
                return float(val) / 3600.0
            except Exception: return None

        ms_vals = [time_str_to_hours(m_idx.loc['Makespan', m]) for m in methods]
        fig_ms = go.Figure(go.Bar(
            x=methods, y=ms_vals,
            text=[f"{v:.1f}h" if v is not None else "N/A" for v in ms_vals],
            textposition='auto',
            marker=dict(color=['#06B6D4', '#8B5CF6', '#EC4899', '#14B8A6'][:len(methods)])
        ))
        fig_ms.update_layout(
            title="<b>Makespan Comparison (Hours)</b>",
            yaxis_title="Makespan (Hours)",
            template="plotly_white", height=380, margin=dict(l=40, r=20, t=50, b=40)
        )
        c2.plotly_chart(fig_ms, use_container_width=True)


def dynamic_gantt(df: pd.DataFrame, title: str):
    """Fallback interactive Plotly Gantt chart from schedule dataframe."""
    if df.empty or 'Machine' not in df.columns:
        st.warning("Schedule data is empty or missing 'Machine' column.")
        return
    
    machines = sorted(df['Machine'].unique())
    y_pos = {m: i for i, m in enumerate(machines)}
    lots = sorted(df['lot ID'].unique())
    palette = px.colors.qualitative.Plotly * (len(lots) // 10 + 1)
    color_map = {lot: palette[i] for i, lot in enumerate(lots)}

    fig = go.Figure()
    for _, r in df.iterrows():
        m, lot = r['Machine'], r['lot ID']
        start_sec, proc_sec = float(r.get('Start Time (sec)', 0)), float(r.get('Processing Time', 0))
        setup_sec = float(r.get('Setup Time', 0))
        prod = r.get('Product ID', '-')
        op = r.get('Operation Sequence', '-')

        if setup_sec > 0:
            fig.add_trace(go.Bar(
                x=[setup_sec / 3600.0], base=[max(0.0, start_sec - setup_sec) / 3600.0],
                y=[y_pos[m]], orientation='h', showlegend=False,
                marker=dict(color=color_map[lot], pattern=dict(shape='/', size=5)),
                hovertemplate=f"<b>[SETUP] Lot {lot}</b><br>Machine: {m}<br>Duration: {setup_sec:.0f}s<extra></extra>"
            ))
        
        fig.add_trace(go.Bar(
            x=[proc_sec / 3600.0], base=[start_sec / 3600.0],
            y=[y_pos[m]], orientation='h', name=f"Lot {lot}", showlegend=False,
            marker_color=color_map[lot],
            hovertemplate=f"<b>Lot {lot}</b> (Op {op})<br>Product: {prod}<br>Machine: {m}<br>Start: {start_sec/3600:.2f}h<br>Proc: {proc_sec/3600:.2f}h<extra></extra>"
        ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Time (hours)",
        barmode='overlay',
        height=max(500, 30 * len(machines)),
        yaxis=dict(tickmode='array', tickvals=list(y_pos.values()), ticktext=machines, autorange='reversed'),
        template="plotly_white", margin=dict(l=120, r=20, t=60, b=40)
    )
    fig.update_xaxes(rangeslider=dict(visible=True))
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown("## ⚙️ Optimization Control")
app_mode = st.sidebar.radio("Navigation Mode", ["🚀 Run Optimization", "📂 Explore Saved Runs"])

if app_mode == "🚀 Run Optimization":
    ds_choices = get_available_datasets()
    sel_ds_label = st.sidebar.selectbox("Dataset Size", list(ds_choices.keys()), index=0)
    sel_dataset_code = ds_choices[sel_ds_label]
    dataset_name = {'1': 'small', '2': 'medium', '3': 'large'}.get(sel_dataset_code, 'small')

    obj_choices = get_available_objectives()
    sel_obj = st.sidebar.selectbox("Objective Model", list(obj_choices.keys()), format_func=lambda k: f"{k} ({obj_choices[k]})")

    exec_mode = st.sidebar.selectbox(
        "Solver Mode",
        ["heuristic", "mip", "both"],
        index=0,
        format_func=lambda m: {
            "heuristic": "Heuristics Only (Greedy + Roulette + LNS)",
            "mip": "MIP Only (Exact Solver)",
            "both": "Both (Comparative Benchmarking)"
        }[m]
    )
    mip_solver = st.sidebar.selectbox("MIP Engine", ["gurobi", "cplex"], index=0) if exec_mode in ("mip", "both") else "gurobi"

    with st.sidebar.expander("🛠️ Hyperparameters & Overrides"):
        p_seed = st.number_input("Random Seed", value=42, step=1)
        p_iters = st.number_input("Heuristic Iterations", value=100, step=10, min_value=1)
        p_time_limit = st.number_input("MIP Time Limit (sec)", value=600.0, step=60.0, min_value=5.0)
        p_route_limit = st.number_input("Greedy Route Limit", value=5, step=1, min_value=1)
        p_threads = st.number_input("CPU Threads (0=Auto)", value=0, step=1, min_value=0)

    btn_run = st.sidebar.button("▶️ Start Scheduling", type="primary", use_container_width=True)

else:
    btn_run = False
    dataset_name = None
    existing_runs = find_existing_runs()
    if not existing_runs:
        st.sidebar.warning("No existing runs found in `output/` directory.")
        sel_run_dir = None
    else:
        sel_run_dir = st.sidebar.selectbox("Select Run Directory", existing_runs, index=0)


# ==========================================
# MAIN PAGE HEADER
# ==========================================
st.markdown("<div class='main-header'>Flexible Job Shop Scheduling (FJSS) Optimization</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>State-of-the-Art Exact Mathematical Programming (MIP) & Targeted Metaheuristics (Greedy, Roulette, LNS)</div>", unsafe_allow_html=True)


# ==========================================
# LIVE RUN EXECUTION LOGIC
# ==========================================
run_output_dir = None
if app_mode == "🚀 Run Optimization" and btn_run:
    with st.status("Running Optimization Pipeline...", expanded=True) as status:
        st.write("Initializing configurations and preprocessing datasets...")
        config = ConfigLoader.load(PROJECT_ROOT / 'parameter.csv', sel_obj)
        config['_project_root'] = str(PROJECT_ROOT)
        config['solver_seed'] = p_seed
        config['iterations'] = p_iters
        config['greedy_route_limit'] = p_route_limit
        if p_time_limit > 0: config['time_limit_seconds'] = p_time_limit
        if p_threads > 0: config['threads'] = p_threads

        dataset_name = {'1': 'small', '2': 'medium', '3': 'large'}.get(sel_dataset_code, 'custom')
        target_dir = PROJECT_ROOT / 'output' / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        config['output_dir'] = str(target_dir)

        data = DataPreprocessor(config)
        pipeline_ok = data.run_pipeline(sel_dataset_code)
        if not pipeline_ok or not data.P or getattr(data, 'greedy_state', None) is None:
            err_msg = getattr(data, 'last_error', None) or "Dataset files could not be loaded or processed."
            st.error(f"❌ **Preprocessing Failed**: {err_msg}")
            status.update(label="Preprocessing Failed", state="error")
            st.stop()

        st.write(f"Dataset preprocessed: **{len(data.P)} lots** on **{len(data.M)} machines** across **{len(data.G)} product families**.")

        write_run_manifest(str(target_dir), config, data, p_seed, exec_mode, mip_solver)
        results, metas, metrics_cache, mip_gap = {}, {}, {}, None

        # MIP Phase
        if exec_mode in ('mip', 'both'):
            st.write(f"Solving with MIP ({mip_solver.upper()})...")
            mip_start = time.perf_counter()
            mip_res = _run_mip(data, str(target_dir / 'mip'), mip_solver, sel_obj)
            if mip_res:
                frame, mip_obj, elapsed, mip_gap = mip_res
                results['mip'] = (frame, mip_obj, elapsed)
                st.write(f"MIP Complete in {elapsed:.2f}s | Objective: {mip_obj:,.2f}")
            else:
                if exec_mode == 'mip':
                    st.error(f"❌ **MIP Solver ({mip_solver.upper()}) Tidak Tersedia di Cloud**\n\nServer Streamlit Cloud gratis tidak memiliki lisensi/binary solver komersial Gurobi atau CPLEX.")
                    st.info("💡 **Solusi**: Di sidebar sebelah kiri, ubah pilihan **Solver Mode** menjadi **`Heuristics Only (Greedy + Roulette + LNS)`**, lalu klik kembali tombol **▶️ Start Scheduling**.\n\n*(Atau jika ingin melihat grafik perbandingan MIP vs Heuristik yang sudah dihitung sebelumnya, pilih menu **📂 Explore Saved Runs** di sidebar)*.")
                    status.update(label="MIP Solver Unavailable", state="error")
                    st.stop()
                else:
                    st.warning(f"⚠️ {mip_solver.upper()} solver tidak tersedia di cloud. Melanjutkan dengan optimasi Heuristik...")

        # Heuristic Phase
        if exec_mode in ('heuristic', 'both'):
            st.write("Executing Deterministic Greedy Portfolio (16 dispatching strategies)...")
            scheduler = load_objective_class(sel_obj, 'heuristic')(config, data)
            heuristic_results, greedy_seed, heuristic_metas = scheduler.run_greedy()
            frame, reported_obj, order, elapsed = greedy_seed
            assert_schedule_feasible(frame, data)
            m = schedule_metrics(frame, data)
            greedy_seed = (frame, float(m['objective']), order, elapsed)
            heuristic_results['best_greedy'] = (frame, float(m['objective']), elapsed)
            st.write(f"Best Greedy Seed: {float(m['objective']):,.2f} ({elapsed:.2f}s)")

            # Roulette
            st.write(f"Running Roulette Wheel Relocation ({p_iters} iterations)...")
            r_frame, r_obj, _, r_elapsed, r_meta = scheduler.run_roulette(greedy_seed)
            heuristic_results['roulette'], heuristic_metas['roulette'] = (r_frame, r_obj, r_elapsed), r_meta
            st.write(f"Roulette Best: {r_obj:,.2f} ({r_elapsed:.2f}s)")

            # LNS
            st.write(f"Running Targeted Large Neighborhood Search (LNS) ({p_iters} iterations)...")
            l_frame, l_obj, _, l_elapsed, l_meta = scheduler.run_lns(greedy_seed)
            heuristic_results['lns'], heuristic_metas['lns'] = (l_frame, l_obj, l_elapsed), l_meta
            st.write(f"LNS Best: {l_obj:,.2f} ({l_elapsed:.2f}s)")

            saved, saved_metas = save_heuristic_results(data.time, data, heuristic_results, heuristic_metas, str(target_dir), metrics_cache=metrics_cache)
            results.update(saved)
            metas.update(saved_metas)

        # Comparison summary
        if results or getattr(data, '_mip_failure', None):
            compare_results(data.time, data, results, str(target_dir / 'compare'), mip_gap=mip_gap, metas=metas, metrics_cache=metrics_cache, mip_failure=getattr(data, '_mip_failure', None), mip_best_bound=getattr(data, 'mip_best_bound', None))

        status.update(label="Optimization Complete! Schedules and analytics generated.", state="complete", expanded=False)
        st.session_state['active_run_dir'] = dataset_name


# Determine which directory to view
active_dir_name = st.session_state.get('active_run_dir', dataset_name) if app_mode == "🚀 Run Optimization" else sel_run_dir

if not active_dir_name:
    st.info("👈 Select parameters in the sidebar and click **Start Scheduling**, or switch to **Explore Saved Runs** to inspect existing schedules.")
    st.stop()

view_dir = PROJECT_ROOT / 'output' / active_dir_name
if not view_dir.exists():
    st.info(f"No existing results found for **{active_dir_name.capitalize()}**. Click **Start Scheduling** in the sidebar to run optimization.")
    st.stop()


# Load Run Data
cmp_csv_path = view_dir / 'compare' / f'comparison_{active_dir_name}_summary.csv'
summary_df = pd.read_csv(cmp_csv_path) if cmp_csv_path.exists() else pd.DataFrame()

# ==========================================
# TABS INTERFACE
# ==========================================
tab_summary, tab_gantt, tab_details = st.tabs([
    "📊 Comparative Summary & KPIs",
    "📅 Interactive Gantt Chart",
    "🔍 Detailed Schedules & Tardy Lots"
])

# ----------------------------------------------------
# TAB 1: SUMMARY & KPIS
# ----------------------------------------------------
with tab_summary:
    if not summary_df.empty and 'Metric' in summary_df.columns:
        m_idx = summary_df.set_index('Metric')
        methods = [c for c in summary_df.columns if c != 'Metric']
        
        # Determine best method
        best_m = methods[0]
        if 'Objective' in m_idx.index:
            try:
                valid_objs = {m: float(str(m_idx.loc['Objective', m]).replace(',', '')) for m in methods if str(m_idx.loc['Objective', m]) not in ('N/A', '-')}
                if valid_objs: best_m = min(valid_objs, key=valid_objs.get)
            except Exception: pass

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            obj_str = str(m_idx.loc['Objective', best_m]) if 'Objective' in m_idx.index else "-"
            render_kpi_card("Winning Objective", f"{float(obj_str):,.0f}" if obj_str.replace('.','',1).isdigit() else obj_str, f"Method: {best_m}")
        with k2:
            ms_str = str(m_idx.loc['Makespan', best_m]) if 'Makespan' in m_idx.index else "-"
            render_kpi_card("Best Makespan", ms_str, f"Span: {m_idx.loc['Schedule Span', best_m] if 'Schedule Span' in m_idx.index else '-'}")
        with k3:
            su_str = str(m_idx.loc['Setup Count', best_m]) if 'Setup Count' in m_idx.index else "-"
            su_dur = str(m_idx.loc['Setup Time', best_m]) if 'Setup Time' in m_idx.index else "-"
            render_kpi_card("Total Setups", f"{su_str} changeovers", f"Setup Duration: {su_dur}")
        with k4:
            tardy_lots = str(m_idx.loc['Tardy Lots', best_m]) if 'Tardy Lots' in m_idx.index else "-"
            mip_gap_val = str(m_idx.loc['MIP Gap', best_m]) if 'MIP Gap' in m_idx.index else "-"
            render_kpi_card("Tardy Lots / MIP Gap", f"{tardy_lots} lots tardy", f"MIP Gap: {mip_gap_val}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Comparative Performance Matrix")
        st.dataframe(summary_df.set_index('Metric'), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        render_comparison_charts(summary_df)
    else:
        st.info(f"No comparison summary CSV found in `{view_dir / 'compare'}`.")


# ----------------------------------------------------
# TAB 2: INTERACTIVE GANTT CHARTS
# ----------------------------------------------------
with tab_gantt:
    st.markdown("### 📅 Production Floor Gantt Schedule")
    available_methods = [d.name for d in view_dir.iterdir() if d.is_dir() and d.name in ('best_greedy', 'roulette', 'lns', 'mip')]
    
    if not available_methods:
        st.warning("No schedule folders found in the active run directory.")
    else:
        labels_map = {'mip': 'MIP (Exact Solver)', 'best_greedy': 'Best Greedy Seed', 'roulette': 'G+Roulette Wheel', 'lns': 'G+Large Neighborhood Search'}
        sel_method = st.selectbox("Select Solution Method to Visualize", available_methods, format_func=lambda k: labels_map.get(k, k))

        method_dir = view_dir / sel_method
        gantt_files = list(method_dir.glob("gantt_*.html"))

        if gantt_files:
            # Read saved Plotly HTML and embed directly
            with open(gantt_files[0], 'r', encoding='utf-8') as gf:
                html_content = gf.read()
            st.caption(f"Displaying saved interactive Gantt chart: `{gantt_files[0].name}`")
            components.html(html_content, height=750, scrolling=True)
        else:
            # Fallback: find CSV and render dynamically
            csv_files = list(method_dir.glob("schedule_*.csv")) or list(method_dir.glob("*.csv"))
            if csv_files:
                df_sched = pd.read_csv(csv_files[0])
                st.caption(f"Rendering schedule from `{csv_files[0].name}`")
                dynamic_gantt(df_sched, f"Gantt Schedule: {labels_map.get(sel_method, sel_method)}")
            else:
                st.warning(f"No Gantt chart HTML or schedule CSV found in `{method_dir}`.")


# ----------------------------------------------------
# TAB 3: DETAILED SCHEDULES & TARDY LOTS
# ----------------------------------------------------
with tab_details:
    st.markdown("### 🔍 Operation Schedule & Tardy Lots Inspection")
    available_methods = [d.name for d in view_dir.iterdir() if d.is_dir() and d.name in ('best_greedy', 'roulette', 'lns', 'mip')]
    if available_methods:
        det_method = st.selectbox("Inspect Schedule Method", available_methods, key="det_method")
        det_dir = view_dir / det_method
        sched_csvs = list(det_dir.glob("schedule_*.csv")) or [f for f in det_dir.glob("*.csv") if not f.name.startswith("summary")]

        if sched_csvs:
            df_full = pd.read_csv(sched_csvs[0])

            # Resolve Machine Column
            m_col = 'Machine ID' if 'Machine ID' in df_full.columns else ('Machine' if 'Machine' in df_full.columns else None)

            def _nat_key(s):
                import re
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s).strip())]

            # Filter Controls
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                all_lots = ["All"] + sorted(df_full['lot ID'].dropna().astype(str).unique().tolist(), key=_nat_key)
                sel_f_lot = st.selectbox("Filter by Lot ID", all_lots)
            with col_f2:
                if m_col:
                    all_mach = ["All"] + sorted(df_full[m_col].dropna().astype(str).unique().tolist(), key=_nat_key)
                else:
                    all_mach = ["All"]
                sel_f_mach = st.selectbox("Filter by Machine", all_mach)
            with col_f3:
                all_prod = ["All"] + sorted(df_full['Product ID'].dropna().astype(str).unique().tolist(), key=_nat_key) if 'Product ID' in df_full.columns else ["All"]
                sel_f_prod = st.selectbox("Filter by Product ID", all_prod)

            df_filtered = df_full.copy()
            if sel_f_lot != "All": df_filtered = df_filtered[df_filtered['lot ID'].astype(str) == sel_f_lot]
            if m_col and sel_f_mach != "All": df_filtered = df_filtered[df_filtered[m_col].astype(str) == sel_f_mach]
            if sel_f_prod != "All": df_filtered = df_filtered[df_filtered['Product ID'].astype(str) == sel_f_prod]

            st.dataframe(df_filtered, use_container_width=True, height=350)
            st.download_button("📥 Download Filtered Schedule (CSV)", df_filtered.to_csv(index=False), f"schedule_{det_method}_{active_dir_name}.csv", "text/csv")

            # Machine Workload Histogram
            if m_col and 'Processing Time' in df_full.columns:
                st.markdown("#### 🏭 Machine Workload Distribution (Hours)")
                m_load = df_full.groupby(m_col)['Processing Time'].sum() / 3600.0
                m_load = m_load.reindex(sorted(m_load.index, key=_nat_key))
                fig_load = px.bar(
                    x=m_load.index, y=m_load.values,
                    labels={'x': 'Machine', 'y': 'Total Processing (Hours)'},
                    title="<b>Machine Load Profile</b>", template="plotly_white"
                )
                fig_load.update_layout(height=350, margin=dict(l=40, r=20, t=50, b=40))
                st.plotly_chart(fig_load, use_container_width=True)
        else:
            st.info(f"No schedule CSV found in `{det_dir}`.")

