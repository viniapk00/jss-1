# Flexible Job Shop Scheduling (FJSS) Optimization Suite

State-of-the-Art Exact Mathematical Programming (MIP) & Targeted Heuristics (Greedy, Roulette Wheel, Targeted LNS).

---

## 🚀 Two Deployment Editions

This repository provides two specialized dashboard editions:

### 1. 💻 Localhost Workstation Edition (`app_local.py`)
Designed for running locally on your workstation/laptop with full computational power unlocked:
- **Full Exact Solvers**: Gurobi 13 & CPLEX with all CPU threads.
- **Large Dataset Solving**: Can run live MIP on Large (350 lots) with complete 600s+ search depth.
- **Run command**:
  ```powershell
  streamlit run app_local.py
  ```

### 2. 🌐 Public Cloud Demo Edition (`app_public.py` / `streamlit_app.py`)
Designed specifically for public deployment on **Streamlit Community Cloud** (3 GB container RAM limit):
- **Lightning Fast**: Targeted Heuristics (Greedy, Roulette, LNS) execute live in ~2–5 seconds with minimal RAM (< 150 MB).
- **Safe Small MIP**: Exact MIP runs live on Small dataset (10 lots) in ~0.15s.
- **Instant Verified Benchmarks**: Precomputed exact MIP benchmarks (with exact MIP Gap %) are available instantly under **Explore Saved Runs**, completely eliminating 10-minute web timeouts and OOM crashes.
- **Cloud Memory Protection**: Dual Simplex (`Method=1`), 2 threads, and disk nodefile to prevent memory spikes.
- **Run command**:
  ```powershell
  streamlit run app_public.py
  ```

> *Note: Running `streamlit run app.py` automatically detects whether it is running on Streamlit Cloud or Localhost and adjusts its defaults accordingly, while also offering an interactive switcher in the sidebar.*

---

## ⚡ Command Line (Headless) Benchmarking

For running multi-objective batch experiments in the terminal:

```powershell
# Run full comparative benchmark (MIP + Heuristics) on Large dataset with Gurobi
python run_all_types_gurobi.py --dataset 3 --mode both

# Run Heuristics only on Medium dataset
python run_all_types_gurobi.py --dataset 2 --mode heuristic

# Run single objective with custom time limit
python main.py --mode mip --dataset 1 --solver gurobi --objective tardy_total_time --time-limit 300
```
