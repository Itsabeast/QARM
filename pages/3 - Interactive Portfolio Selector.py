import streamlit as st
from backend.style_utils import apply_sidebar_style
st.set_page_config(page_title="Results", layout="wide")
apply_sidebar_style()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from datetime import datetime
import json

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


st.title("🎚️ Interactive Portfolio Selector")
st.markdown("Explore different risk-return tradeoffs along the efficient frontier by selecting alternative portfolios.")

# --- 1. Session State Checks ---
if not st.session_state.get("optimization_run"):
    st.warning("⚠️ Please run the optimization on the **Inputs** page first.")
    st.stop()

# --- 2. Load Data ---
opt_df = st.session_state["opt_df"]
initial_asset = st.session_state["initial_asset"]
liab_value = st.session_state["liab_value"]

# Find Optimal (Best)
best_idx = opt_df["objective"].idxmax()
best = opt_df.loc[best_idx]

# --- 3. Interactive Slider ---
# Initialize slider state if not present
if "selected_frontier_idx" not in st.session_state:
    st.session_state["selected_frontier_idx"] = int(best_idx)

st.markdown("### Select a Portfolio")
selected_idx = st.slider(
    "Move the slider to choose a risk/return profile",
    min_value=0,
    max_value=len(opt_df) - 1,
    value=st.session_state["selected_frontier_idx"],
    key="frontier_slider",
    help="0 = Most Aggressive (High Return, Low Solvency), Right = Most Conservative"
)
st.session_state["selected_frontier_idx"] = selected_idx
selected_port = opt_df.iloc[selected_idx]

# --- 4. Visualization (Fixed with Pareto Filter) ---
st.subheader("📉 Efficient Frontier")

# A. PARETO FILTER LOGIC (Fixes the "messy line" issue)
opt_sorted = opt_df.sort_values(by="solvency", ascending=False).copy()
opt_sorted["max_return_seen"] = opt_sorted["return"].cummax()
pareto_frontier = opt_sorted[opt_sorted["return"] >= opt_sorted["max_return_seen"]]
pareto_frontier = pareto_frontier.sort_values(by="solvency")

# B. PLOT
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# Plot the smooth Pareto line
ax.plot(
    pareto_frontier["solvency"] * 100, 
    pareto_frontier["return"] * 100, 
    '-', 
    color='#4ECDC4', 
    linewidth=2.5, 
    label='Efficient Frontier',
    zorder=2
)

# Plot all feasible points faintly
ax.scatter(
    opt_df["solvency"] * 100, 
    opt_df["return"] * 100, 
    s=30, 
    color='gray', 
    alpha=0.2, 
    label='Feasible Portfolios',
    zorder=1
)

# Highlight Selected Portfolio (Purple Circle)
sel_sol = selected_port["solvency"] * 100
sel_ret = selected_port["return"] * 100
ax.scatter(
    sel_sol, sel_ret, 
    s=500, c='#9B59B6', marker='o', 
    edgecolors='#6C3483', linewidth=3, 
    label='Selected Portfolio', 
    zorder=10
)

# Highlight Optimal Portfolio (Gold Star)
ax.scatter(
    best["solvency"] * 100, 
    best["return"] * 100, 
    s=400, c='#FFD700', marker='*', 
    edgecolors='#FF8C00', linewidth=2, 
    label='Optimal Portfolio', 
    zorder=9
)

# Add annotation for selected point
ax.annotate(
    f'SELECTED\n{sel_ret:.2f}% | {sel_sol:.1f}%',
    xy=(sel_sol, sel_ret), xytext=(20, 20),
    textcoords='offset points', fontsize=10, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#9B59B6', edgecolor='#6C3483', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='#6C3483', lw=2), 
    color='white', zorder=11
)

ax.set_xlabel('Solvency Ratio (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.25, linestyle='--')

st.pyplot(fig, use_container_width=True)

# --- 5. Metrics ---
st.markdown("---")
st.subheader("📊 Selected Portfolio Metrics")

c1, c2, c3 = st.columns(3)
c1.metric(
    "Expected Return", 
    f"{selected_port['return']*100:.2f}%", 
    f"{(selected_port['return'] - best['return'])*100:+.2f}pp vs Optimal"
)
c2.metric(
    "Solvency Ratio", 
    f"{selected_port['solvency']*100:.1f}%", 
    f"{(selected_port['solvency'] - best['solvency'])*100:+.1f}pp vs Optimal"
)
c3.metric(
    "SCR Market", 
    f"€{selected_port['SCR_market']:.1f}m",
    f"€{(selected_port['SCR_market'] - best['SCR_market']):+.1f}m",
    delta_color="inverse"
)

# --- 6. Allocation Detail (Fixed Formatting) ---
st.markdown("---")
st.subheader("💼 Allocation Detail")

df_alloc = pd.DataFrame({
    "Asset Class": ["Gov Bonds", "Corp Bonds", "Equity 1", "Equity 2", "Property", "T-Bills"],
    "Weight (%)": selected_port["w_opt"] * 100,
    "Amount (€m)": selected_port["A_opt"]
})

# Apply dictionary-based formatting to prevent 'Unknown format code' error
st.dataframe(
    df_alloc.style.format({
        "Weight (%)": "{:.1f}", 
        "Amount (€m)": "{:.1f}"
    }).background_gradient(subset=["Weight (%)"], cmap="Blues"),
    use_container_width=True,
    hide_index=True
)

# --- 7. Exports ---
st.markdown("---")
st.subheader("💾 Export Selected Portfolio")

c_ex1, c_ex2, c_ex3 = st.columns(3)

with c_ex1:
    csv_data = df_alloc.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Allocation CSV",
        data=csv_data,
        file_name="selected_portfolio.csv",
        mime="text/csv",
        use_container_width=True
    )

with c_ex2:
    report_txt = f"""SELECTED PORTFOLIO REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

METRICS:
Return:   {selected_port['return']:.2%}
Solvency: {selected_port['solvency']:.1%}
SCR:      €{selected_port['SCR_market']:.1f}m
BOF:      €{selected_port['BOF']:.1f}m

ALLOCATION:
"""
    for _, row in df_alloc.iterrows():
        report_txt += f"{row['Asset Class']:<15}: {row['Weight (%)']:>5.1f}% (€{row['Amount (€m)']:>8.1f}m)\n"
    
    st.download_button(
        label="📥 Download Text Report",
        data=report_txt,
        file_name="selected_report.txt",
        mime="text/plain",
        use_container_width=True
    )

with c_ex3:
    json_data = {
        "metrics": {
            "return": float(selected_port['return']), 
            "solvency": float(selected_port['solvency']),
            "scr": float(selected_port['SCR_market'])
        },
        "allocation": {
            "assets": df_alloc["Asset Class"].tolist(),
            "weights": df_alloc["Weight (%)"].tolist(),
            "amounts": df_alloc["Amount (€m)"].tolist()
        }
    }
    st.download_button(
        label="📥 Download JSON",
        data=json.dumps(json_data, indent=2),
        file_name="selected_portfolio.json",
        mime="application/json",
        use_container_width=True
    )
