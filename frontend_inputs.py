import streamlit as st
from typing import List, Dict

st.set_page_config(page_title="Insurer Asset Allocation (Solvency II)", layout="wide")

RET_DEFAULTS = {"r_gov":0.029, "r_corp":0.041, "r_eq1":0.064, "r_eq2":0.064, "r_prop":0.056, "r_tb":0.006}

# ---- Embedded regulatory constants (Solvency II standard formula) ----
STANDARD_SII = {
    "equity": {
        "type1": 0.39,   # Delegated Reg. (EU) 2015/35 Art.169
        "type2": 0.49,   # Delegated Reg. (EU) 2015/35 Art.169
        "lte":   0.22    # Long-Term Equity treatment (Art. 171a)
    },
    "property": 0.25,    # Art.174
    # Interest-rate shocks are term-structure based (Arts.166–167); keep firm-calibrated placeholders here:
    "interest_rate": {"up_default": 0.011, "down_default": 0.009},
    # Spread is rating *and* duration dependent (Art.176) — not a single %; we keep a display placeholder only.
    "spread_placeholder": 0.103
}

# --------------------------
# Validation helpers
# --------------------------
def validate_inputs(A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb, total_A, mode,
                    gov_min, gov_max, corp_max, illiq_max, tb_min, tb_max):
    errs, warns = [], []
    tot = A_gov + A_corp + A_eq1 + A_eq2 + A_prop + A_tb

    if mode == "weights":
        if abs(tot - 1.0) > 1e-6:
            errs.append(f"Weights must sum to 1.000 (current = {tot:.3f}).")
        denom = 1.0
    else:
        if abs(tot - total_A) > 1e-6:
            errs.append(f"Amounts must sum to Total A (€{total_A:.1f}m). Current = €{tot:.1f}m.")
        denom = max(total_A, 1e-9)

    # non-negativity
    if any(x < -1e-12 for x in (A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb)):
        errs.append("Allocations must be non-negative.")

    # weights regardless of mode
    w_gov, w_corp, w_eq1, w_eq2, w_prop, w_tb = [x/denom for x in (A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb)]

    # mandate limits
    if not (gov_min - 1e-9 <= w_gov <= gov_max + 1e-9):
        errs.append(f"Government weight {w_gov:.3f} violates [{gov_min:.2f}, {gov_max:.2f}].")
    if w_corp > corp_max + 1e-9:
        errs.append(f"Corporate weight {w_corp:.3f} exceeds {corp_max:.2f}.")
    if (w_eq1 + w_eq2 + w_prop) > illiq_max + 1e-9:
        errs.append(f"Illiquid (eq1+eq2+prop) {w_eq1+w_eq2+w_prop:.3f} exceeds {illiq_max:.2f}.")
    if not (tb_min - 1e-9 <= w_tb <= tb_max + 1e-9):
        warns.append(f"T-bills {w_tb:.3f} outside policy band [{tb_min:.2f}, {tb_max:.2f}].")

    return errs, warns


# --------------------------
# Helper for adaptive precision
# --------------------------
def smart_step(is_weights: bool, remaining: float):
    if is_weights:
        if remaining <= 0.01: return 0.0005
        if remaining <= 0.05: return 0.001
        return 0.005
    else:
        if remaining <= 5: return 0.1
        if remaining <= 20: return 0.5
        return 1.0


def clamp01(x: float) -> float:
    x = float(x)
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x
# --------------------------
# Use-case presets
# --------------------------
PRESETS = {
    "s1_lower_bond_returns": {
        # Return deltas (add to current values)
        "returns_delta": {"gov": -0.005, "corp": -0.010},
        # Limits to set/raise/lower (absolute targets)
        "limits": {},
        # Residual asset (optional: "A_tb","A_gov","A_corp","A_eq1","A_eq2","A_prop")
        "residual_key": "A_tb",
        # Optional starting allocation (applied once on activation)
        # Use weights if weights mode; amounts if amounts mode.
        "alloc": {"A_gov": 0.45, "A_corp": 0.35, "A_eq1": 0.00, "A_eq2": 0.06, "A_prop": 0.04}  # TBills becomes residual
    },
    "s3_high_liquidity": {
        "returns_delta": {},
        "limits": {"tbills_min": 0.05, "tbills_max": 0.10, "illiquid_max": 0.15},
        "residual_key": "A_tb",
        "alloc": {"A_gov": 0.40, "A_corp": 0.30, "A_eq1": 0.00, "A_eq2": 0.05, "A_prop": 0.05}
    }
}

def activate_preset(name: str):
    p = PRESETS[name]
    st.session_state["active_preset"] = name
    st.session_state["preset_returns_delta"] = p.get("returns_delta", {})
    st.session_state["preset_limits"] = p.get("limits", {})
    st.session_state["preset_residual_key"] = p.get("residual_key")
    st.session_state["preset_alloc_values"] = p.get("alloc", None)

    # apply-once flags for each block
    st.session_state["preset_alloc_applied"] = False
    st.session_state["preset_returns_applied"] = False
    st.session_state["preset_limits_applied"] = False


def clear_preset():
    for k in [
        "active_preset","preset_returns_delta","preset_limits","preset_residual_key",
        "preset_alloc_values","preset_alloc_applied","preset_returns_applied","preset_limits_applied"
    ]:
        st.session_state.pop(k, None)


# --------------------------
# Backend API placeholders
# --------------------------
def search_tickers(query: str) -> List[Dict]:
    """Call your backend search. Return [{'ticker':'IEGA.DE','name':'iShares Core Euro Govt Bond','class':'gov'}, ...]."""
    # TODO: replace with real API call
    demo = [
        {"ticker":"IEGA.DE","name":"iShares Core Euro Govt Bond", "class":"gov"},
        {"ticker":"IEAC.DE","name":"iShares Euro Corporate Bond", "class":"corp"},
        {"ticker":"SUSC.DE","name":"iShares Euro Corp Bond ESG", "class":"corp"},
        {"ticker":"LP01","name":"Bloomberg Pan-Euro HY Index proxy","class":"corp"},
        {"ticker":"RPRA","name":"FTSE EPRA/Nareit Dev Europe proxy","class":"prop"},
    ]
    q = query.lower()
    return [x for x in demo if q in x["ticker"].lower() or q in x["name"].lower()]

def post_to_optimizer(payload: Dict) -> Dict:
    """POST payload to optimizer; return solution dict with optimized weights, SCR breakdown, frontier point, etc."""
    # TODO: replace with real POST; here we just echo
    return {"status":"ok", "received": payload}

# --------------------------
# Session state containers
# --------------------------
if "positions" not in st.session_state:
    st.session_state.positions = []  # [{'ticker','name','class','weight'}]
if "weights_mode" not in st.session_state:
    st.session_state.weights_mode = "weights"  # 'weights' or 'amounts'

# --------------------------
# Sidebar: Use-cases presets
# --------------------------
st.sidebar.title("Use-cases")
st.sidebar.caption("Apply ready-made parameters to speed up testing.")
if st.sidebar.button("Apply: Lower bond returns (Scenario 1)"):
    activate_preset("s1_lower_bond_returns")
if st.sidebar.button("Apply: High liquidity requirement (Scenario 3)"):
    activate_preset("s3_high_liquidity")
if st.sidebar.button("Clear preset"):
    clear_preset()


# --------------------------
# Main layout
# --------------------------
st.title("Solvency II-aware Asset Allocation — Input Builder")

colA, colB = st.columns([2,1])
@st.cache_data(ttl=60)
def _search_backend(query: str):
    #TODO: Replace with real API when ready
    return search_tickers(query)

with colA:
    st.subheader("A) Portfolio & Tickers")
    st.radio("Input mode", ["weights", "amounts (€)"], key="weights_mode", horizontal=True)

    # 👇 add back the total asset input
    total_A = st.number_input(
        "Total Assets A (EUR millions, for amounts mode or to display totals)",
        min_value=0.0, value=1652.7, step=1.0
    )
    # Base allocation inputs (aggregated)
    st.markdown("**Aggregate by asset class**")

    is_weights = (st.session_state.weights_mode == "weights")
    budget_total = 1.0 if is_weights else float(total_A)
    base_step = 0.001 if is_weights else 0.1
    residual_choice = st.selectbox(
        "Residual (auto-computed) asset",
        ["Treasury bills", "Government bonds", "Corporate bonds", "Equity Type 1", "Equity Type 2", "Property"],
        index=0,
        key="residual_choice_label"  # <-- key so preset can adjust
    )
    RESIDUAL_KEY = {
        "Treasury bills": "A_tb",
        "Government bonds": "A_gov",
        "Corporate bonds": "A_corp",
        "Equity Type 1": "A_eq1",
        "Equity Type 2": "A_eq2",
        "Property": "A_prop",
    }[residual_choice]

    # If a preset specified residual, reflect it in the selector (once)
    if st.session_state.get("preset_residual_key"):
        rev_map = {"A_tb": "Treasury bills", "A_gov": "Government bonds", "A_corp": "Corporate bonds",
                   "A_eq1": "Equity Type 1", "A_eq2": "Equity Type 2", "A_prop": "Property"}
        desired_label = rev_map[st.session_state["preset_residual_key"]]
        if st.session_state.get("residual_choice_label") != desired_label:
            st.session_state["residual_choice_label"] = desired_label
            st.rerun()

    # Ordered assets; we'll make TBills the residual bucket
    ALL_ASSETS = [
        ("A_gov", "Government bonds"),
        ("A_corp", "Corporate bonds"),
        ("A_eq1", "Equity Type 1"),
        ("A_eq2", "Equity Type 2"),
        ("A_prop", "Property"),
        ("A_tb", "Treasury bills"),
    ]
    ORDER = [(k, lbl) for (k, lbl) in ALL_ASSETS if k != RESIDUAL_KEY]

    # Defaults
    DEFAULTS_W = {"A_gov": 0.474, "A_corp": 0.355, "A_eq1": 0.000, "A_eq2": 0.062, "A_prop": 0.025, "A_tb": 0.084}
    DEFAULTS_E = {"A_gov": 782.6, "A_corp": 586.0, "A_eq1": 0.0, "A_eq2": 102.5, "A_prop": 42.0, "A_tb": 139.6}

    # If a preset provided starting allocation, apply it ONCE (respect mode)
    preset_alloc = st.session_state.get("preset_alloc_values")
    if preset_alloc and not st.session_state.get("preset_alloc_applied", False):
        if is_weights:
            for k, v in preset_alloc.items():
                st.session_state[k] = float(v)
        else:
            # If in amounts mode, scale by total_A
            for k, v in preset_alloc.items():
                st.session_state[k] = float(v) * float(total_A)
        st.session_state["preset_alloc_applied"] = True

    # Initialize missing keys (use ALL_ASSETS so the residual is initialized too)
    for key, _ in ALL_ASSETS:
        if key not in st.session_state:
            st.session_state[key] = (DEFAULTS_W if is_weights else DEFAULTS_E)[key]

    # Draw first five sliders with a soft cap = budget_total
    # (No enforced per-slider max; we rely on the residual + validation)
    # ------- Dual control (number input + slider) for precise allocation -------

    def _ensure_defaults_for(key: str, default_val: float):
        # Set the single source of truth
        if key not in st.session_state:
            st.session_state[key] = float(default_val)
        # Set the widget mirrors (only once before instantiation)
        if f"{key}_num" not in st.session_state:
            st.session_state[f"{key}_num"] = float(st.session_state[key])
        if f"{key}_sld" not in st.session_state:
            st.session_state[f"{key}_sld"] = float(st.session_state[key])


    def _on_num_change(key: str):
        # Number box changed → update source of truth and mirror slider
        st.session_state[key] = float(st.session_state[f"{key}_num"])
        st.session_state[f"{key}_sld"] = float(st.session_state[key])


    def _on_sld_change(key: str):
        # Slider changed → update source of truth and mirror number box
        st.session_state[key] = float(st.session_state[f"{key}_sld"])
        st.session_state[f"{key}_num"] = float(st.session_state[key])



    # Draw first five assets with number box + slider bound together
    for key, label in ORDER:
        default_val = (DEFAULTS_W if is_weights else DEFAULTS_E)[key]
        _ensure_defaults_for(key, default_val)

        # UI row: [small number box] [slider]
        c_num, c_sld = st.columns([1, 5])

        maxv = 1.0 if is_weights else max(0.0, float(total_A))

        # 🔽 Adaptive step
        budget_total = 1.0 if is_weights else float(total_A)
        current_sum_excl = sum(float(st.session_state[k]) for k, _ in ORDER if k != key)
        remaining = budget_total - current_sum_excl
        step = smart_step(is_weights, max(0.0, remaining))
        # 🔼

        # Number input (precise)
        with c_num:
            st.number_input(
                f"{label} ({'w' if is_weights else '€m'})",
                min_value=0.0,
                max_value=float(maxv),
                value=float(st.session_state[f"{key}_num"]),
                step=step,
                key=f"{key}_num",
                on_change=_on_num_change,
                args=(key,),
                label_visibility="visible"
            )

        # Slider (coarse/faster)
        with c_sld:
            st.slider(
                f"{label} ({'weight' if is_weights else '€m'})",
                min_value=0.0,
                max_value=float(maxv),
                value=float(st.session_state[f"{key}_sld"]),
                step=step,
                key=f"{key}_sld",
                on_change=_on_sld_change,
                args=(key,),
            )

    budget_total = 1.0 if is_weights else float(total_A)
    sum_non_residual = sum(float(st.session_state[k]) for (k, _) in ORDER)  # ORDER excludes residual
    residual_value = budget_total - sum_non_residual

    # Put residual into session_state (no negatives shown in UI)
    st.session_state[RESIDUAL_KEY] = max(0.0, residual_value)

    # Read back locals for payload (keeps your payload code unchanged)
    A_gov = float(st.session_state["A_gov"])
    A_corp = float(st.session_state["A_corp"])
    A_eq1 = float(st.session_state["A_eq1"])
    A_eq2 = float(st.session_state["A_eq2"])
    A_prop = float(st.session_state["A_prop"])
    A_tb = float(st.session_state["A_tb"])

    # 3D) Show the chosen residual asset as read-only
    residual_label = dict(ALL_ASSETS)[RESIDUAL_KEY]
    st.number_input(
        f"{residual_label} (residual, auto-computed)",
        min_value=0.0,
        value=float(st.session_state[RESIDUAL_KEY]),
        step=smart_step(is_weights, max(0.0, residual_value)),  # nice: adaptive precision here too
        disabled=True
    )

    # Validation + progress bar (works with rotating residual)
    total_used = sum_non_residual + float(st.session_state[RESIDUAL_KEY])

    if is_weights:
        st.progress(min(1.0, total_used))
        if residual_value < -1e-9:
            st.error(
                f"Sum of non-residual weights = {sum_non_residual:.3f} exceeds 1.000 by {abs(residual_value):.3f}. Reduce them."
            )
        elif abs(total_used - 1.0) > 1e-6:
            st.warning(f"Current total = {total_used:.3f}. Adjust sliders to hit exactly 1.000.")
        else:
            st.success("Weights sum to 1.000 ✔")
    else:
        denom = max(total_A, 1e-9)
        st.progress(min(1.0, total_used / denom))
        if residual_value < -1e-9:
            st.error(
                f"Allocated €{sum_non_residual:.1f}m exceeds Total A = €{total_A:.1f}m by €{abs(residual_value):.1f}m. Reduce sliders."
            )
        elif abs(total_used - total_A) > 1e-6:
            st.warning(
                f"Allocated €{total_used:.1f}m out of €{total_A:.1f}m. "
                f"Residual {residual_label} is currently €{float(st.session_state[RESIDUAL_KEY]):.1f}m."
            )
        else:
            st.success("Amounts match Total A ✔")

if st.session_state.weights_mode == "weights":
    if st.button("Normalize weights"):
        s = max(1e-12, A_gov + A_corp + A_eq1 + A_eq2 + A_prop + A_tb)
        for k in ["A_gov","A_corp","A_eq1","A_eq2","A_prop","A_tb"]:
            st.session_state[k] = float(st.session_state[k] / s)
        st.rerun()

st.markdown("**Add securities (optional)**")
q = st.text_input("Search ticker / name")
if q:
    results = search_tickers(q)
    for r in results[:8]:
        if st.button(f"Add {r['ticker']} – {r['name']} ({r['class']})"):
            st.session_state.positions.append(
                {"ticker": r["ticker"], "name": r["name"], "class": r["class"], "weight": 0.0}
            )

if st.session_state.positions:
    st.caption("Added positions")
    for i, pos in enumerate(st.session_state.positions):
        c1, c2, c3, c4 = st.columns([2,3,2,2])
        c1.write(pos["ticker"])
        c2.write(pos["name"])
        pos["class"] = c3.selectbox(
            "Class",
            ["gov","corp","eq1","eq2","prop","tbills"],
            index=["gov","corp","eq1","eq2","prop","tbills"].index(pos["class"]),
            key=f"class_{i}"
        )
        pos["weight"] = c4.number_input(
            "Weight",
            min_value=0.0, max_value=1.0,
            value=pos["weight"], step=0.001,
            key=f"w_{i}"
        )

with colB:
    st.subheader("B) Liabilities & Durations")
    BE_value = st.number_input("BE value (€m)", min_value=0.0, value=1424.2, step=1.0)
    BE_dur   = st.number_input("BE duration (mod.)", min_value=0.0, value=6.6, step=0.1)
    dur_gov  = st.number_input("Gov bonds duration", 0.0, 50.0, 5.2, 0.1)
    dur_corp = st.number_input("Corp bonds duration",0.0, 50.0, 5.0, 0.1)
    dur_tb   = st.number_input("T-bills duration",    0.0, 10.0, 0.1, 0.1)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("C) Shocks / Risk factors")
    with st.expander("Advanced / Admin settings", expanded=False):
        admin = st.toggle("Enable overrides", value=False,
                          help="Keep OFF for standard-formula values. Turn ON only if your risk team updates parameters.")

    # Interest-rate (firm-calibrated placeholders; lock by default)
    ir_up = st.number_input(
        "Interest-rate shock (Up) — firm calibration (Arts.166–167)",
        min_value=0.0, max_value=1.0,
        value=STANDARD_SII["interest_rate"]["up_default"],
        step=0.001, disabled=not admin,
        help="IR-Up shocked term structure per Articles 166–167; firm-approved calibration."

    )
    ir_down = st.number_input(
        "Interest-rate shock (Down) — firm calibration (Arts.166–167)",
        min_value=0.0, max_value=1.0,
        value=STANDARD_SII["interest_rate"]["down_default"],
        step=0.001, disabled=not admin,
        help="IR-Down shocked term structure per Articles 166–167; firm-approved calibration."
    )

    st.markdown("**Equity shocks (Art.169)**")
    eq1_sh = st.number_input(
        "Equity Type 1 (standard formula)",
        min_value=0.0, max_value=1.0,
        value=STANDARD_SII["equity"]["type1"],
        step=0.01, disabled=not admin,
        help="Standard Formula Art.169: 39% (plus symmetric adj. if applied)."
    )
    eq2_sh = st.number_input(
        "Equity Type 2 (standard formula)",
        min_value=0.0, max_value=1.0,
        value=STANDARD_SII["equity"]["type2"],
        step=0.01, disabled=not admin,
        help="Standard Formula Art.169: 49% (plus symmetric adj. if applied)."
    )

    # Optional: Long-Term Equity factor (you may not send it in payload unless you support LTE classification)
    _ = st.number_input(
        "Long-Term Equity (LTE) factor (informational)",
        min_value=0.0, max_value=1.0,
        value=STANDARD_SII["equity"]["lte"],
        step=0.01, disabled=True
    )

    prop_sh = st.number_input(
        "Property (Art.174)",
        min_value=0.0, max_value=1.0,
        value=STANDARD_SII["property"],
        step=0.01, disabled=not admin,
        help="Standard Formula Art.174: 25% instantaneous decrease."
    )

    # Spread: show as read-only placeholder; back-end should derive from rating & duration per Art.176.
    corp_sp = st.number_input(
        "Spread (corp) — derived by backend (Art.176)",
        min_value=0.0, max_value=1.0,
        value=STANDARD_SII["spread_placeholder"],
        step=0.001, disabled=True
    )
    st.caption("Note: Spread risk is rating×duration dependent; not a single % in the regulation. The optimizer should compute it.")

with col2:
    st.subheader("D) Expected returns (annual)")

    # ---- Seed defaults so keys exist before widgets ----
    for k, v in RET_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = float(v)
    # ----------------------------------------------------

    # ---- APPLY PRESET RETURN DELTAS BEFORE WIDGETS (once) ----
    if st.session_state.get("preset_returns_delta") and not st.session_state.get("preset_returns_applied", False):
        for short, delta in st.session_state["preset_returns_delta"].items():
            key = f"r_{short}"
            base = float(st.session_state.get(key, RET_DEFAULTS[key]))
            st.session_state[key] = clamp01(base + float(delta))  # <-- clamp to [0,1]
        st.session_state["preset_returns_applied"] = True
    # -----------------------------------------------------------

    # Now create the widgets (they read the values already in session_state)
    r_gov  = st.number_input("gov",   min_value=0.0, max_value=1.0, value=float(st.session_state["r_gov"]),  step=0.001, key="r_gov")
    r_corp = st.number_input("corp",  min_value=0.0, max_value=1.0, value=float(st.session_state["r_corp"]), step=0.001, key="r_corp")
    r_eq1  = st.number_input("eq1",   min_value=0.0, max_value=1.0, value=float(st.session_state["r_eq1"]),  step=0.001, key="r_eq1")
    r_eq2  = st.number_input("eq2",   min_value=0.0, max_value=1.0, value=float(st.session_state["r_eq2"]),  step=0.001, key="r_eq2")
    r_prop = st.number_input("prop",  min_value=0.0, max_value=1.0, value=float(st.session_state["r_prop"]), step=0.001, key="r_prop")
    r_tb   = st.number_input("tbills",min_value=0.0, max_value=1.0, value=float(st.session_state["r_tb"]),   step=0.001, key="r_tb")

    # Re-bind locals (optional clarity)
    r_gov, r_corp, r_eq1, r_eq2, r_prop, r_tb = (
        st.session_state["r_gov"], st.session_state["r_corp"],
        st.session_state["r_eq1"], st.session_state["r_eq2"],
        st.session_state["r_prop"], st.session_state["r_tb"]
    )


with col3:
    st.subheader("E) Limits & Objective")

    # ---- APPLY PRESET LIMITS BEFORE WIDGETS (once) ----
    if st.session_state.get("preset_limits") and not st.session_state.get("preset_limits_applied", False):
        for k, v in st.session_state["preset_limits"].items():
            st.session_state[k] = float(v)
        st.session_state["preset_limits_applied"] = True
    # ---------------------------------------------------

    # Now create the widgets (they read pre-set values)
    gov_min = st.number_input("Gov min", 0.0, 1.0, 0.25, 0.01, key="gov_min")
    gov_max = st.number_input("Gov max", 0.0, 1.0, 0.75, 0.01, key="gov_max")
    corp_max = st.number_input("Corp max", 0.0, 1.0, 0.50, 0.01, key="corp_max")
    illiq_max = st.number_input("Illiquid (eq1+eq2+prop) max", 0.0, 1.0, 0.20, 0.01, key="illiq_max")
    tb_min = st.number_input("T-bills min", 0.0, 1.0, 0.01, 0.01, key="tbills_min")
    tb_max = st.number_input("T-bills max", 0.0, 1.0, 0.05, 0.01, key="tbills_max")

    gamma = st.slider("Penalty γ", 0.0, 10.0, 2.0, 0.1)
    R_choice = st.selectbox("Correlation matrix", ["market_IR_down","market_IR_up"])


st.divider()

# Build payload
def to_amounts(x):
    return x * total_A if st.session_state.weights_mode == "weights" else x

payload = {
    "units": "weights" if st.session_state.weights_mode=="weights" else "EUR",
    "A_total": total_A,
    "allocation": {
        "gov": to_amounts(A_gov),
        "corp": to_amounts(A_corp),
        "eq1": to_amounts(A_eq1),
        "eq2": to_amounts(A_eq2),
        "prop": to_amounts(A_prop),
        "tbills": to_amounts(A_tb),
    },
    "liabilities": {"BE_value": BE_value, "BE_duration": BE_dur},
    "durations": {"gov": dur_gov, "corp": dur_corp, "tbills": dur_tb},
    "returns": {"gov": r_gov, "corp": r_corp, "eq1": r_eq1, "eq2": r_eq2, "prop": r_prop, "tbills": r_tb},
    "shocks": {"ir_up": ir_up, "ir_down": ir_down, "eq1": eq1_sh, "eq2": eq2_sh, "prop": prop_sh, "corp_spread": corp_sp},
    "limits": {"gov_min": gov_min, "gov_max": gov_max, "corp_max": corp_max, "illiquid_max": illiq_max, "tbills_min": tb_min, "tbills_max": tb_max},
    "objective": {"gamma": gamma, "corr_matrix": R_choice},
    "tickers": st.session_state.positions
}

# Validate before enabling Optimize
errs, warns = validate_inputs(
    A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb,
    total_A, st.session_state.weights_mode,
    gov_min, gov_max, corp_max, illiq_max, tb_min, tb_max
)
for e in errs: st.error(e)
for w in warns: st.warning(w)

st.subheader("Payload preview")
st.json(payload)

can_optimize = (len(errs) == 0)
if st.button("Optimize", disabled=not can_optimize):
    resp = post_to_optimizer(payload)
    st.success("Submitted to optimizer.")
    st.json(resp)


