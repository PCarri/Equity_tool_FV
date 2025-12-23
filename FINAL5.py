# =========================
# Equity-Linked Product Playground (V2) — FULL SCRIPT (UPDATED)
# ✅ Investor selections -> correlated GBM basket -> buy&hold distribution
# ✅ Structured products shown in COLUMNS (one column per strategy)
# ✅ Payoff plot is LIGHT: only ONE selected asset (varied), others held at S0
# ✅ Histograms per strategy: Buy&Hold vs Structured product
# =========================

import math
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib as mpl
from scipy.stats import norm

mpl.rcParams["font.family"] = "serif"
st.set_page_config(layout="wide")

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
REGIONS = ["Local", "Foreign 1", "Foreign 2"]
REGION_IDX = {"Local": 1, "Foreign 1": 2, "Foreign 2": 3}

STRATEGIES = [
    "ZCB + Call",
    "ZCB + Call Spread",
    "ZCB + Risk Reversal (Long Call + Short Put)",
]


# ------------------------------------------------------------
# Defaults + Practitioner Panel
# ------------------------------------------------------------
def make_default_params():
    """
    fx is: (local currency) per 1 unit of foreign currency.
    Local fx = 1.0
    fx_T is a deterministic scenario FX at maturity used ONLY for payoff conversion in Compo mode.
    """
    return {
        "Local": {
            "S0": 100.0,
            "r": 0.03,
            "sigma": 0.20,
            "q": 0.00,
            "fx": 1.0,
            "fx_T": 1.0,
            "correl_11": 1.0,
            "correl_12": 0.30,
            "correl_13": 0.25,
            "fx_correl_11": 1.0,
            "sigma_fx": 0.0,
        },
        "Foreign 1": {
            "S0": 120.0,
            "r": 0.04,
            "sigma": 0.22,
            "q": 0.00,
            "fx": 0.92,
            "fx_T": 0.92,
            "correl_21": 0.30,
            "correl_22": 1.0,
            "correl_23": 0.35,
            "fx_correl_21": 0.65,
            "sigma_fx": 0.12,
        },
        "Foreign 2": {
            "S0": 80.0,
            "r": 0.01,
            "sigma": 0.24,
            "q": 0.00,
            "fx": 0.0062,
            "fx_T": 0.0062,
            "correl_31": 0.25,
            "correl_32": 0.35,
            "correl_33": 1.0,
            "fx_correl_31": -0.30,
            "sigma_fx": 0.15,
        },
    }


def practitioner_panel(params_by_region):
    with st.sidebar.expander("Practitioner control panel (advanced)", expanded=False):
        st.caption("Edit market inputs (defaults are used if not changed).")

        for region in REGIONS:
            st.markdown(f"**{region}**")
            params_by_region[region]["S0"] = st.number_input(
                f"{region} S0 (asset ccy)",
                value=float(params_by_region[region]["S0"]),
                step=1.0,
            )
            params_by_region[region]["r"] = st.number_input(
                f"{region} r (cont.)",
                value=float(params_by_region[region]["r"]),
                step=0.001,
                format="%.4f",
            )
            params_by_region[region]["sigma"] = st.number_input(
                f"{region} sigma (asset)",
                value=float(params_by_region[region]["sigma"]),
                step=0.01,
                format="%.4f",
            )
            params_by_region[region]["q"] = st.number_input(
                f"{region} q (div yield)",
                value=float(params_by_region[region]["q"]),
                step=0.001,
                format="%.4f",
            )
            params_by_region[region]["fx"] = st.number_input(
                f"{region} FX t0 (local per 1 ccy)",
                value=float(params_by_region[region]["fx"]),
                step=0.0001,
                format="%.6f",
            )
            params_by_region[region]["fx_T"] = st.number_input(
                f"{region} FX at T scenario (for Compo payoff)",
                value=float(params_by_region[region].get("fx_T", params_by_region[region]["fx"])),
                step=0.0001,
                format="%.6f",
            )
            params_by_region[region]["sigma_fx"] = st.number_input(
                f"{region} FX sigma (for pricing adj.)",
                value=float(params_by_region[region]["sigma_fx"]),
                step=0.01,
                format="%.4f",
            )

        st.markdown("---")
        st.markdown("**Regions Correlations**")
        corr_12 = st.slider("corr(Local, Foreign 1)", -0.95, 0.95, float(params_by_region["Local"]["correl_12"]), 0.01)
        corr_13 = st.slider("corr(Local, Foreign 2)", -0.95, 0.95, float(params_by_region["Local"]["correl_13"]), 0.01)
        corr_23 = st.slider("corr(Foreign 1, Foreign 2)", -0.95, 0.95, float(params_by_region["Foreign 1"]["correl_23"]), 0.01)

        st.markdown("**Foreign asset — FX correlations (for pricing adj.)**")
        fx_correl_21 = st.slider("corr(Foreign 1 asset, FX)", -0.95, 0.95, float(params_by_region["Foreign 1"]["fx_correl_21"]), 0.01)
        fx_correl_31 = st.slider("corr(Foreign 2 asset, FX)", -0.95, 0.95, float(params_by_region["Foreign 2"]["fx_correl_31"]), 0.01)

        params_by_region["Local"]["correl_12"] = corr_12
        params_by_region["Foreign 1"]["correl_21"] = corr_12

        params_by_region["Local"]["correl_13"] = corr_13
        params_by_region["Foreign 2"]["correl_31"] = corr_13

        params_by_region["Foreign 1"]["correl_23"] = corr_23
        params_by_region["Foreign 2"]["correl_32"] = corr_23

        params_by_region["Local"]["correl_11"] = 1.0
        params_by_region["Foreign 1"]["correl_22"] = 1.0
        params_by_region["Foreign 2"]["correl_33"] = 1.0

        params_by_region["Foreign 1"]["fx_correl_21"] = fx_correl_21
        params_by_region["Foreign 2"]["fx_correl_31"] = fx_correl_31
        params_by_region["Local"]["fx_correl_11"] = 1.0

    return params_by_region


# ------------------------------------------------------------
# Investor UI
# ------------------------------------------------------------
def investor_firstSelections():
    st.sidebar.markdown("## Investor inputs")

    investment_amount = float(st.sidebar.slider("Investment amount", 1_000, 100_000, 10_000, 1_000))
    T = float(st.sidebar.slider("Maturity T (years)", 1, 5, 1, 1))

    basket_composition = st.sidebar.multiselect(
        "Select regions:",
        options=REGIONS,
        default=["Local"],
        max_selections=3,
    )
    if len(basket_composition) == 0:
        st.warning("Select at least one region.")
        basket_composition = ["Local"]

    exposure_vector = np.array([1 if r in basket_composition else 0 for r in REGIONS], dtype=int)

    weights_pct = {r: 0.0 for r in REGIONS}
    st.sidebar.markdown("### Basket weights")

    if len(basket_composition) == 1:
        weights_pct[basket_composition[0]] = 100.0
        st.sidebar.info(f"100% in {basket_composition[0]}")
    elif len(basket_composition) == 2:
        r0, r1 = basket_composition
        w0 = st.sidebar.slider(f"Weight {r0} (%)", 0.0, 100.0, 50.0, 1.0)
        weights_pct[r0] = w0
        weights_pct[r1] = 100.0 - w0
    else:
        r0, r1, r2 = basket_composition
        w0 = st.sidebar.slider(f"Weight {r0} (%)", 0.0, 100.0, 34.0, 1.0)
        rem = 100.0 - w0
        w1 = st.sidebar.slider(f"Weight {r1} (%)", 0.0, rem, min(33.0, rem), 1.0)
        weights_pct[r0] = w0
        weights_pct[r1] = w1
        weights_pct[r2] = 100.0 - w0 - w1

    has_foreign = any(r in basket_composition for r in ["Foreign 1", "Foreign 2"])
    if has_foreign:
        fx_mode = st.sidebar.radio(
            "FX treatment",
            options=["Take FX risk (Compo)", "Hedge FX risk (Quanto)"],
            index=0,
        )
    else:
        fx_mode = "Plain (Local only)"

    weights_vector = np.array([weights_pct[r] for r in basket_composition], dtype=float) / 100.0
    return investment_amount, T, basket_composition, exposure_vector, weights_vector, fx_mode


# ------------------------------------------------------------
# Correlation matrix + PSD fix
# ------------------------------------------------------------
def build_corr_matrix(basket_composition, params_by_region, region_idx=None):
    if region_idx is None:
        region_idx = REGION_IDX

    n = len(basket_composition)
    C = np.eye(n, dtype=float)
    for i, ri in enumerate(basket_composition):
        ii = region_idx[ri]
        for j, rj in enumerate(basket_composition):
            jj = region_idx[rj]
            key = f"correl_{ii}{jj}"
            if key in params_by_region[ri]:
                C[i, j] = float(params_by_region[ri][key])
            else:
                key2 = f"correl_{jj}{ii}"
                C[i, j] = float(params_by_region[rj].get(key2, 0.0))
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def ensure_psd_corr(C, eps=1e-10):
    eig_min = np.min(np.linalg.eigvalsh(C))
    if eig_min >= eps:
        return C
    I = np.eye(C.shape[0])
    alpha = 0.0
    while alpha <= 1.0:
        C2 = (1 - alpha) * C + alpha * I
        if np.min(np.linalg.eigvalsh(C2)) >= eps:
            return C2
        alpha += 0.05
    return I


# ------------------------------------------------------------
# Correlated GBM simulator
# ------------------------------------------------------------
def simulate_correlated_gbm(basket_composition, params_by_region, T, n_steps, n_paths, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_assets = len(basket_composition)
    dt = T / n_steps

    C = build_corr_matrix(basket_composition, params_by_region)
    C = ensure_psd_corr(C)
    L = np.linalg.cholesky(C)

    S0 = np.array([params_by_region[r]["S0"] for r in basket_composition], dtype=float)
    r_vec = np.array([params_by_region[r]["r"] for r in basket_composition], dtype=float)
    q_vec = np.array([params_by_region[r].get("q", 0.0) for r in basket_composition], dtype=float)
    sig = np.array([params_by_region[r]["sigma"] for r in basket_composition], dtype=float)

    Z = rng.standard_normal(size=(n_steps, n_paths, n_assets))
    Z_corr = Z @ L.T

    drift = ((r_vec - q_vec) - 0.5 * sig**2) * dt
    diffusion = sig * np.sqrt(dt)

    log_paths = np.zeros((n_steps + 1, n_paths, n_assets), dtype=float)
    log_paths[0, :, :] = np.log(S0)[None, :]

    for t in range(1, n_steps + 1):
        log_paths[t, :, :] = log_paths[t - 1, :, :] + drift + diffusion * Z_corr[t - 1, :, :]

    paths = np.exp(log_paths)
    time = np.linspace(0.0, T, n_steps + 1)
    terminal = paths[-1, :, :]
    return time, paths, terminal, C


# ------------------------------------------------------------
# Buy & Hold basket (local currency using FX at t0)
# ------------------------------------------------------------
def buy_and_hold_basket_from_paths(investment_amount, basket_composition, weights_vector, params_by_region, paths):
    fx_vec = np.array([params_by_region[r].get("fx", 1.0) for r in basket_composition], dtype=float)
    S0_vec = np.array([params_by_region[r]["S0"] for r in basket_composition], dtype=float)
    price0_loc = S0_vec * fx_vec

    budget_i = investment_amount * weights_vector
    N_shares = budget_i / price0_loc

    paths_local = paths * fx_vec[None, None, :]
    basket_V = np.einsum("i,tpi->tp", N_shares, paths_local)

    logret_step = np.log(basket_V[1:, :] / basket_V[:-1, :])
    logret_T = np.log(basket_V[-1, :] / basket_V[0, :])
    return fx_vec, price0_loc, N_shares, basket_V, logret_step, logret_T


# ------------------------------------------------------------
# ZCB helper
# ------------------------------------------------------------
def zcb_pv(face_value, r_cont, T):
    return float(face_value) * np.exp(-float(r_cont) * float(T))


# ------------------------------------------------------------
# Black–Scholes pricing (Plain / Compo / Quanto)
# Output PV is in LOCAL currency.
# ------------------------------------------------------------
def bs_from_forward(F, K, df, sigma, tau, is_call=True):
    F = float(F); K = float(K); df = float(df); sigma = float(sigma); tau = float(tau)
    if tau <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return df * intrinsic
    vol_sqrt = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * tau) / vol_sqrt
    d2 = d1 - vol_sqrt
    if is_call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def get_fx_corr(params_by_region, region):
    idx = REGION_IDX[region]
    key = f"fx_correl_{idx}1"   # fx_correl_21, fx_correl_31
    return float(params_by_region[region].get(key, 0.0))


def compo_sigma(sigma_s, sigma_x, rho_sx):
    return float(np.sqrt(sigma_s**2 + sigma_x**2 + 2.0 * rho_sx * sigma_s * sigma_x))


def price_option_region(region, K, T, params_by_region, fx_mode, is_call=True):
    tau = float(T)
    p = params_by_region[region]

    S0 = float(p["S0"])
    r_f = float(p["r"])
    q_f = float(p.get("q", 0.0))
    sigma_s = float(p["sigma"])

    r_d = float(params_by_region["Local"]["r"])
    df_d = np.exp(-r_d * tau)

    # Local / plain
    if region == "Local" or fx_mode == "Plain (Local only)":
        df_f = np.exp(-r_f * tau)
        F = S0 * np.exp((r_f - q_f) * tau)
        pv_asset_ccy = bs_from_forward(F=F, K=K, df=df_f, sigma=sigma_s, tau=tau, is_call=is_call)
        fx0 = float(p.get("fx", 1.0))
        return fx0 * pv_asset_ccy

    fx0 = float(p.get("fx", 1.0))
    sigma_x = float(p.get("sigma_fx", 0.0))
    rho_sx = float(get_fx_corr(params_by_region, region))

    if fx_mode == "Take FX risk (Compo)":
        sigma_c = compo_sigma(sigma_s, sigma_x, rho_sx)
        S0_dom = fx0 * S0
        K_dom = fx0 * K
        F_dom = S0_dom * np.exp((r_d - q_f) * tau)
        return bs_from_forward(F=F_dom, K=K_dom, df=df_d, sigma=sigma_c, tau=tau, is_call=is_call)

    # Quanto
    adj = rho_sx * sigma_s * sigma_x
    F_quanto = S0 * np.exp((r_f - q_f - adj) * tau)
    pv_asset = bs_from_forward(F=F_quanto, K=K, df=df_d, sigma=sigma_s, tau=tau, is_call=is_call)
    return fx0 * pv_asset


# ------------------------------------------------------------
# Structured product terminal values (payoff uses deterministic fx_T)
# ------------------------------------------------------------
def structured_product_terminal_values(
    investment_amount,
    protection_pct,
    basket_composition,
    weights_vector,
    params_by_region,
    fx_mode,
    paths,
    T,
    strategy_name,
    call_up=1.10,
    spread_up=1.10,
    rr_call=1.05,
    rr_put=0.95
):
    r_d = float(params_by_region["Local"]["r"])

    face = investment_amount * (protection_pct / 100.0)
    pv_zcb = zcb_pv(face, r_d, T)
    opt_budget_total = investment_amount - pv_zcb
    budget_i = opt_budget_total * weights_vector

    S0_vec = np.array([params_by_region[r]["S0"] for r in basket_composition], dtype=float)
    fx0_vec = np.array([params_by_region[r].get("fx", 1.0) for r in basket_composition], dtype=float)
    fxT_vec = np.array([params_by_region[r].get("fx_T", params_by_region[r].get("fx", 1.0)) for r in basket_composition], dtype=float)

    S0_local = S0_vec * fx0_vec
    base_shares = (investment_amount * weights_vector) / S0_local

    S_T = paths[-1, :, :]
    payoff_assets = np.zeros((S_T.shape[0], len(basket_composition)), dtype=float)
    cash_T = np.zeros(len(basket_composition), dtype=float)
    strikes = {}

    for i, region in enumerate(basket_composition):
        S0 = float(params_by_region[region]["S0"])

        # Pricing uses theoretical compo/quanto adjustments; payoff conversion uses deterministic fx_T
        if region == "Local":
            payoff_fx = 1.0
        else:
            payoff_fx = fxT_vec[i] if fx_mode == "Take FX risk (Compo)" else fx0_vec[i]

        if strategy_name == "ZCB + Call":
            K = call_up * S0
            strikes[(region, "K")] = K

            c0 = price_option_region(region, K, T, params_by_region, fx_mode, is_call=True)
            denom = base_shares[i] * c0
            part = 0.0 if denom <= 0 else min(1.0, budget_i[i] / denom)

            n_calls = part * base_shares[i]
            spent = n_calls * c0
            leftover = budget_i[i] - spent

            payoff_assets[:, i] = n_calls * payoff_fx * np.maximum(S_T[:, i] - K, 0.0)
            cash_T[i] = max(0.0, leftover) * np.exp(r_d * T)

        elif strategy_name == "ZCB + Call Spread":
            K1 = 1.00 * S0
            K2 = spread_up * S0
            strikes[(region, "K1")] = K1
            strikes[(region, "K2")] = K2

            c1 = price_option_region(region, K1, T, params_by_region, fx_mode, is_call=True)
            c2 = price_option_region(region, K2, T, params_by_region, fx_mode, is_call=True)
            net0 = max(c1 - c2, 1e-12)

            denom = base_shares[i] * net0
            part = 0.0 if denom <= 0 else min(1.0, budget_i[i] / denom)

            n_spreads = part * base_shares[i]
            spent = n_spreads * net0
            leftover = budget_i[i] - spent

            payoff_assets[:, i] = n_spreads * payoff_fx * (
                np.maximum(S_T[:, i] - K1, 0.0) - np.maximum(S_T[:, i] - K2, 0.0)
            )
            cash_T[i] = max(0.0, leftover) * np.exp(r_d * T)

        elif strategy_name == "ZCB + Risk Reversal (Long Call + Short Put)":
            Kc = rr_call * S0
            Kp = rr_put * S0
            strikes[(region, "Kc")] = Kc
            strikes[(region, "Kp")] = Kp

            c0 = price_option_region(region, Kc, T, params_by_region, fx_mode, is_call=True)
            p0 = price_option_region(region, Kp, T, params_by_region, fx_mode, is_call=False)
            net0 = c0 - p0

            if net0 <= 0:
                n_rr = base_shares[i]
                credit = (-net0) * n_rr
                cash_T[i] = (budget_i[i] + credit) * np.exp(r_d * T)
            else:
                denom = base_shares[i] * net0
                part = 0.0 if denom <= 0 else min(1.0, budget_i[i] / denom)

                n_rr = part * base_shares[i]
                spent = n_rr * net0
                leftover = budget_i[i] - spent
                cash_T[i] = max(0.0, leftover) * np.exp(r_d * T)

            payoff_assets[:, i] = n_rr * payoff_fx * (
                np.maximum(S_T[:, i] - Kc, 0.0) - np.maximum(Kp - S_T[:, i], 0.0)
            )
        else:
            raise ValueError(f"Unknown strategy_name: {strategy_name}")

    V_T = face + payoff_assets.sum(axis=1) + cash_T.sum()
    details = {
        "face": face,
        "pv_zcb": pv_zcb,
        "opt_budget_total": opt_budget_total,
        "budget_i": budget_i,
        "base_shares": base_shares,
        "strikes": strikes,
        "cash_T_total": float(cash_T.sum()),
    }
    return V_T, details


# ------------------------------------------------------------
# Positions + payoff curves (vary ONE asset)
# ------------------------------------------------------------
def compute_positions_for_payoff(
    investment_amount,
    protection_pct,
    basket_composition,
    weights_vector,
    params_by_region,
    fx_mode,
    T,
    strategy_name,
    call_up=1.10,
    spread_up=1.10,
    rr_call=1.05,
    rr_put=0.95
):
    r_d = float(params_by_region["Local"]["r"])

    face = investment_amount * (protection_pct / 100.0)
    pv_zcb = zcb_pv(face, r_d, T)
    opt_budget_total = investment_amount - pv_zcb
    budget_i = opt_budget_total * weights_vector

    S0_vec = np.array([params_by_region[r]["S0"] for r in basket_composition], dtype=float)
    fx0_vec = np.array([params_by_region[r].get("fx", 1.0) for r in basket_composition], dtype=float)
    fxT_vec = np.array([params_by_region[r].get("fx_T", params_by_region[r].get("fx", 1.0)) for r in basket_composition], dtype=float)

    S0_local = S0_vec * fx0_vec
    base_shares = (investment_amount * weights_vector) / S0_local

    positions = []
    cash_T = np.zeros(len(basket_composition), dtype=float)

    for i, region in enumerate(basket_composition):
        S0 = float(params_by_region[region]["S0"])

        payoff_fx = 1.0 if region == "Local" else (fxT_vec[i] if fx_mode == "Take FX risk (Compo)" else fx0_vec[i])

        if strategy_name == "ZCB + Call":
            K = call_up * S0
            c0 = price_option_region(region, K, T, params_by_region, fx_mode, is_call=True)

            denom = base_shares[i] * c0
            part = 0.0 if denom <= 0 else min(1.0, budget_i[i] / denom)
            n_calls = part * base_shares[i]

            spent = n_calls * c0
            leftover = budget_i[i] - spent
            cash_T[i] = max(0.0, leftover) * np.exp(r_d * T)

            positions.append({"region": region, "type": "call", "K": K, "n": n_calls, "payoff_fx": payoff_fx})

        elif strategy_name == "ZCB + Call Spread":
            K1 = 1.00 * S0
            K2 = spread_up * S0

            c1 = price_option_region(region, K1, T, params_by_region, fx_mode, is_call=True)
            c2 = price_option_region(region, K2, T, params_by_region, fx_mode, is_call=True)
            net0 = max(c1 - c2, 1e-12)

            denom = base_shares[i] * net0
            part = 0.0 if denom <= 0 else min(1.0, budget_i[i] / denom)
            n_spreads = part * base_shares[i]

            spent = n_spreads * net0
            leftover = budget_i[i] - spent
            cash_T[i] = max(0.0, leftover) * np.exp(r_d * T)

            positions.append({"region": region, "type": "call_spread", "K1": K1, "K2": K2, "n": n_spreads, "payoff_fx": payoff_fx})

        elif strategy_name == "ZCB + Risk Reversal (Long Call + Short Put)":
            Kc = rr_call * S0
            Kp = rr_put * S0

            c0 = price_option_region(region, Kc, T, params_by_region, fx_mode, is_call=True)
            p0 = price_option_region(region, Kp, T, params_by_region, fx_mode, is_call=False)
            net0 = c0 - p0

            if net0 <= 0:
                n_rr = base_shares[i]
                credit = (-net0) * n_rr
                cash_T[i] = (budget_i[i] + credit) * np.exp(r_d * T)
            else:
                denom = base_shares[i] * net0
                part = 0.0 if denom <= 0 else min(1.0, budget_i[i] / denom)
                n_rr = part * base_shares[i]

                spent = n_rr * net0
                leftover = budget_i[i] - spent
                cash_T[i] = max(0.0, leftover) * np.exp(r_d * T)

            positions.append({"region": region, "type": "risk_reversal", "Kc": Kc, "Kp": Kp, "n": n_rr, "payoff_fx": payoff_fx})

        else:
            raise ValueError(f"Unknown strategy_name: {strategy_name}")

    return positions, float(face), float(cash_T.sum()), S0_vec


def payoff_curve_one_asset(
    investment_amount,
    protection_pct,
    basket_composition,
    weights_vector,
    params_by_region,
    fx_mode,
    T,
    strategy_name,
    vary_region,
    call_up=1.10,
    spread_up=1.10,
    rr_call=1.05,
    rr_put=0.95,
    grid_min=0.5,
    grid_max=1.8,
    n_points=220
):
    positions, face, cash_T_total, S0_vec = compute_positions_for_payoff(
        investment_amount=investment_amount,
        protection_pct=protection_pct,
        basket_composition=basket_composition,
        weights_vector=weights_vector,
        params_by_region=params_by_region,
        fx_mode=fx_mode,
        T=T,
        strategy_name=strategy_name,
        call_up=call_up,
        spread_up=spread_up,
        rr_call=rr_call,
        rr_put=rr_put
    )

    i_vary = basket_composition.index(vary_region)
    S0_i = float(S0_vec[i_vary])

    S_grid = np.linspace(grid_min * S0_i, grid_max * S0_i, n_points)
    V_grid = np.zeros_like(S_grid, dtype=float)

    ST_vec = S0_vec.copy()

    for k, ST_i in enumerate(S_grid):
        ST_vec[i_vary] = ST_i

        payoff_sum = 0.0
        for pos in positions:
            reg = pos["region"]
            j = basket_composition.index(reg)
            ST = float(ST_vec[j])
            fx_pay = float(pos["payoff_fx"])

            if pos["type"] == "call":
                payoff_sum += pos["n"] * fx_pay * max(ST - pos["K"], 0.0)
            elif pos["type"] == "call_spread":
                payoff_sum += pos["n"] * fx_pay * (max(ST - pos["K1"], 0.0) - max(ST - pos["K2"], 0.0))
            elif pos["type"] == "risk_reversal":
                payoff_sum += pos["n"] * fx_pay * (max(ST - pos["Kc"], 0.0) - max(pos["Kp"] - ST, 0.0))

        V_grid[k] = face + cash_T_total + payoff_sum

    return S_grid, V_grid


# =========================
# App starts here
# =========================
st.title("Equity-Linked Product Playground (V2)")
st.write("Correlated basket simulation + structured products per strategy (columns) + light payoff plots.")

# 1) Investor selections
investment_amount, T, basket_composition, exposure_vector, weights_vector, fx_mode = investor_firstSelections()

# 2) Default params + practitioner overrides
params_by_region = deepcopy(make_default_params())
params_by_region = practitioner_panel(params_by_region)

# 3) Simulation settings
dt = 1 / 252
n_steps = int(T / dt)
n_paths = 500
rng = np.random.default_rng(7)

# 4) Run correlated GBM
time, paths, terminal_prices, corr_used = simulate_correlated_gbm(
    basket_composition=basket_composition,
    params_by_region=params_by_region,
    T=T,
    n_steps=n_steps,
    n_paths=n_paths,
    rng=rng
)

# 5) Buy & hold basket
st.subheader("Buy & Hold basket (local currency, FX frozen at t0)")
fx_vec, price0_loc, N_shares, basket_V, logret_step, logret_T = buy_and_hold_basket_from_paths(
    investment_amount=investment_amount,
    basket_composition=basket_composition,
    weights_vector=weights_vector,
    params_by_region=params_by_region,
    paths=paths
)

colA, colB = st.columns([1.3, 1.0])
with colA:
    shares_df = pd.DataFrame({
        "Region": basket_composition,
        "Weight": weights_vector,
        "FX t0": fx_vec,
        "S0 (asset)": [params_by_region[r]["S0"] for r in basket_composition],
        "S0 (local)": price0_loc,
        "Budget (local)": investment_amount * weights_vector,
        "Shares N": N_shares,
    })
    st.dataframe(
        shares_df.style.format({
            "Weight": "{:.2%}",
            "FX t0": "{:.6f}",
            "S0 (asset)": "{:.2f}",
            "S0 (local)": "{:.2f}",
            "Budget (local)": "{:,.2f}",
            "Shares N": "{:.6f}",
        }),
        use_container_width=True
    )

with colB:
    corr_df = pd.DataFrame(corr_used, index=basket_composition, columns=basket_composition)
    st.dataframe(corr_df.style.format("{:.2f}"), use_container_width=True)

col1, col2 = st.columns([1.6, 1.0])
with col1:
    max_show = min(100, basket_V.shape[1])
    st.line_chart(pd.DataFrame(basket_V[:, :max_show], index=time), height=320)
with col2:
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=140)
    ax.hist(logret_T, bins=40)
    ax.set_xlabel("log(V_T / V_0)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Buy & Hold terminal log-return", fontsize=9)
    ax.tick_params(labelsize=8)
    st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Sidebar: structured product inputs (NO strategy selectbox)
# ------------------------------------------------------------
st.sidebar.markdown("## Structured products inputs")
protection_pct = st.sidebar.slider("Capital protection at maturity (%)", 0, 100, 100, 5)

call_up = st.sidebar.slider("Call-only strike moneyness", 1.00, 1.30, 1.10, 0.01)
spread_up = st.sidebar.slider("Call-spread upper strike moneyness", 1.01, 1.50, 1.10, 0.01)
rr_call = st.sidebar.slider("Risk reversal call moneyness", 1.00, 1.30, 1.05, 0.01)
rr_put = st.sidebar.slider("Risk reversal put moneyness", 0.70, 1.00, 0.95, 0.01)

payoff_region = st.sidebar.selectbox(
    "Payoff plot asset (vary this S_T)",
    options=basket_composition,
    index=0
)

st.header("Structured products (one column per strategy)")
st.info(
    "Pricing uses theoretical Compo/Quanto adjustments (sigma_fx, corr(asset,FX)). "
    "Payoff conversion uses deterministic FX scenario fx_T (not simulated)."
)

# ------------------------------------------------------------
# Caching helpers (to keep UI responsive)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_terminal_values(strategy_name, investment_amount, protection_pct, basket_composition, weights_vector,
                            params_by_region, fx_mode, T, call_up, spread_up, rr_call, rr_put, paths_last):
    V_T, details = structured_product_terminal_values(
        investment_amount=investment_amount,
        protection_pct=protection_pct,
        basket_composition=basket_composition,
        weights_vector=weights_vector,
        params_by_region=params_by_region,
        fx_mode=fx_mode,
        paths=paths,
        T=T,
        strategy_name=strategy_name,
        call_up=call_up,
        spread_up=spread_up,
        rr_call=rr_call,
        rr_put=rr_put
    )
    return V_T, details


@st.cache_data(show_spinner=False)
def _cached_payoff_curve(strategy_name, investment_amount, protection_pct, basket_composition, weights_vector,
                         params_by_region, fx_mode, T, call_up, spread_up, rr_call, rr_put, vary_region):
    S_grid, V_grid = payoff_curve_one_asset(
        investment_amount=investment_amount,
        protection_pct=protection_pct,
        basket_composition=basket_composition,
        weights_vector=weights_vector,
        params_by_region=params_by_region,
        fx_mode=fx_mode,
        T=T,
        strategy_name=strategy_name,
        vary_region=vary_region,
        call_up=call_up,
        spread_up=spread_up,
        rr_call=rr_call,
        rr_put=rr_put
    )
    return S_grid, V_grid


# ------------------------------------------------------------
# Display columns: one per strategy
# ------------------------------------------------------------
cols = st.columns(len(STRATEGIES), gap="large")
paths_last = np.asarray(paths[-1, :, :])  # cache key

for col, strat in zip(cols, STRATEGIES):
    with col:
        st.subheader(strat)

        # 1) Payoff plot (ONE asset only)
        st.markdown("**Payoff (vary one asset, others held at S0)**")
        S_grid, V_grid = _cached_payoff_curve(
            strategy_name=strat,
            investment_amount=investment_amount,
            protection_pct=protection_pct,
            basket_composition=basket_composition,
            weights_vector=weights_vector,
            params_by_region=params_by_region,
            fx_mode=fx_mode,
            T=T,
            call_up=call_up,
            spread_up=spread_up,
            rr_call=rr_call,
            rr_put=rr_put,
            vary_region=payoff_region
        )

        fig, ax = plt.subplots(figsize=(4.0, 2.6), dpi=140)
        ax.plot(S_grid, V_grid, linewidth=2.0)
        ax.axvline(float(params_by_region[payoff_region]["S0"]), linestyle="--", linewidth=1.0)
        ax.set_xlabel(f"{payoff_region}: S_T (asset ccy)", fontsize=8)
        ax.set_ylabel("Value at T (local)", fontsize=8)
        ax.set_title(f"Payoff vs {payoff_region}", fontsize=9)
        ax.tick_params(labelsize=8)
        st.pyplot(fig, use_container_width=True)

        # 2) Histogram: buy&hold vs structured
        st.markdown("**Return distribution (Buy & Hold vs Structured)**")

        V_T, details = _cached_terminal_values(
            strategy_name=strat,
            investment_amount=investment_amount,
            protection_pct=protection_pct,
            basket_composition=basket_composition,
            weights_vector=weights_vector,
            params_by_region=params_by_region,
            fx_mode=fx_mode,
            T=T,
            call_up=call_up,
            spread_up=spread_up,
            rr_call=rr_call,
            rr_put=rr_put,
            paths_last=paths_last
        )

        sp_logret_T = np.log(V_T / investment_amount)

        fig, ax = plt.subplots(figsize=(4.0, 2.8), dpi=140)
        ax.hist(logret_T, bins=35, alpha=0.6, label="Buy & Hold")
        ax.hist(sp_logret_T, bins=35, alpha=0.6, label="Structured")
        ax.set_xlabel("Terminal log return", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title("Histogram comparison", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
        st.pyplot(fig, use_container_width=True)

        st.caption(
            f"ZCB face: {details['face']:,.2f} | "
            f"ZCB PV: {details['pv_zcb']:,.2f} | "
            f"Options budget: {details['opt_budget_total']:,.2f}"
        )
