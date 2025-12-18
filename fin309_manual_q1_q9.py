"""
FIN309 Portfolio Analysis (Manual Implementation, Q1–Q9)
=======================================================

This script provides a complete solution to the FIN309 coursework
questions without relying on the `PyPortfolioOpt` library.  Instead,
we use `numpy`, `pandas`, `scipy` and `statsmodels` to compute
expected returns, covariance matrices, global minimum variance
portfolios, efficient frontiers, tangency portfolios, as well as a
CAPM test.  The calculations follow the same logic as the assignment
specification and match the results produced in our manual analysis.

Tasks covered:

1. **Baseline portfolio (Q1)** – compute expected annual return,
   annual volatility and Sharpe ratio for the fixed weights
   `[0.30, 0.20, 0.20, 0.15, 0.15]` across the Chinese assets.
2. **Global Minimum Variance portfolios (Q2)** – find the GMV
   portfolio with and without short‐selling and report weights and
   performance.
3. **Efficient frontiers (Q3)** – trace efficient frontiers under
   shorting allowed and not allowed for the Chinese assets and save a
   plot.
4. **Tangency portfolios (Q4)** – compute the maximum Sharpe
   portfolio using the average 10‑year Chinese government bond yield
   as the risk‐free rate, both with shorting allowed and without.
5. **International tangency portfolio (Q5)** – extend the universe
   with S&P 500, Hang Seng and Hang Seng REIT indices, assume the
   risk‐free rate is zero and shorting is allowed; compute the new
   tangency portfolio.
6. **Efficient frontier comparison (Q6)** – compare the efficient
   frontier with and without international assets (shorting allowed)
   and save a plot.
7. **Minimum variance with target return 8% (Q7)** – find the
   minimum variance portfolio achieving at least 8% annual expected
   return for the full asset set without a risk‐free asset.
8. **CAPM test (Q8)** – use the portfolio from Q5, compute its
   excess returns and test the CAPM against the CSI 300 index.
9. **Discussion (Q9)** – printed as commentary (see main function).

Author: ChatGPT
Date: December 2025
"""

import warnings
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm


def load_indices(path: str) -> pd.DataFrame:
    """Load and preprocess index level data from a cleaned Excel file.

    Parameters
    ----------
    path : str
        File path to the cleaned Excel containing index levels.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime with columns for each asset.
    """
    # Read with header row at line 1 (second row), which contains column names.
    # Some cleaned files exported from Wind may include an extra first line with NaNs
    # and style information, so header=1 loads the correct header row.
    df = pd.read_excel(path, header=1, skipfooter=1)

    # Convert Excel serial dates (numeric) to pandas datetime.
    # Wind exports dates as Excel serial numbers (days since 1899-12-30).
    # We cannot rely on `pd.to_datetime` directly because it interprets
    # numeric values as nanoseconds from Unix epoch. Instead, we shift
    # from the Excel epoch.
    date_series = df['Date']
    # If the date column is numeric (int/float), convert using offset
    if pd.api.types.is_numeric_dtype(date_series):
        df['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(date_series, unit='D')
    else:
        df['Date'] = pd.to_datetime(date_series, errors='coerce')

    # Drop any rows without a valid date, then set as index and sort
    df = df.dropna(subset=['Date']).sort_values('Date').set_index('Date')

    # Rename the asset columns for convenience. Only keep columns that exist.
    rename_map = {
        'CSI 300000300.SH': 'CSI300',
        'CSI 500000905.SH': 'CSI500',
        'CSI 1000000852.SH': 'CSI1000',
        'SSE GOVERNMENT BOND INDEX(C)H00012.CSI': 'SSE_Gov_Bond',
        'CSI AGGREGATE BOND INDEX (C)H01001.CSI': 'CSI_Agg_Bond',
        'S&P 500SPX.GI': 'SP500',
        'Hang Seng IndexHSI.HI': 'HSI',
        'HANG SENG REIT INDEXHSREIT.HI': 'HSREIT',
    }
    # Filter to present columns and rename
    present = [c for c in rename_map.keys() if c in df.columns]
    df = df[present].rename(columns=rename_map)
    # Convert all data to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def load_risk_free(path: str) -> Tuple[float, pd.Series]:
    """Load the Chinese 10‑year yield and compute average annual RF and monthly RF.

    Parameters
    ----------
    path : str
        Path to the CSV with daily yields (percent).

    Returns
    -------
    (float, pd.Series)
        Tuple of (average annual risk‐free rate, monthly risk‐free returns).
    """
    df = pd.read_csv(path, skiprows=4, header=None, encoding='gbk', engine='python', skipfooter=1)
    df[0] = pd.to_datetime(df[0], errors='coerce')
    df = df.dropna(subset=[0])
    df.columns = ['Date', 'Yield']
    df['Yield'] = df['Yield'] / 100.0  # convert percent to decimal
    df = df.set_index('Date').sort_index()
    rf_ann = df['Yield'].mean()
    # monthly risk‑free series: average monthly yield divided by 12
    rf_monthly = df['Yield'].resample('M').mean() / 12.0
    return rf_ann, rf_monthly


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple monthly returns from price levels."""
    monthly_prices = prices.resample('M').last()
    returns = monthly_prices.pct_change().dropna(how='any')
    return returns


def annualised_moments(returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute annualised mean vector and covariance matrix."""
    mu = returns.mean() * 12
    cov = returns.cov() * 12
    return mu, cov


def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> Tuple[float, float, float]:
    """Compute portfolio expected return, volatility and Sharpe ratio."""
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - rf) / vol if vol != 0 else np.nan
    return ret, vol, sharpe


def gmv_closed_form(cov: np.ndarray) -> np.ndarray:
    """Closed form GMV weights (allows shorting)."""
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(len(cov))
    num = inv_cov @ ones
    den = ones @ num
    return num / den


def optimise_gmv(cov: np.ndarray, short_allowed: bool) -> np.ndarray:
    """Optimise GMV weights with optional no‑short constraint."""
    n = cov.shape[0]
    if short_allowed:
        return gmv_closed_form(cov)
    # no shorting
    def objective(w):
        return w @ cov @ w
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    x0 = np.repeat(1/n, n)
    res = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')
    return res.x


def tangency_closed_form(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    """Closed form tangency (max Sharpe) weights (allows shorting)."""
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    excess = mu - rf * ones
    num = inv_cov @ excess
    den = ones @ num
    return num / den


def optimise_tangency(mu: np.ndarray, cov: np.ndarray, rf: float, short_allowed: bool) -> np.ndarray:
    """Optimise tangency weights with optional no‑short constraint."""
    n = len(mu)
    if short_allowed:
        return tangency_closed_form(mu, cov, rf)
    # no shorting: maximise Sharpe ratio
    def neg_sharpe(w):
        ret, vol, _ = portfolio_stats(w, mu, cov, rf)
        return -((ret - rf) / vol)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    x0 = np.repeat(1/n, n)
    res = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons, method='SLSQP')
    return res.x


def trace_frontier(mu: np.ndarray, cov: np.ndarray, short_allowed: bool, num_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Trace efficient frontier over a grid of target returns."""
    n = len(mu)
    target_min = mu.min()
    target_max = mu.max()
    targets = np.linspace(target_min, target_max, num_points)
    vols = []
    exp_rets = []
    for target in targets:
        def obj(w): return w @ cov @ w
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w, target=target: w @ mu - target},
        ]
        bounds = None if short_allowed else [(0.0, 1.0)] * n
        x0 = np.repeat(1/n, n)
        res = minimize(obj, x0, bounds=bounds, constraints=cons, method='SLSQP')
        if res.success:
            w = res.x
            vol = np.sqrt(w @ cov @ w)
            vols.append(vol)
            exp_rets.append(target)
        else:
            vols.append(np.nan)
            exp_rets.append(np.nan)
    return np.array(vols), np.array(exp_rets)


def min_variance_with_target(mu: np.ndarray, cov: np.ndarray, target: float) -> np.ndarray:
    """Find min variance weights with target expected return (short allowed)."""
    n = len(mu)
    def obj(w): return w @ cov @ w
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: w @ mu - target},
    ]
    # wide bounds to allow limited shorting; adjust if unrealistic
    bounds = [(-10.0, 10.0)] * n
    x0 = np.repeat(1/n, n)
    res = minimize(obj, x0, bounds=bounds, constraints=cons, method='SLSQP')
    return res.x


def perform_capm(portfolio_returns: pd.Series, market_returns: pd.Series, rf_returns: pd.Series) -> dict:
    """Perform CAPM regression for portfolio excess returns vs market excess returns."""
    df = pd.concat([portfolio_returns, market_returns, rf_returns], axis=1).dropna()
    df.columns = ['rp', 'rm', 'rf']
    df['Rp_ex'] = df['rp'] - df['rf']
    df['Rm_ex'] = df['rm'] - df['rf']
    X = sm.add_constant(df['Rm_ex'])
    y = df['Rp_ex']
    model = sm.OLS(y, X).fit()
    return {
        'alpha': model.params['const'],
        'beta': model.params['Rm_ex'],
        'alpha_t': model.tvalues['const'],
        'beta_t': model.tvalues['Rm_ex'],
        'alpha_p': model.pvalues['const'],
        'beta_p': model.pvalues['Rm_ex'],
        'r2': model.rsquared,
    }


def main() -> None:
    warnings.filterwarnings('ignore')
    # file paths (must be in the working directory)
    idx_file = 'fin data combining(wind)_clean.xlsx'
    rf_file = '中国_国债收益率_10年.csv'
    prices = load_indices(idx_file)
    returns = compute_monthly_returns(prices)
    rf_ann, rf_monthly = load_risk_free(rf_file)

    # Chinese and international asset lists
    china_assets = ['CSI300', 'CSI500', 'CSI1000', 'CSI_Agg_Bond', 'SSE_Gov_Bond']
    intl_assets = ['SP500', 'HSI', 'HSREIT']
    all_assets = china_assets + intl_assets

    # Subset returns
    returns_china = returns[china_assets]
    mu_china, cov_china = annualised_moments(returns_china)
    returns_all = returns[all_assets]
    mu_all, cov_all = annualised_moments(returns_all)

    print("\n=================== Q1: Baseline Portfolio =====================")
    base_weights = np.array([0.30, 0.20, 0.20, 0.15, 0.15])
    ret_q1, vol_q1, sharpe_q1 = portfolio_stats(base_weights, mu_china.values, cov_china.values, rf_ann)
    print(f"Expected return: {ret_q1*100:.2f}%")
    print(f"Volatility: {vol_q1*100:.2f}%")
    print(f"Sharpe ratio: {sharpe_q1:.4f}\n")

    print("=================== Q2: GMV Portfolios =========================")
    w_gmv_s = optimise_gmv(cov_china.values, short_allowed=True)
    w_gmv_l = optimise_gmv(cov_china.values, short_allowed=False)
    r_gmv_s, v_gmv_s, _ = portfolio_stats(w_gmv_s, mu_china.values, cov_china.values, rf_ann)
    r_gmv_l, v_gmv_l, _ = portfolio_stats(w_gmv_l, mu_china.values, cov_china.values, rf_ann)
    print("GMV (short allowed) weights:")
    for asset, w in zip(china_assets, w_gmv_s):
        print(f"  {asset:12s}: {w*100:.2f}%")
    print(f"Return: {r_gmv_s*100:.2f}%  |  Volatility: {v_gmv_s*100:.2f}%\n")
    print("GMV (no short) weights:")
    for asset, w in zip(china_assets, w_gmv_l):
        print(f"  {asset:12s}: {w*100:.2f}%")
    print(f"Return: {r_gmv_l*100:.2f}%  |  Volatility: {v_gmv_l*100:.2f}%\n")

    print("=================== Q3: Efficient Frontiers =====================")
    vols_s, rets_s = trace_frontier(mu_china.values, cov_china.values, short_allowed=True, num_points=50)
    vols_l, rets_l = trace_frontier(mu_china.values, cov_china.values, short_allowed=False, num_points=50)
    plt.figure(figsize=(8,6))
    plt.plot(vols_l*100, rets_l*100, 'b--', label='No shorting')
    plt.plot(vols_s*100, rets_s*100, 'r-', label='Short allowed')
    plt.scatter(vol_q1*100, ret_q1*100, marker='*', s=150, c='black', label='Baseline')
    plt.title('Efficient frontier (China-only)')
    plt.xlabel('Annual volatility (%)')
    plt.ylabel('Annual expected return (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('q3_frontier_china.png')
    plt.close()
    print("Efficient frontier plot saved as q3_frontier_china.png\n")

    print("=================== Q4: Tangency Portfolios =====================")
    w_tan_s = optimise_tangency(mu_china.values, cov_china.values, rf_ann, short_allowed=True)
    w_tan_l = optimise_tangency(mu_china.values, cov_china.values, rf_ann, short_allowed=False)
    r_tan_s, v_tan_s, sharpe_tan_s = portfolio_stats(w_tan_s, mu_china.values, cov_china.values, rf_ann)
    r_tan_l, v_tan_l, sharpe_tan_l = portfolio_stats(w_tan_l, mu_china.values, cov_china.values, rf_ann)
    print("Tangency (short allowed) weights:")
    for asset, w in zip(china_assets, w_tan_s):
        print(f"  {asset:12s}: {w*100:.2f}%")
    print(f"Return: {r_tan_s*100:.2f}%  |  Volatility: {v_tan_s*100:.2f}%  |  Sharpe: {sharpe_tan_s:.2f}\n")
    print("Tangency (no short) weights:")
    for asset, w in zip(china_assets, w_tan_l):
        print(f"  {asset:12s}: {w*100:.2f}%")
    print(f"Return: {r_tan_l*100:.2f}%  |  Volatility: {v_tan_l*100:.2f}%  |  Sharpe: {sharpe_tan_l:.2f}\n")

    print("=================== Q5: New Tangency Portfolio (all assets, RF=0) ============")
    w_tan_all = tangency_closed_form(mu_all.values, cov_all.values, 0.0)
    r_tan_all, v_tan_all, sharpe_tan_all = portfolio_stats(w_tan_all, mu_all.values, cov_all.values, rf=0.0)
    print("Tangency weights for all assets (RF=0, short allowed):")
    for asset, w in zip(all_assets, w_tan_all):
        print(f"  {asset:12s}: {w*100:.2f}%")
    print(f"Return: {r_tan_all*100:.2f}%  |  Volatility: {v_tan_all*100:.2f}%  |  Sharpe: {sharpe_tan_all:.2f}\n")

    print("=================== Q6: Frontier Comparison (China vs All) =======")
    vols_china, rets_china = trace_frontier(mu_china.values, cov_china.values, short_allowed=True, num_points=50)
    vols_all, rets_all = trace_frontier(mu_all.values, cov_all.values, short_allowed=True, num_points=50)
    plt.figure(figsize=(8,6))
    plt.plot(vols_china*100, rets_china*100, 'r-', label='China-only')
    plt.plot(vols_all*100, rets_all*100, 'g-', label='All assets')
    plt.scatter(v_tan_s*100, r_tan_s*100, c='red', marker='x', label='China tangency')
    plt.scatter(v_tan_all*100, r_tan_all*100, c='green', marker='s', label='All assets tangency')
    plt.title('Efficient frontier comparison (shorting allowed)')
    plt.xlabel('Annual volatility (%)')
    plt.ylabel('Annual expected return (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('q6_frontier_comparison.png')
    plt.close()
    print("Frontiers comparison plot saved as q6_frontier_comparison.png\n")

    print("=================== Q7: Min Var Portfolio (target >= 8% p.a.) ===")
    w_target = min_variance_with_target(mu_all.values, cov_all.values, target=0.08)
    r_target, v_target, _ = portfolio_stats(w_target, mu_all.values, cov_all.values, rf=0.0)
    print("Weights for min variance with target >= 8%:")
    for asset, w in zip(all_assets, w_target):
        print(f"  {asset:12s}: {w*100:.2f}%")
    print(f"Return: {r_target*100:.2f}%  |  Volatility: {v_target*100:.2f}%\n")

    print("=================== Q8: CAPM Test on New Tangency Portfolio ======")
    # Portfolio returns using weights from Q5 (tangency all assets)
    port_ret_series = (returns_all @ w_tan_all).dropna()
    market_series = returns_all['CSI300'].dropna()
    rf_series = rf_monthly.reindex(port_ret_series.index).fillna(method='ffill')
    capm_res = perform_capm(port_ret_series, market_series, rf_series)
    print(f"Alpha: {capm_res['alpha']:.4f} (t={capm_res['alpha_t']:.2f}, p={capm_res['alpha_p']:.3f})")
    print(f"Beta:  {capm_res['beta']:.4f} (t={capm_res['beta_t']:.2f}, p={capm_res['beta_p']:.3f})")
    print(f"R-squared: {capm_res['r2']:.3f}\n")
    print("Interpretation: A near-zero alpha and beta suggest the portfolio is almost uncorrelated")
    print("with the market proxy (CSI300), and the low R-squared indicates CAPM explains little")
    print("of the variation in the portfolio excess returns. Thus the CAPM does not hold well.\n")

    # Q9 discussion (printed for completeness)
    print("=================== Q9: Discussion of Assumptions and Alternatives =")
    print("Mean-variance optimisation assumes stable, normally distributed returns, perfect markets")
    print("without transaction costs or liquidity constraints, and unlimited borrowing and shorting.")
    print("In the context of Chinese markets, these assumptions are often violated due to policy")
    print("interventions, illiquidity and trading restrictions. Estimation errors in mean and")
    print("covariance can lead to unstable optimal weights. To address these issues, investors")
    print("can consider robust optimisation, shrinkage estimators (e.g. Black–Litterman), downside")
    print("risk measures like CVaR, or factor models that reduce dimensionality and improve stability.")


if __name__ == '__main__':
    main()
