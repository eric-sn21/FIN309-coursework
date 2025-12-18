"""
FIN309 Portfolio Analysis - PyPortfolioOpt Implementation (Q1-Q9)
This module implements portfolio optimization using PyPortfolioOpt library
for efficient and robust portfolio analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Import PyPortfolioOpt modules
from pypfopt import expected_returns, risk_models
from pypfopt import EfficientFrontier
from pypfopt import plotting
from pypfopt import CLA


def load_data(file_path):
    """
    Load price data from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing price data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with dates as index and asset prices as columns
    """
    df = pd.read_excel(file_path, index_col=0, parse_dates=True, skiprows=1, header=0)
    df = df.dropna()
    return df


def calculate_returns(prices):
    """
    Calculate returns from price data.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with asset prices
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with asset returns
    """
    returns = prices.pct_change().dropna()
    return returns


# Q1: Calculate Expected Returns (PyPortfolioOpt)
def calculate_expected_returns(prices):
    """
    Calculate expected annual returns from historical prices using PyPortfolioOpt.
    Uses mean_historical_return which calculates the annualized mean (daily) historical return.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with asset prices
        
    Returns:
    --------
    pd.Series
        Expected annual returns for each asset
    """
    # Use PyPortfolioOpt's mean_historical_return
    mu = expected_returns.mean_historical_return(prices)
    return mu


# Q2: Calculate Covariance Matrix (PyPortfolioOpt)
def calculate_covariance_matrix(prices):
    """
    Calculate sample covariance matrix from historical prices using PyPortfolioOpt.
    Uses sample_cov which calculates the annualized sample covariance matrix.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with asset prices
        
    Returns:
    --------
    pd.DataFrame
        Covariance matrix of returns (annualized)
    """
    # Use PyPortfolioOpt's sample_cov
    cov = risk_models.sample_cov(prices)
    return cov


# Q3: Global Minimum Variance Portfolio (PyPortfolioOpt)
def optimise_gmv(mu, cov):
    """
    Find the Global Minimum Variance (GMV) portfolio using PyPortfolioOpt.
    Uses EfficientFrontier.min_volatility() method.
    
    Parameters:
    -----------
    mu : pd.Series or np.array
        Expected returns
    cov : pd.DataFrame or np.array
        Covariance matrix
        
    Returns:
    --------
    dict
        Dictionary with 'weights', 'return', 'volatility', 'sharpe'
    """
    # Create EfficientFrontier object
    ef = EfficientFrontier(mu, cov)
    
    # Find minimum volatility portfolio
    weights = ef.min_volatility()
    
    # Get cleaned weights
    cleaned_weights = ef.clean_weights()
    
    # Get portfolio performance
    portfolio_return, portfolio_vol, sharpe = ef.portfolio_performance(verbose=False)
    
    # Convert cleaned weights to array
    weights_array = np.array([cleaned_weights.get(asset, 0.0) for asset in mu.index])
    
    return {
        'weights': weights_array,
        'return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe': sharpe
    }


# Q4: Tangency Portfolio / Maximum Sharpe Ratio (PyPortfolioOpt)
def optimise_tangency(mu, cov, risk_free_rate=0.02):
    """
    Find the Tangency Portfolio (Maximum Sharpe Ratio) using PyPortfolioOpt.
    Uses EfficientFrontier.max_sharpe() method.
    
    Parameters:
    -----------
    mu : pd.Series or np.array
        Expected returns
    cov : pd.DataFrame or np.array
        Covariance matrix
    risk_free_rate : float
        Risk-free rate (default: 0.02 or 2%)
        
    Returns:
    --------
    dict
        Dictionary with 'weights', 'return', 'volatility', 'sharpe'
    """
    # Create EfficientFrontier object
    ef = EfficientFrontier(mu, cov)
    
    # Find maximum Sharpe ratio portfolio
    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    
    # Get cleaned weights
    cleaned_weights = ef.clean_weights()
    
    # Get portfolio performance
    portfolio_return, portfolio_vol, sharpe = ef.portfolio_performance(
        verbose=False, risk_free_rate=risk_free_rate
    )
    
    # Convert cleaned weights to array
    weights_array = np.array([cleaned_weights.get(asset, 0.0) for asset in mu.index])
    
    return {
        'weights': weights_array,
        'return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe': sharpe
    }


# Q5: Trace Efficient Frontier (PyPortfolioOpt)
def trace_frontier(mu, cov, n_points=100):
    """
    Generate the efficient frontier using PyPortfolioOpt's CLA algorithm.
    CLA (Critical Line Algorithm) is more efficient for generating the frontier.
    
    Parameters:
    -----------
    mu : pd.Series or np.array
        Expected returns
    cov : pd.DataFrame or np.array
        Covariance matrix
    n_points : int
        Number of points on the frontier (used for compatibility)
        
    Returns:
    --------
    tuple
        (returns, volatilities) arrays for plotting
    """
    # Generate frontier by varying target returns
    min_ret = np.min(mu)
    max_ret = np.max(mu)
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    frontier_returns = []
    frontier_vols = []
    
    for target_ret in target_returns:
        try:
            # Create a new CLA instance for each target return
            cla_temp = CLA(mu, cov)
            cla_temp.efficient_return(target_ret)
            ret, vol, _ = cla_temp.portfolio_performance(verbose=False)
            frontier_returns.append(ret)
            frontier_vols.append(vol)
        except (ValueError, Exception) as e:
            # Skip target returns that are not achievable
            continue
    
    return np.array(frontier_returns), np.array(frontier_vols)


def plot_efficient_frontier(mu, cov, show_assets=True):
    """
    Plot the efficient frontier using PyPortfolioOpt's plotting functionality.
    Uses plot_efficient_frontier helper method for visualization.
    
    Parameters:
    -----------
    mu : pd.Series or np.array
        Expected returns
    cov : pd.DataFrame or np.array
        Covariance matrix
    show_assets : bool
        Whether to show individual assets on the plot
    """
    # Use CLA (Critical Line Algorithm) for plotting the frontier
    cla = CLA(mu, cov)
    
    # Plot using PyPortfolioOpt's built-in plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = plotting.plot_efficient_frontier(cla, ax=ax, show_assets=show_assets)
    
    # Add GMV and Tangency portfolios as markers
    gmv = optimise_gmv(mu, cov)
    ax.scatter(gmv['volatility'], gmv['return'], c='green', marker='*', 
               s=300, label='Min Volatility', zorder=5, edgecolors='black')
    
    tangency = optimise_tangency(mu, cov)
    ax.scatter(tangency['volatility'], tangency['return'], c='gold', marker='*', 
               s=300, label='Max Sharpe', zorder=5, edgecolors='black')
    
    ax.set_xlabel('Volatility (Risk)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier (PyPortfolioOpt)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Q6: Minimum Variance Portfolio with Target Return (PyPortfolioOpt)
def min_variance_with_target(mu, cov, target_return):
    """
    Find the minimum variance portfolio that achieves a target return using PyPortfolioOpt.
    Uses EfficientFrontier.efficient_return() method.
    
    Parameters:
    -----------
    mu : pd.Series or np.array
        Expected returns
    cov : pd.DataFrame or np.array
        Covariance matrix
    target_return : float
        Target portfolio return
        
    Returns:
    --------
    dict
        Dictionary with 'weights', 'return', 'volatility', 'sharpe'
    """
    try:
        # Create EfficientFrontier object
        ef = EfficientFrontier(mu, cov)
        
        # Find efficient portfolio with target return
        weights = ef.efficient_return(target_return)
        
        # Get cleaned weights
        cleaned_weights = ef.clean_weights()
        
        # Get portfolio performance
        portfolio_return, portfolio_vol, sharpe = ef.portfolio_performance(verbose=False)
        
        # Convert cleaned weights to array
        weights_array = np.array([cleaned_weights.get(asset, 0.0) for asset in mu.index])
        
        return {
            'weights': weights_array,
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe': sharpe
        }
    except Exception as e:
        # Return None if target return is not achievable
        return None


# Q7-Q9: CAPM Analysis and Additional Portfolio Statistics
def calculate_portfolio_statistics(weights, mu, cov):
    """
    Calculate comprehensive portfolio statistics.
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    mu : pd.Series or np.array
        Expected returns
    cov : pd.DataFrame or np.array
        Covariance matrix
        
    Returns:
    --------
    dict
        Dictionary with various portfolio statistics
    """
    portfolio_return = np.dot(weights, mu)
    portfolio_vol = np.sqrt(weights.T @ cov @ weights)
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe': sharpe,
        'weights': weights
    }


def run_capm_analysis(asset_returns, market_returns, risk_free_rate=0.02):
    """
    Perform CAPM (Capital Asset Pricing Model) analysis.
    Calculate beta, alpha, and test statistical significance.
    
    Parameters:
    -----------
    asset_returns : pd.Series or np.array
        Returns of the asset
    market_returns : pd.Series or np.array
        Returns of the market portfolio
    risk_free_rate : float
        Risk-free rate (annualized)
        
    Returns:
    --------
    dict
        Dictionary with CAPM statistics including alpha, beta, r-squared, p-values
    """
    # Convert to excess returns
    rf_daily = risk_free_rate / 252  # Daily risk-free rate
    excess_asset = asset_returns - rf_daily
    excess_market = market_returns - rf_daily
    
    # Run linear regression: excess_asset = alpha + beta * excess_market
    X = excess_market.values.reshape(-1, 1)
    y = excess_asset.values
    
    # Calculate beta using covariance
    covariance = np.cov(excess_asset, excess_market)[0, 1]
    market_variance = np.var(excess_market)
    beta = covariance / market_variance if market_variance > 0 else 0
    
    # Calculate alpha
    alpha = np.mean(excess_asset) - beta * np.mean(excess_market)
    
    # Annualize alpha
    alpha_annual = alpha * 252
    
    # Calculate R-squared
    y_pred = alpha + beta * excess_market
    ss_res = np.sum((excess_asset - y_pred) ** 2)
    ss_tot = np.sum((excess_asset - np.mean(excess_asset)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Statistical tests (simplified)
    n = len(excess_asset)
    residuals = excess_asset - y_pred
    std_error = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    
    # t-statistic for alpha
    se_alpha = std_error / np.sqrt(n)
    t_alpha = alpha / se_alpha if se_alpha > 0 else 0
    p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), n - 2))
    
    # t-statistic for beta
    se_beta = std_error / np.sqrt(np.sum((excess_market - np.mean(excess_market)) ** 2))
    t_beta = beta / se_beta if se_beta > 0 else 0
    p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), n - 2))
    
    return {
        'alpha': alpha_annual,
        'beta': beta,
        'r_squared': r_squared,
        't_stat_alpha': t_alpha,
        'p_value_alpha': p_alpha,
        't_stat_beta': t_beta,
        'p_value_beta': p_beta
    }


# Main execution flow for Q1-Q9
def main():
    """
    Main function to run all portfolio analysis questions Q1-Q9.
    """
    # Load data
    file_path = '/home/runner/work/FIN309-coursework/FIN309-coursework/FIN DATA FIRST PART (1).xlsx'
    
    print("="*80)
    print("FIN309 Portfolio Analysis - Q1 to Q9")
    print("="*80)
    
    try:
        df = load_data(file_path)
        print(f"\n✓ Data loaded successfully")
        print(f"  Timeframe: {df.index[0]} to {df.index[-1]}")
        print(f"  Assets: {df.columns.tolist()}")
        print(f"  Number of observations: {len(df)}")
    except FileNotFoundError:
        print(f"\n✗ Error: File '{file_path}' not found.")
        return
    
    # Q1: Calculate Expected Returns
    print("\n" + "="*80)
    print("Q1: Expected Returns (PyPortfolioOpt)")
    print("="*80)
    mu = calculate_expected_returns(df)
    print("\nExpected Annual Returns:")
    for asset, ret in mu.items():
        print(f"  {asset}: {ret:.4f} ({ret*100:.2f}%)")
    
    # Q2: Calculate Covariance Matrix
    print("\n" + "="*80)
    print("Q2: Covariance Matrix (PyPortfolioOpt)")
    print("="*80)
    cov = calculate_covariance_matrix(df)
    print("\nCovariance Matrix:")
    print(cov)
    
    # Q3: Global Minimum Variance Portfolio
    print("\n" + "="*80)
    print("Q3: Global Minimum Variance (GMV) Portfolio")
    print("="*80)
    gmv = optimise_gmv(mu, cov)
    print("\nGMV Portfolio Weights:")
    for i, (asset, weight) in enumerate(zip(df.columns, gmv['weights'])):
        print(f"  {asset}: {weight:.4f} ({weight*100:.2f}%)")
    print(f"\nExpected Return: {gmv['return']:.4f} ({gmv['return']*100:.2f}%)")
    print(f"Volatility: {gmv['volatility']:.4f} ({gmv['volatility']*100:.2f}%)")
    print(f"Sharpe Ratio: {gmv['sharpe']:.4f}")
    
    # Q4: Tangency Portfolio (Maximum Sharpe Ratio)
    print("\n" + "="*80)
    print("Q4: Tangency Portfolio (Maximum Sharpe Ratio)")
    print("="*80)
    risk_free_rate = 0.02  # 2% risk-free rate
    tangency = optimise_tangency(mu, cov, risk_free_rate)
    print(f"\nRisk-free rate: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")
    print("\nTangency Portfolio Weights:")
    for i, (asset, weight) in enumerate(zip(df.columns, tangency['weights'])):
        print(f"  {asset}: {weight:.4f} ({weight*100:.2f}%)")
    print(f"\nExpected Return: {tangency['return']:.4f} ({tangency['return']*100:.2f}%)")
    print(f"Volatility: {tangency['volatility']:.4f} ({tangency['volatility']*100:.2f}%)")
    print(f"Sharpe Ratio: {tangency['sharpe']:.4f}")
    
    # Q5: Efficient Frontier
    print("\n" + "="*80)
    print("Q5: Efficient Frontier")
    print("="*80)
    print("\nGenerating efficient frontier plot...")
    plot_efficient_frontier(mu, cov, show_assets=True)
    
    # Q6: Minimum Variance Portfolio with Target Return
    print("\n" + "="*80)
    print("Q6: Minimum Variance Portfolio with Target Return")
    print("="*80)
    target_return = 0.15  # 15% target return
    print(f"\nTarget Return: {target_return:.4f} ({target_return*100:.2f}%)")
    target_portfolio = min_variance_with_target(mu, cov, target_return)
    if target_portfolio:
        print("\nTarget Portfolio Weights:")
        for i, (asset, weight) in enumerate(zip(df.columns, target_portfolio['weights'])):
            print(f"  {asset}: {weight:.4f} ({weight*100:.2f}%)")
        print(f"\nExpected Return: {target_portfolio['return']:.4f} ({target_portfolio['return']*100:.2f}%)")
        print(f"Volatility: {target_portfolio['volatility']:.4f} ({target_portfolio['volatility']*100:.2f}%)")
        print(f"Sharpe Ratio: {target_portfolio['sharpe']:.4f}")
    else:
        print("\n✗ Could not find a portfolio with the target return")
    
    # Q7: Portfolio Statistics Summary
    print("\n" + "="*80)
    print("Q7: Portfolio Statistics Summary")
    print("="*80)
    print("\nComparing GMV vs Tangency Portfolio:")
    print(f"\n{'Metric':<20} {'GMV Portfolio':<20} {'Tangency Portfolio':<20}")
    print("-" * 60)
    print(f"{'Expected Return':<20} {gmv['return']:.4f} ({gmv['return']*100:.2f}%)    {tangency['return']:.4f} ({tangency['return']*100:.2f}%)")
    print(f"{'Volatility':<20} {gmv['volatility']:.4f} ({gmv['volatility']*100:.2f}%)    {tangency['volatility']:.4f} ({tangency['volatility']*100:.2f}%)")
    print(f"{'Sharpe Ratio':<20} {gmv['sharpe']:.4f}              {tangency['sharpe']:.4f}")
    
    # Q8-Q9: CAPM Analysis (if market data available)
    print("\n" + "="*80)
    print("Q8-Q9: CAPM Analysis")
    print("="*80)
    print("\nNote: CAPM analysis requires market portfolio returns.")
    print("If available, use run_capm_analysis() function with appropriate data.")
    
    # Example CAPM analysis if first asset is used as market proxy
    if len(df.columns) > 1:
        returns = calculate_returns(df)
        market_proxy = returns.iloc[:, 0]  # First asset as market proxy
        asset = returns.iloc[:, 1]  # Second asset
        
        print(f"\nExample CAPM: {df.columns[1]} vs {df.columns[0]} (as market proxy)")
        capm_results = run_capm_analysis(asset, market_proxy, risk_free_rate)
        print(f"\nAlpha (annual): {capm_results['alpha']:.4f} ({capm_results['alpha']*100:.2f}%)")
        print(f"Beta: {capm_results['beta']:.4f}")
        print(f"R-squared: {capm_results['r_squared']:.4f}")
        print(f"Alpha p-value: {capm_results['p_value_alpha']:.4f}")
        print(f"Beta p-value: {capm_results['p_value_beta']:.4f}")
    
    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)


if __name__ == "__main__":
    main()
