# FIN309-coursework

Portfolio analysis and optimization coursework using PyPortfolioOpt library.

## Overview

This repository contains portfolio optimization code for FIN309 coursework. The main script `fin309_manual_q1_q9.py` implements Questions 1-9 of the portfolio analysis using the PyPortfolioOpt library.

## Requirements

- Python 3.8+
- PyPortfolioOpt
- pandas
- numpy
- matplotlib
- scipy
- openpyxl (for reading Excel files)

Install dependencies:
```bash
pip install PyPortfolioOpt pandas numpy matplotlib scipy openpyxl
```

## Usage

Run the complete analysis:
```bash
python fin309_manual_q1_q9.py
```

## Features

The script implements the following portfolio analysis questions:

- **Q1**: Expected Returns calculation using `mean_historical_return`
- **Q2**: Covariance Matrix calculation using `sample_cov`
- **Q3**: Global Minimum Variance (GMV) Portfolio using `EfficientFrontier.min_volatility`
- **Q4**: Tangency Portfolio (Max Sharpe Ratio) using `EfficientFrontier.max_sharpe`
- **Q5**: Efficient Frontier visualization using `plot_efficient_frontier` and CLA
- **Q6**: Minimum Variance Portfolio with Target Return using `EfficientFrontier.efficient_return`
- **Q7**: Portfolio Statistics Summary comparing GMV vs Tangency portfolios
- **Q8-Q9**: CAPM (Capital Asset Pricing Model) analysis with regression and statistical tests

## Implementation Details

This codebase uses the PyPortfolioOpt library for robust and efficient portfolio optimization:

- **Expected Returns**: Uses historical mean returns annualized to 252 trading days
- **Covariance Matrix**: Sample covariance matrix with annualization
- **Optimization**: Convex optimization via CVXPY backend
- **Constraints**: Long-only portfolios (weights >= 0, sum to 1)
- **Efficient Frontier**: Critical Line Algorithm (CLA) for accurate frontier generation

## Data

The analysis uses historical price data from `FIN DATA FIRST PART (1).xlsx` containing:
- CSI Aggregate Bond Index
- SSE Government Bond Index
- CSI 300
- CSI 500
- CSI 1000

Time period: January 2015 - December 2024 (monthly data)

## Output

The script produces:
1. Expected returns and covariance matrix
2. Optimal portfolio weights for GMV and Tangency portfolios
3. Portfolio performance metrics (return, volatility, Sharpe ratio)
4. Efficient frontier visualization plot
5. CAPM analysis results (alpha, beta, R-squared, p-values)

## License

This is coursework for educational purposes.

