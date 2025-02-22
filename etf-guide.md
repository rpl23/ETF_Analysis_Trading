# ETF Analysis Code Guide

## 1. Data Import and Preparation
- ETFs analyzed: SPY, QQQ, IWM, TLT, GLD, XLF, XLK, XLE
- Uses yfinance to fetch data from 2018 onwards
- Features created include:
  - Returns
  - Moving averages (short and long window)
  - Volatility metrics
  - Volume features
  - Cross-ETF relative strength

## 2. Model Training Setup
- Creates 5-day future returns as target variables
- Splits data into training (80%) and testing (20%) sets
- Scales features using StandardScaler

## 3. Random Forest Regression Model
- Predicts continuous returns
- Parameters:
  - 100 estimators
  - Max depth of 6
  - Min samples requirements for better generalization
- Includes feature importance analysis

## 4. Time Series Cross-Validation
- Implements TimeSeriesSplit with 5 splits
- Uses simplified model parameters
- Evaluates R² scores across time periods

## 5. Binary Classification Model
- Predicts 20-day price movement (up/down)
- Additional regime features:
  - Market trend indicators
  - Volatility regimes
  - Volume regimes
  - Sector rotation metrics

## 6. Adaptive Model
- Enhanced feature set including:
  - Trend strength indicators
  - Moving average distances
  - Volatility regime indicators
  - Cross-asset relationships
- Improved RandomForest parameters
- Comprehensive metrics for each ETF

## 7. Final Trading Strategy
- Focuses on top performing ETFs (GLD, SPY, XLF, XLK)
- Features:
  - Position sizing based on prediction confidence
  - Risk management rules
  - Portfolio allocation logic
  - Performance metrics calculation
  - Strategy returns visualization

## Key Outputs and Visualizations
- Feature importance rankings
- Cross-validation scores
- Prediction accuracy metrics
- Strategy performance metrics
- Cumulative returns plot

## Usage Notes
- All code is self-contained and runs in Colab
- Requires yfinance, pandas, numpy, scikit-learn
- Can be modified for different time periods or ETFs
- Includes comprehensive error handling and data validation