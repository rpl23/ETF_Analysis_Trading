# ETF Trading Strategy

A machine learning-based trading strategy for ETFs using Random Forest models. This project implements a dual-model approach combining conservative and aggressive predictions for optimal market positioning.

## Features

- Dual Random Forest model implementation (Conservative/Aggressive)
- Dynamic position sizing based on model confidence
- Comprehensive risk management framework
- Real-time trading signals for 8 major ETFs
- Extensive backtesting capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/etf-trading-strategy.git

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from etf_trading_strategy import ETFPredictor, DataProcessor

# Initialize components
data_processor = DataProcessor()
predictor = ETFPredictor()

# Fetch and process data
data = data_processor.fetch_data(['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'XLF', 'XLK', 'XLE'])
features = data_processor.create_features(data)

# Generate trading signals
signals = predictor.predict(features)
```

## Project Structure

```
etf_trading_strategy/
├── etf_trading_strategy/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── models.py
│   └── strategy.py
├── tests/
│   └── test_*.py
├── examples/
│   └── basic_usage.ipynb
├── requirements.txt
└── README.md
```

## Models

1. Conservative Model
   - Higher precision focus
   - Reduced false positives
   - Suitable for core positions

2. Aggressive Model
   - Higher recall focus
   - Captures more opportunities
   - Used for tactical positions

## Trading Strategy

- Position sizing based on model confidence
- Maximum position size: 25% per ETF
- Portfolio-level exposure limits
- Sector-specific constraints
- Dynamic risk management

Strategy Risk Metrics:
Annualized Return: 0.0745
Annualized Volatility: 0.0312
Sharpe Ratio: 2.3929
Max Drawdown: 0.0281
Win Rate: 0.1324

## Requirements

- Python 3.8+
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- yfinance>=0.1.70


## Acknowledgments

- Data provided by Yahoo Finance
- Based on research in ETF trading strategies
- Inspired by various machine learning approaches to financial markets

## Disclaimer

This software is for educational purposes only. Do not use it for actual trading without thoroughly understanding the risks involved. Past performance does not guarantee future results.
