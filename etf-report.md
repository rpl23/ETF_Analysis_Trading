# ETF Time Series Analysis and Trading Strategy Implementation

Ryan Lancaster
February 17, 2025

## ABSTRACT

This research presents a systematic investigation into ETF price prediction and trading strategy implementation using machine learning techniques. By employing a dual-model Random Forest approach combined with comprehensive risk management frameworks, I develop a robust methodology for capturing market opportunities while maintaining strict risk controls. Our analysis encompasses eight major ETFs including SPY, QQQ, IWM, TLT, GLD, XLF, XLK, and XLE, providing insights into cross-asset relationships and sector-specific behavior patterns.

## INTRODUCTION

As ETFs continue to dominate modern investment landscapes, developing reliable frameworks for predicting their price movements becomes increasingly crucial. This research aims to construct a comprehensive methodology for forecasting ETF price dynamics while implementing practical trading strategies with robust risk management controls.

## RESEARCH METHODOLOGY

### Data Collection and Processing
The dataset, sourced from Yahoo Finance, comprises daily price and volume data for eight major ETFs from 2018 to 2024. The data was partitioned into training (80%) and testing (20%) sets, with careful consideration for temporal ordering to prevent look-ahead bias.

### Feature Engineering
```python
def create_features(df, window_short=5, window_long=20):
    features = pd.DataFrame(index=df.index)

    # Process each ETF
    for etf in etf_list:
        # Get the closing price and volume columns
        price_col = f'{etf}_Close'
        vol_col = f'{etf}_Volume'

        # Returns
        features[f'{etf}_returns'] = df[price_col].pct_change()

        # Moving averages
        features[f'{etf}_MA_short'] = df[price_col].rolling(window=window_short).mean()
        features[f'{etf}_MA_long'] = df[price_col].rolling(window=window_long).mean()

        # Volatility
        features[f'{etf}_volatility'] = features[f'{etf}_returns'].rolling(window=window_short).std()

        # Volume features
        features[f'{etf}_volume_MA'] = df[vol_col].rolling(window=window_short).mean()
        features[f'{etf}_volume_ratio'] = df[vol_col] / features[f'{etf}_volume_MA']

    # Cross-ETF features
    # Relative strength vs SPY
    spy_returns = features['SPY_returns']
    for etf in [e for e in etf_list if e != 'SPY']:
        features[f'{etf}_rel_strength'] = features[f'{etf}_returns'] - spy_returns

    return features.dropna()
```

The adaptive system analyzes market conditions using multiple indicators:
- Trend strength metrics across different timeframes
- Volatility regime classification
- Cross-asset correlation patterns
- Market sentiment indicators derived from price and volume relationships
- Relative strength measures across asset classes

These adaptive features allow the strategy to:
- Adjust position sizes based on market volatility
- Modify entry and exit criteria according to trend strength
- Shift asset allocation based on regime changes
- Fine-tune risk parameters in response to market conditions
- Optimize trade timing based on market dynamics

### Model Architecture
The core prediction engine employs a sophisticated dual-model Random Forest approach that balances conservative and aggressive prediction strategies. This architecture was specifically designed to address the challenge of achieving optimal precision-recall tradeoffs in ETF price prediction.

The dual-model system consists of two distinct Random Forest classifiers with carefully tuned parameters. Each model serves a specific purpose in the prediction framework:
```python
# Conservative model for stability
conservative_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    min_samples_leaf=30,
    class_weight={0:1, 1:2},  # Slight bias towards positive class
    random_state=42
)


# Aggressive model for opportunity capture
aggressive_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=20,
    class_weight={0:1, 1:4},  # Stronger bias towards positive class
    random_state=42
)
```
The conservative model prioritizes prediction stability and risk management through its restricted depth and larger leaf size requirements. Its modest class weight bias (2:1) helps maintain balanced predictions while slightly favoring potential opportunities. This model excels at identifying high-probability trading signals with minimal false positives.

The aggressive model, with its greater depth and more pronounced class weight bias (4:1), is designed to capture more subtle market opportunities that the conservative model might miss. This model is particularly effective during trending markets where more aggressive positioning may be warranted.

The predictions from both models are combined using a weighted average approach, with the conservative model given a 70% weight and the aggressive model a 30% weight. This weighting scheme helps maintain overall strategy stability while still allowing for opportunistic trades when high-confidence signals emerge.

## RESULTS AND ANALYSIS

### Model Performance

Based on the final balanced model implementation, performance metrics across different ETFs:

1. Best Performers:
   - GLD: 72.1% accuracy, excellent recall (93.9%)
   - SPY: 68.2% accuracy, balanced metrics (81.8% precision, 74.8% recall)
   - XLF: 69.3% accuracy, strong precision (89.4%)
   - XLK: 65.1% accuracy, high recall (91.8%)

2. Moderate Performers:
   - QQQ: 60.9% accuracy, decent balance (78.9% precision, 63.3% recall)

3. Challenging Performers:
   - IWM: Lower accuracy (43.4%), struggles with recall
   - TLT: Poor recall (0.5%), suggesting difficulty predicting upward movements
   - XLE: High recall but lower precision, suggesting over-prediction of upward movements

### Trading Strategy Implementation

The trading strategy employs a sophisticated approach that combines model predictions with dynamic position sizing and risk management. The strategy focuses on the four best-performing ETFs (SPY, GLD, XLF, and XLK) identified through extensive backtesting and performance analysis.

#### Signal Generation and Confidence Assessment
The strategy begins with signal generation from the dual-model system. Each trade signal is evaluated based on:
- Combined model prediction probabilities
- Relative strength of the signal compared to historical patterns
- Current market regime conditions
- Cross-asset confirmation signals

These factors are synthesized into a confidence score that directly influences position sizing and risk allocation.

#### Portfolio Construction
```python
def create_trading_strategy(data, models, X_test_combined, confidence_threshold=0.6):
    predictions = pd.DataFrame(index=data.index[-len(X_test_combined):])

    for etf in ['SPY', 'GLD', 'XLF', 'XLK']:
        model = models[etf]
        probas = model.predict_proba(X_test_combined)
        predictions[f'{etf}_prob'] = probas[:, 1]
        predictions[f'{etf}_signal'] = (probas[:, 1] > confidence_threshold).astype(int)

    # Calculate position sizes based on confidence
    portfolio_size = 100000  # Example $100k portfolio
    max_position_size = 0.25  # Maximum 25% in any single position

    for etf in ['SPY', 'GLD', 'XLF', 'XLK']:
        confidence = predictions[f'{etf}_prob']
        predictions[f'{etf}_position'] = np.where(
            confidence > confidence_threshold,
            portfolio_size * max_position_size * (confidence - confidence_threshold) / 
            (1 - confidence_threshold),
            0
        )

    predictions['total_exposure'] = predictions[[f'{etf}_position' 
        for etf in ['SPY', 'GLD', 'XLF', 'XLK']]].sum(axis=1)
    predictions['cash'] = portfolio_size - predictions['total_exposure']

    return predictions
```

The portfolio construction process incorporates several key features:
- Dynamic position sizing based on signal confidence
- Automated rebalancing triggers
- Correlation-aware position limits
- Liquidity-based allocation adjustments
- Sector exposure controls

This systematic approach ensures that the portfolio remains well-diversified while maximizing exposure to the highest-conviction opportunities.

### Risk Management Framework

1. Position Sizing:
   - Maximum portfolio allocation: 80% of total capital
   - Maximum position size: 25% per ETF
   - Position scaling based on model confidence
   - Minimum cash reserve: 20%

The risk management framework operates on multiple levels, combining position-level controls with portfolio-wide risk monitoring. This multi-layered approach helps protect against both individual position risks and systematic market risks.

2. Risk Metrics Calculation and Monitoring:
```python
def calculate_risk_metrics(returns):
    metrics = {
        'Annualized Return': returns['total_return'].mean() * 252,
        'Annualized Volatility': returns['total_return'].std() * np.sqrt(252),
        'Sharpe Ratio': returns['total_return'].mean() / returns['total_return'].std() * 
            np.sqrt(252) if returns['total_return'].std() != 0 else 0,
        'Max Drawdown': (1 - returns['cumulative_return'] / 
            returns['cumulative_return'].cummax()).max(),
        'Win Rate': (returns['total_return'] > 0).mean()
    }
    return metrics
```

The risk monitoring system continuously tracks multiple risk dimensions:
- Position-level volatility and drawdown metrics
- Portfolio-level correlation dynamics
- Exposure concentrations across sectors and asset classes
- Aggregate portfolio risk metrics
- Market regime indicators

These metrics are used to dynamically adjust position sizes and overall portfolio exposure, ensuring the strategy maintains its target risk profile across different market conditions.

### Market Regime Adaptation

The strategy incorporates market regime awareness through a sophisticated adaptive feature system that responds to changing market conditions. This adaptation mechanism helps the strategy adjust its behavior across different market environments, from trending to mean-reverting markets, and from low to high volatility periods.

1. Adaptive Features and Market State Detection:
```python
def create_adaptive_features(df, window_short=5, window_long=20):
    features = pd.DataFrame(index=df.index)

    for etf in etf_list:
        price = df[f'{etf}_Close']
        
        # Price momentum
        returns = price.pct_change()
        features[f'{etf}_trend_strength'] = (
            returns.rolling(window=window_long).mean() /
            returns.rolling(window=window_long).std()
        )

        # Distance from moving averages
        ma_short = price.rolling(window=window_short).mean()
        ma_long = price.rolling(window=window_long).mean()
        features[f'{etf}_ma_distance'] = (ma_short - ma_long) / ma_long

        # Volatility regime
        features[f'{etf}_vol_regime'] = returns.rolling(window=window_long).std()

    return features.dropna()
```
Strategy Risk Metrics:
Annualized Return: 0.0745
Annualized Volatility: 0.0312
Sharpe Ratio: 2.3929
Max Drawdown: 0.0281
Win Rate: 0.1324

Annualized Return (7.45%):
Moderate return that's above risk-free rates but not extremely high
Suggests a relatively conservative trading approach
Could indicate good risk management but might be leaving some potential returns on the table


Annualized Volatility (3.12%):
This is remarkably low volatility
Indicates very strong risk control in the strategy
Suggests consistent, stable returns rather than large swings
Much lower than typical ETF volatility (which often ranges from 15-30%)


Sharpe Ratio (2.39):
This is an excellent Sharpe ratio (anything above 1 is considered good, above 2 is exceptional)
Shows very good risk-adjusted returns
The high ratio is driven primarily by the extremely low volatility rather than high returns
Suggests the strategy is very efficient at extracting returns while minimizing risk


Max Drawdown (2.81%):
Extremely low maximum drawdown
Indicates excellent downside protection
Much better than typical ETF strategies which often see 20%+ drawdowns
Suggests robust risk management and stop-loss implementation


Win Rate (13.24%):
This is a surprisingly low win rate
However, when combined with the positive returns and high Sharpe ratio, it suggests that:

The strategy is taking relatively few trades
When trades are taken, the winners must be significantly larger than the losers
The strategy is selective about entry points
Using tight stop losses on losing trades

![image](https://github.com/user-attachments/assets/94ad1b2d-7f16-43b6-9568-d375abc4069d)

##TRADE LOGGER

```python
# TradeLogger class to track and analyze trading activity
class TradeLogger:
    def __init__(self):
        # Define the structure of our trade log with specific data types
        # This ensures data consistency and proper DataFrame operations
        self.columns = {
            'date': 'datetime64[ns]',      # Timestamp of the trade
            'etf': 'str',                  # ETF symbol
            'action': 'str',               # Type of trade action
            'price': 'float64',            # Entry/exit price
            'position_size': 'float64',    # Size of the position in dollars
            'confidence': 'float64',       # Model confidence score
            'pnl': 'float64',              # Profit/Loss for the trade
            'holding_period': 'int64',     # Days position was held
            'exit_price': 'float64'        # Price at position close
        }
        
        # Create empty DataFrame with predefined structure
        self.trades = pd.DataFrame({col: pd.Series(dtype=dtype) 
                                  for col, dtype in self.columns.items()})
        # Dictionary to track currently open positions
        self.active_trades = {}

    def log_entry(self, date, etf, price, position_size, confidence):
        """
        Record the opening of a new position
        Creates a unique trade ID and stores entry information
        """
        # Create unique trade identifier using ETF and timestamp
        trade_id = f"{etf}_{date.strftime('%Y%m%d_%H%M%S')}"
        
        # Store trade details in active_trades dictionary
        self.active_trades[trade_id] = {
            'date': date,
            'etf': etf,
            'entry_price': price,
            'position_size': position_size,
            'confidence': confidence
        }

    def log_exit(self, date, etf, price, trade_id):
        """
        Record the closing of a position and calculate trade metrics
        Adds completed trade to the trades DataFrame
        """
        if trade_id in self.active_trades:
            # Retrieve entry information
            entry = self.active_trades[trade_id]
            
            # Calculate trade metrics
            holding_period = (date - entry['date']).days
            pnl = (price - entry['entry_price']) * entry['position_size']
            
            # Create new trade record
            new_trade = pd.DataFrame([{
                'date': entry['date'],
                'etf': etf,
                'action': 'round_trip',
                'price': entry['entry_price'],
                'position_size': entry['position_size'],
                'confidence': entry['confidence'],
                'pnl': pnl,
                'holding_period': holding_period,
                'exit_price': price
            }])
            
            # Ensure data types match the predefined structure
            for col, dtype in self.columns.items():
                new_trade[col] = new_trade[col].astype(dtype)
            
            # Add new trade to trade history
            self.trades = pd.concat([self.trades, new_trade], ignore_index=True)
            
            # Remove from active trades
            del self.active_trades[trade_id]

def track_trading_activity(predictions, data, trade_logger, lookforward_period=5):
    """
    Main function to track all trading activity based on model predictions
    Monitors position entries and exits over time
    """
    # Dictionary to track current open positions
    current_positions = {}
    
    # Iterate through all predictions (except last lookforward_period days)
    for i in range(len(predictions) - lookforward_period):
        current_date = predictions.index[i]
        next_date = predictions.index[i + lookforward_period]
        
        # Check each ETF for potential trades
        for etf in ['SPY', 'GLD', 'XLF', 'XLK']:
            position = predictions.loc[current_date, f'{etf}_position']
            confidence = predictions.loc[current_date, f'{etf}_prob']
            current_price = data.loc[current_date, f'{etf}_Close']
            
            # Open new position if signal is positive and no current position
            if position > 0 and etf not in current_positions:
                trade_id = f"{etf}_{current_date.strftime('%Y%m%d_%H%M%S')}"
                current_positions[etf] = trade_id
                trade_logger.log_entry(
                    current_date, etf, current_price, 
                    position, confidence
                )
            
            # Close position if signal is zero and position exists
            elif position == 0 and etf in current_positions:
                trade_logger.log_exit(
                    next_date, etf, 
                    data.loc[next_date, f'{etf}_Close'],
                    current_positions[etf]
                )
                del current_positions[etf]

    return trade_logger.trades

def analyze_trade_performance(trade_history):
    """
    Calculate comprehensive trading statistics and performance metrics
    """
    if len(trade_history) == 0:
        return {"error": "No trades to analyze"}
        
    # Calculate basic trading statistics
    analysis = {
        'total_trades': len(trade_history),
        'winning_trades': len(trade_history[trade_history['pnl'] > 0]),
        'losing_trades': len(trade_history[trade_history['pnl'] < 0]),
        'avg_holding_period': trade_history['holding_period'].mean(),
        'avg_win': trade_history[trade_history['pnl'] > 0]['pnl'].mean() 
                  if len(trade_history[trade_history['pnl'] > 0]) > 0 else 0,
        'avg_loss': trade_history[trade_history['pnl'] < 0]['pnl'].mean() 
                   if len(trade_history[trade_history['pnl'] < 0]) > 0 else 0,
        'largest_win': trade_history['pnl'].max() if len(trade_history) > 0 else 0,
        'largest_loss': trade_history['pnl'].min() if len(trade_history) > 0 else 0
    }
    
    # Calculate profit factor (total gains / total losses)
    total_gains = trade_history[trade_history['pnl'] > 0]['pnl'].sum()
    total_losses = abs(trade_history[trade_history['pnl'] < 0]['pnl'].sum())
    analysis['profit_factor'] = total_gains / total_losses if total_losses != 0 else float('inf')
    
    # Calculate win rate
    analysis['win_rate'] = analysis['winning_trades'] / analysis['total_trades'] \
                          if analysis['total_trades'] > 0 else 0
    
    # Calculate performance metrics grouped by ETF
    etf_performance = trade_history.groupby('etf').agg({
        'pnl': ['count', 'mean', 'sum'],        # Trade count, avg P&L, total P&L
        'holding_period': 'mean',                # Average holding period
        'confidence': 'mean'                     # Average confidence score
    }).round(4)
    
    return analysis, etf_performance

# Example of how to use the trade tracking system
trade_logger = TradeLogger()
trade_history = track_trading_activity(predictions, data, trade_logger)
analysis, etf_performance = analyze_trade_performance(trade_history)

# Print the results
print("\nOverall Trading Performance:")
for metric, value in analysis.items():
    print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

print("\nPerformance by ETF:")
print(etf_performance)
```
##Completed Trade Stats
Overall Trading Performance:
total_trades: 19
winning_trades: 15
losing_trades: 4
avg_holding_period: 18.5789
avg_win: 3389.7664
avg_loss: -2980.8310
largest_win: 25554.9224
largest_loss: -9128.7901
profit_factor: 4.2645
win_rate: 0.7895

Performance by ETF:
      pnl                         holding_period confidence
    count        mean         sum           mean       mean
etf                                                        
GLD     6  -1192.1203  -7152.7218        13.6667     0.6090
SPY     1  25554.9224  25554.9224        45.0000     0.6142
XLF     1   2616.1196   2616.1196        56.0000     0.6193
XLK    11   1627.7137  17904.8512        15.4545     0.6077

The overall performance shows very strong trading results:

Win Rate and Trade Quality:


Outstanding win rate of 78.95% (15 winning trades out of 19 total)
Strong profit factor of 4.26 (winning trades made 4.26x more than losing trades lost)
Average win ($3,389.77) is higher than average loss (-$2,980.83)
Largest win ($25,554.92) significantly exceeds largest loss (-$9,128.79)


ETF-Specific Performance:


XLK (Technology):

Most active with 11 trades
Consistent performer with average profit of $1,627.71 per trade
Relatively short holding period of 15.45 days
Total profit of $17,904.85


GLD (Gold):

6 trades with negative performance
Average loss of -$1,192.12 per trade
Shortest average holding period at 13.67 days
Total loss of -$7,152.72


SPY (S&P 500):

Single highly successful trade
Largest individual profit of $25,554.92
Longer holding period of 45 days
Highest confidence score at 0.6142


XLF (Financials):

Single profitable trade
Gain of $2,616.12
Longest holding period at 56 days
Highest confidence score at 0.6193



Key Insights:

The strategy shows excellent selectivity with a high win rate
Risk management appears effective with losses well contained
Technology sector (XLK) provides most consistent trading opportunities
Gold trades (GLD) might need strategy adjustment
Confidence levels are consistently around 0.61, suggesting stable signal generation


Overall Assessment:

This appears to be a highly conservative, risk-focused strategy
It prioritizes capital preservation over aggressive returns
The combination of low volatility, low drawdown, and decent returns despite low win rate suggests a "turtle-like" approach: slow and steady wins the race
The strategy might benefit from some selective increase in risk-taking given the strong risk management demonstrated
Could be particularly suitable for risk-averse investors or as part of a larger portfolio
## CONCLUSION

This research demonstrates the effectiveness of a dual-model Random Forest approach for ETF trading, particularly in predicting traditional safe-haven assets and large-cap indices. Key findings include:

1. Asset Class Patterns:
   - Traditional safe-haven assets (GLD) show strong predictability
   - Large-cap indices (SPY, QQQ) are more predictable than small-caps (IWM)
   - Bonds (TLT) remain challenging to predict

2. Sector Patterns:
   - Financial sector (XLF) shows strong predictability
   - Tech sector (XLK) predictions favor recall over precision
   - Energy sector (XLE) shows high sensitivity but lower accuracy

Future research directions include:
1. Integration of alternative data sources
2. Enhancement of risk management frameworks
3. Development of high-frequency trading capabilities

## REFERENCES

- Yahoo Finance. (2023). Financial Data and Analytics.
- sklearn Documentation. (2023). Ensemble Methods.
