from utils.data import scrape_candles_to_dataframe
from utils.features import create_robust_features
# Import the integrated model instead
from model_with_trading import EnhancedHierarchicalMetaLearner, simulate_enhanced_real_time_forecast
from utils.plots import plot_results, plot_classification_results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Configuration
OPTIMIZE_HYPERPARAMS = False  
SYMBOL = 'SOL/USDT'  # or 'BTC/USDT'
USE_ENHANCED_MODEL = True  # Toggle between original and enhanced model

# Trading Configuration
INITIAL_CAPITAL = 1000.0  # Starting capital in USD
RISK_LEVEL = 'medium'  # Options: 'low', 'medium', 'high'
ENABLE_TRADING = True  # Enable integrated trading simulation


def plot_integrated_results(simulation_result):
    """
    Plot comprehensive results from the integrated simulation
    """
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    
    # 1. Price chart with predictions and trading signals
    ax1 = axes[0]
    ax1.plot(simulation_result['actual_values'], label='Actual Price', color='black', linewidth=2)
    ax1.plot(simulation_result['predictions'], label='Predicted Price', color='blue', 
             linewidth=1.5, alpha=0.7, linestyle='--')
    
    # Plot buy/sell signals
    if 'trade_signals' in simulation_result:
        buy_signals = [s for s in simulation_result['trade_signals'] if s['type'] == 'buy']
        sell_signals = [s for s in simulation_result['trade_signals'] if s['type'] == 'sell']
        
        if buy_signals:
            buy_indices = [s['idx'] - len(simulation_result['actual_values']) + len(simulation_result['actual_values']) 
                          for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            ax1.scatter(range(len(buy_signals)), buy_prices[:len(buy_signals)], 
                       color='green', marker='^', s=150, label='Buy Signal', zorder=5)
            
        if sell_signals:
            sell_indices = [s['idx'] - len(simulation_result['actual_values']) + len(simulation_result['actual_values']) 
                           for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            ax1.scatter(range(len(sell_signals)), sell_prices[:len(sell_signals)], 
                       color='red', marker='v', s=150, label='Sell Signal', zorder=5)
    
    ax1.set_title('Price Predictions with Trading Signals', fontsize=16)
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Equity curve
    ax2 = axes[1]
    if 'equity_curve' in simulation_result:
        equity_curve = simulation_result['equity_curve']
        ax2.plot(equity_curve, label='Portfolio Value', color='green', linewidth=2)
        ax2.axhline(y=simulation_result['initial_capital'], color='gray', 
                   linestyle='--', label='Initial Capital')
        
        # Fill profit/loss areas
        ax2.fill_between(range(len(equity_curve)), 
                        simulation_result['initial_capital'], 
                        equity_curve,
                        where=np.array(equity_curve) > simulation_result['initial_capital'],
                        color='green', alpha=0.3, label='Profit')
        ax2.fill_between(range(len(equity_curve)), 
                        simulation_result['initial_capital'], 
                        equity_curve,
                        where=np.array(equity_curve) <= simulation_result['initial_capital'],
                        color='red', alpha=0.3, label='Loss')
        
        # Add performance text
        final_return = simulation_result.get('total_return', 0)
        max_dd = simulation_result.get('max_drawdown', 0)
        ax2.text(0.02, 0.98, f'Total Return: {final_return:.1f}%\nMax Drawdown: {max_dd:.1f}%',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_title('Portfolio Equity Curve', fontsize=16)
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Model confidence and prediction accuracy
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    # Confidence scores
    ax3.plot(simulation_result['confidence_scores'], label='Model Confidence', 
            color='purple', alpha=0.7)
    ax3.set_ylabel('Confidence Score', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.set_ylim([0, 1])
    
    # Prediction errors
    errors = np.abs(np.array(simulation_result['predictions']) - np.array(simulation_result['actual_values']))
    error_pct = errors / np.array(simulation_result['actual_values']) * 100
    ax3_twin.plot(error_pct, label='Prediction Error %', color='orange', alpha=0.7)
    ax3_twin.set_ylabel('Error (%)', color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    
    ax3.set_title('Model Confidence vs Prediction Error', fontsize=16)
    ax3.grid(True, alpha=0.3)
    
    # 4. NFI Technical Indicators
    ax4 = axes[3]
    if 'nfi_indicators' in simulation_result:
        indicators = simulation_result['nfi_indicators']
        
        # Plot RSI
        ax4.plot(indicators['rsi'], label='RSI', color='blue', linewidth=1.5)
        ax4.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax4.set_ylabel('RSI', color='blue')
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4.set_ylim([0, 100])
        
        # Plot MFI on secondary axis
        ax4_twin = ax4.twinx()
        ax4_twin.plot(indicators['mfi'], label='MFI', color='orange', linewidth=1.5, alpha=0.7)
        ax4_twin.set_ylabel('MFI', color='orange')
        ax4_twin.tick_params(axis='y', labelcolor='orange')
        ax4_twin.set_ylim([0, 100])
    
    ax4.set_title('NFI Technical Indicators (RSI & MFI)', fontsize=16)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    # 5. Trade analysis
    ax5 = axes[4]
    if 'trades' in simulation_result and simulation_result['trades']:
        trades = simulation_result['trades']
        profits = [t['profit'] for t in trades]
        
        # Create profit/loss bars
        colors = ['green' if p > 0 else 'red' for p in profits]
        bars = ax5.bar(range(len(profits)), profits, color=colors, alpha=0.8)
        
        # Add average lines
        if simulation_result['avg_profit'] > 0:
            ax5.axhline(y=simulation_result['avg_profit'], color='green', 
                       linestyle='--', alpha=0.7, 
                       label=f"Avg Win: ${simulation_result['avg_profit']:.2f}")
        if simulation_result['avg_loss'] < 0:
            ax5.axhline(y=simulation_result['avg_loss'], color='red', 
                       linestyle='--', alpha=0.7, 
                       label=f"Avg Loss: ${simulation_result['avg_loss']:.2f}")
        
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add statistics text
        win_rate = simulation_result.get('win_rate', 0)
        total_trades = simulation_result.get('total_trades', 0)
        ax5.text(0.02, 0.98, f'Total Trades: {total_trades}\nWin Rate: {win_rate:.1f}%',
                transform=ax5.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax5.set_title('Individual Trade Performance', fontsize=16)
    ax5.set_xlabel('Trade Number')
    ax5.set_ylabel('Profit/Loss ($)')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def create_performance_summary_plot(simulation_result):
    """
    Create a comprehensive performance summary dashboard
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Key Metrics Summary (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    metrics_text = f"""
    PERFORMANCE SUMMARY
    ━━━━━━━━━━━━━━━━━━
    Initial Capital: ${simulation_result['initial_capital']:.2f}
    Final Capital: ${simulation_result['final_capital']:.2f}
    Total Return: {simulation_result['total_return']:.2f}%
    
    Total Trades: {simulation_result['total_trades']}
    Win Rate: {simulation_result['win_rate']:.1f}%
    Sharpe Ratio: {simulation_result['sharpe_ratio']:.2f}
    Max Drawdown: {simulation_result['max_drawdown']:.2f}%
    
    Model MAE: ${simulation_result['mae']:.2f}
    Model MAPE: {simulation_result['mape']:.2f}%
    """
    
    ax1.text(0.1, 0.9, metrics_text, transform=ax1.transAxes, 
            fontsize=12, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 2. Return vs Risk Scatter (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot current strategy
    ax2.scatter(abs(simulation_result['max_drawdown']), simulation_result['total_return'],
               s=200, color='red', marker='*', label='Current Strategy', zorder=5)
    
    # Add reference points for different risk levels
    risk_levels = {'Low': (5, 10), 'Medium': (15, 30), 'High': (25, 50)}
    for level, (risk, ret) in risk_levels.items():
        ax2.scatter(risk, ret, s=100, alpha=0.5, label=f'{level} Risk')
    
    ax2.set_xlabel('Risk (Max Drawdown %)')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Risk-Return Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Monthly Returns Heatmap (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Calculate daily returns from equity curve
    if 'equity_curve' in simulation_result:
        equity_curve = np.array(simulation_result['equity_curve'])
        daily_returns = np.diff(equity_curve) / equity_curve[:-1] * 100
        
        # Create a simple heatmap of returns
        n_days = len(daily_returns)
        n_cols = min(30, n_days)
        n_rows = n_days // n_cols + (1 if n_days % n_cols else 0)
        
        returns_matrix = np.zeros((n_rows, n_cols))
        returns_matrix[:] = np.nan
        
        for i, ret in enumerate(daily_returns):
            row = i // n_cols
            col = i % n_cols
            returns_matrix[row, col] = ret
        
        im = ax3.imshow(returns_matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
        ax3.set_title('Daily Returns Heatmap (%)')
        ax3.set_xlabel('Day of Period')
        ax3.set_ylabel('Period')
        plt.colorbar(im, ax=ax3, label='Return %')
    
    # 4. Cumulative Returns Comparison (middle left)
    ax4 = fig.add_subplot(gs[1, :2])
    
    if 'equity_curve' in simulation_result:
        # Calculate cumulative returns
        equity_curve = np.array(simulation_result['equity_curve'])
        cum_returns = (equity_curve / equity_curve[0] - 1) * 100
        
        # Buy and hold comparison
        prices = np.array(simulation_result['actual_values'])
        buy_hold_returns = (prices / prices[0] - 1) * 100
        
        ax4.plot(cum_returns, label='Strategy Returns', color='green', linewidth=2)
        ax4.plot(buy_hold_returns, label='Buy & Hold', color='blue', 
                linewidth=2, linestyle='--', alpha=0.7)
        
        ax4.fill_between(range(len(cum_returns)), 0, cum_returns,
                        where=cum_returns > 0, color='green', alpha=0.3)
        ax4.fill_between(range(len(cum_returns)), 0, cum_returns,
                        where=cum_returns <= 0, color='red', alpha=0.3)
        
        ax4.set_title('Cumulative Returns: Strategy vs Buy & Hold')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Cumulative Return (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Trade Duration Distribution (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    
    if 'trades' in simulation_result and simulation_result['trades']:
        durations = [t['holding_period'] for t in simulation_result['trades']]
        
        ax5.hist(durations, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax5.axvline(np.mean(durations), color='red', linestyle='--', 
                   label=f'Avg: {np.mean(durations):.1f}')
        ax5.set_title('Trade Duration Distribution')
        ax5.set_xlabel('Holding Period (steps)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Profit Distribution (bottom)
    ax6 = fig.add_subplot(gs[2, :])
    
    if 'trades' in simulation_result and simulation_result['trades']:
        profits = [t['profit'] for t in simulation_result['trades']]
        profit_pcts = [t['profit_percentage'] for t in simulation_result['trades']]
        
        # Create bins for histogram
        ax6.hist(profit_pcts, bins=30, alpha=0.7, color='purple', edgecolor='black')
        
        # Add statistics
        ax6.axvline(np.mean(profit_pcts), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(profit_pcts):.1f}%')
        ax6.axvline(np.median(profit_pcts), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(profit_pcts):.1f}%')
        ax6.axvline(0, color='black', linestyle='-', linewidth=1)
        
        # Add text statistics
        pos_trades = sum(1 for p in profits if p > 0)
        neg_trades = sum(1 for p in profits if p <= 0)
        ax6.text(0.02, 0.95, 
                f'Winning Trades: {pos_trades}\nLosing Trades: {neg_trades}',
                transform=ax6.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax6.set_title('Trade Profit Distribution')
        ax6.set_xlabel('Profit/Loss (%)')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Trading Performance Dashboard - {SYMBOL}', fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    # 1. Fetch data
    print("1. Fetching data...")
    data = scrape_candles_to_dataframe('binance', 3, SYMBOL, '1h', '2025-01-01T00:00:00Z', 1000)
    print(f"Total data: {len(data)}")

    # 2. Create features
    print("\n2. Creating features...")
    df = create_robust_features(data)

    # 3. Initialize model
    if USE_ENHANCED_MODEL:
        print("\n3. Initializing Enhanced Hierarchical Meta-Learning Model...")
        model = EnhancedHierarchicalMetaLearner(sequence_length=168, forecast_horizon=1)
    
    X, y = model.prepare_data(df)

    # 4. Split data
    print("\n4. Splitting data...")
    test_size = 730  # 730 hours ~ 1 month
    val_size = 730

    X_train = X[:-test_size-val_size]
    y_train = y[:-test_size-val_size]
    X_val = X[-test_size-val_size:-test_size]
    y_val = y[-test_size-val_size:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")

    # 5. Train models
    print("\n5. Training models...")
    model.train(X_train, y_train, X_val, y_val)

    # 6. Run INTEGRATED simulation with trading
    print("\n6. Running integrated real-time simulation with NFI trading...")
    n_simulation_steps = len(X_test) - model.sequence_length
    test_df = df.iloc[-test_size:].copy()

    # Run the integrated simulation
    simulation_result = simulate_enhanced_real_time_forecast(
        model=model, 
        test_data=test_df, 
        scaler_X=model.scalers['features'], 
        scaler_y=model.scalers['target'], 
        forecast_horizon=n_simulation_steps,
        update_interval=10,
        enable_trading=ENABLE_TRADING,
        initial_capital=INITIAL_CAPITAL,
        risk_level=RISK_LEVEL
    )

    # 7. Visualize integrated results
    print("\n7. Creating visualizations...")
    
    # Plot integrated results
    plot_integrated_results(simulation_result)
    
    # Create performance summary dashboard
    create_performance_summary_plot(simulation_result)
    
    # 8. Save results
    print("\n8. Saving results...")
    
    # Create detailed results summary
    results_summary = f"""
INTEGRATED TRADING SIMULATION RESULTS
=====================================
Symbol: {SYMBOL}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION:
--------------
Initial Capital:     ${INITIAL_CAPITAL:.2f}
Risk Level:          {RISK_LEVEL}
Sequence Length:     {model.sequence_length}
Forecast Horizon:    {n_simulation_steps}

MODEL PERFORMANCE:
------------------
MAE:                 ${simulation_result['mae']:.2f}
RMSE:                ${simulation_result['rmse']:.2f}
MAPE:                {simulation_result['mape']:.2f}%
Avg Confidence:      {np.mean(simulation_result['confidence_scores']):.3f}

TRADING PERFORMANCE:
--------------------
Final Capital:       ${simulation_result['final_capital']:.2f}
Total Return:        {simulation_result['total_return']:.2f}%
Total Trades:        {simulation_result['total_trades']}
Win Rate:            {simulation_result['win_rate']:.1f}%
Average Win:         ${simulation_result['avg_profit']:.2f}
Average Loss:        ${simulation_result['avg_loss']:.2f}
Sharpe Ratio:        {simulation_result['sharpe_ratio']:.2f}
Max Drawdown:        {simulation_result['max_drawdown']:.2f}%

STRATEGY DETAILS:
-----------------
Strategy: NFI (NostalgiaForInfinity)
- 21 Buy Conditions (OR logic)
- 8 Sell Conditions (OR logic)
- Dynamic Risk Management
- Model Prediction Integration
- Confidence-Based Filtering

KEY INSIGHTS:
-------------
1. The strategy leverages model predictions to filter NFI signals
2. Trades are only executed when both NFI conditions AND model predictions align
3. Risk is dynamically adjusted based on capital levels
4. Stop loss at 2% protects against significant losses
=====================================
"""
    
    print(results_summary)
    
    # Save to file
    with open('integrated_trading_results.txt', 'w') as f:
        f.write(results_summary)
        
        # Add trade log
        f.write("\n\nDETAILED TRADE LOG:\n")
        f.write("==================\n")
        if 'trades' in simulation_result:
            for i, trade in enumerate(simulation_result['trades']):
                f.write(f"\nTrade #{i+1}:\n")
                f.write(f"  Entry: ${trade['entry_price']:.2f} @ step {trade['entry_time']}\n")
                f.write(f"  Exit:  ${trade['exit_price']:.2f} @ step {trade['exit_time']}\n")
                f.write(f"  Profit: ${trade['profit']:.2f} ({trade['profit_percentage']:.1f}%)\n")
                f.write(f"  Duration: {trade['holding_period']} steps\n")
                f.write(f"  Exit Reason: {trade['exit_reason']}\n")
    
    # Save results to CSV for further analysis
    results_df = pd.DataFrame({
        'actual_price': simulation_result['actual_values'],
        'predicted_price': simulation_result['predictions'],
        'confidence': simulation_result['confidence_scores'],
        'direction': simulation_result['direction_predictions'],
        'equity': simulation_result['equity_curve'][1:] if 'equity_curve' in simulation_result else None
    })
    results_df.to_csv('integrated_simulation_results.csv', index=False)
    
    print("\nResults saved to:")
    print("  - integrated_trading_results.txt")
    print("  - integrated_simulation_results.csv")
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()