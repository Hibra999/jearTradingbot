from utils.data import scrape_candles_to_dataframe
from utils.features import create_robust_features
# Import the updated model with all enhancements
from model_with_trading import (
    EnhancedHierarchicalMetaLearner, 
    simulate_enhanced_real_time_forecast,
    TradingConfig,
    TradingPerformanceTracker
)
from utils.plots import plot_results, plot_classification_results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# Configuration
OPTIMIZE_HYPERPARAMS = False  # Enable parameter optimization
SYMBOL = 'SOL/USDT'  # or 'BTC/USDT'
USE_ENHANCED_MODEL = True  # Toggle between original and enhanced model

# Enhanced Trading Configuration
INITIAL_CAPITAL = 500
RISK_LEVEL = 'medium'  # Options: 'low', 'medium', 'high'
ENABLE_TRADING = True  # Enable integrated trading simulation

# Custom trading configuration (override defaults if needed)
CUSTOM_TRADING_CONFIG = TradingConfig(
    # Confidence thresholds
    min_confidence_buy=0.6,
    high_confidence_threshold=0.8,
    
    # Uncertainty thresholds
    max_uncertainty_buy=0.3,
    uncertainty_position_scale=True,
    
    # Risk management
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    trailing_stop_pct=0.015,
    min_risk_reward_ratio=2.0,
    
    # Position sizing
    base_position_pct=0.2,
    max_position_pct=0.4,
    confidence_scale_factor=2.0,
    
    # Trend filter
    use_trend_filter=True,
    trend_ma_period=200,
    
    # Model prediction thresholds
    min_predicted_gain=0.01,
    prediction_lookback=20
)


def plot_summary_results(simulation_result):
    """
    Generate a focused 2x2 summary dashboard of the most important trading results.
    """
    # Set a clean style for the plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a 2x2 figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

    # Define a clear color scheme
    colors = {
        'buy': '#2ECC71',      # Green
        'sell': '#E74C3C',     # Red
        'price': '#34495E',    # Dark Blue/Gray
        'prediction': '#3498DB', # Light Blue
        'profit': '#27AE60',    # Darker Green
        'loss': "#BD1604",      # Darker Red
        'neutral': '#95A5A6'    # Gray
    }

    # ===================================================================
    # AX1: Main Price Chart with Trading Signals
    # ===================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    time_indices = range(len(simulation_result['actual_values']))
    
    # Plot actual and predicted prices
    ax1.plot(time_indices, simulation_result['actual_values'], 
             label='Actual Price', color=colors['price'], linewidth=2)
    ax1.plot(time_indices, simulation_result['predictions'], 
             label='Predicted Price', color=colors['prediction'], 
             linewidth=1.5, alpha=0.9, linestyle='--')
    
    # Plot buy/sell signals
    if 'trade_signals' in simulation_result:
        buy_signals = [s for s in simulation_result['trade_signals'] if s['type'] == 'buy']
        sell_signals = [s for s in simulation_result['trade_signals'] if s['type'] == 'sell']
        
        if buy_signals:
            buy_indices = [s['idx'] - (len(simulation_result['actual_values']) - len(simulation_result['predictions'])) for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            ax1.scatter(buy_indices, buy_prices, 
                       color=colors['buy'], marker='^', s=150, 
                       edgecolor='black', linewidth=1, zorder=5, label='Buy')

        if sell_signals:
            sell_indices = [s['idx'] - (len(simulation_result['actual_values']) - len(simulation_result['predictions'])) for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            ax1.scatter(sell_indices, sell_prices, 
                       color=colors['sell'], marker='v', s=150,
                       edgecolor='black', linewidth=1, zorder=5, label='Sell')

    ax1.set_title('Price & Trade Execution', fontsize=18, weight='bold')
    ax1.set_ylabel('Price ($)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # ===================================================================
    # AX2: Portfolio Equity Curve & Drawdown
    # ===================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    equity_curve = simulation_result['equity_curve']
    
    # Plot equity curve
    ax2.plot(equity_curve, color=colors['profit'], linewidth=2.5, label='Portfolio Value')
    ax2.axhline(y=simulation_result['initial_capital'], color=colors['neutral'], 
               linestyle='--', linewidth=2, label='Initial Capital')
    
    # Fill profit/loss areas
    ax2.fill_between(range(len(equity_curve)), simulation_result['initial_capital'], equity_curve,
                    where=np.array(equity_curve) >= simulation_result['initial_capital'],
                    color=colors['profit'], alpha=0.3)
    ax2.fill_between(range(len(equity_curve)), simulation_result['initial_capital'], equity_curve,
                    where=np.array(equity_curve) < simulation_result['initial_capital'],
                    color=colors['loss'], alpha=0.3)

    # Plot drawdown on a secondary y-axis
    ax2_dd = ax2.twinx()
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100
    ax2_dd.fill_between(range(len(drawdown)), 0, drawdown, color=colors['loss'], alpha=0.2, label='Drawdown')
    ax2_dd.set_ylabel('Drawdown (%)', color=colors['loss'], fontsize=14)
    ax2_dd.tick_params(axis='y', labelcolor=colors['loss'])
    
    ax2.set_title('Portfolio Equity and Drawdown', fontsize=18, weight='bold')
    ax2.set_ylabel('Portfolio Value ($)', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2_dd.legend(loc='lower right', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # ===================================================================
    # AX3: Cumulative Returns: Strategy vs. Buy & Hold
    # ===================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    equity_curve_aligned = np.array(equity_curve[1:len(simulation_result['actual_values'])+1])
    prices = np.array(simulation_result['actual_values'])

    strategy_returns = (equity_curve_aligned / equity_curve[0] - 1) * 100
    buy_hold_returns = (prices / prices[0] - 1) * 100

    ax3.plot(strategy_returns, label='Strategy Returns', color=colors['profit'], linewidth=2.5)
    ax3.plot(buy_hold_returns, label='Buy & Hold Returns', color=colors['neutral'], linewidth=2, linestyle='--')
    
    # Fill area to show out/underperformance
    ax3.fill_between(range(len(strategy_returns)), strategy_returns, buy_hold_returns,
                    where=strategy_returns > buy_hold_returns,
                    color=colors['profit'], alpha=0.2, label='Outperformance')
    ax3.fill_between(range(len(strategy_returns)), strategy_returns, buy_hold_returns,
                    where=strategy_returns <= buy_hold_returns,
                    color=colors['loss'], alpha=0.2, label='Underperformance')

    ax3.set_title('Strategy vs. Buy & Hold', fontsize=18, weight='bold')
    ax3.set_ylabel('Cumulative Return (%)', fontsize=14)
    ax3.set_xlabel('Time Steps', fontsize=14)
    ax3.legend(loc='upper left', fontsize=12)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

    # ===================================================================
    # AX4: Individual Trade Performance (Profit/Loss)
    # ===================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    if 'trades' in simulation_result and simulation_result['trades']:
        profits = [t['profit'] for t in simulation_result['trades']]
        trade_colors = [colors['profit'] if p > 0 else colors['loss'] for p in profits]
        
        ax4.bar(range(len(profits)), profits, color=trade_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add average win/loss lines
        if simulation_result.get('avg_profit', 0) > 0:
            ax4.axhline(y=simulation_result['avg_profit'], color=colors['profit'], 
                       linestyle='--', linewidth=2, label=f"Avg. Win: ${simulation_result['avg_profit']:.2f}")
        if simulation_result.get('avg_loss', 0) < 0:
            ax4.axhline(y=simulation_result['avg_loss'], color=colors['loss'], 
                       linestyle='--', linewidth=2, label=f"Avg. Loss: ${simulation_result['avg_loss']:.2f}")
        
        # Add text box with key trade stats
        stats_text = (f"Total Trades: {simulation_result['total_trades']}\n"
                     f"Win Rate: {simulation_result['win_rate']:.1f}%\n"
                     f"Sharpe Ratio: {simulation_result['sharpe_ratio']:.2f}")
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))

        ax4.legend(loc='lower left', fontsize=12)

    ax4.set_title('Individual Trade Profit/Loss', fontsize=18, weight='bold')
    ax4.set_xlabel('Trade Number', fontsize=14)
    ax4.set_ylabel('Profit/Loss ($)', fontsize=14)
    ax4.grid(True, axis='y', linestyle='--', linewidth=0.5)
    
    # Overall Title
    fig.suptitle('Trading Strategy Performance Summary', fontsize=24, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def analyze_model_contribution(simulation_result):
    """
    Analyze how much value the model adds vs pure NFI strategy
    """
    print("\n" + "="*60)
    print("MODEL CONTRIBUTION ANALYSIS")
    print("="*60)
    
    if 'exit_analysis' in simulation_result:
        exit_reasons = simulation_result['exit_analysis'].get('exit_reasons', {})
        
        model_driven_exits = ['model_prediction', 'uncertainty', 'trailing_stop']
        nfi_exits = ['nfi_sell']
        risk_exits = ['stop_loss', 'take_profit']
        
        model_trades = sum(exit_reasons.get(reason, {}).get('count', 0) 
                          for reason in model_driven_exits)
        nfi_trades = sum(exit_reasons.get(reason, {}).get('count', 0) 
                        for reason in nfi_exits)
        risk_trades = sum(exit_reasons.get(reason, {}).get('count', 0) 
                         for reason in risk_exits)
        
        total_exits = model_trades + nfi_trades + risk_trades
        
        if total_exits > 0:
            print(f"Exit Reason Breakdown:")
            print(f"  Model-driven exits: {model_trades} ({model_trades/total_exits*100:.1f}%)")
            print(f"  NFI strategy exits: {nfi_trades} ({nfi_trades/total_exits*100:.1f}%)")
            print(f"  Risk management exits: {risk_trades} ({risk_trades/total_exits*100:.1f}%)")
            
            # Analyze profitability by exit type
            print(f"\nProfitability by Exit Type:")
            for reason, stats in exit_reasons.items():
                if stats['count'] > 0:
                    print(f"  {reason}: Win Rate {stats['win_rate']:.1f}%, "
                          f"Avg Profit ${stats['avg_profit']:.2f}")
    
    # Calculate model prediction accuracy impact
    mae = simulation_result['mae']
    mape = simulation_result['mape']
    avg_confidence = np.mean(simulation_result['confidence_scores'])
    avg_uncertainty = np.mean(simulation_result['uncertainty_scores'])
    
    print(f"\nModel Performance Metrics:")
    print(f"  Prediction MAE: ${mae:.2f}")
    print(f"  Prediction MAPE: {mape:.2f}%")
    print(f"  Average Confidence: {avg_confidence:.3f}")
    print(f"  Average Uncertainty: {avg_uncertainty:.3f}")
    
    # Estimate value added
    if simulation_result['total_return'] > 0:
        print(f"\nValue Creation:")
        print(f"  Total Return: {simulation_result['total_return']:.2f}%")
        print(f"  Sharpe Ratio: {simulation_result['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {simulation_result['max_drawdown']:.2f}%")


def main():
    # 1. Fetch data
    print("1. Fetching data...")
    data = scrape_candles_to_dataframe('binance', 3, SYMBOL, '1h', '2025-01-01T00:00:00Z', 1000)
    print(f"Total data: {len(data)}")

    # 2. Create features
    print("\n2. Creating features...")
    df = create_robust_features(data)

    # 3. Initialize model
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

    # 6. Set trading configuration
    print("\n6. Using default trading parameters...")
    trading_config = CUSTOM_TRADING_CONFIG

    # 7. Run enhanced simulation with advanced trading
    print("\n7. Running enhanced real-time simulation...")
    n_simulation_steps = len(X_test) - model.sequence_length
    test_df = df.iloc[-test_size:].copy()

    # Import the simulation function
    from model_with_trading import simulate_enhanced_real_time_forecast_with_advanced_trading
    
    # Run the simulation
    simulation_result = simulate_enhanced_real_time_forecast_with_advanced_trading(
        model=model, 
        test_data=test_df, 
        scaler_X=model.scalers['features'], 
        scaler_y=model.scalers['target'], 
        initial_capital=INITIAL_CAPITAL,
        trading_config=trading_config,
        forecast_horizon=n_simulation_steps
    )

    # 8. Analyze model contribution
    analyze_model_contribution(simulation_result)

    # 9. Visualize results with the new summary plot
    print("\n9. Creating focused performance summary visualization...")
    plot_summary_results(simulation_result)
    
    # 10. Save results text file
    print("\n10. Saving results summary...")
    
    # Create detailed results summary text
    results_summary = f"""
ENHANCED TRADING SIMULATION RESULTS
=====================================
Symbol: {SYMBOL}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION:
--------------
Initial Capital:     ${INITIAL_CAPITAL:.2f}
Sequence Length:     {model.sequence_length}
Forecast Horizon:    {n_simulation_steps}

TRADING PARAMETERS:
--------------------
Min Confidence Buy:  {trading_config.min_confidence_buy}
Max Uncertainty Buy: {trading_config.max_uncertainty_buy}
Stop Loss:           {trading_config.stop_loss_pct*100:.1f}%
Take Profit:         {trading_config.take_profit_pct*100:.1f}%
Min Risk-Reward:     {trading_config.min_risk_reward_ratio}
Trend Filter:        {trading_config.use_trend_filter} (MA {trading_config.trend_ma_period})

MODEL PERFORMANCE:
------------------
MAE:                 ${simulation_result['mae']:.2f}
RMSE:                ${simulation_result['rmse']:.2f}
MAPE:                {simulation_result['mape']:.2f}%
Avg Confidence:      {np.mean(simulation_result['confidence_scores']):.3f}
Avg Uncertainty:     {np.mean(simulation_result['uncertainty_scores']):.3f}

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
"""
    
    # Save to file
    with open('trading_summary_results.txt', 'w') as f:
        f.write(results_summary)

    print("\nResults summary saved to: trading_summary_results.txt")
    print("\n" + "="*60)
    print("SIMULATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()