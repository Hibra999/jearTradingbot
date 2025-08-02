from utils.data import scrape_candles_to_dataframe
from utils.features import create_robust_features
# Import the updated model with all enhancements
from model_with_trading import (
    EnhancedHierarchicalMetaLearner, 
    simulate_enhanced_real_time_forecast,
    TradingConfig,
    TradingPerformanceTracker
)
from utils.plots import plot_detailed_trading_signals, analyze_uncertainty_issue
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
INITIAL_CAPITAL = 5300
RISK_LEVEL = 'high'  # Options: 'low', 'medium', 'high'
ENABLE_TRADING = True  # Enable integrated trading simulation

# Custom trading configuration (override defaults if needed)
CUSTOM_TRADING_CONFIG = TradingConfig(
    # Confidence thresholds
    min_confidence_buy=0.9,
    high_confidence_threshold=0.8,
    
    # Uncertainty thresholds
    max_uncertainty_buy=0.3,
    uncertainty_position_scale=True,
    
    # Risk management
    stop_loss_pct=0.25,
    take_profit_pct=0.05,
    trailing_stop_pct=0.015,
    min_risk_reward_ratio=2.0,
    
    # Position sizing
    base_position_pct=0.2,
    max_position_pct=0.4,
    confidence_scale_factor=2.0,
    
    # Trend filter
    use_trend_filter=False,
    trend_ma_period=200,
    
    # Model prediction thresholds
    min_predicted_gain=0.01,
    prediction_lookback=5
)


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
    data = scrape_candles_to_dataframe('binance', 3, SYMBOL, '15m', '2025-07-01T00:00:00Z', 1000)
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
    test_size = 185 # 730 hours ~ 1 month
    val_size = 185

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

        # Plot detailed signals
    plot_detailed_trading_signals(simulation_result)

    # Analyze uncertainty
    analyze_uncertainty_issue(simulation_result)

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