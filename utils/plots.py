import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

def plot_detailed_trading_signals(simulation_result, window_size=200):
    """
    Create a detailed, large-format chart focused on trading signals
    with zoom capability and better visibility
    """
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with multiple subplots for detailed view
    fig = plt.figure(figsize=(24, 16))
    
    # Define color scheme
    colors = {
        'buy': '#2ECC71',
        'sell': '#E74C3C',
        'price': '#2C3E50',
        'prediction': '#3498DB',
        'profit_zone': '#27AE60',
        'loss_zone': '#E74C3C'
    }
    
    # Main plot - Full view
    ax1 = plt.subplot(3, 1, 1)
    
    # Data
    actual_prices = simulation_result['actual_values']
    predicted_prices = simulation_result['predictions']
    
    # Create time indices for the simulation period
    time_indices = range(len(actual_prices))
    
    # Plot prices
    ax1.plot(time_indices, actual_prices, 
             label='Actual Price', color=colors['price'], linewidth=2.5, alpha=0.9)
    ax1.plot(time_indices, predicted_prices, 
             label='Predicted Price', color=colors['prediction'], 
             linewidth=2, alpha=0.7, linestyle='--')
    
    # Calculate the sequence_length from the difference in total data vs predictions
    # This is needed to properly align the trade signal indices
    sequence_length = 168  # Default, but we can infer it
    if 'trade_signals' in simulation_result and simulation_result['trade_signals']:
        # Find the offset by looking at the first trade signal
        first_signal_idx = simulation_result['trade_signals'][0]['idx']
        # The sequence_length is approximately the first signal index if trading starts early
        # But we need a more reliable method - let's use a fixed offset based on typical setup
        sequence_length = 168  # This matches the model's sequence_length
    
    # Plot trading signals with corrected indices
    if 'trade_signals' in simulation_result:
        buy_signals = [s for s in simulation_result['trade_signals'] if s['type'] == 'buy']
        sell_signals = [s for s in simulation_result['trade_signals'] if s['type'] == 'sell']
        
        # Buy signals - adjust indices to match the actual_values array
        if buy_signals:
            # The signal idx is in the full dataset, but our arrays start at sequence_length
            buy_indices = [s['idx'] - sequence_length for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            
            # Filter out any negative indices (shouldn't happen but just in case)
            valid_buys = [(idx, price, i) for i, (idx, price) in enumerate(zip(buy_indices, buy_prices)) if 0 <= idx < len(actual_prices)]
            
            if valid_buys:
                buy_indices_plot = [x[0] for x in valid_buys]
                buy_prices_plot = [x[1] for x in valid_buys]
                buy_original_indices = [x[2] for x in valid_buys]
                
                ax1.scatter(buy_indices_plot, buy_prices_plot, 
                           color=colors['buy'], marker='^', s=200, 
                           edgecolor='black', linewidth=2, zorder=5, label='Buy Signal')
                
                # Add annotations for buy signals
                for plot_idx, price, orig_idx in zip(buy_indices_plot, buy_prices_plot, buy_original_indices):
                    if 'reason' in buy_signals[orig_idx]:
                        reason = buy_signals[orig_idx]['reason'].split('|')[0]  # Get first part of reason
                        ax1.annotate(f'B{orig_idx+1}: {reason}', 
                                   xy=(plot_idx, price), 
                                   xytext=(plot_idx, price * 1.02),
                                   fontsize=10, ha='center',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['buy'], alpha=0.7))

        # Sell signals - adjust indices similarly
        if sell_signals:
            sell_indices = [s['idx'] - sequence_length for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            
            # Filter out any invalid indices
            valid_sells = [(idx, price, i) for i, (idx, price) in enumerate(zip(sell_indices, sell_prices)) if 0 <= idx < len(actual_prices)]
            
            if valid_sells:
                sell_indices_plot = [x[0] for x in valid_sells]
                sell_prices_plot = [x[1] for x in valid_sells]
                sell_original_indices = [x[2] for x in valid_sells]
                
                ax1.scatter(sell_indices_plot, sell_prices_plot, 
                           color=colors['sell'], marker='v', s=200,
                           edgecolor='black', linewidth=2, zorder=5, label='Sell Signal')
                
                # Add annotations for sell signals
                for plot_idx, price, orig_idx in zip(sell_indices_plot, sell_prices_plot, sell_original_indices):
                    if 'reason' in sell_signals[orig_idx]:
                        reason = sell_signals[orig_idx]['reason'].split(':')[0]  # Get exit type
                        ax1.annotate(f'S{orig_idx+1}: {reason}', 
                                   xy=(plot_idx, price), 
                                   xytext=(plot_idx, price * 0.98),
                                   fontsize=10, ha='center',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['sell'], alpha=0.7))
    
    # Highlight profitable trades
    if 'trades' in simulation_result:
        for trade in simulation_result['trades']:
            if trade['profit'] > 0:
                # Shade profitable trade periods
                entry_idx = trade.get('entry_time', 0) - sequence_length
                exit_idx = trade.get('exit_time', 0) - sequence_length
                
                if 0 <= entry_idx < len(actual_prices) and 0 <= exit_idx < len(actual_prices):
                    ax1.axvspan(entry_idx, exit_idx, alpha=0.1, color=colors['profit_zone'])
    
    ax1.set_title('Trading Signals Overview - Full Period', fontsize=20, weight='bold', pad=20)
    ax1.set_ylabel('Price ($)', fontsize=16)
    ax1.legend(loc='upper left', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Zoomed view 1 - First half of trades
    ax2 = plt.subplot(3, 1, 2)
    
    # Find range with most activity
    if 'trade_signals' in simulation_result and simulation_result['trade_signals']:
        signal_indices = [s['idx'] - sequence_length for s in simulation_result['trade_signals']]
        valid_signal_indices = [idx for idx in signal_indices if 0 <= idx < len(actual_prices)]
        
        # First half zoom
        if valid_signal_indices:
            mid_point = len(time_indices) // 2
            zoom_start = max(0, min(valid_signal_indices[0] - 50, mid_point - window_size//2))
            zoom_end = min(len(time_indices), zoom_start + window_size)
            
            # Plot zoomed section
            zoom_indices = range(zoom_start, zoom_end)
            ax2.plot(zoom_indices, actual_prices[zoom_start:zoom_end], 
                    color=colors['price'], linewidth=3, label='Actual Price')
            ax2.plot(zoom_indices, predicted_prices[zoom_start:zoom_end], 
                    color=colors['prediction'], linewidth=2.5, alpha=0.8, 
                    linestyle='--', label='Predicted Price')
            
            # Plot signals in zoom range
            for signal in simulation_result['trade_signals']:
                sig_idx = signal['idx'] - sequence_length
                if zoom_start <= sig_idx < zoom_end:
                    if signal['type'] == 'buy':
                        ax2.scatter(sig_idx, signal['price'], 
                                   color=colors['buy'], marker='^', s=300, 
                                   edgecolor='black', linewidth=2, zorder=5)
                        ax2.text(sig_idx, signal['price'] * 1.01, 'BUY', 
                                ha='center', fontsize=12, weight='bold')
                    else:
                        ax2.scatter(sig_idx, signal['price'], 
                                   color=colors['sell'], marker='v', s=300,
                                   edgecolor='black', linewidth=2, zorder=5)
                        ax2.text(sig_idx, signal['price'] * 0.99, 'SELL', 
                                ha='center', fontsize=12, weight='bold')
            
            ax2.set_title(f'Detailed View - Time Steps {zoom_start} to {zoom_end}', 
                         fontsize=18, weight='bold')
            ax2.set_ylabel('Price ($)', fontsize=16)
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3)
    
    # Zoomed view 2 - Most recent trades
    ax3 = plt.subplot(3, 1, 3)
    
    # Recent activity zoom
    recent_start = max(0, len(time_indices) - window_size)
    recent_end = len(time_indices)
    recent_indices = range(recent_start, recent_end)
    
    ax3.plot(recent_indices, actual_prices[recent_start:recent_end], 
            color=colors['price'], linewidth=3, label='Actual Price')
    ax3.plot(recent_indices, predicted_prices[recent_start:recent_end], 
            color=colors['prediction'], linewidth=2.5, alpha=0.8, 
            linestyle='--', label='Predicted Price')
    
    # Plot signals in recent range
    for signal in simulation_result['trade_signals']:
        sig_idx = signal['idx'] - sequence_length
        if recent_start <= sig_idx < recent_end:
            if signal['type'] == 'buy':
                ax3.scatter(sig_idx, signal['price'], 
                           color=colors['buy'], marker='^', s=300, 
                           edgecolor='black', linewidth=2, zorder=5)
                ax3.text(sig_idx, signal['price'] * 1.01, 'BUY', 
                        ha='center', fontsize=12, weight='bold')
            else:
                ax3.scatter(sig_idx, signal['price'], 
                           color=colors['sell'], marker='v', s=300,
                           edgecolor='black', linewidth=2, zorder=5)
                ax3.text(sig_idx, signal['price'] * 0.99, 'SELL', 
                        ha='center', fontsize=12, weight='bold')
    
    ax3.set_title(f'Recent Activity - Last {window_size} Time Steps', 
                 fontsize=18, weight='bold')
    ax3.set_xlabel('Time Step', fontsize=16)
    ax3.set_ylabel('Price ($)', fontsize=16)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create a second figure for trade analysis
    fig2 = plt.figure(figsize=(20, 10))
    
    # Trade entry/exit analysis
    if 'trades' in simulation_result and simulation_result['trades']:
        ax4 = plt.subplot(1, 2, 1)
        
        # Extract trade data
        entry_prices = [t['entry_price'] for t in simulation_result['trades']]
        exit_prices = [t['exit_price'] for t in simulation_result['trades']]
        profits = [t['profit'] for t in simulation_result['trades']]
        
        # Create trade visualization
        trade_nums = range(len(simulation_result['trades']))
        bar_colors = [colors['profit_zone'] if p > 0 else colors['loss_zone'] for p in profits]
        
        # Plot entry and exit prices
        ax4.scatter(trade_nums, entry_prices, color=colors['buy'], 
                   marker='^', s=150, label='Entry Price', zorder=3)
        ax4.scatter(trade_nums, exit_prices, color=colors['sell'], 
                   marker='v', s=150, label='Exit Price', zorder=3)
        
        # Connect entry and exit with lines
        for i, (entry, exit, profit) in enumerate(zip(entry_prices, exit_prices, profits)):
            color = colors['profit_zone'] if profit > 0 else colors['loss_zone']
            ax4.plot([i, i], [entry, exit], color=color, linewidth=2, alpha=0.6)
        
        ax4.set_title('Trade Entry and Exit Prices', fontsize=18, weight='bold')
        ax4.set_xlabel('Trade Number', fontsize=14)
        ax4.set_ylabel('Price ($)', fontsize=14)
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Trade profit distribution
        ax5 = plt.subplot(1, 2, 2)
        
        ax5.bar(trade_nums, profits, color=bar_colors, alpha=0.8, edgecolor='black')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add average lines
        avg_profit = simulation_result.get('avg_profit', 0)
        avg_loss = simulation_result.get('avg_loss', 0)
        
        if avg_profit > 0:
            ax5.axhline(y=avg_profit, color=colors['profit_zone'], 
                       linestyle='--', linewidth=2, label=f'Avg Win: ${avg_profit:.2f}')
        if avg_loss < 0:
            ax5.axhline(y=avg_loss, color=colors['loss_zone'], 
                       linestyle='--', linewidth=2, label=f'Avg Loss: ${avg_loss:.2f}')
        
        ax5.set_title('Individual Trade Profit/Loss', fontsize=18, weight='bold')
        ax5.set_xlabel('Trade Number', fontsize=14)
        ax5.set_ylabel('Profit/Loss ($)', fontsize=14)
        ax5.legend(fontsize=12)
        ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def analyze_uncertainty_issue(simulation_result, model=None):
    """
    Analyze and diagnose the uncertainty calculation issue
    """
    print("\n" + "="*60)
    print("UNCERTAINTY ANALYSIS")
    print("="*60)
    
    # Get uncertainty scores
    uncertainty_scores = np.array(simulation_result.get('uncertainty_scores', []))
    
    if len(uncertainty_scores) == 0:
        print("No uncertainty scores found in simulation results.")
        return
    
    # Basic statistics
    print(f"\nUncertainty Statistics:")
    print(f"  Min: {np.min(uncertainty_scores):.6f}")
    print(f"  Max: {np.max(uncertainty_scores):.6f}")
    print(f"  Mean: {np.mean(uncertainty_scores):.6f}")
    print(f"  Std: {np.std(uncertainty_scores):.6f}")
    print(f"  Unique values: {len(np.unique(uncertainty_scores))}")
    
    # Check if all values are 1.0
    if np.all(uncertainty_scores == 1.0):
        print("\n⚠️  WARNING: All uncertainty values are 1.0!")
        print("This indicates an issue with the uncertainty calculation.")
    
    # Sample some values
    print(f"\nFirst 10 uncertainty values: {uncertainty_scores[:10]}")
    print(f"Last 10 uncertainty values: {uncertainty_scores[-10:]}")
    
    # Plot uncertainty distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    ax1 = axes[0, 0]
    ax1.plot(uncertainty_scores, linewidth=1)
    ax1.set_title('Uncertainty Over Time', fontsize=14)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Uncertainty Score')
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2 = axes[0, 1]
    ax2.hist(uncertainty_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_title('Uncertainty Distribution', fontsize=14)
    ax2.set_xlabel('Uncertainty Score')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Uncertainty vs Prediction Error
    if 'predictions' in simulation_result and 'actual_values' in simulation_result:
        predictions = np.array(simulation_result['predictions'])
        actuals = np.array(simulation_result['actual_values'])
        
        # Calculate prediction errors
        errors = np.abs(predictions - actuals)
        relative_errors = errors / (actuals + 1e-6)
        
        ax3 = axes[1, 0]
        scatter = ax3.scatter(uncertainty_scores, relative_errors, 
                            alpha=0.5, c=range(len(uncertainty_scores)), 
                            cmap='viridis', s=20)
        ax3.set_title('Uncertainty vs Prediction Error', fontsize=14)
        ax3.set_xlabel('Uncertainty Score')
        ax3.set_ylabel('Relative Prediction Error')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Time Step')
        
        # Rolling correlation
        ax4 = axes[1, 1]
        window = 50
        if len(uncertainty_scores) > window:
            rolling_corr = pd.Series(uncertainty_scores).rolling(window).corr(pd.Series(relative_errors))
            ax4.plot(rolling_corr)
            ax4.set_title(f'Rolling Correlation (window={window})', fontsize=14)
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Correlation')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Provide recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if np.all(uncertainty_scores == 1.0):
        print("\n1. The uncertainty normalization is causing all values to be 1.0")
        print("   This happens when all raw uncertainty values are similar.")
        print("\n2. Possible fixes:")
        print("   - Remove or adjust the normalization in the model's predict method")
        print("   - Use a different uncertainty calculation method")
        print("   - Check if the model is properly learning uncertainty during training")
        print("\n3. The model might need:")
        print("   - Better uncertainty targets during training")
        print("   - A different loss function for uncertainty")
        print("   - More diverse training data to learn uncertainty patterns")


def create_fixed_uncertainty_calculation():
    """
    Provide a fixed uncertainty calculation method
    """
    code = '''
# Fixed uncertainty calculation for model_with_trading.py
# Replace the uncertainty normalization in the predict method

# Option 1: Use min-max scaling with bounds
def normalize_uncertainty(uncertainty_raw):
    """Normalize uncertainty to 0-1 range with proper scaling"""
    # Add small noise to prevent all values being identical
    uncertainty = uncertainty_raw + np.random.normal(0, 1e-6, uncertainty_raw.shape)
    
    # Use percentile-based normalization for better range
    low_percentile = np.percentile(uncertainty, 5)
    high_percentile = np.percentile(uncertainty, 95)
    
    if high_percentile > low_percentile:
        normalized = (uncertainty - low_percentile) / (high_percentile - low_percentile)
        normalized = np.clip(normalized, 0, 1)
    else:
        # If all values are similar, use standard scaling
        mean = np.mean(uncertainty)
        std = np.std(uncertainty)
        if std > 0:
            normalized = (uncertainty - mean) / (3 * std) + 0.5
            normalized = np.clip(normalized, 0, 1)
        else:
            # Last resort: return moderate uncertainty
            normalized = np.full_like(uncertainty, 0.5)
    
    return normalized

# Option 2: Use ensemble variance as uncertainty
def calculate_ensemble_uncertainty(base_predictions_reg):
    """Calculate uncertainty from ensemble variance"""
    # Get predictions from all models
    all_preds = np.array(list(base_predictions_reg.values()))
    
    # Calculate variance across models
    ensemble_variance = np.var(all_preds, axis=0)
    
    # Normalize to 0-1 range
    normalized_uncertainty = 1 - np.exp(-ensemble_variance)
    
    return normalized_uncertainty

# Option 3: Use prediction confidence and historical accuracy
def calculate_adaptive_uncertainty(predictions, confidence, lookback=20):
    """Calculate uncertainty based on recent prediction accuracy"""
    if len(predictions) < lookback:
        return np.full_like(predictions, 0.5)
    
    # Calculate rolling prediction variance
    pred_series = pd.Series(predictions)
    rolling_std = pred_series.rolling(lookback).std()
    
    # Combine with confidence
    uncertainty = (1 - confidence) * 0.7 + (rolling_std / pred_series.mean()) * 0.3
    
    return np.clip(uncertainty, 0, 1)
'''
    
    print("\n" + "="*60)
    print("FIXED UNCERTAINTY CALCULATION")
    print("="*60)
    print(code)
    print("\nYou can replace the uncertainty calculation in model_with_trading.py")
    print("in the predict method where it currently normalizes uncertainty.")


# Additional function to plot model confidence analysis
def plot_confidence_uncertainty_analysis(simulation_result):
    """
    Create detailed plots for confidence and uncertainty analysis
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Get data
    confidence_scores = np.array(simulation_result.get('confidence_scores', []))
    uncertainty_scores = np.array(simulation_result.get('uncertainty_scores', []))
    predictions = np.array(simulation_result.get('predictions', []))
    actuals = np.array(simulation_result.get('actual_values', []))
    
    # Calculate prediction errors
    errors = np.abs(predictions - actuals)
    relative_errors = errors / (actuals + 1e-6)
    
    # Plot 1: Confidence and Uncertainty over time
    ax1 = plt.subplot(3, 1, 1)
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(confidence_scores, color='green', linewidth=2, label='Confidence', alpha=0.8)
    line2 = ax1_twin.plot(uncertainty_scores, color='red', linewidth=2, label='Uncertainty', alpha=0.8)
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Confidence Score', fontsize=12, color='green')
    ax1_twin.set_ylabel('Uncertainty Score', fontsize=12, color='red')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.set_title('Model Confidence and Uncertainty Over Time', fontsize=16, weight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence vs Prediction Accuracy
    ax2 = plt.subplot(3, 1, 2)
    
    # Create bins for confidence levels
    confidence_bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(confidence_scores, confidence_bins)
    
    # Calculate accuracy for each confidence bin
    accuracies = []
    bin_centers = []
    bin_counts = []
    
    for i in range(1, len(confidence_bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_errors = relative_errors[mask]
            accuracy = 1 - np.mean(bin_errors)
            accuracies.append(accuracy)
            bin_centers.append((confidence_bins[i-1] + confidence_bins[i]) / 2)
            bin_counts.append(np.sum(mask))
    
    # Plot accuracy by confidence
    scatter = ax2.scatter(bin_centers, accuracies, s=np.array(bin_counts)*10, 
                         alpha=0.6, c=bin_centers, cmap='viridis')
    ax2.plot(bin_centers, accuracies, 'r--', alpha=0.5)
    
    # Ideal line (confidence = accuracy)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Ideal Calibration')
    
    ax2.set_xlabel('Confidence Level', fontsize=12)
    ax2.set_ylabel('Actual Accuracy', fontsize=12)
    ax2.set_title('Model Calibration: Confidence vs Actual Accuracy', fontsize=16, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Add colorbar for scatter
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Confidence Level', fontsize=10)
    
    # Plot 3: Trade signals colored by confidence/uncertainty
    ax3 = plt.subplot(3, 1, 3)
    
    # Calculate sequence_length for proper alignment
    sequence_length = 168  # Default value, matching the model
    
    if 'trade_signals' in simulation_result:
        buy_signals = [s for s in simulation_result['trade_signals'] if s['type'] == 'buy']
        
        if buy_signals:
            # Adjust indices to match the confidence/uncertainty arrays
            buy_indices_adjusted = []
            buy_confidences = []
            buy_uncertainties = []
            
            for signal in buy_signals:
                # Adjust the index
                adjusted_idx = signal['idx'] - sequence_length
                
                # Check if the adjusted index is valid
                if 0 <= adjusted_idx < len(confidence_scores):
                    buy_indices_adjusted.append(len(buy_indices_adjusted))  # Sequential numbering for x-axis
                    buy_confidences.append(confidence_scores[adjusted_idx])
                    buy_uncertainties.append(uncertainty_scores[adjusted_idx])
            
            if buy_confidences:
                # Create color map based on uncertainty
                scatter = ax3.scatter(range(len(buy_confidences)), buy_confidences, 
                                    c=buy_uncertainties, cmap='RdYlGn_r', 
                                    s=200, edgecolor='black', linewidth=1)
                
                ax3.set_xlabel('Buy Signal Number', fontsize=12)
                ax3.set_ylabel('Confidence at Buy', fontsize=12)
                ax3.set_title('Buy Signal Analysis: Confidence vs Uncertainty', 
                            fontsize=16, weight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax3)
                cbar.set_label('Uncertainty', fontsize=10)
                
                # Add reference lines
                ax3.axhline(y=0.6, color='orange', linestyle='--', 
                          alpha=0.5, label='Min Confidence Threshold')
                ax3.legend()
    
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()