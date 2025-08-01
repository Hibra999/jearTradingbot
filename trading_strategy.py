import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NFITradingStrategy:
    """
    NFI (NostalgiaForInfinity) Trading Strategy Implementation
    This strategy uses 21 buy conditions and 8 sell conditions with ANY OR logic
    """
    
    def __init__(self):
        # NFI Strategy Parameters
        # Price offset multipliers (buying below moving averages)
        self.low_offset_sma = 0.955    # Buy at 95.5% of SMA
        self.low_offset_ema = 0.929    # Buy at 92.9% of EMA  
        self.low_offset_trima = 0.949  # Buy at 94.9% of TRIMA
        self.low_offset_t3 = 0.975     # Buy at 97.5% of T3
        self.low_offset_kama = 0.972   # Buy at 97.2% of KAMA
        
        # Sell offset
        self.high_offset_ema = 1.047   # Sell at 104.7% of EMA
        
        # EWO thresholds
        self.ewo_low = -8.5
        self.ewo_high = 4.3
        
        # Technical indicators storage
        self.indicators = {}
        
    def calculate_indicators(self, df):
        """
        Calculate all technical indicators needed for NFI strategy
        """
        print("Calculating NFI technical indicators...")
        
        # Ensure we have OHLCV data
        close = df['Close'].values
        high = df['High'].values if 'High' in df.columns else close
        low = df['Low'].values if 'Low' in df.columns else close
        volume = df['Volume'].values if 'Volume' in df.columns else np.ones_like(close)
        
        # Price and volume
        self.indicators['close'] = close
        self.indicators['volume'] = volume
        
        # Moving Averages
        self.indicators['ema_20'] = self._calculate_ema(close, 20)
        self.indicators['sma_20'] = self._calculate_sma(close, 20)
        self.indicators['trima_20'] = self._calculate_trima(close, 20)
        self.indicators['t3_20'] = self._calculate_t3(close, 20)
        self.indicators['kama_20'] = self._calculate_kama(close, 20)
        
        # RSI - Relative Strength Index
        self.indicators['rsi'] = self._calculate_rsi(close, 14)
        
        # MFI - Money Flow Index
        self.indicators['mfi'] = self._calculate_mfi(high, low, close, volume, 14)
        
        # EWO - Elliott Wave Oscillator (difference between 5 and 35 EMAs)
        ema_5 = self._calculate_ema(close, 5)
        ema_35 = self._calculate_ema(close, 35)
        self.indicators['ewo'] = ((ema_5 - ema_35) / close) * 100
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
        self.indicators['bb_upper'] = bb_upper
        self.indicators['bb_lower'] = bb_lower
        
        # Chopiness Index
        self.indicators['chop'] = self._calculate_chopiness_index(high, low, close, 14)
        
        print("Technical indicators calculated successfully!")
        
    def _calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            
        return ema
    
    def _calculate_sma(self, data, period):
        """Calculate Simple Moving Average"""
        sma = np.zeros_like(data)
        for i in range(period-1, len(data)):
            sma[i] = np.mean(data[i-period+1:i+1])
        # Fill initial values
        sma[:period-1] = sma[period-1]
        return sma
    
    def _calculate_trima(self, data, period):
        """Calculate Triangular Moving Average"""
        # TRIMA is SMA of SMA
        sma1 = self._calculate_sma(data, period)
        trima = self._calculate_sma(sma1, period)
        return trima
    
    def _calculate_t3(self, data, period, vfactor=0.7):
        """Calculate T3 Moving Average (Tillson T3)"""
        # T3 is a smoothed moving average with less lag
        ema1 = self._calculate_ema(data, period)
        ema2 = self._calculate_ema(ema1, period)
        ema3 = self._calculate_ema(ema2, period)
        ema4 = self._calculate_ema(ema3, period)
        ema5 = self._calculate_ema(ema4, period)
        ema6 = self._calculate_ema(ema5, period)
        
        c1 = -vfactor * vfactor * vfactor
        c2 = 3 * vfactor * vfactor + 3 * vfactor * vfactor * vfactor
        c3 = -6 * vfactor * vfactor - 3 * vfactor - 3 * vfactor * vfactor * vfactor
        c4 = 1 + 3 * vfactor + vfactor * vfactor * vfactor + 3 * vfactor * vfactor
        
        t3 = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3
        return t3
    
    def _calculate_kama(self, data, period, fast=2, slow=30):
        """Calculate Kaufman Adaptive Moving Average"""
        kama = np.zeros_like(data)
        kama[0] = data[0]
        
        for i in range(1, len(data)):
            if i < period:
                kama[i] = data[i]
                continue
                
            # Calculate efficiency ratio
            change = abs(data[i] - data[i-period])
            volatility = np.sum(np.abs(np.diff(data[i-period:i+1])))
            
            if volatility != 0:
                er = change / volatility
            else:
                er = 0
                
            # Calculate smoothing constant
            sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
            
            # Calculate KAMA
            kama[i] = kama[i-1] + sc * (data[i] - kama[i-1])
            
        return kama
    
    def _calculate_rsi(self, data, period):
        """Calculate Relative Strength Index"""
        deltas = np.diff(data, prepend=data[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros_like(data)
        avg_losses = np.zeros_like(data)
        
        # Initial averages
        avg_gains[period-1] = np.mean(gains[:period])
        avg_losses[period-1] = np.mean(losses[:period])
        
        # Calculate running averages
        for i in range(period, len(data)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
        
        # Calculate RSI
        rs = np.divide(avg_gains, avg_losses, out=np.ones_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        
        # Fill initial values
        rsi[:period-1] = 50
        
        return rsi
    
    def _calculate_mfi(self, high, low, close, volume, period):
        """Calculate Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Calculate positive and negative money flow
        price_changes = np.diff(typical_price, prepend=typical_price[0])
        positive_flow = np.where(price_changes > 0, money_flow, 0)
        negative_flow = np.where(price_changes < 0, money_flow, 0)
        
        # Calculate MFI
        mfi = np.zeros_like(close)
        
        for i in range(period, len(close)):
            positive_sum = np.sum(positive_flow[i-period+1:i+1])
            negative_sum = np.sum(negative_flow[i-period+1:i+1])
            
            if negative_sum != 0:
                money_ratio = positive_sum / negative_sum
                mfi[i] = 100 - (100 / (1 + money_ratio))
            else:
                mfi[i] = 100
                
        # Fill initial values
        mfi[:period] = 50
        
        return mfi
    
    def _calculate_bollinger_bands(self, data, period, std_dev):
        """Calculate Bollinger Bands"""
        middle = self._calculate_sma(data, period)
        
        # Calculate rolling standard deviation
        std = np.zeros_like(data)
        for i in range(period-1, len(data)):
            std[i] = np.std(data[i-period+1:i+1])
        std[:period-1] = std[period-1]
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    def _calculate_chopiness_index(self, high, low, close, period):
        """Calculate Chopiness Index"""
        # ATR calculation
        tr = np.maximum(high - low, 
                       np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        
        chop = np.zeros_like(close)
        
        for i in range(period, len(close)):
            sum_tr = np.sum(tr[i-period+1:i+1])
            high_low_range = np.max(high[i-period+1:i+1]) - np.min(low[i-period+1:i+1])
            
            if high_low_range != 0:
                chop[i] = 100 * np.log10(sum_tr / high_low_range) / np.log10(period)
            else:
                chop[i] = 50
                
        # Fill initial values
        chop[:period] = 50
        
        return chop
    
    def check_buy_conditions(self, idx):
        """
        Check all 21 buy conditions for NFI strategy
        Returns True if ANY condition is met (OR logic)
        """
        # Extract current values
        close = self.indicators['close'][idx]
        volume = self.indicators['volume'][idx]
        rsi = self.indicators['rsi'][idx]
        mfi = self.indicators['mfi'][idx]
        ewo = self.indicators['ewo'][idx]
        ema = self.indicators['ema_20'][idx]
        sma = self.indicators['sma_20'][idx]
        trima = self.indicators['trima_20'][idx]
        t3 = self.indicators['t3_20'][idx]
        kama = self.indicators['kama_20'][idx]
        bb_lower = self.indicators['bb_lower'][idx]
        chop = self.indicators['chop'][idx]
        
        # Track which condition triggered for logging
        triggered_condition = None
        
        # Condition 1: Price < 92.9% EMA + MFI < 27 + Price < EMA + EWO extreme + Volume > 0
        if (close < self.low_offset_ema * ema and 
            mfi < 27 and 
            close < ema and 
            (ewo < self.ewo_low or ewo > 100) and 
            volume > 0):
            triggered_condition = 1
            
        # Condition 2: Price < 95.5% SMA + MFI < 30 + Price < EMA + EWO extreme + Volume > 0
        elif (close < self.low_offset_sma * sma and 
              mfi < 30 and 
              close < ema and 
              (ewo < self.ewo_low or ewo > 100) and 
              volume > 0):
            triggered_condition = 2
            
        # Condition 3: Price above BB lower band + MFI < 35 + TRIMA available + EWO extreme
        elif (close > bb_lower and 
              mfi < 35 and 
              (ewo < self.ewo_low or ewo > 100)):
            triggered_condition = 3
            
        # Condition 4: Price < 97.5% T3 + RSI < 40 + MFI < 35 + Volume > 0
        elif (close < self.low_offset_t3 * t3 and 
              rsi < 40 and 
              mfi < 35 and 
              volume > 0):
            triggered_condition = 4
            
        # Condition 5: Price < 97.2% KAMA + RSI < 36 + MFI < 26 + Volume > 0
        elif (close < self.low_offset_kama * kama and 
              rsi < 36 and 
              mfi < 26 and 
              volume > 0):
            triggered_condition = 5
            
        # Condition 6: Price < 92.9% EMA + MFI < 49 + Volume > 0
        elif (close < self.low_offset_ema * ema and 
              mfi < 49 and 
              volume > 0):
            triggered_condition = 6
            
        # Condition 7: Price < 94.9% TRIMA + RSI < 35 + Volume > 0
        elif (close < self.low_offset_trima * trima and 
              rsi < 35 and 
              volume > 0):
            triggered_condition = 7
            
        # Condition 8: Price < 97.5% T3 + RSI < 36 + Chopiness < 60 + Volume > 0
        elif (close < self.low_offset_t3 * t3 and 
              rsi < 36 and 
              chop < 60 and 
              volume > 0):
            triggered_condition = 8
            
        # Condition 9: Price < 95.5% SMA + MFI < 30 + Volume > 0
        elif (close < self.low_offset_sma * sma and 
              mfi < 30 and 
              volume > 0):
            triggered_condition = 9
            
        # Condition 10: Price < 92.9% EMA + RSI < 35 + Volume > 0
        elif (close < self.low_offset_ema * ema and 
              rsi < 35 and 
              volume > 0):
            triggered_condition = 10
            
        # Condition 11: Price < 95.5% SMA + MFI < 38 + Volume > 0
        elif (close < self.low_offset_sma * sma and 
              mfi < 38 and 
              volume > 0):
            triggered_condition = 11
            
        # Condition 12: Price < 92.9% EMA + EWO > 2 + Volume > 0 (bullish momentum)
        elif (close < self.low_offset_ema * ema and 
              ewo > 2 and 
              volume > 0):
            triggered_condition = 12
            
        # Condition 13: Price < 95.5% SMA + EWO < -7 + Volume > 0 (oversold)
        elif (close < self.low_offset_sma * sma and 
              ewo < -7 and 
              volume > 0):
            triggered_condition = 13
            
        # Condition 14: Price < both EMA & SMA offsets + RSI < 40 + Volume > 0
        elif (close < self.low_offset_ema * ema and 
              close < self.low_offset_sma * sma and 
              rsi < 40 and 
              volume > 0):
            triggered_condition = 14
            
        # Condition 15: Price < 92.9% EMA + RSI < 30 + Volume > 0
        elif (close < self.low_offset_ema * ema and 
              rsi < 30 and 
              volume > 0):
            triggered_condition = 15
            
        # Condition 16: Price < 92.9% EMA + EWO > -8.5 + Volume > 0
        elif (close < self.low_offset_ema * ema and 
              ewo > self.ewo_low and 
              volume > 0):
            triggered_condition = 16
            
        # Condition 17: Price < 95.5% SMA + EWO < -10 + Volume > 0 (very oversold)
        elif (close < self.low_offset_sma * sma and 
              ewo < -10 and 
              volume > 0):
            triggered_condition = 17
            
        # Condition 18: Price < 92.9% EMA + RSI < 26 + Volume > 0
        elif (close < self.low_offset_ema * ema and 
              rsi < 26 and 
              volume > 0):
            triggered_condition = 18
            
        # Condition 19: Chopiness < 58.2 + RSI < 65.3 (trending market filter)
        elif (chop < 58.2 and 
              rsi < 65.3):
            triggered_condition = 19
            
        # Condition 20: Price < 92.9% EMA + RSI < 26 + Volume > 0 (duplicate of 18)
        elif (close < self.low_offset_ema * ema and 
              rsi < 26 and 
              volume > 0):
            triggered_condition = 20
            
        # Condition 21: Price < 95.5% SMA + RSI < 23 + Volume > 0 (very oversold)
        elif (close < self.low_offset_sma * sma and 
              rsi < 23 and 
              volume > 0):
            triggered_condition = 21
            
        return triggered_condition is not None, triggered_condition
    
    def check_sell_conditions(self, idx):
        """
        Check all 8 sell conditions for NFI strategy
        Returns True if ANY condition is met (OR logic)
        """
        # Extract current values
        close = self.indicators['close'][idx]
        volume = self.indicators['volume'][idx]
        rsi = self.indicators['rsi'][idx]
        ema = self.indicators['ema_20'][idx]
        bb_upper = self.indicators['bb_upper'][idx]
        
        # Track which condition triggered
        triggered_condition = None
        
        # Condition 1: Price > 104.7% of EMA + Volume > 0
        if close > self.high_offset_ema * ema and volume > 0:
            triggered_condition = 1
            
        # Condition 2: RSI > 81.0
        elif rsi > 81.0:
            triggered_condition = 2
            
        # Condition 3: RSI > 82.0
        elif rsi > 82.0:
            triggered_condition = 3
            
        # Condition 4: Dual RSI Check - RSI > 73.4 AND RSI > 74.6
        elif rsi > 73.4 and rsi > 74.6:
            triggered_condition = 4
            
        # Condition 5: EMA Relative Gain - Price > EMA + (EMA × 2.4%) AND RSI > 50
        elif close > ema * 1.024 and rsi > 50:
            triggered_condition = 5
            
        # Condition 6: RSI > 79.0
        elif rsi > 79.0:
            triggered_condition = 6
            
        # Condition 7: RSI > 81.7
        elif rsi > 81.7:
            triggered_condition = 7
            
        # Condition 8: Price > 110% of BB Upper Band + Volume > 0
        elif close > 1.10 * bb_upper and volume > 0:
            triggered_condition = 8
            
        return triggered_condition is not None, triggered_condition


class DynamicRiskManager:
    """
    Dynamic Risk Management System
    Automatically adjusts risk per trade based on capital and risk level
    """
    
    def __init__(self, initial_capital, risk_level):
        """
        Initialize risk manager
        
        Args:
            initial_capital: Starting capital in USD
            risk_level: 'low', 'medium', or 'high'
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_level = risk_level.lower()
        
        # Risk matrix based on capital ranges
        self.risk_matrix = {
            # (min_capital, max_capital): {risk_level: risk_percentage}
            (10, 50): {'low': 0.0001, 'medium': 0.0002, 'high': 0.0003},
            (51, 100): {'low': 0.0001, 'medium': 0.0002, 'high': 0.0005},
            (101, 200): {'low': 0.0002, 'medium': 0.0005, 'high': 0.001},
            (201, 500): {'low': 0.0005, 'medium': 0.001, 'high': 0.0025},
            (501, 1000): {'low': 0.001, 'medium': 0.0025, 'high': 0.005},
            (1001, 5000): {'low': 0.002, 'medium': 0.005, 'high': 0.01},
        }
        
    def get_risk_percentage(self, capital=None):
        """
        Get the risk percentage based on current capital and risk level
        
        Args:
            capital: Capital amount (uses current_capital if None)
            
        Returns:
            float: Risk percentage (e.g., 0.01 for 1%)
        """
        if capital is None:
            capital = self.current_capital
            
        # Find the appropriate capital range
        for (min_cap, max_cap), risk_dict in self.risk_matrix.items():
            if min_cap <= capital <= max_cap:
                return risk_dict.get(self.risk_level, 0.001)
        
        # Default for capital > 5000
        if capital > 5000:
            default_risks = {'low': 0.002, 'medium': 0.005, 'high': 0.01}
            return default_risks.get(self.risk_level, 0.005)
        
        # Default for capital < 10
        return 0.0001
    
    def calculate_position_size(self, current_price, stop_loss_price=None):
        """
        Calculate position size based on risk management rules
        
        Args:
            current_price: Current asset price
            stop_loss_price: Stop loss price (if None, uses 2% below current)
            
        Returns:
            float: Number of units to buy
        """
        risk_percentage = self.get_risk_percentage()
        risk_amount = self.current_capital * risk_percentage
        
        # If no stop loss provided, assume 2% risk per trade
        if stop_loss_price is None:
            stop_loss_price = current_price * 0.98
            
        # Calculate position size
        risk_per_unit = current_price - stop_loss_price
        
        if risk_per_unit > 0:
            position_size = risk_amount / risk_per_unit
        else:
            # Use fixed percentage of capital if stop loss calculation fails
            position_size = (self.current_capital * risk_percentage * 100) / current_price
            
        # Ensure we don't use more than 99% of available capital
        max_units = (self.current_capital * 0.99) / current_price
        position_size = min(position_size, max_units)
        
        return position_size
    
    def update_capital(self, new_capital):
        """Update current capital"""
        self.current_capital = new_capital


class TradingSimulator:
    """
    Manual Trading Simulation with NFI Strategy
    Implements trading logic with loops and conditional statements
    """
    
    def __init__(self, initial_capital, risk_level, model):
        """
        Initialize the trading simulator
        
        Args:
            initial_capital: Starting capital in USD
            risk_level: 'low', 'medium', or 'high'
            model: The trained prediction model
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_level = risk_level
        self.model = model
        
        # Initialize components
        self.strategy = NFITradingStrategy()
        self.risk_manager = DynamicRiskManager(initial_capital, risk_level)
        
        # Trading state
        self.position = None  # {'entry_price', 'quantity', 'entry_time', 'entry_idx'}
        self.trades = []  # List of completed trades
        self.equity_curve = []
        self.trade_signals = []  # For visualization
        
        # Model prediction tracking
        self.prediction_errors = []
        self.confidence_window = 20  # Window for moving average of errors
        
    def calculate_prediction_confidence(self, actual_price, predicted_price, idx):
        """
        Calculate prediction confidence based on error analysis
        
        Args:
            actual_price: Actual price
            predicted_price: Model's predicted price
            idx: Current index
            
        Returns:
            bool: True if confidence is high enough to trade
        """
        # Calculate prediction error
        error = abs(actual_price - predicted_price) / actual_price
        self.prediction_errors.append(error)
        
        # Need enough data for moving average
        if len(self.prediction_errors) < self.confidence_window:
            return False
            
        # Calculate moving average of errors
        recent_errors = self.prediction_errors[-self.confidence_window:]
        avg_error = np.mean(recent_errors)
        current_error = error
        
        # Confidence check: current error should be below average (can adjust threshold)
        confidence_threshold = 0.8  # Current error should be < 80% of average
        is_confident = current_error < (avg_error * confidence_threshold)
        
        return is_confident
    
    def should_buy(self, idx, data, predictions, actual_prices):
        """
        Determine if we should buy based on strategy and model predictions
        
        Args:
            idx: Current index in the data
            data: DataFrame with OHLCV data
            predictions: Model predictions
            actual_prices: Actual prices
            
        Returns:
            tuple: (should_buy, reason)
        """
        # Check if we already have a position
        if self.position is not None:
            return False, "Already in position"
            
        # Check if we have enough data
        if idx < self.confidence_window or idx >= len(predictions) - 1:
            return False, "Insufficient data"
            
        # 1. Check NFI strategy buy conditions
        nfi_buy, buy_condition = self.strategy.check_buy_conditions(idx)
        if not nfi_buy:
            return False, "No NFI buy condition met"
            
        # 2. Check model prediction direction (looking ahead)
        current_price = actual_prices[idx]
        predicted_price = predictions[idx]
        
        # Prediction direction: expecting price increase
        if predicted_price <= current_price:
            return False, f"Model predicts decrease (NFI condition {buy_condition} ignored)"
            
        # 3. Check prediction confidence
        is_confident = self.calculate_prediction_confidence(current_price, predicted_price, idx)
        if not is_confident:
            return False, f"Low confidence (NFI condition {buy_condition} ignored)"
            
        return True, f"NFI condition {buy_condition} + Model bullish + High confidence"
    
    def should_sell(self, idx, data, predictions, actual_prices):
        """
        Determine if we should sell based on strategy and model predictions
        
        Args:
            idx: Current index in the data
            data: DataFrame with OHLCV data
            predictions: Model predictions
            actual_prices: Actual prices
            
        Returns:
            tuple: (should_sell, reason)
        """
        # Check if we have a position to sell
        if self.position is None:
            return False, "No position to sell"
            
        # 1. Check NFI strategy sell conditions
        nfi_sell, sell_condition = self.strategy.check_sell_conditions(idx)
        if nfi_sell:
            return True, f"NFI sell condition {sell_condition}"
            
        # 2. Check if model predicts significant downturn
        if idx < len(predictions) - 1:
            current_price = actual_prices[idx]
            predicted_price = predictions[idx]
            
            # If model predicts > 3% drop, consider selling
            predicted_drop = (current_price - predicted_price) / current_price
            if predicted_drop > 0.03:
                return True, f"Model predicts {predicted_drop*100:.1f}% drop"
                
        # 3. Stop loss check (2% below entry)
        current_price = actual_prices[idx]
        entry_price = self.position['entry_price']
        loss_percentage = (entry_price - current_price) / entry_price
        
        if loss_percentage > 0.02:
            return True, f"Stop loss triggered ({loss_percentage*100:.1f}% loss)"
            
        return False, "Hold position"
    
    def execute_buy(self, idx, price, timestamp):
        """Execute a buy order"""
        # Calculate position size using risk manager
        position_size = self.risk_manager.calculate_position_size(price)
        
        # Calculate cost
        cost = position_size * price
        
        # Check if we have enough capital
        if cost > self.current_capital * 0.99:
            position_size = (self.current_capital * 0.99) / price
            cost = position_size * price
            
        # Execute trade
        self.position = {
            'entry_price': price,
            'quantity': position_size,
            'entry_time': timestamp,
            'entry_idx': idx,
            'cost': cost
        }
        
        # Update capital
        self.current_capital -= cost
        
        # Record signal for visualization
        self.trade_signals.append({
            'idx': idx,
            'type': 'buy',
            'price': price,
            'quantity': position_size,
            'capital': self.current_capital + cost
        })
        
        return position_size, cost
    
    def execute_sell(self, idx, price, timestamp, reason):
        """Execute a sell order"""
        if self.position is None:
            return 0, 0
            
        # Calculate proceeds
        proceeds = self.position['quantity'] * price
        
        # Calculate profit/loss
        profit = proceeds - self.position['cost']
        profit_percentage = (profit / self.position['cost']) * 100
        
        # Record trade
        self.trades.append({
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'entry_price': self.position['entry_price'],
            'exit_price': price,
            'quantity': self.position['quantity'],
            'profit': profit,
            'profit_percentage': profit_percentage,
            'holding_period': idx - self.position['entry_idx'],
            'exit_reason': reason
        })
        
        # Update capital
        self.current_capital += proceeds
        self.risk_manager.update_capital(self.current_capital)
        
        # Record signal for visualization
        self.trade_signals.append({
            'idx': idx,
            'type': 'sell',
            'price': price,
            'quantity': self.position['quantity'],
            'capital': self.current_capital
        })
        
        # Clear position
        self.position = None
        
        return self.position['quantity'] if self.position else 0, profit
    
    def run_simulation(self, data, predictions, actual_prices):
        """
        Run the complete trading simulation
        
        Args:
            data: DataFrame with OHLCV data and timestamps
            predictions: Model predictions array
            actual_prices: Actual prices array
            
        Returns:
            dict: Simulation results
        """
        print(f"\n{'='*60}")
        print(f"STARTING TRADING SIMULATION")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Risk Level: {self.risk_level}")
        print(f"Strategy: NFI (NostalgiaForInfinity)")
        print(f"{'='*60}\n")
        
        # Calculate NFI indicators
        self.strategy.calculate_indicators(data)
        
        # Initialize equity curve
        self.equity_curve = [self.initial_capital]
        
        # Main trading loop
        for idx in range(self.confidence_window, len(actual_prices)):
            current_price = actual_prices[idx]
            timestamp = data.index[idx] if hasattr(data.index, '__iter__') else idx
            
            # Update current equity
            if self.position is not None:
                position_value = self.position['quantity'] * current_price
                current_equity = self.current_capital + position_value
            else:
                current_equity = self.current_capital
                
            self.equity_curve.append(current_equity)
            
            # Check for sell signal first (if in position)
            if self.position is not None:
                should_sell, sell_reason = self.should_sell(idx, data, predictions, actual_prices)
                
                if should_sell:
                    quantity, profit = self.execute_sell(idx, current_price, timestamp, sell_reason)
                    print(f"[{timestamp}] SELL: {quantity:.4f} units @ ${current_price:.2f} | "
                          f"Profit: ${profit:.2f} | Reason: {sell_reason}")
                    
            # Check for buy signal (if not in position)
            else:
                should_buy, buy_reason = self.should_buy(idx, data, predictions, actual_prices)
                
                if should_buy:
                    quantity, cost = self.execute_buy(idx, current_price, timestamp)
                    print(f"[{timestamp}] BUY: {quantity:.4f} units @ ${current_price:.2f} | "
                          f"Cost: ${cost:.2f} | Reason: {buy_reason}")
        
        # Force close any open position at end
        if self.position is not None:
            self.execute_sell(len(actual_prices)-1, actual_prices[-1], 
                            data.index[-1] if hasattr(data.index, '__iter__') else len(actual_prices)-1,
                            "End of simulation")
        
        # Calculate final metrics
        results = self.calculate_performance_metrics()
        
        return results
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        final_capital = self.current_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit'] > 0]
        losing_trades = [t for t in self.trades if t['profit'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Average profit/loss
        avg_profit = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
            
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'trade_signals': self.trade_signals
        }
        
        return results
    
    def print_summary(self, results):
        """Print a comprehensive summary of trading performance"""
        print(f"\n{'='*60}")
        print(f"TRADING SIMULATION SUMMARY")
        print(f"{'='*60}")
        print(f"Initial Capital:      ${results['initial_capital']:.2f}")
        print(f"Final Capital:        ${results['final_capital']:.2f}")
        print(f"Total Return:         {results['total_return']:.2f}%")
        print(f"Profit/Loss:          ${results['final_capital'] - results['initial_capital']:.2f}")
        print(f"\nTrade Statistics:")
        print(f"Total Trades:         {results['total_trades']}")
        print(f"Winning Trades:       {results['winning_trades']}")
        print(f"Losing Trades:        {results['losing_trades']}")
        print(f"Win Rate:             {results['win_rate']:.1f}%")
        print(f"Average Profit:       ${results['avg_profit']:.2f}")
        print(f"Average Loss:         ${results['avg_loss']:.2f}")
        print(f"\nRisk Metrics:")
        print(f"Max Drawdown:         {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio:         {results['sharpe_ratio']:.2f}")
        print(f"{'='*60}\n")
    
    def plot_results(self, results, prices, predictions):
        """Create comprehensive visualization of trading results"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # 1. Price chart with buy/sell signals
        ax1 = axes[0]
        ax1.plot(prices, label='Actual Price', color='black', linewidth=1.5)
        ax1.plot(predictions, label='Model Predictions', color='blue', alpha=0.7, linestyle='--')
        
        # Plot buy signals
        buy_signals = [s for s in results['trade_signals'] if s['type'] == 'buy']
        if buy_signals:
            buy_indices = [s['idx'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            ax1.scatter(buy_indices, buy_prices, color='green', marker='^', 
                       s=100, label='Buy Signal', zorder=5)
        
        # Plot sell signals
        sell_signals = [s for s in results['trade_signals'] if s['type'] == 'sell']
        if sell_signals:
            sell_indices = [s['idx'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            ax1.scatter(sell_indices, sell_prices, color='red', marker='v', 
                       s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title('Trading Signals on Price Chart', fontsize=14)
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Equity curve
        ax2 = axes[1]
        ax2.plot(results['equity_curve'], label='Portfolio Value', color='green', linewidth=2)
        ax2.axhline(y=results['initial_capital'], color='gray', linestyle='--', 
                   label='Initial Capital')
        ax2.fill_between(range(len(results['equity_curve'])), 
                        results['initial_capital'], 
                        results['equity_curve'],
                        where=np.array(results['equity_curve']) > results['initial_capital'],
                        color='green', alpha=0.3, label='Profit')
        ax2.fill_between(range(len(results['equity_curve'])), 
                        results['initial_capital'], 
                        results['equity_curve'],
                        where=np.array(results['equity_curve']) <= results['initial_capital'],
                        color='red', alpha=0.3, label='Loss')
        ax2.set_title('Portfolio Equity Curve', fontsize=14)
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown chart
        ax3 = axes[2]
        equity_curve = np.array(results['equity_curve'])
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak * 100
        ax3.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.7)
        ax3.set_title('Drawdown Analysis', fontsize=14)
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Trade P&L distribution
        ax4 = axes[3]
        if results['trades']:
            profits = [t['profit'] for t in results['trades']]
            colors = ['green' if p > 0 else 'red' for p in profits]
            bars = ax4.bar(range(len(profits)), profits, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_title('Individual Trade Profit/Loss', fontsize=14)
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('Profit/Loss ($)')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add average lines
            if results['avg_profit'] > 0:
                ax4.axhline(y=results['avg_profit'], color='green', linestyle='--', 
                           alpha=0.7, label=f"Avg Profit: ${results['avg_profit']:.2f}")
            if results['avg_loss'] < 0:
                ax4.axhline(y=results['avg_loss'], color='red', linestyle='--', 
                           alpha=0.7, label=f"Avg Loss: ${results['avg_loss']:.2f}")
            ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Additional plots - Technical indicators
        fig2, axes2 = plt.subplots(3, 1, figsize=(15, 12))
        
        # RSI
        ax5 = axes2[0]
        ax5.plot(self.strategy.indicators['rsi'], label='RSI', color='purple')
        ax5.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        ax5.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        ax5.set_title('Relative Strength Index (RSI)', fontsize=14)
        ax5.set_ylabel('RSI')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # MFI
        ax6 = axes2[1]
        ax6.plot(self.strategy.indicators['mfi'], label='MFI', color='orange')
        ax6.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Overbought')
        ax6.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Oversold')
        ax6.set_title('Money Flow Index (MFI)', fontsize=14)
        ax6.set_ylabel('MFI')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # EWO
        ax7 = axes2[2]
        ax7.plot(self.strategy.indicators['ewo'], label='EWO', color='blue')
        ax7.axhline(y=self.strategy.ewo_high, color='red', linestyle='--', alpha=0.5)
        ax7.axhline(y=self.strategy.ewo_low, color='green', linestyle='--', alpha=0.5)
        ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax7.set_title('Elliott Wave Oscillator (EWO)', fontsize=14)
        ax7.set_ylabel('EWO')
        ax7.set_xlabel('Time')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()