"""
Advanced pattern detection with tolerance-based matching.
"""

from typing import Callable, List, Dict
from collections import deque
from datetime import datetime

from ..core.models import CandlestickTick, PatternMatch
from ..core.types import PatternType, CandleType
from ..core.exceptions import PatternDetectionError


class PatternDetector:
    """Advanced pattern detection with tolerance-based matching"""
    
    def __init__(self, tolerance: float = 0.01, lookback_window: int = 50):
        if tolerance <= 0:
            raise ValueError("Tolerance must be positive")
        if lookback_window <= 0:
            raise ValueError("Lookback window must be positive")
            
        self.tolerance = tolerance
        self.lookback_window = lookback_window
        self.candle_history: Dict[str, deque] = {}
        self.pattern_callbacks: Dict[PatternType, List[Callable]] = {}
        
    def add_candle(self, candle: CandlestickTick) -> List[PatternMatch]:
        """Add new candle and detect patterns"""
        if not isinstance(candle, CandlestickTick):
            raise PatternDetectionError("Invalid candle type")
            
        symbol = candle.symbol
        
        if symbol not in self.candle_history:
            self.candle_history[symbol] = deque(maxlen=self.lookback_window)
        
        self.candle_history[symbol].append(candle)
        
        patterns = []
        if len(self.candle_history[symbol]) >= 4:
            patterns.extend(self._detect_flat_top_breakout(symbol))
            patterns.extend(self._detect_bullish_reversal(symbol))
            patterns.extend(self._detect_bearish_reversal(symbol))
            patterns.extend(self._detect_double_top_bottom(symbol))
            patterns.extend(self._detect_bull_flag(symbol))
            patterns.extend(self._detect_bear_flag(symbol))
        
        for pattern in patterns:
            self._trigger_callbacks(pattern)
            
        return patterns
    
    def register_pattern_callback(self, pattern_type: PatternType, callback: Callable[[PatternMatch], None]):
        """Register callback for when specific pattern is detected"""
        if not callable(callback):
            raise ValueError("Callback must be callable")
            
        if pattern_type not in self.pattern_callbacks:
            self.pattern_callbacks[pattern_type] = []
        self.pattern_callbacks[pattern_type].append(callback)
    
    def _trigger_callbacks(self, pattern: PatternMatch):
        """Trigger registered callbacks for detected pattern"""
        if pattern.pattern_type in self.pattern_callbacks:
            for callback in self.pattern_callbacks[pattern.pattern_type]:
                try:
                    callback(pattern)
                except Exception as e:
                    print(f"Error in pattern callback: {e}")
    
    def _is_flat_top(self, c1: CandlestickTick, c2: CandlestickTick) -> bool:
        """Check if two candles form a flat top with tolerance"""
        return abs(c1.high - c2.high) <= self.tolerance
    
    def _detect_flat_top_breakout(self, symbol: str) -> List[PatternMatch]:
        """Detect flat top breakout pattern"""
        candles = list(self.candle_history[symbol])
        patterns = []
        
        if len(candles) < 3:
            return patterns
            
        # Only check the most recent possible pattern to avoid duplicates
        if len(candles) >= 3:
            c1, c2, c3 = candles[-3:]  # Only check last 3 candles
            
            if (self._is_flat_top(c1, c2) and 
                c3.close > max(c1.high, c2.high) and
                c3.candle_type == CandleType.BULLISH):
                
                confidence = self._calculate_breakout_confidence(c1, c2, c3)
                patterns.append(PatternMatch(
                    pattern_type=PatternType.FLAT_TOP_BREAKOUT,
                    confidence=confidence,
                    timestamp=c3.timestamp,
                    symbol=symbol,
                    trigger_price=c3.close,
                    candles_involved=[c1, c2, c3],
                    metadata={
                        'resistance_level': max(c1.high, c2.high),
                        'breakout_volume': c3.volume,
                        'breakout_strength': (c3.close - max(c1.high, c2.high)) / max(c1.high, c2.high)
                    }
                ))
        
        return patterns
    
    def _detect_bullish_reversal(self, symbol: str) -> List[PatternMatch]:
        """Detect bullish reversal confirmation pattern"""
        candles = list(self.candle_history[symbol])
        patterns = []
        
        if len(candles) < 4:
            return patterns
            
        # Only check the most recent possible pattern to avoid duplicates
        if len(candles) >= 4:
            c0, c1, c2, c3 = candles[-4:]  # Only check last 4 candles
            
            # Original pattern logic with enhancements
            breakout = c0.candle_type == CandleType.BULLISH
            pullback = (c1.candle_type == CandleType.BEARISH and 
                       c2.candle_type == CandleType.BEARISH and 
                       c1.close > c2.close)
            confirm = (c3.candle_type == CandleType.BULLISH and 
                      c3.close > c2.open)
            
            if breakout and pullback and confirm:
                confidence = self._calculate_reversal_confidence(c0, c1, c2, c3)
                patterns.append(PatternMatch(
                    pattern_type=PatternType.BULLISH_REVERSAL,
                    confidence=confidence,
                    timestamp=c3.timestamp,
                    symbol=symbol,
                    trigger_price=c3.close,
                    candles_involved=[c0, c1, c2, c3],
                    metadata={
                        'initial_high': c0.high,
                        'pullback_low': c2.low,
                        'recovery_strength': (c3.close - c2.low) / (c0.high - c2.low) if c0.high != c2.low else 0
                    }
                ))
        
        return patterns
    
    def _detect_bearish_reversal(self, symbol: str) -> List[PatternMatch]:
        """Detect bearish reversal pattern (inverse of bullish)"""
        candles = list(self.candle_history[symbol])
        patterns = []
        
        if len(candles) < 4:
            return patterns
            
        for i in range(len(candles) - 3):
            c0, c1, c2, c3 = candles[i:i+4]
            
            breakdown = c0.candle_type == CandleType.BEARISH
            bounce = (c1.candle_type == CandleType.BULLISH and 
                     c2.candle_type == CandleType.BULLISH and 
                     c1.close < c2.close)
            confirm = (c3.candle_type == CandleType.BEARISH and 
                      c3.close < c2.open)
            
            if breakdown and bounce and confirm:
                confidence = self._calculate_reversal_confidence(c0, c1, c2, c3, bearish=True)
                patterns.append(PatternMatch(
                    pattern_type=PatternType.BEARISH_REVERSAL,
                    confidence=confidence,
                    timestamp=c3.timestamp,
                    symbol=symbol,
                    trigger_price=c3.close,
                    candles_involved=[c0, c1, c2, c3],
                    metadata={
                        'initial_low': c0.low,
                        'bounce_high': c2.high,
                        'breakdown_strength': (c2.high - c3.close) / (c2.high - c0.low) if c2.high != c0.low else 0
                    }
                ))
        
        return patterns
    
    def _detect_double_top_bottom(self, symbol: str) -> List[PatternMatch]:
        """Detect double top/bottom patterns with tolerance"""
        candles = list(self.candle_history[symbol])
        patterns = []
        
        if len(candles) < 10:  # Need more history for double patterns
            return patterns
            
        # Look for peaks and valleys
        peaks = self._find_peaks(candles)
        
        # Check for double tops
        for i in range(len(peaks) - 1):
            peak1, peak2 = peaks[i], peaks[i + 1]
            if abs(peak1.high - peak2.high) <= self.tolerance:
                # Find valley between peaks
                valley_between = min([c for c in candles 
                                    if peak1.timestamp < c.timestamp < peak2.timestamp],
                                   key=lambda x: x.low, default=None)
                
                if valley_between and valley_between.low < min(peak1.low, peak2.low):
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.DOUBLE_TOP,
                        confidence=0.7,  # Base confidence
                        timestamp=peak2.timestamp,
                        symbol=symbol,
                        trigger_price=peak2.high,
                        candles_involved=[peak1, valley_between, peak2],
                        metadata={'resistance_level': (peak1.high + peak2.high) / 2}
                    ))
        
        return patterns
    
    def _detect_bull_flag(self, symbol: str) -> List[PatternMatch]:
        """Detect bull flag pattern: strong upward move, consolidation, breakout"""
        candles = list(self.candle_history[symbol])
        patterns = []
        
        if len(candles) < 15:  # Need more history for bull flag
            return patterns
        
        # Look for bull flag patterns in recent candles
        for i in range(len(candles) - 10):
            window = candles[i:i+15] if i+15 <= len(candles) else candles[i:]
            if len(window) < 10:
                continue
                
            # Phase 1: Find flagpole (strong upward move)
            flagpole_candles = window[:5]  # First 5 candles for flagpole
            flagpole_start = flagpole_candles[0].low
            flagpole_end = flagpole_candles[-1].high
            flagpole_gain = (flagpole_end - flagpole_start) / flagpole_start
            
            # Must have significant gain (at least 8%)
            if flagpole_gain < 0.08:
                continue
            
            # Phase 2: Find consolidation/flag (pullback and sideways movement)
            flag_candles = window[5:10]  # Next 5 candles for flag
            flag_high = max(c.high for c in flag_candles)
            flag_low = min(c.low for c in flag_candles)
            flag_range = (flag_high - flag_low) / flag_high
            
            # Flag should pullback 23.6% to 61.8% of flagpole gain
            pullback_amount = (flagpole_end - flag_low) / (flagpole_end - flagpole_start)
            if not (0.236 <= pullback_amount <= 0.618):
                continue
                
            # Flag should be relatively narrow (consolidation)
            if flag_range > 0.10:  # More than 10% range is too wide
                continue
            
            # Phase 3: Check for breakout
            if len(window) >= 12:
                breakout_candles = window[10:12]
                breakout_candle = breakout_candles[-1]
                
                # Breakout above flag high with volume
                if (breakout_candle.close > flag_high and 
                    breakout_candle.candle_type == CandleType.BULLISH):
                    
                    confidence = self._calculate_bull_flag_confidence(
                        flagpole_candles, flag_candles, breakout_candle, 
                        flagpole_gain, pullback_amount, flag_range
                    )
                    
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.BULL_FLAG,
                        confidence=confidence,
                        timestamp=breakout_candle.timestamp,
                        symbol=symbol,
                        trigger_price=breakout_candle.close,
                        candles_involved=flagpole_candles + flag_candles + [breakout_candle],
                        metadata={
                            'flagpole_gain': flagpole_gain,
                            'pullback_ratio': pullback_amount,
                            'flag_range': flag_range,
                            'flag_high': flag_high,
                            'flag_low': flag_low,
                            'flagpole_start': flagpole_start,
                            'flagpole_end': flagpole_end
                        }
                    ))
        
        return patterns
    
    def _detect_bear_flag(self, symbol: str) -> List[PatternMatch]:
        """Detect bear flag pattern: strong downward move, consolidation, breakdown"""
        candles = list(self.candle_history[symbol])
        patterns = []
        
        if len(candles) < 15:  # Need more history for bear flag
            return patterns
        
        # Look for bear flag patterns in recent candles
        for i in range(len(candles) - 10):
            window = candles[i:i+15] if i+15 <= len(candles) else candles[i:]
            if len(window) < 10:
                continue
                
            # Phase 1: Find flagpole (strong downward move)
            flagpole_candles = window[:5]  # First 5 candles for flagpole
            flagpole_start = flagpole_candles[0].high
            flagpole_end = flagpole_candles[-1].low
            flagpole_decline = (flagpole_start - flagpole_end) / flagpole_start
            
            # Must have significant decline (at least 8%)
            if flagpole_decline < 0.08:
                continue
            
            # Phase 2: Find consolidation/flag (bounce and sideways movement)
            flag_candles = window[5:10]  # Next 5 candles for flag
            flag_high = max(c.high for c in flag_candles)
            flag_low = min(c.low for c in flag_candles)
            flag_range = (flag_high - flag_low) / flag_low
            
            # Flag should retrace 23.6% to 61.8% of flagpole decline
            retrace_amount = (flag_high - flagpole_end) / (flagpole_start - flagpole_end)
            if not (0.236 <= retrace_amount <= 0.618):
                continue
                
            # Flag should be relatively narrow (consolidation)
            if flag_range > 0.10:  # More than 10% range is too wide
                continue
            
            # Phase 3: Check for breakdown
            if len(window) >= 12:
                breakdown_candles = window[10:12]
                breakdown_candle = breakdown_candles[-1]
                
                # Breakdown below flag low with volume
                if (breakdown_candle.close < flag_low and 
                    breakdown_candle.candle_type == CandleType.BEARISH):
                    
                    confidence = self._calculate_bear_flag_confidence(
                        flagpole_candles, flag_candles, breakdown_candle, 
                        flagpole_decline, retrace_amount, flag_range
                    )
                    
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.BEAR_FLAG,
                        confidence=confidence,
                        timestamp=breakdown_candle.timestamp,
                        symbol=symbol,
                        trigger_price=breakdown_candle.close,
                        candles_involved=flagpole_candles + flag_candles + [breakdown_candle],
                        metadata={
                            'flagpole_decline': flagpole_decline,
                            'retrace_ratio': retrace_amount,
                            'flag_range': flag_range,
                            'flag_high': flag_high,
                            'flag_low': flag_low,
                            'flagpole_start': flagpole_start,
                            'flagpole_end': flagpole_end
                        }
                    ))
        
        return patterns
    
    def _calculate_bull_flag_confidence(self, flagpole_candles: List[CandlestickTick], 
                                       flag_candles: List[CandlestickTick], 
                                       breakout_candle: CandlestickTick,
                                       flagpole_gain: float, pullback_ratio: float, 
                                       flag_range: float) -> float:
        """Calculate confidence score for bull flag patterns"""
        base_confidence = 0.6
        
        # Strong flagpole increases confidence
        if flagpole_gain > 0.15:  # > 15% gain
            base_confidence += 0.1
        if flagpole_gain > 0.25:  # > 25% gain
            base_confidence += 0.1
            
        # Ideal pullback ratio (38.2% - 50% Fibonacci levels)
        if 0.35 <= pullback_ratio <= 0.55:
            base_confidence += 0.1
            
        # Tight consolidation (flag) increases confidence
        if flag_range < 0.05:  # Less than 5% range
            base_confidence += 0.1
            
        # Volume confirmation on breakout
        flag_avg_volume = sum(c.volume for c in flag_candles) / len(flag_candles)
        if breakout_candle.volume > flag_avg_volume * 1.5:
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)
    
    def _calculate_bear_flag_confidence(self, flagpole_candles: List[CandlestickTick], 
                                       flag_candles: List[CandlestickTick], 
                                       breakdown_candle: CandlestickTick,
                                       flagpole_decline: float, retrace_ratio: float, 
                                       flag_range: float) -> float:
        """Calculate confidence score for bear flag patterns"""
        base_confidence = 0.6
        
        # Strong flagpole increases confidence
        if flagpole_decline > 0.15:  # > 15% decline
            base_confidence += 0.1
        if flagpole_decline > 0.25:  # > 25% decline
            base_confidence += 0.1
            
        # Ideal retrace ratio (38.2% - 50% Fibonacci levels)
        if 0.35 <= retrace_ratio <= 0.55:
            base_confidence += 0.1
            
        # Tight consolidation (flag) increases confidence
        if flag_range < 0.05:  # Less than 5% range
            base_confidence += 0.1
            
        # Volume confirmation on breakdown
        flag_avg_volume = sum(c.volume for c in flag_candles) / len(flag_candles)
        if breakdown_candle.volume > flag_avg_volume * 1.5:
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)
    
    def _find_peaks(self, candles: List[CandlestickTick], window: int = 3) -> List[CandlestickTick]:
        """Find local peaks in candlestick data"""
        peaks = []
        for i in range(window, len(candles) - window):
            current = candles[i]
            is_peak = all(current.high >= candles[j].high 
                         for j in range(i - window, i + window + 1) if j != i)
            if is_peak:
                peaks.append(current)
        return peaks
    
    def _find_valleys(self, candles: List[CandlestickTick], window: int = 3) -> List[CandlestickTick]:
        """Find local valleys in candlestick data"""
        valleys = []
        for i in range(window, len(candles) - window):
            current = candles[i]
            is_valley = all(current.low <= candles[j].low 
                          for j in range(i - window, i + window + 1) if j != i)
            if is_valley:
                valleys.append(current)
        return valleys
    
    def _calculate_breakout_confidence(self, c1: CandlestickTick, c2: CandlestickTick, 
                                     c3: CandlestickTick) -> float:
        """Calculate confidence score for breakout patterns"""
        base_confidence = 0.6
        
        # Volume confirmation
        if c3.volume > max(c1.volume, c2.volume):
            base_confidence += 0.2
        
        # Strength of breakout
        resistance = max(c1.high, c2.high)
        if resistance > 0:
            breakout_strength = (c3.close - resistance) / resistance
            base_confidence += min(breakout_strength * 10, 0.2)
        
        return min(base_confidence, 1.0)
    
    def _calculate_reversal_confidence(self, c0: CandlestickTick, c1: CandlestickTick,
                                     c2: CandlestickTick, c3: CandlestickTick,
                                     bearish: bool = False) -> float:
        """Calculate confidence for reversal patterns"""
        base_confidence = 0.5
        
        if not bearish:
            # Bullish reversal factors
            if c3.volume > c2.volume:
                base_confidence += 0.15
            if c3.close > c1.close:
                base_confidence += 0.15
            if c0.high != c2.low:
                recovery_ratio = (c3.close - c2.low) / (c0.high - c2.low)
                base_confidence += min(recovery_ratio * 0.2, 0.2)
        else:
            # Bearish reversal factors
            if c3.volume > c2.volume:
                base_confidence += 0.15
            if c3.close < c1.close:
                base_confidence += 0.15
            if c2.high != c0.low:
                breakdown_ratio = (c2.high - c3.close) / (c2.high - c0.low)
                base_confidence += min(breakdown_ratio * 0.2, 0.2)
        
        return min(base_confidence, 1.0)
    
    def get_pattern_history(self, symbol: str = None, pattern_type: PatternType = None) -> List[PatternMatch]:
        """Get historical patterns (would need to store them to implement fully)"""
        # This would require storing detected patterns in the class
        # For now, return empty list
        return []
    
    def clear_history(self, symbol: str = None):
        """Clear candle history for symbol or all symbols"""
        if symbol:
            if symbol in self.candle_history:
                self.candle_history[symbol].clear()
        else:
            self.candle_history.clear()