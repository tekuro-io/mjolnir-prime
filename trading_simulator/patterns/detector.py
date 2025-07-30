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
        
        # Track alerted patterns to prevent spam
        self.alerted_patterns: Dict[str, Dict] = {}  # symbol -> {pattern_type: timestamp}
        
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
        """Trigger registered callbacks for detected pattern (with spam prevention)"""
        symbol = pattern.symbol
        pattern_type = pattern.pattern_type
        
        # Initialize symbol tracking if needed
        if symbol not in self.alerted_patterns:
            self.alerted_patterns[symbol] = {}
        
        # Check if we've already alerted for this pattern type recently
        cooldown_minutes = 5  # Don't re-alert same pattern for 5 minutes
        current_time = pattern.timestamp
        
        if pattern_type in self.alerted_patterns[symbol]:
            last_alert_time = self.alerted_patterns[symbol][pattern_type]
            time_diff = (current_time - last_alert_time).total_seconds() / 60
            
            if time_diff < cooldown_minutes:
                return  # Skip this alert - too soon since last one
        
        # Record this alert
        self.alerted_patterns[symbol][pattern_type] = current_time
        
        # Trigger callbacks
        if pattern_type in self.pattern_callbacks:
            for callback in self.pattern_callbacks[pattern_type]:
                try:
                    callback(pattern)
                except Exception as e:
                    print(f"Error in pattern callback: {e}")
    
    def _is_flat_top(self, c1: CandlestickTick, c2: CandlestickTick) -> bool:
        """Check if two candles form a flat top with tolerance"""
        return abs(c1.high - c2.high) <= self.tolerance
    
    def _is_consolidating(self, flag_candles: List[CandlestickTick], flag_len: int) -> bool:
        """Check if flag candles show consolidation (flattening) rather than continuous decline"""
        if flag_len < 5:
            return False  # Need at least 5 minutes to confirm consolidation
        
        # Split flag into first half and second half
        mid_point = flag_len // 2
        first_half = flag_candles[:mid_point] if mid_point > 0 else [flag_candles[0]]
        second_half = flag_candles[mid_point:] if mid_point < flag_len else [flag_candles[-1]]
        
        # Calculate average price for each half
        first_half_avg = sum(c.close for c in first_half) / len(first_half)
        second_half_avg = sum(c.close for c in second_half) / len(second_half)
        
        # Calculate slope (decline rate) - negative means still declining
        slope = (second_half_avg - first_half_avg) / first_half_avg
        
        # Consolidation requirements:
        # 1. Slope should be gentle (not steep decline) - less than 2% decline in second half
        # 2. OR actually flattening/rising slightly
        gentle_slope = slope > -0.02  # Less than 2% decline from first to second half
        
        # Additional check: Look at volatility - consolidation should have lower volatility in later stages
        if len(second_half) >= 3:
            # Check if recent candles are showing sideways movement
            recent_candles = flag_candles[-min(3, flag_len):]  # Last 3 candles or fewer
            recent_high = max(c.high for c in recent_candles)
            recent_low = min(c.low for c in recent_candles)
            recent_range = (recent_high - recent_low) / recent_high
            
            # Recent range should be tight (indicating consolidation, not sharp moves)
            tight_recent_range = recent_range < 0.03  # Less than 3% range in recent candles
            
            return gentle_slope and tight_recent_range
        
        return gentle_slope
    
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
        """Detect bull flag pattern with flexible, organic pattern recognition"""
        candles = list(self.candle_history[symbol])
        patterns = []
        
        if len(candles) < 8:  # Minimum: 3 flagpole + 5 flag = 8 candles
            return patterns
        
        # Try different flagpole and flag combinations
        for flagpole_len in range(3, min(13, len(candles) - 4)):  # 3-12 candle flagpoles
            for flag_len in range(4, min(16, len(candles) - flagpole_len + 1)):  # 4-15 candle flags
                total_pattern_len = flagpole_len + flag_len
                
                if total_pattern_len > len(candles):
                    continue
                    
                # Always look at the most recent pattern ending NOW
                start_idx = len(candles) - total_pattern_len
                flagpole_candles = candles[start_idx:start_idx + flagpole_len]
                flag_candles = candles[start_idx + flagpole_len:start_idx + total_pattern_len]
                
                # Phase 1: Analyze flagpole for organic upward trend
                if not self._is_organic_flagpole(flagpole_candles):
                    continue
                
                # Phase 2: Analyze flag for consolidation pattern
                if not self._is_organic_flag(flag_candles, flagpole_candles):
                    continue
                
                # Phase 3: Validate overall pattern relationship
                pattern_quality = self._validate_bull_flag_relationship(flagpole_candles, flag_candles)
                if pattern_quality['valid']:
                    trigger_candle = flag_candles[-1]
                    
                    confidence = self._calculate_organic_bull_flag_confidence(
                        flagpole_candles, flag_candles, pattern_quality
                    )
                    
                    if confidence >= 0.6:
                        patterns.append(PatternMatch(
                            pattern_type=PatternType.BULL_FLAG,
                            confidence=confidence,
                            timestamp=trigger_candle.timestamp,
                            symbol=symbol,
                            trigger_price=trigger_candle.close,
                            candles_involved=flagpole_candles + flag_candles,
                            metadata={
                                'flagpole_length': flagpole_len,
                                'flag_length': flag_len,
                                'setup_type': 'pre_breakout',
                                **pattern_quality
                            }
                        ))
        
        # Return highest confidence pattern
        if patterns:
            return [max(patterns, key=lambda p: p.confidence)]
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
            
            # Phase 3: Bear flag setup is complete - trigger BEFORE breakdown for actionable trading
            # We have: flagpole (strong decline) + flag (consolidation) = ready for potential breakdown
            
            # Use the last flag candle as the trigger point  
            trigger_candle = flag_candles[-1]
            
            # Calculate confidence based on setup quality (not breakdown confirmation)
            confidence = self._calculate_bear_flag_setup_confidence(
                flagpole_candles, flag_candles, 
                flagpole_decline, retrace_amount, flag_range
            )
            
            patterns.append(PatternMatch(
                pattern_type=PatternType.BEAR_FLAG,
                confidence=confidence,
                timestamp=trigger_candle.timestamp,
                symbol=symbol,
                trigger_price=trigger_candle.close,
                candles_involved=flagpole_candles + flag_candles,
                metadata={
                    'flagpole_decline': flagpole_decline,
                    'retrace_ratio': retrace_amount,
                    'flag_range': flag_range,
                    'flag_high': flag_high,
                    'flag_low': flag_low,
                    'flagpole_start': flagpole_start,
                    'flagpole_end': flagpole_end,
                    'breakdown_target': flag_low,  # Price level to watch for breakdown
                    'setup_type': 'pre_breakdown'   # Indicates this is setup detection, not confirmation
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
    
    def _calculate_bull_flag_setup_confidence(self, flagpole_candles: List[CandlestickTick], 
                                            flag_candles: List[CandlestickTick],
                                            flagpole_gain: float, pullback_ratio: float, 
                                            flag_range: float) -> float:
        """Calculate confidence for bull flag setup (before breakout)"""
        base_confidence = 0.6  # Start with reasonable base for valid setup
        
        # Flagpole strength (stronger flagpole = higher confidence)
        if flagpole_gain >= 0.15:  # 15%+ gain
            base_confidence += 0.2
        elif flagpole_gain >= 0.10:  # 10-15% gain
            base_confidence += 0.1
        # 8-10% is minimum, no bonus
        
        # Ideal pullback ratio (38.2% - 50% is ideal for bull flags)
        if 0.382 <= pullback_ratio <= 0.50:
            base_confidence += 0.15  # Perfect Fibonacci zone
        elif 0.236 <= pullback_ratio <= 0.382 or 0.50 <= pullback_ratio <= 0.618:
            base_confidence += 0.05  # Acceptable zones
        
        # Tight consolidation (smaller range = better flag)
        if flag_range <= 0.03:  # Very tight 3%
            base_confidence += 0.1
        elif flag_range <= 0.05:  # Tight 5%
            base_confidence += 0.05
        # 5-10% gets no bonus but is acceptable
        
        # Volume pattern (flagpole should have higher volume than flag)
        flagpole_avg_volume = sum(c.volume for c in flagpole_candles) / len(flagpole_candles)
        flag_avg_volume = sum(c.volume for c in flag_candles) / len(flag_candles) 
        
        if flagpole_avg_volume > flag_avg_volume * 1.5:  # Strong volume decline in flag
            base_confidence += 0.1
        elif flagpole_avg_volume > flag_avg_volume:  # Some volume decline
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _calculate_flexible_bull_flag_confidence(self, flagpole_candles: List[CandlestickTick], 
                                               flag_candles: List[CandlestickTick],
                                               flagpole_gain: float, pullback_ratio: float, 
                                               flag_range: float, flagpole_len: int, flag_len: int) -> float:
        """Calculate confidence for flexible bull flag patterns"""
        base_confidence = 0.5  # Start lower since we're more flexible
        
        # Flagpole strength (stronger flagpole = higher confidence)
        if flagpole_gain >= 0.15:  # 15%+ gain
            base_confidence += 0.25
        elif flagpole_gain >= 0.10:  # 10-15% gain
            base_confidence += 0.15
        elif flagpole_gain >= 0.08:  # 8-10% gain
            base_confidence += 0.10
        # 6-8% gets base confidence only
        
        # Ideal pullback ratio (38.2% - 50% is ideal)
        if 0.382 <= pullback_ratio <= 0.50:
            base_confidence += 0.15  # Perfect Fibonacci zone
        elif 0.236 <= pullback_ratio <= 0.382 or 0.50 <= pullback_ratio <= 0.618:
            base_confidence += 0.10  # Good zones
        elif 0.15 <= pullback_ratio <= 0.236 or 0.618 <= pullback_ratio <= 0.75:
            base_confidence += 0.05  # Acceptable zones
        
        # Tight consolidation (smaller range = better flag)
        if flag_range <= 0.03:  # Very tight 3%
            base_confidence += 0.10
        elif flag_range <= 0.05:  # Tight 5%
            base_confidence += 0.08
        elif flag_range <= 0.08:  # Reasonable 8%
            base_confidence += 0.05
        # Above 8% gets no bonus
        
        # Pattern proportions (ideal flagpole:flag ratios)
        ratio = flagpole_len / flag_len
        if 0.4 <= ratio <= 1.5:  # Good proportions (flagpole roughly same as flag)
            base_confidence += 0.05
        elif 0.2 <= ratio <= 2.5:  # Acceptable proportions
            base_confidence += 0.02
        
        # Pattern length bonus (not too short, not too long)
        total_len = flagpole_len + flag_len
        if 8 <= total_len <= 15:  # Sweet spot
            base_confidence += 0.03
        elif 6 <= total_len <= 20:  # Acceptable
            base_confidence += 0.01
        
        # Volume pattern (flagpole should have higher volume than flag)
        if len(flagpole_candles) > 0 and len(flag_candles) > 0:
            flagpole_avg_volume = sum(c.volume for c in flagpole_candles) / len(flagpole_candles)
            flag_avg_volume = sum(c.volume for c in flag_candles) / len(flag_candles) 
            
            if flagpole_avg_volume > flag_avg_volume * 1.5:  # Strong volume decline in flag
                base_confidence += 0.08
            elif flagpole_avg_volume > flag_avg_volume:  # Some volume decline
                base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _calculate_bear_flag_setup_confidence(self, flagpole_candles: List[CandlestickTick], 
                                            flag_candles: List[CandlestickTick],
                                            flagpole_decline: float, retrace_ratio: float, 
                                            flag_range: float) -> float:
        """Calculate confidence for bear flag setup (before breakdown)"""
        base_confidence = 0.6  # Start with reasonable base for valid setup
        
        # Flagpole strength (stronger decline = higher confidence)
        if flagpole_decline >= 0.15:  # 15%+ decline
            base_confidence += 0.2
        elif flagpole_decline >= 0.10:  # 10-15% decline
            base_confidence += 0.1
        # 8-10% is minimum, no bonus
        
        # Ideal retrace ratio (38.2% - 50% is ideal for bear flags)
        if 0.382 <= retrace_ratio <= 0.50:
            base_confidence += 0.15  # Perfect Fibonacci zone
        elif 0.236 <= retrace_ratio <= 0.382 or 0.50 <= retrace_ratio <= 0.618:
            base_confidence += 0.05  # Acceptable zones
        
        # Tight consolidation (smaller range = better flag)
        if flag_range <= 0.03:  # Very tight 3%
            base_confidence += 0.1
        elif flag_range <= 0.05:  # Tight 5%
            base_confidence += 0.05
        # 5-10% gets no bonus but is acceptable
        
        # Volume pattern (flagpole should have higher volume than flag)
        flagpole_avg_volume = sum(c.volume for c in flagpole_candles) / len(flagpole_candles)
        flag_avg_volume = sum(c.volume for c in flag_candles) / len(flag_candles) 
        
        if flagpole_avg_volume > flag_avg_volume * 1.5:  # Strong volume decline in flag
            base_confidence += 0.1
        elif flagpole_avg_volume > flag_avg_volume:  # Some volume decline
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _is_organic_flagpole(self, flagpole_candles: List[CandlestickTick]) -> bool:
        """Check if flagpole shows organic upward trend (not rigid candle-by-candle)"""
        if len(flagpole_candles) < 3:
            return False
        
        # Get key price points
        start_low = min(c.low for c in flagpole_candles[:2])  # Start area low
        end_high = max(c.high for c in flagpole_candles[-2:])  # End area high
        overall_gain = (end_high - start_low) / start_low
        
        # Must have meaningful gain (3-6% depending on length)
        min_gain = max(0.025, 0.04 - len(flagpole_candles) * 0.002)
        if overall_gain < min_gain:
            return False
        
        # Check for general upward progression (not every candle, but overall trend)
        # Divide flagpole into thirds and check progression
        third = len(flagpole_candles) // 3
        if third > 0:
            first_third_high = max(c.high for c in flagpole_candles[:third])
            middle_third_high = max(c.high for c in flagpole_candles[third:2*third])
            last_third_high = max(c.high for c in flagpole_candles[2*third:])
            
            # Allow for some flexibility - at least 2 of 3 sections should progress upward
            progressions = 0
            if middle_third_high >= first_third_high * 0.995:  # Small tolerance
                progressions += 1
            if last_third_high >= middle_third_high * 0.995:
                progressions += 1
            
            if progressions < 1:  # Need at least some upward progression
                return False
        
        # Check that we end higher than we started (net positive)
        start_price = (flagpole_candles[0].open + flagpole_candles[0].close) / 2
        end_price = (flagpole_candles[-1].open + flagpole_candles[-1].close) / 2
        if end_price <= start_price * 1.005:  # At least 0.5% net gain
            return False
        
        return True
    
    def _is_organic_flag(self, flag_candles: List[CandlestickTick], flagpole_candles: List[CandlestickTick]) -> bool:
        """Check if flag shows organic consolidation pattern"""
        if len(flag_candles) < 4:
            return False
        
        # Get flagpole peak for reference
        flagpole_high = max(c.high for c in flagpole_candles)
        
        # Flag characteristics
        flag_high = max(c.high for c in flag_candles)
        flag_low = min(c.low for c in flag_candles)
        flag_range = (flag_high - flag_low) / flag_high
        
        # Flag should not exceed flagpole high (with tolerance)
        if flag_high > flagpole_high * 1.03:  # 3% tolerance for sideways at top
            return False
        
        # Flag should show range-bound behavior (not one-directional)
        # Check if price oscillates rather than just declining
        price_changes = []
        for i in range(1, len(flag_candles)):
            prev_mid = (flag_candles[i-1].high + flag_candles[i-1].low) / 2
            curr_mid = (flag_candles[i].high + flag_candles[i].low) / 2
            change = curr_mid - prev_mid
            price_changes.append(change)
        
        # Count direction changes (oscillation indicates consolidation)
        direction_changes = 0
        for i in range(1, len(price_changes)):
            if (price_changes[i] > 0) != (price_changes[i-1] > 0):
                direction_changes += 1
        
        # Need some oscillation (not just straight decline)
        min_oscillations = max(1, len(flag_candles) // 3)
        if direction_changes < min_oscillations:
            return False
        
        # Flag range should be reasonable (not too wide, not too narrow)
        if flag_range > 0.15 or flag_range < 0.01:  # 1% to 15% range
            return False
        
        return True
    
    def _validate_bull_flag_relationship(self, flagpole_candles: List[CandlestickTick], 
                                       flag_candles: List[CandlestickTick]) -> dict:
        """Validate the relationship between flagpole and flag"""
        flagpole_low = min(c.low for c in flagpole_candles)
        flagpole_high = max(c.high for c in flagpole_candles)
        flagpole_gain = (flagpole_high - flagpole_low) / flagpole_low
        
        flag_high = max(c.high for c in flag_candles)
        flag_low = min(c.low for c in flag_candles)
        flag_range = (flag_high - flag_low) / flag_high
        
        # Calculate pullback from flagpole high
        pullback_amount = (flagpole_high - flag_low) / (flagpole_high - flagpole_low)
        
        # Validate pullback is reasonable (not too shallow, not too deep)
        if not (0.1 <= pullback_amount <= 0.8):  # 10% to 80% pullback
            return {'valid': False}
        
        # Check proportions are reasonable
        if flagpole_gain < 0.02 or flagpole_gain > 0.5:  # 2% to 50% flagpole gain
            return {'valid': False}
        
        return {
            'valid': True,
            'flagpole_gain': flagpole_gain,
            'pullback_ratio': pullback_amount,
            'flag_range': flag_range,
            'flagpole_low': flagpole_low,
            'flagpole_high': flagpole_high,
            'flag_high': flag_high,
            'flag_low': flag_low
        }
    
    def _calculate_organic_bull_flag_confidence(self, flagpole_candles: List[CandlestickTick], 
                                              flag_candles: List[CandlestickTick], 
                                              pattern_quality: dict) -> float:
        """Calculate confidence for organic bull flag patterns"""
        base_confidence = 0.6
        
        # Strong flagpole bonus
        flagpole_gain = pattern_quality['flagpole_gain']
        if flagpole_gain > 0.08:  # 8%+ gain
            base_confidence += 0.15
        elif flagpole_gain > 0.05:  # 5%+ gain
            base_confidence += 0.10
        elif flagpole_gain > 0.03:  # 3%+ gain
            base_confidence += 0.05
        
        # Ideal pullback bonus (38.2% - 50% is classic)
        pullback_ratio = pattern_quality['pullback_ratio']
        if 0.35 <= pullback_ratio <= 0.55:
            base_confidence += 0.15
        elif 0.25 <= pullback_ratio <= 0.65:
            base_confidence += 0.10
        elif 0.15 <= pullback_ratio <= 0.75:
            base_confidence += 0.05
        
        # Tight consolidation bonus
        flag_range = pattern_quality['flag_range']
        if flag_range <= 0.05:  # Very tight 5%
            base_confidence += 0.10
        elif flag_range <= 0.08:  # Tight 8%
            base_confidence += 0.05
        
        # Pattern proportions (balanced flagpole:flag ratio)
        flagpole_len = len(flagpole_candles)
        flag_len = len(flag_candles)
        ratio = flagpole_len / flag_len
        if 0.3 <= ratio <= 2.0:  # Reasonable proportions
            base_confidence += 0.05
        
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