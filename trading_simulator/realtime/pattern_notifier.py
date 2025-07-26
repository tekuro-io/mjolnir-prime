"""
Pattern detection notifications and alerts for real-time trading.
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..core.models import PatternMatch
from ..core.types import PatternType


class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PatternNotification:
    """Notification for detected pattern"""
    pattern_type: PatternType
    symbol: str
    timestamp: datetime
    confidence: float
    trigger_price: float
    alert_level: AlertLevel
    message: str
    metadata: Dict = None
    
    def __str__(self) -> str:
        return f"[{self.alert_level.value.upper()}] {self.message}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'pattern_type': self.pattern_type.value,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'trigger_price': self.trigger_price,
            'alert_level': self.alert_level.value,
            'message': self.message,
            'metadata': self.metadata or {}
        }


class PatternNotifier:
    """Manages pattern detection notifications and alerts"""
    
    def __init__(self):
        # Alert thresholds
        self.confidence_thresholds = {
            AlertLevel.LOW: 0.5,
            AlertLevel.MEDIUM: 0.7,
            AlertLevel.HIGH: 0.85,
            AlertLevel.CRITICAL: 0.95
        }
        
        # Pattern priority mapping
        self.pattern_priorities = {
            PatternType.FLAT_TOP_BREAKOUT: AlertLevel.HIGH,
            PatternType.BULLISH_REVERSAL: AlertLevel.MEDIUM,
            PatternType.BEARISH_REVERSAL: AlertLevel.MEDIUM,
            PatternType.DOUBLE_TOP: AlertLevel.HIGH,
            PatternType.DOUBLE_BOTTOM: AlertLevel.HIGH
        }
        
        # Notification callbacks
        self.notification_callbacks: List[Callable[[PatternNotification], None]] = []
        
        # Recent notifications for deduplication
        self.recent_notifications: List[PatternNotification] = []
        self.max_recent_notifications = 100
    
    def register_notification_callback(self, callback: Callable[[PatternNotification], None]):
        """Register callback for pattern notifications"""
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self.notification_callbacks.append(callback)
    
    def create_notification(self, pattern: PatternMatch) -> PatternNotification:
        """Create notification from detected pattern"""
        alert_level = self._determine_alert_level(pattern)
        message = self._create_message(pattern, alert_level)
        
        notification = PatternNotification(
            pattern_type=pattern.pattern_type,
            symbol=pattern.symbol,
            timestamp=pattern.timestamp,
            confidence=pattern.confidence,
            trigger_price=pattern.trigger_price,
            alert_level=alert_level,
            message=message,
            metadata=pattern.metadata
        )
        
        # Store notification
        self._store_notification(notification)
        
        # Trigger callbacks
        self._trigger_notification_callbacks(notification)
        
        return notification
    
    def _determine_alert_level(self, pattern: PatternMatch) -> AlertLevel:
        """Determine alert level based on pattern and confidence"""
        # Start with pattern-based priority
        base_level = self.pattern_priorities.get(pattern.pattern_type, AlertLevel.LOW)
        
        # Adjust based on confidence
        if pattern.confidence >= self.confidence_thresholds[AlertLevel.CRITICAL]:
            return AlertLevel.CRITICAL
        elif pattern.confidence >= self.confidence_thresholds[AlertLevel.HIGH]:
            return AlertLevel.HIGH if self._alert_level_value(AlertLevel.HIGH) > self._alert_level_value(base_level) else base_level
        elif pattern.confidence >= self.confidence_thresholds[AlertLevel.MEDIUM]:
            return AlertLevel.MEDIUM if self._alert_level_value(AlertLevel.MEDIUM) > self._alert_level_value(base_level) else base_level
        else:
            return AlertLevel.LOW
    
    def _create_message(self, pattern: PatternMatch, alert_level: AlertLevel) -> str:
        """Create human-readable message for pattern"""
        pattern_name = pattern.pattern_type.value.replace('_', ' ').title()
        confidence_pct = int(pattern.confidence * 100)
        
        base_message = f"{pattern_name} detected for {pattern.symbol} at ${pattern.trigger_price:.2f} ({confidence_pct}% confidence)"
        
        # Add context based on pattern type
        if pattern.pattern_type == PatternType.FLAT_TOP_BREAKOUT:
            resistance = pattern.metadata.get('resistance_level', 0)
            if resistance:
                base_message += f" - Broke resistance at ${resistance:.2f}"
                
        elif pattern.pattern_type == PatternType.BULLISH_REVERSAL:
            recovery = pattern.metadata.get('recovery_strength', 0)
            if recovery:
                base_message += f" - Recovery strength: {recovery:.1%}"
                
        elif pattern.pattern_type == PatternType.BEARISH_REVERSAL:
            breakdown = pattern.metadata.get('breakdown_strength', 0)
            if breakdown:
                base_message += f" - Breakdown strength: {breakdown:.1%}"
        
        # Add urgency based on alert level
        if alert_level == AlertLevel.CRITICAL:
            base_message += " 🔥 STRONG SIGNAL!"
        elif alert_level == AlertLevel.HIGH:
            base_message += " ⚡ High confidence signal"
        
        return base_message
    
    def _store_notification(self, notification: PatternNotification):
        """Store notification for history and deduplication"""
        self.recent_notifications.append(notification)
        
        # Trim old notifications
        if len(self.recent_notifications) > self.max_recent_notifications:
            self.recent_notifications = self.recent_notifications[-self.max_recent_notifications:]
    
    def _trigger_notification_callbacks(self, notification: PatternNotification):
        """Trigger registered notification callbacks"""
        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                # Don't let callback errors break the notification system
                print(f"Error in notification callback: {e}")
    
    def get_recent_notifications(self, limit: int = 10, 
                                min_alert_level: AlertLevel = AlertLevel.LOW) -> List[PatternNotification]:
        """Get recent notifications filtered by alert level"""
        filtered = [
            n for n in self.recent_notifications
            if self._alert_level_value(n.alert_level) >= self._alert_level_value(min_alert_level)
        ]
        
        return filtered[-limit:]
    
    def _alert_level_value(self, level: AlertLevel) -> int:
        """Get numeric value for alert level comparison"""
        return {
            AlertLevel.LOW: 1,
            AlertLevel.MEDIUM: 2,
            AlertLevel.HIGH: 3,
            AlertLevel.CRITICAL: 4
        }[level]
    
    def get_notification_summary(self) -> Dict:
        """Get summary of notifications"""
        if not self.recent_notifications:
            return {'total': 0}
        
        # Count by alert level
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.value] = sum(
                1 for n in self.recent_notifications if n.alert_level == level
            )
        
        # Count by symbol
        symbol_counts = {}
        for notification in self.recent_notifications:
            symbol_counts[notification.symbol] = symbol_counts.get(notification.symbol, 0) + 1
        
        # Count by pattern type
        pattern_counts = {}
        for notification in self.recent_notifications:
            pattern_type = notification.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return {
            'total': len(self.recent_notifications),
            'by_alert_level': level_counts,
            'by_symbol': symbol_counts,
            'by_pattern_type': pattern_counts,
            'latest_notification': self.recent_notifications[-1].to_dict() if self.recent_notifications else None
        }
    
    def clear_notifications(self):
        """Clear notification history"""
        self.recent_notifications.clear()
    
    def set_confidence_threshold(self, alert_level: AlertLevel, threshold: float):
        """Set confidence threshold for alert level"""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.confidence_thresholds[alert_level] = threshold
    
    def set_pattern_priority(self, pattern_type: PatternType, alert_level: AlertLevel):
        """Set base alert level for pattern type"""
        self.pattern_priorities[pattern_type] = alert_level