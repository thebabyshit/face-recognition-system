"""Alert management and notification system."""

import logging
import asyncio
import smtplib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
from collections import deque

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime
    status: AlertStatus
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data

@dataclass
class NotificationChannel:
    """Notification channel configuration."""
    name: str
    type: str  # email, webhook, sms, slack
    enabled: bool
    config: Dict[str, Any]
    severity_filter: List[AlertSeverity]

class AlertManager:
    """Comprehensive alert management system."""
    
    def __init__(self, config_file: str = "alert_config.json"):
        """
        Initialize alert manager.
        
        Args:
            config_file: Alert configuration file path
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Notification channels
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self._initialize_notification_channels()
        
        # Alert rules and suppression
        self.alert_rules = {}
        self.suppression_rules = {}
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'alerts_by_severity': {severity.value: 0 for severity in AlertSeverity},
            'alerts_by_type': {},
            'notification_failures': 0,
            'last_alert_time': None
        }
        
        logger.info("Alert manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load alert configuration."""
        default_config = {
            "alert_retention_days": 30,
            "max_active_alerts": 100,
            "notification_retry_attempts": 3,
            "notification_retry_delay": 60,
            "alert_aggregation_window": 300,  # 5 minutes
            "auto_resolve_timeout": 3600,  # 1 hour
            "escalation_enabled": True,
            "escalation_timeout": 1800,  # 30 minutes
            "default_notification_channels": ["email", "webhook"]
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                default_config.update(config)
            else:
                self._save_config(default_config)
            
            return default_config
            
        except Exception as e:
            logger.error(f"Failed to load alert config: {e}")
            return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save alert configuration."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alert config: {e}")
    
    def _initialize_notification_channels(self):
        """Initialize notification channels from configuration."""
        channels_config = self.config.get('notification_channels', {})
        
        for channel_name, channel_config in channels_config.items():
            channel = NotificationChannel(
                name=channel_name,
                type=channel_config['type'],
                enabled=channel_config.get('enabled', True),
                config=channel_config.get('config', {}),
                severity_filter=[AlertSeverity(s) for s in channel_config.get('severity_filter', ['warning', 'critical', 'emergency'])]
            )
            self.notification_channels[channel_name] = channel
    
    async def create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new alert.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            source: Alert source
            metadata: Additional metadata
            
        Returns:
            Alert ID
        """
        try:
            import uuid
            
            alert_id = str(uuid.uuid4())
            
            alert = Alert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                source=source,
                timestamp=datetime.now(timezone.utc),
                status=AlertStatus.ACTIVE,
                metadata=metadata or {}
            )
            
            # Check for duplicate/similar alerts
            if not self._should_create_alert(alert):
                logger.debug(f"Alert suppressed due to duplication: {alert_type}")
                return alert_id
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update statistics
            self.stats['total_alerts'] += 1
            self.stats['alerts_by_severity'][severity.value] += 1
            self.stats['alerts_by_type'][alert_type] = self.stats['alerts_by_type'].get(alert_type, 0) + 1
            self.stats['last_alert_time'] = alert.timestamp.isoformat()
            
            # Send notifications
            await self._send_notifications(alert)
            
            # Schedule auto-resolution if configured
            if self.config.get('auto_resolve_timeout', 0) > 0:
                asyncio.create_task(self._schedule_auto_resolve(alert_id))
            
            logger.info(f"Alert created: {alert_type} - {title} ({alert_id})")
            return alert_id
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            raise
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            acknowledged_by: User who acknowledged the alert
            
        Returns:
            True if acknowledged successfully, False otherwise
        """
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now(timezone.utc)
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID
            resolved_by: User who resolved the alert
            
        Returns:
            True if resolved successfully, False otherwise
        """
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}" + (f" by {resolved_by}" if resolved_by else ""))
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        try:
            for channel_name, channel in self.notification_channels.items():
                if not channel.enabled:
                    continue
                
                # Check severity filter
                if alert.severity not in channel.severity_filter:
                    continue
                
                try:
                    if channel.type == 'email':
                        await self._send_email_notification(alert, channel)
                    elif channel.type == 'webhook':
                        await self._send_webhook_notification(alert, channel)
                    elif channel.type == 'slack':
                        await self._send_slack_notification(alert, channel)
                    else:
                        logger.warning(f"Unsupported notification channel type: {channel.type}")
                        
                except Exception as e:
                    logger.error(f"Notification failed for channel {channel_name}: {e}")
                    self.stats['notification_failures'] += 1
                    
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
    
    async def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """Send email notification."""
        try:
            config = channel.config
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
Alert Details:
- Type: {alert.alert_type}
- Severity: {alert.severity.value.upper()}
- Source: {alert.source}
- Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Message:
{alert.message}

Alert ID: {alert.alert_id}
"""
            
            if alert.metadata:
                body += f"\nAdditional Information:\n{json.dumps(alert.metadata, indent=2)}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                if config.get('use_tls', True):
                    server.starttls()
                
                if config.get('username') and config.get('password'):
                    server.login(config['username'], config['password'])
                
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            raise
    
    async def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """Send webhook notification."""
        try:
            import aiohttp
            
            config = channel.config
            webhook_url = config['webhook_url']
            
            # Prepare payload
            payload = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            # Add custom fields if configured
            if 'custom_fields' in config:
                payload.update(config['custom_fields'])
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers=config.get('headers', {}),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for alert {alert.alert_id}")
                    else:
                        logger.error(f"Webhook notification failed: HTTP {response.status}")
                        raise Exception(f"Webhook returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            raise
    
    async def _send_slack_notification(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification."""
        try:
            import aiohttp
            
            config = channel.config
            webhook_url = config['webhook_url']
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500",
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.EMERGENCY: "#8b0000"
            }
            
            # Prepare Slack payload
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#36a64f"),
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                "short": True
                            },
                            {
                                "title": "Alert ID",
                                "value": alert.alert_id,
                                "short": True
                            }
                        ],
                        "footer": "Face Recognition System",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.alert_id}")
                    else:
                        logger.error(f"Slack notification failed: HTTP {response.status}")
                        raise Exception(f"Slack webhook returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            raise
    
    def _should_create_alert(self, alert: Alert) -> bool:
        """Check if alert should be created (deduplication and suppression)."""
        try:
            # Check for duplicate alerts in the last 5 minutes
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=5)
            
            for existing_alert in self.active_alerts.values():
                if (existing_alert.alert_type == alert.alert_type and
                    existing_alert.source == alert.source and
                    existing_alert.timestamp > cutoff_time):
                    return False  # Suppress duplicate
            
            # Check suppression rules
            for rule_name, rule in self.suppression_rules.items():
                if self._matches_suppression_rule(alert, rule):
                    logger.debug(f"Alert suppressed by rule: {rule_name}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking alert creation: {e}")
            return True  # Default to creating alert
    
    def _matches_suppression_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert matches a suppression rule."""
        try:
            # Check alert type
            if 'alert_types' in rule and alert.alert_type not in rule['alert_types']:
                return False
            
            # Check severity
            if 'severities' in rule and alert.severity.value not in rule['severities']:
                return False
            
            # Check source
            if 'sources' in rule and alert.source not in rule['sources']:
                return False
            
            # Check time window
            if 'time_window' in rule:
                start_time = datetime.fromisoformat(rule['time_window']['start'])
                end_time = datetime.fromisoformat(rule['time_window']['end'])
                
                if not (start_time <= alert.timestamp <= end_time):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching suppression rule: {e}")
            return False
    
    async def _schedule_auto_resolve(self, alert_id: str):
        """Schedule automatic alert resolution."""
        try:
            await asyncio.sleep(self.config['auto_resolve_timeout'])
            
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                if alert.status == AlertStatus.ACTIVE:
                    self.resolve_alert(alert_id, "system_auto_resolve")
                    
        except Exception as e:
            logger.error(f"Auto-resolve scheduling failed: {e}")
    
    def get_active_alerts(self, severity_filter: Optional[List[AlertSeverity]] = None) -> List[Alert]:
        """
        Get active alerts.
        
        Args:
            severity_filter: Filter by severity levels
            
        Returns:
            List of active alerts
        """
        try:
            alerts = list(self.active_alerts.values())
            
            if severity_filter:
                alerts = [alert for alert in alerts if alert.severity in severity_filter]
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda a: a.timestamp, reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
    
    def get_alert_history(self, hours: int = 24, limit: int = 100) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            hours: Number of hours of history
            limit: Maximum number of alerts to return
            
        Returns:
            List of historical alerts
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            filtered_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_time
            ]
            
            # Sort by timestamp (newest first) and limit
            filtered_alerts.sort(key=lambda a: a.timestamp, reverse=True)
            
            return filtered_alerts[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get alert history: {e}")
            return []
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        try:
            # Calculate additional statistics
            active_count = len(self.active_alerts)
            acknowledged_count = sum(1 for alert in self.active_alerts.values() if alert.status == AlertStatus.ACKNOWLEDGED)
            
            # Recent alert trends (last 24 hours)
            recent_alerts = self.get_alert_history(24)
            recent_count = len(recent_alerts)
            
            # Alert resolution time (for resolved alerts)
            resolved_alerts = [alert for alert in self.alert_history if alert.status == AlertStatus.RESOLVED and alert.resolved_at]
            
            if resolved_alerts:
                resolution_times = [
                    (alert.resolved_at - alert.timestamp).total_seconds()
                    for alert in resolved_alerts
                ]
                avg_resolution_time = sum(resolution_times) / len(resolution_times)
            else:
                avg_resolution_time = 0
            
            return {
                **self.stats,
                'active_alerts': active_count,
                'acknowledged_alerts': acknowledged_count,
                'recent_alerts_24h': recent_count,
                'average_resolution_time_seconds': avg_resolution_time,
                'notification_channels': len(self.notification_channels),
                'enabled_channels': sum(1 for ch in self.notification_channels.values() if ch.enabled)
            }
            
        except Exception as e:
            logger.error(f"Failed to get alert statistics: {e}")
            return self.stats.copy()
    
    def add_suppression_rule(self, rule_name: str, rule_config: Dict[str, Any]) -> bool:
        """
        Add alert suppression rule.
        
        Args:
            rule_name: Rule name
            rule_config: Rule configuration
            
        Returns:
            True if rule added successfully, False otherwise
        """
        try:
            self.suppression_rules[rule_name] = rule_config
            logger.info(f"Suppression rule added: {rule_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add suppression rule: {e}")
            return False
    
    def remove_suppression_rule(self, rule_name: str) -> bool:
        """Remove alert suppression rule."""
        try:
            if rule_name in self.suppression_rules:
                del self.suppression_rules[rule_name]
                logger.info(f"Suppression rule removed: {rule_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove suppression rule: {e}")
            return False
    
    def cleanup_old_alerts(self):
        """Clean up old alerts based on retention policy."""
        try:
            retention_days = self.config.get('alert_retention_days', 30)
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=retention_days)
            
            # Clean up alert history
            original_count = len(self.alert_history)
            self.alert_history = deque(
                (alert for alert in self.alert_history if alert.timestamp > cutoff_time),
                maxlen=self.alert_history.maxlen
            )
            
            cleaned_count = original_count - len(self.alert_history)
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old alerts")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Alert cleanup failed: {e}")
            return 0