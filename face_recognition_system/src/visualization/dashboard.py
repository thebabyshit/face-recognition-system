"""Real-time dashboard for system monitoring."""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass, asdict

from database.services import get_database_service
from .chart_generator import ChartGenerator
from logging.access_logger import AccessLogger

logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure."""
    total_persons: int
    total_access_attempts: int
    successful_access_rate: float
    failed_access_count: int
    active_sessions: int
    system_uptime: str
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class SystemHealth:
    """System health metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    database_status: str
    camera_status: str
    recognition_service_status: str
    overall_health_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class Dashboard:
    """Real-time system monitoring dashboard."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.db_service = get_database_service()
        self.chart_generator = ChartGenerator()
        self.access_logger = AccessLogger()
        
        # Dashboard configuration
        self.config = {
            'refresh_interval': 30,  # seconds
            'metrics_retention': 24,  # hours
            'alert_thresholds': {
                'failed_access_rate': 0.1,  # 10%
                'cpu_usage': 80.0,  # 80%
                'memory_usage': 85.0,  # 85%
                'disk_usage': 90.0,  # 90%
                'response_time': 1000  # 1 second
            }
        }
        
        # Runtime state
        self.metrics_cache = {}
        self.alerts = []
        self.dashboard_data = {}
        self.start_time = datetime.now(timezone.utc)
        
        logger.info("Dashboard initialized")
    
    async def get_dashboard_data(self, time_range: str = '24h') -> Dict[str, Any]:
        """
        Get comprehensive dashboard data.
        
        Args:
            time_range: Time range for data ('1h', '24h', '7d', '30d')
            
        Returns:
            Dashboard data dictionary
        """
        try:
            # Get current metrics
            metrics = await self.get_current_metrics()
            
            # Get system health
            health = await self.get_system_health()
            
            # Get access data for charts
            access_data = await self.get_access_data(time_range)
            
            # Generate charts
            charts = await self.generate_dashboard_charts(access_data, time_range)
            
            # Get recent alerts
            recent_alerts = await self.get_recent_alerts()
            
            # Get top statistics
            statistics = await self.get_top_statistics(time_range)
            
            dashboard_data = {
                'metrics': metrics.to_dict(),
                'system_health': health.to_dict(),
                'charts': charts,
                'alerts': recent_alerts,
                'statistics': statistics,
                'time_range': time_range,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            # Cache the data
            self.dashboard_data = dashboard_data
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return self._get_error_dashboard_data(str(e))
    
    async def get_current_metrics(self) -> DashboardMetrics:
        """Get current system metrics."""
        try:
            # Get total persons
            total_persons = self.db_service.persons.get_person_count()
            
            # Get access attempts for today
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            access_logs = self.db_service.access_logs.get_access_logs_by_date_range(
                start_date=today_start,
                end_date=datetime.now(timezone.utc)
            )
            
            total_attempts = len(access_logs)
            successful_attempts = sum(1 for log in access_logs if log.access_granted)
            failed_attempts = total_attempts - successful_attempts
            
            success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
            
            # Get active sessions (mock implementation)
            active_sessions = await self._get_active_sessions_count()
            
            # Calculate uptime
            uptime = datetime.now(timezone.utc) - self.start_time
            uptime_str = self._format_uptime(uptime)
            
            return DashboardMetrics(
                total_persons=total_persons,
                total_access_attempts=total_attempts,
                successful_access_rate=success_rate,
                failed_access_count=failed_attempts,
                active_sessions=active_sessions,
                system_uptime=uptime_str,
                last_updated=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return DashboardMetrics(
                total_persons=0,
                total_access_attempts=0,
                successful_access_rate=0.0,
                failed_access_count=0,
                active_sessions=0,
                system_uptime="Unknown",
                last_updated=datetime.now(timezone.utc)
            )
    
    async def get_system_health(self) -> SystemHealth:
        """Get system health metrics."""
        try:
            # Get system resource usage (mock implementation)
            cpu_usage = await self._get_cpu_usage()
            memory_usage = await self._get_memory_usage()
            disk_usage = await self._get_disk_usage()
            
            # Check service status
            database_status = await self._check_database_status()
            camera_status = await self._check_camera_status()
            recognition_status = await self._check_recognition_service_status()
            
            # Calculate overall health score
            health_score = await self._calculate_health_score(
                cpu_usage, memory_usage, disk_usage,
                database_status, camera_status, recognition_status
            )
            
            return SystemHealth(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                database_status=database_status,
                camera_status=camera_status,
                recognition_service_status=recognition_status,
                overall_health_score=health_score
            )
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return SystemHealth(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                database_status="unknown",
                camera_status="unknown",
                recognition_service_status="unknown",
                overall_health_score=0.0
            )
    
    async def get_access_data(self, time_range: str) -> List[Dict[str, Any]]:
        """Get access data for specified time range."""
        try:
            # Calculate time range
            now = datetime.now(timezone.utc)
            if time_range == '1h':
                start_time = now - timedelta(hours=1)
            elif time_range == '24h':
                start_time = now - timedelta(hours=24)
            elif time_range == '7d':
                start_time = now - timedelta(days=7)
            elif time_range == '30d':
                start_time = now - timedelta(days=30)
            else:
                start_time = now - timedelta(hours=24)
            
            # Get access logs
            access_logs = self.db_service.access_logs.get_access_logs_by_date_range(
                start_date=start_time,
                end_date=now
            )
            
            # Convert to chart data format
            access_data = []
            for log in access_logs:
                person = self.db_service.persons.get_person_by_id(log.person_id) if log.person_id else None
                location = self.db_service.locations.get_location_by_id(log.location_id)
                
                access_data.append({
                    'timestamp': log.timestamp,
                    'person_id': log.person_id,
                    'person_name': person.name if person else 'Unknown',
                    'location_id': log.location_id,
                    'location_name': location.name if location else 'Unknown',
                    'access_granted': log.access_granted,
                    'access_method': log.access_method,
                    'confidence_score': log.confidence_score,
                    'reason': log.reason
                })\n            \n            return access_data\n            \n        except Exception as e:\n            logger.error(f\"Error getting access data: {e}\")\n            return []\n    \n    async def generate_dashboard_charts(self, access_data: List[Dict[str, Any]], time_range: str) -> Dict[str, str]:\n        \"\"\"Generate all dashboard charts.\"\"\"\n        try:\n            charts = {}\n            \n            # Access timeline\n            charts['access_timeline'] = self.chart_generator.generate_access_timeline(\n                access_data, time_range, 'plotly'\n            )\n            \n            # Success rate pie chart\n            charts['success_rate'] = self.chart_generator.generate_access_success_rate(\n                access_data, 'plotly'\n            )\n            \n            # Activity heatmap\n            charts['activity_heatmap'] = self.chart_generator.generate_person_activity_heatmap(\n                access_data, 'plotly'\n            )\n            \n            # Location usage\n            charts['location_usage'] = self.chart_generator.generate_location_usage_chart(\n                access_data, 'plotly'\n            )\n            \n            # Confidence distribution\n            charts['confidence_distribution'] = self.chart_generator.generate_recognition_confidence_distribution(\n                access_data, 'plotly'\n            )\n            \n            return charts\n            \n        except Exception as e:\n            logger.error(f\"Error generating dashboard charts: {e}\")\n            return {}\n    \n    async def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:\n        \"\"\"Get recent system alerts.\"\"\"\n        try:\n            # Get recent alerts from cache\n            recent_alerts = self.alerts[-limit:] if self.alerts else []\n            \n            # Add system health alerts\n            health_alerts = await self._check_health_alerts()\n            recent_alerts.extend(health_alerts)\n            \n            # Sort by timestamp\n            recent_alerts.sort(key=lambda x: x['timestamp'], reverse=True)\n            \n            return recent_alerts[:limit]\n            \n        except Exception as e:\n            logger.error(f\"Error getting recent alerts: {e}\")\n            return []\n    \n    async def get_top_statistics(self, time_range: str) -> Dict[str, Any]:\n        \"\"\"Get top statistics for the dashboard.\"\"\"\n        try:\n            # Calculate time range\n            now = datetime.now(timezone.utc)\n            if time_range == '1h':\n                start_time = now - timedelta(hours=1)\n            elif time_range == '24h':\n                start_time = now - timedelta(hours=24)\n            elif time_range == '7d':\n                start_time = now - timedelta(days=7)\n            elif time_range == '30d':\n                start_time = now - timedelta(days=30)\n            else:\n                start_time = now - timedelta(hours=24)\n            \n            # Get access logs\n            access_logs = self.db_service.access_logs.get_access_logs_by_date_range(\n                start_date=start_time,\n                end_date=now\n            )\n            \n            # Calculate statistics\n            statistics = {\n                'most_active_person': await self._get_most_active_person(access_logs),\n                'most_accessed_location': await self._get_most_accessed_location(access_logs),\n                'peak_access_hour': await self._get_peak_access_hour(access_logs),\n                'average_confidence_score': await self._get_average_confidence_score(access_logs),\n                'total_unique_persons': await self._get_unique_persons_count(access_logs),\n                'busiest_day': await self._get_busiest_day(access_logs)\n            }\n            \n            return statistics\n            \n        except Exception as e:\n            logger.error(f\"Error getting top statistics: {e}\")\n            return {}\n    \n    async def add_alert(self, alert_type: str, message: str, severity: str = 'info', additional_data: Optional[Dict[str, Any]] = None):\n        \"\"\"Add a new alert to the dashboard.\"\"\"\n        alert = {\n            'id': len(self.alerts) + 1,\n            'type': alert_type,\n            'message': message,\n            'severity': severity,\n            'timestamp': datetime.now(timezone.utc),\n            'additional_data': additional_data or {},\n            'acknowledged': False\n        }\n        \n        self.alerts.append(alert)\n        \n        # Keep only recent alerts\n        max_alerts = 100\n        if len(self.alerts) > max_alerts:\n            self.alerts = self.alerts[-max_alerts:]\n        \n        logger.info(f\"Alert added: {alert_type} - {message}\")\n    \n    async def acknowledge_alert(self, alert_id: int) -> bool:\n        \"\"\"Acknowledge an alert.\"\"\"\n        try:\n            for alert in self.alerts:\n                if alert['id'] == alert_id:\n                    alert['acknowledged'] = True\n                    alert['acknowledged_at'] = datetime.now(timezone.utc)\n                    return True\n            return False\n        except Exception as e:\n            logger.error(f\"Error acknowledging alert: {e}\")\n            return False\n    \n    async def get_dashboard_config(self) -> Dict[str, Any]:\n        \"\"\"Get dashboard configuration.\"\"\"\n        return self.config.copy()\n    \n    async def update_dashboard_config(self, config_updates: Dict[str, Any]):\n        \"\"\"Update dashboard configuration.\"\"\"\n        try:\n            for key, value in config_updates.items():\n                if key in self.config:\n                    self.config[key] = value\n                    logger.info(f\"Dashboard config updated: {key} = {value}\")\n                else:\n                    logger.warning(f\"Unknown config key: {key}\")\n        except Exception as e:\n            logger.error(f\"Error updating dashboard config: {e}\")\n    \n    # Helper methods\n    \n    async def _get_active_sessions_count(self) -> int:\n        \"\"\"Get count of active sessions (mock implementation).\"\"\"\n        # In a real implementation, this would check active user sessions\n        return 5\n    \n    def _format_uptime(self, uptime: timedelta) -> str:\n        \"\"\"Format uptime duration.\"\"\"\n        days = uptime.days\n        hours, remainder = divmod(uptime.seconds, 3600)\n        minutes, _ = divmod(remainder, 60)\n        \n        if days > 0:\n            return f\"{days}d {hours}h {minutes}m\"\n        elif hours > 0:\n            return f\"{hours}h {minutes}m\"\n        else:\n            return f\"{minutes}m\"\n    \n    async def _get_cpu_usage(self) -> float:\n        \"\"\"Get CPU usage percentage (mock implementation).\"\"\"\n        import random\n        return random.uniform(20, 80)\n    \n    async def _get_memory_usage(self) -> float:\n        \"\"\"Get memory usage percentage (mock implementation).\"\"\"\n        import random\n        return random.uniform(30, 70)\n    \n    async def _get_disk_usage(self) -> float:\n        \"\"\"Get disk usage percentage (mock implementation).\"\"\"\n        import random\n        return random.uniform(40, 60)\n    \n    async def _check_database_status(self) -> str:\n        \"\"\"Check database connection status.\"\"\"\n        try:\n            # Simple database connectivity check\n            self.db_service.persons.get_person_count()\n            return \"healthy\"\n        except Exception:\n            return \"error\"\n    \n    async def _check_camera_status(self) -> str:\n        \"\"\"Check camera service status (mock implementation).\"\"\"\n        # In a real implementation, this would check camera connectivity\n        return \"healthy\"\n    \n    async def _check_recognition_service_status(self) -> str:\n        \"\"\"Check recognition service status (mock implementation).\"\"\"\n        # In a real implementation, this would check recognition service health\n        return \"healthy\"\n    \n    async def _calculate_health_score(self, cpu: float, memory: float, disk: float, \n                                    db_status: str, camera_status: str, recognition_status: str) -> float:\n        \"\"\"Calculate overall system health score.\"\"\"\n        score = 100.0\n        \n        # Deduct points for high resource usage\n        if cpu > self.config['alert_thresholds']['cpu_usage']:\n            score -= (cpu - self.config['alert_thresholds']['cpu_usage']) * 0.5\n        \n        if memory > self.config['alert_thresholds']['memory_usage']:\n            score -= (memory - self.config['alert_thresholds']['memory_usage']) * 0.5\n        \n        if disk > self.config['alert_thresholds']['disk_usage']:\n            score -= (disk - self.config['alert_thresholds']['disk_usage']) * 0.5\n        \n        # Deduct points for service issues\n        if db_status != \"healthy\":\n            score -= 20\n        if camera_status != \"healthy\":\n            score -= 15\n        if recognition_status != \"healthy\":\n            score -= 15\n        \n        return max(0, min(100, score))\n    \n    async def _check_health_alerts(self) -> List[Dict[str, Any]]:\n        \"\"\"Check for system health alerts.\"\"\"\n        alerts = []\n        \n        # Check resource usage\n        cpu_usage = await self._get_cpu_usage()\n        memory_usage = await self._get_memory_usage()\n        disk_usage = await self._get_disk_usage()\n        \n        if cpu_usage > self.config['alert_thresholds']['cpu_usage']:\n            alerts.append({\n                'id': f\"cpu_{int(datetime.now().timestamp())}\",\n                'type': 'resource_usage',\n                'message': f\"High CPU usage: {cpu_usage:.1f}%\",\n                'severity': 'warning',\n                'timestamp': datetime.now(timezone.utc),\n                'additional_data': {'cpu_usage': cpu_usage}\n            })\n        \n        if memory_usage > self.config['alert_thresholds']['memory_usage']:\n            alerts.append({\n                'id': f\"memory_{int(datetime.now().timestamp())}\",\n                'type': 'resource_usage',\n                'message': f\"High memory usage: {memory_usage:.1f}%\",\n                'severity': 'warning',\n                'timestamp': datetime.now(timezone.utc),\n                'additional_data': {'memory_usage': memory_usage}\n            })\n        \n        if disk_usage > self.config['alert_thresholds']['disk_usage']:\n            alerts.append({\n                'id': f\"disk_{int(datetime.now().timestamp())}\",\n                'type': 'resource_usage',\n                'message': f\"High disk usage: {disk_usage:.1f}%\",\n                'severity': 'critical',\n                'timestamp': datetime.now(timezone.utc),\n                'additional_data': {'disk_usage': disk_usage}\n            })\n        \n        return alerts\n    \n    async def _get_most_active_person(self, access_logs: List) -> Dict[str, Any]:\n        \"\"\"Get most active person from access logs.\"\"\"\n        try:\n            person_counts = {}\n            for log in access_logs:\n                if log.person_id:\n                    person_counts[log.person_id] = person_counts.get(log.person_id, 0) + 1\n            \n            if person_counts:\n                most_active_id = max(person_counts, key=person_counts.get)\n                person = self.db_service.persons.get_person_by_id(most_active_id)\n                return {\n                    'person_id': most_active_id,\n                    'name': person.name if person else 'Unknown',\n                    'access_count': person_counts[most_active_id]\n                }\n            \n            return {'name': 'None', 'access_count': 0}\n            \n        except Exception as e:\n            logger.error(f\"Error getting most active person: {e}\")\n            return {'name': 'Error', 'access_count': 0}\n    \n    async def _get_most_accessed_location(self, access_logs: List) -> Dict[str, Any]:\n        \"\"\"Get most accessed location from access logs.\"\"\"\n        try:\n            location_counts = {}\n            for log in access_logs:\n                location_counts[log.location_id] = location_counts.get(log.location_id, 0) + 1\n            \n            if location_counts:\n                most_accessed_id = max(location_counts, key=location_counts.get)\n                location = self.db_service.locations.get_location_by_id(most_accessed_id)\n                return {\n                    'location_id': most_accessed_id,\n                    'name': location.name if location else 'Unknown',\n                    'access_count': location_counts[most_accessed_id]\n                }\n            \n            return {'name': 'None', 'access_count': 0}\n            \n        except Exception as e:\n            logger.error(f\"Error getting most accessed location: {e}\")\n            return {'name': 'Error', 'access_count': 0}\n    \n    async def _get_peak_access_hour(self, access_logs: List) -> Dict[str, Any]:\n        \"\"\"Get peak access hour from access logs.\"\"\"\n        try:\n            hour_counts = {}\n            for log in access_logs:\n                hour = log.timestamp.hour\n                hour_counts[hour] = hour_counts.get(hour, 0) + 1\n            \n            if hour_counts:\n                peak_hour = max(hour_counts, key=hour_counts.get)\n                return {\n                    'hour': peak_hour,\n                    'formatted_hour': f\"{peak_hour:02d}:00\",\n                    'access_count': hour_counts[peak_hour]\n                }\n            \n            return {'formatted_hour': 'None', 'access_count': 0}\n            \n        except Exception as e:\n            logger.error(f\"Error getting peak access hour: {e}\")\n            return {'formatted_hour': 'Error', 'access_count': 0}\n    \n    async def _get_average_confidence_score(self, access_logs: List) -> float:\n        \"\"\"Get average confidence score from access logs.\"\"\"\n        try:\n            confidence_scores = [log.confidence_score for log in access_logs \n                               if log.confidence_score is not None]\n            \n            if confidence_scores:\n                return sum(confidence_scores) / len(confidence_scores)\n            \n            return 0.0\n            \n        except Exception as e:\n            logger.error(f\"Error getting average confidence score: {e}\")\n            return 0.0\n    \n    async def _get_unique_persons_count(self, access_logs: List) -> int:\n        \"\"\"Get count of unique persons from access logs.\"\"\"\n        try:\n            unique_persons = set(log.person_id for log in access_logs if log.person_id)\n            return len(unique_persons)\n        except Exception as e:\n            logger.error(f\"Error getting unique persons count: {e}\")\n            return 0\n    \n    async def _get_busiest_day(self, access_logs: List) -> Dict[str, Any]:\n        \"\"\"Get busiest day from access logs.\"\"\"\n        try:\n            day_counts = {}\n            for log in access_logs:\n                day = log.timestamp.date()\n                day_counts[day] = day_counts.get(day, 0) + 1\n            \n            if day_counts:\n                busiest_day = max(day_counts, key=day_counts.get)\n                return {\n                    'date': busiest_day.isoformat(),\n                    'formatted_date': busiest_day.strftime('%Y-%m-%d'),\n                    'access_count': day_counts[busiest_day]\n                }\n            \n            return {'formatted_date': 'None', 'access_count': 0}\n            \n        except Exception as e:\n            logger.error(f\"Error getting busiest day: {e}\")\n            return {'formatted_date': 'Error', 'access_count': 0}\n    \n    def _get_error_dashboard_data(self, error_message: str) -> Dict[str, Any]:\n        \"\"\"Get error dashboard data when main data retrieval fails.\"\"\"\n        return {\n            'error': True,\n            'error_message': error_message,\n            'metrics': {\n                'total_persons': 0,\n                'total_access_attempts': 0,\n                'successful_access_rate': 0.0,\n                'failed_access_count': 0,\n                'active_sessions': 0,\n                'system_uptime': 'Unknown',\n                'last_updated': datetime.now(timezone.utc).isoformat()\n            },\n            'system_health': {\n                'overall_health_score': 0.0\n            },\n            'charts': {},\n            'alerts': [],\n            'statistics': {},\n            'last_updated': datetime.now(timezone.utc).isoformat()\n        }"