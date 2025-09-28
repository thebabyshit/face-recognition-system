"""System monitoring and health checking."""

import logging
import psutil
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import json
import os

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    disk_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    uptime_seconds: float

@dataclass
class ServiceStatus:
    """Service status information."""
    service_name: str
    status: HealthStatus
    response_time: Optional[float]
    last_check: datetime
    error_message: Optional[str]
    metadata: Dict[str, Any]

class SystemMonitor:
    """Comprehensive system monitoring."""
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize system monitor.
        
        Args:
            check_interval: Monitoring check interval in seconds
        """
        self.check_interval = check_interval
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Metrics storage
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 1440  # 24 hours at 1-minute intervals
        
        # Service status tracking
        self.service_statuses: Dict[str, ServiceStatus] = {}
        
        # Alert thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'response_time_warning': 1000.0,  # ms
            'response_time_critical': 5000.0,  # ms
        }
        
        # Callbacks for alerts
        self.alert_callbacks: List[Callable] = []
        
        logger.info("System monitor initialized")
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            logger.warning("System monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self._store_metrics(metrics)
                
                # Check system health
                health_status = self._assess_system_health(metrics)
                
                # Trigger alerts if necessary
                self._check_alert_conditions(metrics, health_status)
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage (root partition)
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_stats = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_stats = {
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0,
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_avg = list(os.getloadavg())
            except (OSError, AttributeError):
                load_avg = [0.0, 0.0, 0.0]
            
            # System uptime
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_stats,
                disk_io=disk_stats,
                process_count=process_count,
                load_average=load_avg,
                uptime_seconds=uptime_seconds
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                disk_io={},
                process_count=0,
                load_average=[0.0, 0.0, 0.0],
                uptime_seconds=0.0
            )
    
    def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics in history."""
        self.metrics_history.append(metrics)
        
        # Limit history size
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def _assess_system_health(self, metrics: SystemMetrics) -> HealthStatus:
        """Assess overall system health based on metrics."""
        try:
            critical_conditions = []
            warning_conditions = []
            
            # Check CPU usage
            if metrics.cpu_usage >= self.thresholds['cpu_critical']:
                critical_conditions.append(f"CPU usage critical: {metrics.cpu_usage:.1f}%")
            elif metrics.cpu_usage >= self.thresholds['cpu_warning']:
                warning_conditions.append(f"CPU usage high: {metrics.cpu_usage:.1f}%")
            
            # Check memory usage
            if metrics.memory_usage >= self.thresholds['memory_critical']:
                critical_conditions.append(f"Memory usage critical: {metrics.memory_usage:.1f}%")
            elif metrics.memory_usage >= self.thresholds['memory_warning']:
                warning_conditions.append(f"Memory usage high: {metrics.memory_usage:.1f}%")
            
            # Check disk usage
            if metrics.disk_usage >= self.thresholds['disk_critical']:
                critical_conditions.append(f"Disk usage critical: {metrics.disk_usage:.1f}%")
            elif metrics.disk_usage >= self.thresholds['disk_warning']:
                warning_conditions.append(f"Disk usage high: {metrics.disk_usage:.1f}%")
            
            # Determine overall status
            if critical_conditions:
                return HealthStatus.CRITICAL
            elif warning_conditions:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
                
        except Exception as e:
            logger.error(f"Failed to assess system health: {e}")
            return HealthStatus.UNKNOWN
    
    def _check_alert_conditions(self, metrics: SystemMetrics, health_status: HealthStatus):
        """Check if alert conditions are met and trigger callbacks."""
        try:
            alerts = []
            
            # CPU alerts
            if metrics.cpu_usage >= self.thresholds['cpu_critical']:
                alerts.append({
                    'type': 'cpu_critical',
                    'message': f"CPU usage critical: {metrics.cpu_usage:.1f}%",
                    'severity': 'critical',
                    'value': metrics.cpu_usage,
                    'threshold': self.thresholds['cpu_critical']
                })
            elif metrics.cpu_usage >= self.thresholds['cpu_warning']:
                alerts.append({
                    'type': 'cpu_warning',
                    'message': f"CPU usage high: {metrics.cpu_usage:.1f}%",
                    'severity': 'warning',
                    'value': metrics.cpu_usage,
                    'threshold': self.thresholds['cpu_warning']
                })
            
            # Memory alerts
            if metrics.memory_usage >= self.thresholds['memory_critical']:
                alerts.append({
                    'type': 'memory_critical',
                    'message': f"Memory usage critical: {metrics.memory_usage:.1f}%",
                    'severity': 'critical',
                    'value': metrics.memory_usage,
                    'threshold': self.thresholds['memory_critical']
                })
            elif metrics.memory_usage >= self.thresholds['memory_warning']:
                alerts.append({
                    'type': 'memory_warning',
                    'message': f"Memory usage high: {metrics.memory_usage:.1f}%",
                    'severity': 'warning',
                    'value': metrics.memory_usage,
                    'threshold': self.thresholds['memory_warning']
                })
            
            # Disk alerts
            if metrics.disk_usage >= self.thresholds['disk_critical']:
                alerts.append({
                    'type': 'disk_critical',
                    'message': f"Disk usage critical: {metrics.disk_usage:.1f}%",
                    'severity': 'critical',
                    'value': metrics.disk_usage,
                    'threshold': self.thresholds['disk_critical']
                })
            elif metrics.disk_usage >= self.thresholds['disk_warning']:
                alerts.append({
                    'type': 'disk_warning',
                    'message': f"Disk usage high: {metrics.disk_usage:.1f}%",
                    'severity': 'warning',
                    'value': metrics.disk_usage,
                    'threshold': self.thresholds['disk_warning']
                })
            
            # Trigger alert callbacks
            for alert in alerts:
                self._trigger_alert_callbacks(alert, metrics)
                
        except Exception as e:
            logger.error(f"Failed to check alert conditions: {e}")
    
    def _trigger_alert_callbacks(self, alert: Dict[str, Any], metrics: SystemMetrics):
        """Trigger registered alert callbacks."""
        try:
            alert_data = {
                **alert,
                'timestamp': metrics.timestamp.isoformat(),
                'system_metrics': asdict(metrics)
            }
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to trigger alert callbacks: {e}")
    
    async def check_service_health(self, service_name: str, check_function: Callable) -> ServiceStatus:
        """
        Check health of a specific service.
        
        Args:
            service_name: Name of the service
            check_function: Async function that returns health status
            
        Returns:
            Service status information
        """
        try:
            start_time = time.time()
            
            try:
                # Call the service health check function
                health_result = await check_function()
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Determine status based on response time and result
                if isinstance(health_result, dict):
                    if health_result.get('healthy', True):
                        if response_time > self.thresholds['response_time_critical']:
                            status = HealthStatus.CRITICAL
                        elif response_time > self.thresholds['response_time_warning']:
                            status = HealthStatus.WARNING
                        else:
                            status = HealthStatus.HEALTHY
                    else:
                        status = HealthStatus.CRITICAL
                    
                    error_message = health_result.get('error')
                    metadata = health_result.get('metadata', {})
                else:
                    # Simple boolean result
                    status = HealthStatus.HEALTHY if health_result else HealthStatus.CRITICAL
                    error_message = None if health_result else "Service check failed"
                    metadata = {}
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                status = HealthStatus.CRITICAL
                error_message = str(e)
                metadata = {}
            
            service_status = ServiceStatus(
                service_name=service_name,
                status=status,
                response_time=response_time,
                last_check=datetime.now(timezone.utc),
                error_message=error_message,
                metadata=metadata
            )
            
            # Store service status
            self.service_statuses[service_name] = service_status
            
            return service_status
            
        except Exception as e:
            logger.error(f"Failed to check service health for {service_name}: {e}")
            return ServiceStatus(
                service_name=service_name,
                status=HealthStatus.UNKNOWN,
                response_time=None,
                last_check=datetime.now(timezone.utc),
                error_message=str(e),
                metadata={}
            )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """
        Get metrics history for specified time period.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of system metrics
        """
        if not self.metrics_history:
            return []
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def get_service_statuses(self) -> Dict[str, ServiceStatus]:
        """Get current status of all monitored services."""
        return self.service_statuses.copy()
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system monitoring summary."""
        try:
            current_metrics = self.get_current_metrics()
            
            if not current_metrics:
                return {
                    'status': 'no_data',
                    'message': 'No metrics available'
                }
            
            # Calculate averages over last hour
            recent_metrics = self.get_metrics_history(1)
            
            if recent_metrics:
                avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
                avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
                avg_disk = sum(m.disk_usage for m in recent_metrics) / len(recent_metrics)
            else:
                avg_cpu = current_metrics.cpu_usage
                avg_memory = current_metrics.memory_usage
                avg_disk = current_metrics.disk_usage
            
            # Count service statuses
            service_counts = {
                'healthy': 0,
                'warning': 0,
                'critical': 0,
                'unknown': 0
            }
            
            for service_status in self.service_statuses.values():
                service_counts[service_status.status.value] += 1
            
            # Determine overall system health
            overall_health = self._assess_system_health(current_metrics)
            
            return {
                'overall_health': overall_health.value,
                'monitoring_active': self.monitoring_active,
                'last_update': current_metrics.timestamp.isoformat(),
                'current_metrics': {
                    'cpu_usage': current_metrics.cpu_usage,
                    'memory_usage': current_metrics.memory_usage,
                    'disk_usage': current_metrics.disk_usage,
                    'process_count': current_metrics.process_count,
                    'uptime_hours': current_metrics.uptime_seconds / 3600
                },
                'hourly_averages': {
                    'cpu_usage': avg_cpu,
                    'memory_usage': avg_memory,
                    'disk_usage': avg_disk
                },
                'service_summary': service_counts,
                'total_services': len(self.service_statuses),
                'metrics_history_size': len(self.metrics_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get system summary: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
        logger.info("Alert callback added")
    
    def remove_alert_callback(self, callback: Callable):
        """Remove alert callback function."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            logger.info("Alert callback removed")
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update alert thresholds."""
        for key, value in new_thresholds.items():
            if key in self.thresholds:
                old_value = self.thresholds[key]
                self.thresholds[key] = value
                logger.info(f"Threshold updated: {key} = {value} (was {old_value})")
            else:
                logger.warning(f"Unknown threshold key: {key}")
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current alert thresholds."""
        return self.thresholds.copy()
    
    def export_metrics(self, hours: int = 24, format: str = 'json') -> str:
        """
        Export metrics data.
        
        Args:
            hours: Number of hours of data to export
            format: Export format ('json', 'csv')
            
        Returns:
            Exported data as string
        """
        try:
            metrics_data = self.get_metrics_history(hours)
            
            if format == 'json':
                return json.dumps([asdict(m) for m in metrics_data], indent=2, default=str)
            elif format == 'csv':
                if not metrics_data:
                    return "No data available"
                
                # CSV header
                csv_lines = [
                    "timestamp,cpu_usage,memory_usage,disk_usage,process_count,uptime_seconds"
                ]
                
                # CSV data
                for metrics in metrics_data:
                    csv_lines.append(
                        f"{metrics.timestamp.isoformat()},"
                        f"{metrics.cpu_usage},"
                        f"{metrics.memory_usage},"
                        f"{metrics.disk_usage},"
                        f"{metrics.process_count},"
                        f"{metrics.uptime_seconds}"
                    )
                
                return "\n".join(csv_lines)
            else:
                return f"Unsupported format: {format}"
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return f"Export failed: {str(e)}"