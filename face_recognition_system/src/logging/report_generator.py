"""Report generation system for access control and system analytics."""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import json
import asyncio
from pathlib import Path

from database.services import get_database_service
from .access_logger import AccessLogger

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Report type enumeration."""
    ACCESS_SUMMARY = "access_summary"
    PERSON_ACTIVITY = "person_activity"
    LOCATION_USAGE = "location_usage"
    SECURITY_INCIDENTS = "security_incidents"
    SYSTEM_PERFORMANCE = "system_performance"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_SUMMARY = "monthly_summary"
    CUSTOM = "custom"

class ReportFormat(Enum):
    """Report format enumeration."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"

class ReportGenerator:
    """Comprehensive report generation system."""
    
    def __init__(self):
        """Initialize report generator."""
        self.db_service = get_database_service()
        self.access_logger = AccessLogger()
        
        # Report templates and configurations
        self.report_templates = {
            ReportType.ACCESS_SUMMARY: self._generate_access_summary_report,
            ReportType.PERSON_ACTIVITY: self._generate_person_activity_report,
            ReportType.LOCATION_USAGE: self._generate_location_usage_report,
            ReportType.SECURITY_INCIDENTS: self._generate_security_incidents_report,
            ReportType.SYSTEM_PERFORMANCE: self._generate_system_performance_report,
            ReportType.DAILY_SUMMARY: self._generate_daily_summary_report,
            ReportType.WEEKLY_SUMMARY: self._generate_weekly_summary_report,
            ReportType.MONTHLY_SUMMARY: self._generate_monthly_summary_report
        }
        
        logger.info("Report generator initialized")
    
    async def generate_report(
        self,
        report_type: Union[str, ReportType],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        format_type: Union[str, ReportFormat] = ReportFormat.JSON,
        include_charts: bool = True,
        custom_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report.
        
        Args:
            report_type: Type of report to generate
            start_date: Start date for report data
            end_date: End date for report data
            filters: Additional filters for report data
            format_type: Output format for the report
            include_charts: Whether to include chart data
            custom_fields: Custom fields to include
            
        Returns:
            Dict containing report data and metadata
        """
        try:
            if isinstance(report_type, str):
                report_type = ReportType(report_type)
            if isinstance(format_type, str):
                format_type = ReportFormat(format_type)
            
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now(timezone.utc)
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            logger.info(f"Generating {report_type.value} report from {start_date} to {end_date}")
            
            # Generate report data
            if report_type in self.report_templates:
                report_data = await self.report_templates[report_type](
                    start_date, end_date, filters or {}, include_charts, custom_fields
                )
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            # Add metadata
            report_data['metadata'] = {
                'report_type': report_type.value,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'format': format_type.value,
                'filters': filters or {},
                'include_charts': include_charts,
                'custom_fields': custom_fields or []
            }
            
            # Format report based on requested format
            if format_type == ReportFormat.JSON:
                formatted_report = report_data
            elif format_type == ReportFormat.CSV:
                formatted_report = await self._format_as_csv(report_data)
            elif format_type == ReportFormat.HTML:
                formatted_report = await self._format_as_html(report_data)
            elif format_type == ReportFormat.PDF:
                formatted_report = await self._format_as_pdf(report_data)
            else:
                formatted_report = report_data
            
            logger.info(f"Report generated successfully: {report_type.value}")
            return formatted_report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                'error': str(e),
                'report_type': report_type.value if isinstance(report_type, ReportType) else report_type,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
    
    async def _generate_access_summary_report(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Dict[str, Any],
        include_charts: bool,
        custom_fields: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate access summary report."""
        try:
            # Get access statistics
            access_stats = await self.access_logger.get_access_statistics(
                start_date=start_date,
                end_date=end_date,
                group_by="day"
            )
            
            # Get detailed access logs
            access_logs = await self.access_logger.get_access_logs(
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )
            
            # Calculate summary metrics
            total_attempts = access_stats.get('total_attempts', 0)
            successful_attempts = access_stats.get('successful_attempts', 0)
            failed_attempts = access_stats.get('failed_attempts', 0)
            success_rate = access_stats.get('success_rate', 0.0)
            
            # Peak usage analysis
            peak_hour = await self._get_peak_usage_hour(start_date, end_date)
            peak_day = await self._get_peak_usage_day(start_date, end_date)
            
            # Top users and locations
            top_users = await self._get_top_users(start_date, end_date, limit=10)
            top_locations = await self._get_top_locations(start_date, end_date, limit=10)
            
            report_data = {
                'title': 'Access Summary Report',
                'summary': {
                    'total_attempts': total_attempts,
                    'successful_attempts': successful_attempts,
                    'failed_attempts': failed_attempts,
                    'success_rate': success_rate,
                    'average_daily_attempts': total_attempts / max(1, (end_date - start_date).days),
                    'peak_hour': peak_hour,
                    'peak_day': peak_day
                },
                'timeline': access_stats.get('timeline', []),
                'by_method': access_stats.get('by_method', []),
                'top_users': top_users,
                'top_locations': top_locations,
                'recent_activities': access_logs['logs'][:50]  # Last 50 activities
            }
            
            if include_charts:
                report_data['charts'] = {
                    'success_rate_timeline': self._prepare_timeline_chart_data(access_stats.get('timeline', [])),
                    'access_methods_pie': self._prepare_pie_chart_data(access_stats.get('by_method', [])),
                    'hourly_distribution': await self._get_hourly_distribution(start_date, end_date)
                }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating access summary report: {e}")
            raise
    
    async def _generate_person_activity_report(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Dict[str, Any],
        include_charts: bool,
        custom_fields: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate person activity report."""
        try:
            person_id = filters.get('person_id')
            
            if person_id:
                # Individual person report
                person = self.db_service.persons.get_person_by_id(person_id)
                if not person:
                    raise ValueError(f"Person not found: {person_id}")
                
                # Get person's access logs
                access_logs = await self.access_logger.get_access_logs(
                    person_id=person_id,
                    start_date=start_date,
                    end_date=end_date,
                    limit=1000
                )
                
                # Calculate person statistics
                total_attempts = len(access_logs['logs'])\n                successful_attempts = sum(1 for log in access_logs['logs'] if log['access_granted'])
                failed_attempts = total_attempts - successful_attempts
                
                # Get person's favorite locations
                location_usage = await self._get_person_location_usage(person_id, start_date, end_date)
                
                # Get person's access patterns
                access_patterns = await self._get_person_access_patterns(person_id, start_date, end_date)
                
                report_data = {
                    'title': f'Person Activity Report - {person.name}',
                    'person': {
                        'id': person.id,
                        'name': person.name,
                        'employee_id': person.employee_id,
                        'department': person.department,
                        'access_level': person.access_level,
                        'is_active': person.is_active
                    },
                    'summary': {
                        'total_attempts': total_attempts,
                        'successful_attempts': successful_attempts,
                        'failed_attempts': failed_attempts,
                        'success_rate': (successful_attempts / max(1, total_attempts)) * 100,
                        'first_access': access_logs['logs'][-1]['timestamp'] if access_logs['logs'] else None,
                        'last_access': access_logs['logs'][0]['timestamp'] if access_logs['logs'] else None,
                        'most_used_location': location_usage[0] if location_usage else None
                    },
                    'location_usage': location_usage,
                    'access_patterns': access_patterns,
                    'recent_activities': access_logs['logs'][:20]
                }
            else:
                # All persons summary report
                all_persons_stats = await self._get_all_persons_activity_stats(start_date, end_date)
                
                report_data = {
                    'title': 'All Persons Activity Report',
                    'summary': {
                        'total_active_persons': len(all_persons_stats),
                        'total_attempts': sum(p['total_attempts'] for p in all_persons_stats),
                        'average_attempts_per_person': sum(p['total_attempts'] for p in all_persons_stats) / max(1, len(all_persons_stats))
                    },
                    'persons': all_persons_stats[:50]  # Top 50 most active persons
                }
            
            if include_charts:
                if person_id:
                    report_data['charts'] = {
                        'daily_activity': await self._get_person_daily_activity_chart(person_id, start_date, end_date),
                        'location_usage_pie': self._prepare_pie_chart_data(location_usage),
                        'hourly_pattern': access_patterns.get('hourly', [])
                    }
                else:
                    report_data['charts'] = {
                        'top_users_bar': self._prepare_bar_chart_data(all_persons_stats[:10], 'name', 'total_attempts')
                    }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating person activity report: {e}")
            raise
    
    async def _generate_location_usage_report(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Dict[str, Any],
        include_charts: bool,
        custom_fields: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate location usage report."""
        try:
            location_id = filters.get('location_id')
            
            if location_id:
                # Individual location report
                location = self.db_service.locations.get_location_by_id(location_id)
                if not location:
                    raise ValueError(f"Location not found: {location_id}")
                
                # Get location's access logs
                access_logs = await self.access_logger.get_access_logs(
                    location_id=location_id,
                    start_date=start_date,
                    end_date=end_date,
                    limit=1000
                )
                
                # Calculate location statistics
                total_attempts = len(access_logs['logs'])
                successful_attempts = sum(1 for log in access_logs['logs'] if log['access_granted'])
                failed_attempts = total_attempts - successful_attempts
                unique_users = len(set(log['person_id'] for log in access_logs['logs'] if log['person_id']))
                
                # Get location usage patterns
                usage_patterns = await self._get_location_usage_patterns(location_id, start_date, end_date)
                
                # Get frequent users of this location
                frequent_users = await self._get_location_frequent_users(location_id, start_date, end_date)
                
                report_data = {
                    'title': f'Location Usage Report - {location.name}',
                    'location': {
                        'id': location.id,
                        'name': location.name,
                        'description': location.description,
                        'required_access_level': location.required_access_level,
                        'is_active': location.is_active
                    },
                    'summary': {
                        'total_attempts': total_attempts,
                        'successful_attempts': successful_attempts,
                        'failed_attempts': failed_attempts,
                        'success_rate': (successful_attempts / max(1, total_attempts)) * 100,
                        'unique_users': unique_users,
                        'average_daily_usage': total_attempts / max(1, (end_date - start_date).days),
                        'peak_usage_hour': usage_patterns.get('peak_hour'),
                        'peak_usage_day': usage_patterns.get('peak_day')
                    },
                    'usage_patterns': usage_patterns,
                    'frequent_users': frequent_users,
                    'recent_activities': access_logs['logs'][:20]
                }
            else:
                # All locations summary report
                all_locations_stats = await self._get_all_locations_usage_stats(start_date, end_date)
                
                report_data = {
                    'title': 'All Locations Usage Report',
                    'summary': {
                        'total_locations': len(all_locations_stats),
                        'total_attempts': sum(l['total_attempts'] for l in all_locations_stats),
                        'average_attempts_per_location': sum(l['total_attempts'] for l in all_locations_stats) / max(1, len(all_locations_stats))
                    },
                    'locations': all_locations_stats
                }
            
            if include_charts:
                if location_id:
                    report_data['charts'] = {
                        'daily_usage': await self._get_location_daily_usage_chart(location_id, start_date, end_date),
                        'hourly_pattern': usage_patterns.get('hourly', []),
                        'user_distribution': self._prepare_pie_chart_data(frequent_users)
                    }
                else:
                    report_data['charts'] = {
                        'location_usage_bar': self._prepare_bar_chart_data(all_locations_stats, 'name', 'total_attempts')
                    }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating location usage report: {e}")
            raise
    
    async def _generate_security_incidents_report(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Dict[str, Any],
        include_charts: bool,
        custom_fields: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate security incidents report."""
        try:
            # Get security statistics
            security_stats = await self.access_logger.get_security_statistics(
                start_date=start_date,
                end_date=end_date
            )
            
            # Get security-related system logs
            security_logs = await self.access_logger.get_system_logs(
                event_type="security_alert",
                start_date=start_date,
                end_date=end_date,
                limit=500
            )
            
            # Get failed access attempts
            failed_attempts = await self.access_logger.get_access_logs(
                access_granted=False,
                start_date=start_date,
                end_date=end_date,
                limit=500
            )
            
            # Analyze security incidents
            incident_analysis = await self._analyze_security_incidents(
                security_logs['logs'], failed_attempts['logs']
            )
            
            # Get threat patterns
            threat_patterns = await self._get_threat_patterns(start_date, end_date)
            
            report_data = {
                'title': 'Security Incidents Report',
                'summary': {
                    'total_incidents': security_stats.get('total_events', 0),
                    'security_alerts': security_stats.get('security_alerts', 0),
                    'authentication_failures': security_stats.get('authentication_failures', 0),
                    'system_errors': security_stats.get('system_errors', 0),
                    'failed_access_attempts': len(failed_attempts['logs']),
                    'high_risk_incidents': incident_analysis.get('high_risk_count', 0),
                    'resolved_incidents': incident_analysis.get('resolved_count', 0)
                },
                'incident_breakdown': security_stats.get('by_event_type', []),
                'severity_breakdown': security_stats.get('by_level', []),
                'threat_patterns': threat_patterns,
                'recent_incidents': security_logs['logs'][:20],
                'failed_attempts_analysis': incident_analysis.get('failed_attempts_analysis', {}),
                'recommendations': await self._generate_security_recommendations(incident_analysis)
            }
            
            if include_charts:
                report_data['charts'] = {
                    'incidents_timeline': security_stats.get('timeline', []),
                    'incident_types_pie': self._prepare_pie_chart_data(security_stats.get('by_event_type', [])),
                    'severity_distribution': self._prepare_pie_chart_data(security_stats.get('by_level', [])),
                    'threat_heatmap': await self._get_threat_heatmap_data(start_date, end_date)
                }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating security incidents report: {e}")
            raise
    
    async def _generate_system_performance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Dict[str, Any],
        include_charts: bool,
        custom_fields: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate system performance report."""
        try:
            # Get system performance metrics
            performance_metrics = await self._get_system_performance_metrics(start_date, end_date)
            
            # Get system logs for performance analysis
            system_logs = await self.access_logger.get_system_logs(
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )
            
            # Analyze response times and throughput
            response_analysis = await self._analyze_response_times(start_date, end_date)
            
            # Get error rates and uptime
            error_analysis = await self._analyze_system_errors(system_logs['logs'])
            
            # Calculate uptime and availability
            uptime_stats = await self._calculate_uptime_stats(start_date, end_date)
            
            report_data = {
                'title': 'System Performance Report',
                'summary': {
                    'uptime_percentage': uptime_stats.get('uptime_percentage', 0),
                    'total_requests': performance_metrics.get('total_requests', 0),
                    'average_response_time': response_analysis.get('average_response_time', 0),
                    'error_rate': error_analysis.get('error_rate', 0),
                    'peak_throughput': performance_metrics.get('peak_throughput', 0),
                    'system_availability': uptime_stats.get('availability_score', 0)
                },
                'performance_metrics': performance_metrics,
                'response_time_analysis': response_analysis,
                'error_analysis': error_analysis,
                'uptime_stats': uptime_stats,
                'resource_usage': await self._get_resource_usage_stats(start_date, end_date),
                'bottlenecks': await self._identify_performance_bottlenecks(start_date, end_date)
            }
            
            if include_charts:
                report_data['charts'] = {
                    'response_time_timeline': response_analysis.get('timeline', []),
                    'throughput_chart': performance_metrics.get('throughput_timeline', []),
                    'error_rate_chart': error_analysis.get('timeline', []),
                    'uptime_chart': uptime_stats.get('timeline', [])
                }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating system performance report: {e}")
            raise
    
    async def _generate_daily_summary_report(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Dict[str, Any],
        include_charts: bool,
        custom_fields: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate daily summary report."""
        try:
            # Use single day if range is provided
            target_date = filters.get('target_date', end_date.date())
            day_start = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
            day_end = day_start + timedelta(days=1)
            
            # Get daily access statistics
            daily_stats = await self.access_logger.get_access_statistics(
                start_date=day_start,
                end_date=day_end,
                group_by="hour"
            )
            
            # Get daily activities
            daily_activities = await self.access_logger.get_access_logs(
                start_date=day_start,
                end_date=day_end,
                limit=200
            )
            
            # Get daily security events
            daily_security = await self.access_logger.get_system_logs(
                start_date=day_start,
                end_date=day_end,
                limit=100
            )
            
            # Calculate daily insights
            insights = await self._generate_daily_insights(daily_stats, daily_activities['logs'])
            
            report_data = {
                'title': f'Daily Summary Report - {target_date}',
                'date': target_date.isoformat(),
                'summary': {
                    'total_attempts': daily_stats.get('total_attempts', 0),
                    'successful_attempts': daily_stats.get('successful_attempts', 0),
                    'failed_attempts': daily_stats.get('failed_attempts', 0),
                    'unique_users': len(set(log['person_id'] for log in daily_activities['logs'] if log['person_id'])),
                    'peak_hour': insights.get('peak_hour'),
                    'busiest_location': insights.get('busiest_location'),
                    'security_incidents': len(daily_security['logs'])
                },
                'hourly_breakdown': daily_stats.get('timeline', []),
                'top_activities': daily_activities['logs'][:10],
                'security_events': daily_security['logs'][:5],
                'insights': insights,
                'recommendations': await self._generate_daily_recommendations(insights)
            }
            
            if include_charts:
                report_data['charts'] = {
                    'hourly_activity': daily_stats.get('timeline', []),
                    'access_methods': self._prepare_pie_chart_data(daily_stats.get('by_method', []))
                }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating daily summary report: {e}")
            raise
    
    async def _generate_weekly_summary_report(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Dict[str, Any],
        include_charts: bool,
        custom_fields: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate weekly summary report."""
        try:
            # Ensure we have a full week
            week_start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            week_end = week_start + timedelta(days=7)
            
            # Get weekly statistics
            weekly_stats = await self.access_logger.get_access_statistics(
                start_date=week_start,
                end_date=week_end,
                group_by="day"
            )
            
            # Compare with previous week
            prev_week_start = week_start - timedelta(days=7)
            prev_week_stats = await self.access_logger.get_access_statistics(
                start_date=prev_week_start,
                end_date=week_start,
                group_by="day"
            )
            
            # Calculate trends
            trends = await self._calculate_weekly_trends(weekly_stats, prev_week_stats)
            
            report_data = {
                'title': f'Weekly Summary Report - Week of {week_start.date()}',
                'week_start': week_start.date().isoformat(),
                'week_end': week_end.date().isoformat(),
                'summary': {
                    'total_attempts': weekly_stats.get('total_attempts', 0),
                    'successful_attempts': weekly_stats.get('successful_attempts', 0),
                    'failed_attempts': weekly_stats.get('failed_attempts', 0),
                    'success_rate': weekly_stats.get('success_rate', 0),
                    'daily_average': weekly_stats.get('total_attempts', 0) / 7,
                    'busiest_day': trends.get('busiest_day'),
                    'quietest_day': trends.get('quietest_day')
                },
                'daily_breakdown': weekly_stats.get('timeline', []),
                'trends': trends,
                'week_over_week_comparison': {
                    'current_week': weekly_stats.get('total_attempts', 0),
                    'previous_week': prev_week_stats.get('total_attempts', 0),
                    'change_percentage': trends.get('change_percentage', 0)
                }
            }
            
            if include_charts:
                report_data['charts'] = {
                    'daily_trend': weekly_stats.get('timeline', []),
                    'week_comparison': [
                        {'week': 'Current', 'attempts': weekly_stats.get('total_attempts', 0)},
                        {'week': 'Previous', 'attempts': prev_week_stats.get('total_attempts', 0)}
                    ]
                }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating weekly summary report: {e}")
            raise
    
    async def _generate_monthly_summary_report(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Dict[str, Any],
        include_charts: bool,
        custom_fields: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate monthly summary report."""
        try:
            # Ensure we have a full month
            month_start = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if month_start.month == 12:
                month_end = month_start.replace(year=month_start.year + 1, month=1)
            else:
                month_end = month_start.replace(month=month_start.month + 1)
            
            # Get monthly statistics
            monthly_stats = await self.access_logger.get_access_statistics(
                start_date=month_start,
                end_date=month_end,
                group_by="day"
            )
            
            # Get monthly insights
            insights = await self._generate_monthly_insights(month_start, month_end)
            
            report_data = {
                'title': f'Monthly Summary Report - {month_start.strftime("%B %Y")}',
                'month': month_start.strftime("%Y-%m"),
                'summary': {
                    'total_attempts': monthly_stats.get('total_attempts', 0),
                    'successful_attempts': monthly_stats.get('successful_attempts', 0),
                    'failed_attempts': monthly_stats.get('failed_attempts', 0),
                    'success_rate': monthly_stats.get('success_rate', 0),
                    'daily_average': monthly_stats.get('total_attempts', 0) / (month_end - month_start).days,
                    'peak_day': insights.get('peak_day'),
                    'total_unique_users': insights.get('unique_users', 0)
                },
                'daily_breakdown': monthly_stats.get('timeline', []),
                'insights': insights,
                'monthly_trends': await self._calculate_monthly_trends(month_start, month_end)
            }
            
            if include_charts:
                report_data['charts'] = {
                    'monthly_trend': monthly_stats.get('timeline', []),
                    'weekly_comparison': await self._get_weekly_comparison_data(month_start, month_end)
                }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating monthly summary report: {e}")
            raise
    
    # Helper methods for data processing and analysis
    
    async def _get_peak_usage_hour(self, start_date: datetime, end_date: datetime) -> Optional[int]:
        """Get the peak usage hour."""
        try:
            hourly_stats = await self.db_service.access_logs.get_hourly_statistics(start_date, end_date)
            if hourly_stats:
                return max(hourly_stats, key=lambda x: x['count'])['hour']
            return None
        except Exception:
            return None
    
    async def _get_peak_usage_day(self, start_date: datetime, end_date: datetime) -> Optional[str]:
        """Get the peak usage day."""
        try:
            daily_stats = await self.db_service.access_logs.get_daily_statistics(start_date, end_date)
            if daily_stats:
                peak_day = max(daily_stats, key=lambda x: x['count'])
                return peak_day['date'].strftime('%Y-%m-%d')
            return None
        except Exception:
            return None
    
    async def _get_top_users(self, start_date: datetime, end_date: datetime, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top users by access attempts."""
        try:
            return await self.db_service.access_logs.get_top_users(start_date, end_date, limit)
        except Exception:
            return []
    
    async def _get_top_locations(self, start_date: datetime, end_date: datetime, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top locations by access attempts."""
        try:
            return await self.db_service.access_logs.get_top_locations(start_date, end_date, limit)
        except Exception:
            return []
    
    def _prepare_timeline_chart_data(self, timeline_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare timeline data for charts."""
        return [
            {
                'date': item.get('date', ''),
                'successful': item.get('successful', 0),
                'failed': item.get('failed', 0),
                'total': item.get('total', 0)
            }
            for item in timeline_data
        ]
    
    def _prepare_pie_chart_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare data for pie charts."""
        return [
            {
                'label': item.get('name', item.get('label', 'Unknown')),
                'value': item.get('count', item.get('value', 0))
            }
            for item in data
        ]
    
    def _prepare_bar_chart_data(self, data: List[Dict[str, Any]], label_key: str, value_key: str) -> List[Dict[str, Any]]:
        """Prepare data for bar charts."""
        return [
            {
                'label': item.get(label_key, 'Unknown'),
                'value': item.get(value_key, 0)
            }
            for item in data
        ]
    
    # Format conversion methods
    
    async def _format_as_csv(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format report data as CSV."""
        # This would implement CSV formatting
        # For now, return a placeholder
        return {
            'format': 'csv',
            'content': 'CSV formatting not implemented yet',
            'filename': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    
    async def _format_as_html(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format report data as HTML."""
        # This would implement HTML formatting with templates
        # For now, return a placeholder
        return {
            'format': 'html',
            'content': '<html><body>HTML formatting not implemented yet</body></html>',
            'filename': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        }
    
    async def _format_as_pdf(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format report data as PDF."""
        # This would implement PDF generation
        # For now, return a placeholder
        return {
            'format': 'pdf',
            'content': 'PDF formatting not implemented yet',
            'filename': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        }
    
    # Additional helper methods would be implemented here...
    # (Due to length constraints, showing structure only)
    
    async def _analyze_security_incidents(self, security_logs: List[Dict], failed_attempts: List[Dict]) -> Dict[str, Any]:
        """Analyze security incidents for patterns and insights."""
        return {
            'high_risk_count': 0,
            'resolved_count': 0,
            'failed_attempts_analysis': {}
        }
    
    async def _generate_security_recommendations(self, incident_analysis: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on incident analysis."""
        return [
            "Review access control policies",
            "Consider additional security measures",
            "Monitor high-risk areas more closely"
        ]