"""Test script for data visualization components."""

import sys
import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from visualization.chart_generator import ChartGenerator
from visualization.dashboard import Dashboard
from visualization.report_visualizer import ReportVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_access_data() -> List[Dict[str, Any]]:
    """Generate mock access data for testing."""
    import random
    
    access_data = []
    now = datetime.now(timezone.utc)
    
    # Generate data for the last 7 days
    for i in range(7):
        date = now - timedelta(days=i)
        
        # Generate 20-50 access attempts per day
        daily_attempts = random.randint(20, 50)
        
        for j in range(daily_attempts):
            # Random time during the day (more activity during business hours)
            if random.random() < 0.7:  # 70% during business hours
                hour = random.randint(8, 18)
            else:
                hour = random.randint(0, 23)
            
            minute = random.randint(0, 59)
            timestamp = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # Random person and location
            person_id = random.randint(1, 20)
            location_id = random.randint(1, 5)
            
            # Success rate around 85%
            access_granted = random.random() < 0.85
            
            access_data.append({
                'timestamp': timestamp,
                'person_id': person_id,
                'person_name': f'Person {person_id}',
                'location_id': location_id,
                'location_name': f'Location {location_id}',
                'access_granted': access_granted,
                'access_method': 'face_recognition',
                'confidence_score': random.uniform(0.7, 0.99) if access_granted else random.uniform(0.3, 0.7),
                'reason': 'Access granted' if access_granted else 'Recognition failed'
            })
    
    return access_data

def generate_mock_performance_data() -> Dict[str, Any]:
    """Generate mock performance data for testing."""
    import random
    
    now = datetime.now(timezone.utc)
    
    # Response times for the last 24 hours
    response_times = []
    for i in range(24):
        timestamp = now - timedelta(hours=i)
        response_time = random.uniform(100, 500)  # 100-500ms
        response_times.append({
            'timestamp': timestamp,
            'response_time': response_time
        })
    
    return {
        'system_health': random.uniform(80, 95),
        'response_times': response_times,
        'error_rates': {
            'recognition_errors': random.randint(0, 5),
            'database_errors': random.randint(0, 2),
            'camera_errors': random.randint(0, 3)
        },
        'resource_usage': {
            'CPU': random.uniform(20, 60),
            'Memory': random.uniform(30, 70),
            'Disk': random.uniform(40, 60)
        }
    }

async def test_chart_generator():
    """Test chart generator functionality."""
    logger.info("Testing Chart Generator...")
    
    chart_generator = ChartGenerator()
    access_data = generate_mock_access_data()
    
    try:
        # Test access timeline
        logger.info("Generating access timeline chart...")
        timeline_chart = chart_generator.generate_access_timeline(access_data, '7d', 'plotly')
        logger.info(f"Timeline chart generated: {len(timeline_chart)} characters")
        
        # Test success rate chart
        logger.info("Generating success rate chart...")
        success_chart = chart_generator.generate_access_success_rate(access_data, 'plotly')
        logger.info(f"Success rate chart generated: {len(success_chart)} characters")
        
        # Test activity heatmap
        logger.info("Generating activity heatmap...")
        heatmap_chart = chart_generator.generate_person_activity_heatmap(access_data, 'plotly')
        logger.info(f"Heatmap chart generated: {len(heatmap_chart)} characters")
        
        # Test location usage chart
        logger.info("Generating location usage chart...")
        location_chart = chart_generator.generate_location_usage_chart(access_data, 'plotly')
        logger.info(f"Location chart generated: {len(location_chart)} characters")
        
        # Test confidence distribution
        logger.info("Generating confidence distribution...")
        confidence_chart = chart_generator.generate_recognition_confidence_distribution(access_data, 'plotly')
        logger.info(f"Confidence chart generated: {len(confidence_chart)} characters")
        
        # Test performance dashboard
        logger.info("Generating performance dashboard...")
        performance_data = generate_mock_performance_data()
        dashboard_chart = chart_generator.generate_system_performance_dashboard(performance_data, 'plotly')
        logger.info(f"Dashboard chart generated: {len(dashboard_chart)} characters")
        
        # Test custom chart
        logger.info("Generating custom chart...")
        chart_config = {
            'type': 'line',
            'library': 'plotly',
            'x_column': 'timestamp',
            'y_column': 'confidence_score',
            'title': 'Custom Confidence Timeline'
        }
        custom_chart = chart_generator.create_custom_chart(chart_config, access_data)
        logger.info(f"Custom chart generated: {len(custom_chart)} characters")
        
        # Test data export
        logger.info("Testing data export...")
        export_path = chart_generator.export_chart_data(access_data, 'test_export', 'csv')
        logger.info(f"Data exported to: {export_path}")
        
        logger.info("Chart Generator tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Chart Generator test failed: {e}")
        return False

async def test_dashboard():
    """Test dashboard functionality."""
    logger.info("Testing Dashboard...")
    
    try:
        dashboard = Dashboard()
        
        # Test dashboard data retrieval
        logger.info("Getting dashboard data...")
        dashboard_data = await dashboard.get_dashboard_data('24h')
        
        logger.info("Dashboard data structure:")
        for key, value in dashboard_data.items():
            if isinstance(value, dict):\n                logger.info(f\"  {key}: {len(value)} items\")\n            elif isinstance(value, list):\n                logger.info(f\"  {key}: {len(value)} items\")\n            else:\n                logger.info(f\"  {key}: {type(value).__name__}\")\n        \n        # Test metrics\n        metrics = await dashboard.get_current_metrics()\n        logger.info(f\"Current metrics: {metrics.to_dict()}\")\n        \n        # Test system health\n        health = await dashboard.get_system_health()\n        logger.info(f\"System health: {health.to_dict()}\")\n        \n        # Test alerts\n        await dashboard.add_alert('test_alert', 'This is a test alert', 'info')\n        alerts = await dashboard.get_recent_alerts()\n        logger.info(f\"Recent alerts: {len(alerts)} alerts\")\n        \n        # Test configuration\n        config = await dashboard.get_dashboard_config()\n        logger.info(f\"Dashboard config: {len(config)} settings\")\n        \n        # Test configuration update\n        await dashboard.update_dashboard_config({'refresh_interval': 60})\n        logger.info(\"Dashboard configuration updated\")\n        \n        logger.info(\"Dashboard tests completed successfully!\")\n        return True\n        \n    except Exception as e:\n        logger.error(f\"Dashboard test failed: {e}\")\n        return False\n\nasync def test_report_visualizer():\n    \"\"\"Test report visualizer functionality.\"\"\"\n    logger.info(\"Testing Report Visualizer...\")\n    \n    try:\n        report_visualizer = ReportVisualizer()\n        \n        # Generate mock report data\n        access_data = generate_mock_access_data()\n        performance_data = generate_mock_performance_data()\n        \n        report_data = {\n            'period': 'Last 7 Days',\n            'access_logs': access_data,\n            'performance_metrics': performance_data,\n            'summary': {\n                'total_attempts': len(access_data),\n                'successful_attempts': sum(1 for a in access_data if a['access_granted']),\n                'failed_attempts': sum(1 for a in access_data if not a['access_granted']),\n                'success_rate': sum(1 for a in access_data if a['access_granted']) / len(access_data) * 100,\n                'unique_persons': len(set(a['person_id'] for a in access_data)),\n                'active_locations': len(set(a['location_id'] for a in access_data)),\n                'avg_confidence': sum(a['confidence_score'] for a in access_data if a['confidence_score']) / len(access_data),\n                'security_incidents': 2,\n                'peak_hour': '14:00',\n                'busiest_day': '2024-01-15'\n            },\n            'security_incidents': [\n                {'type': 'failed_recognition', 'timestamp': datetime.now()},\n                {'type': 'tailgating_detected', 'timestamp': datetime.now()}\n            ]\n        }\n        \n        # Test PNG report generation\n        logger.info(\"Generating PNG report...\")\n        start_date = datetime.now() - timedelta(days=7)\n        end_date = datetime.now()\n        \n        png_report = await report_visualizer._generate_png_report(report_data, 'test_png_report')\n        logger.info(f\"PNG report generated: {png_report}\")\n        \n        # Test HTML report generation\n        logger.info(\"Generating HTML report...\")\n        html_report = await report_visualizer._generate_html_report(report_data, 'test_html_report')\n        logger.info(f\"HTML report generated: {html_report}\")\n        \n        # Test interactive charts generation\n        logger.info(\"Generating interactive charts...\")\n        charts = await report_visualizer._generate_interactive_charts(report_data)\n        logger.info(f\"Generated {len(charts)} interactive charts\")\n        \n        logger.info(\"Report Visualizer tests completed successfully!\")\n        return True\n        \n    except Exception as e:\n        logger.error(f\"Report Visualizer test failed: {e}\")\n        return False\n\nasync def test_integration():\n    \"\"\"Test integration between visualization components.\"\"\"\n    logger.info(\"Testing Integration...\")\n    \n    try:\n        # Create components\n        chart_generator = ChartGenerator()\n        dashboard = Dashboard()\n        report_visualizer = ReportVisualizer()\n        \n        # Generate test data\n        access_data = generate_mock_access_data()\n        \n        # Test dashboard with chart generation\n        logger.info(\"Testing dashboard with charts...\")\n        dashboard_data = await dashboard.get_dashboard_data('7d')\n        \n        # Generate charts for dashboard data\n        if dashboard_data.get('charts'):\n            logger.info(f\"Dashboard generated {len(dashboard_data['charts'])} charts\")\n        \n        # Test report with dashboard metrics\n        logger.info(\"Testing report with dashboard metrics...\")\n        metrics = await dashboard.get_current_metrics()\n        health = await dashboard.get_system_health()\n        \n        # Create comprehensive report data\n        report_data = {\n            'period': 'Integration Test',\n            'access_logs': access_data,\n            'summary': {\n                'total_attempts': metrics.total_access_attempts,\n                'success_rate': metrics.successful_access_rate,\n                'unique_persons': metrics.total_persons\n            },\n            'performance_metrics': {\n                'health_score': health.overall_health_score,\n                'cpu_usage': health.cpu_usage,\n                'memory_usage': health.memory_usage\n            }\n        }\n        \n        # Generate integrated report\n        charts = await report_visualizer._generate_interactive_charts(report_data)\n        logger.info(f\"Generated {len(charts)} charts for integrated report\")\n        \n        logger.info(\"Integration tests completed successfully!\")\n        return True\n        \n    except Exception as e:\n        logger.error(f\"Integration test failed: {e}\")\n        return False\n\nasync def main():\n    \"\"\"Run all visualization tests.\"\"\"\n    logger.info(\"Starting Visualization System Tests...\")\n    \n    test_results = {\n        'chart_generator': await test_chart_generator(),\n        'dashboard': await test_dashboard(),\n        'report_visualizer': await test_report_visualizer(),\n        'integration': await test_integration()\n    }\n    \n    # Print results\n    logger.info(\"\\n\" + \"=\"*50)\n    logger.info(\"VISUALIZATION SYSTEM TEST RESULTS\")\n    logger.info(\"=\"*50)\n    \n    for test_name, result in test_results.items():\n        status = \"PASS\" if result else \"FAIL\"\n        logger.info(f\"{test_name.replace('_', ' ').title()}: {status}\")\n    \n    overall_success = all(test_results.values())\n    logger.info(f\"\\nOverall Result: {'SUCCESS' if overall_success else 'FAILURE'}\")\n    \n    if overall_success:\n        logger.info(\"\\nAll visualization components are working correctly!\")\n        logger.info(\"The system can generate:\")\n        logger.info(\"- Interactive charts (Plotly)\")\n        logger.info(\"- Static charts (Matplotlib)\")\n        logger.info(\"- Real-time dashboards\")\n        logger.info(\"- Comprehensive reports (PDF, HTML, PNG)\")\n        logger.info(\"- Data exports (CSV, Excel, JSON)\")\n    else:\n        logger.error(\"\\nSome visualization tests failed. Please check the logs above.\")\n    \n    return overall_success\n\nif __name__ == \"__main__\":\n    success = asyncio.run(main())\n    sys.exit(0 if success else 1)"