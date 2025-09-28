"""Chart generation utilities for data visualization."""

import logging
import io
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generate various types of charts for system data visualization."""
    
    def __init__(self):
        """Initialize chart generator."""
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure plotly
        pio.templates.default = "plotly_white"
        
        self.colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c', 
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'info': '#17a2b8',
            'secondary': '#6c757d'
        }
        
        logger.info("Chart generator initialized")
    
    def generate_access_timeline(
        self, 
        access_data: List[Dict[str, Any]], 
        time_range: str = '24h',
        chart_type: str = 'plotly'
    ) -> str:
        """
        Generate access timeline chart.
        
        Args:
            access_data: List of access records
            time_range: Time range for the chart ('24h', '7d', '30d')
            chart_type: Chart library to use ('matplotlib', 'plotly')
            
        Returns:
            Base64 encoded chart image or HTML
        """
        try:
            if not access_data:
                return self._generate_empty_chart("No access data available")
            
            # Convert to DataFrame
            df = pd.DataFrame(access_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by time range
            now = datetime.now(timezone.utc)
            if time_range == '24h':
                start_time = now - timedelta(hours=24)
                freq = 'H'
                title = "Access Activity - Last 24 Hours"
            elif time_range == '7d':
                start_time = now - timedelta(days=7)
                freq = 'D'
                title = "Access Activity - Last 7 Days"
            elif time_range == '30d':
                start_time = now - timedelta(days=30)
                freq = 'D'
                title = "Access Activity - Last 30 Days"
            else:
                start_time = now - timedelta(hours=24)
                freq = 'H'
                title = "Access Activity"
            
            df_filtered = df[df['timestamp'] >= start_time]
            
            if chart_type == 'plotly':
                return self._generate_plotly_timeline(df_filtered, title, freq)
            else:
                return self._generate_matplotlib_timeline(df_filtered, title, freq)
                
        except Exception as e:
            logger.error(f"Error generating access timeline: {e}")
            return self._generate_error_chart(f"Error: {str(e)}")
    
    def generate_access_success_rate(
        self, 
        access_data: List[Dict[str, Any]],
        chart_type: str = 'plotly'
    ) -> str:
        """Generate access success rate pie chart."""
        try:
            if not access_data:
                return self._generate_empty_chart("No access data available")
            
            df = pd.DataFrame(access_data)
            success_counts = df['access_granted'].value_counts()
            
            if chart_type == 'plotly':
                fig = go.Figure(data=[go.Pie(
                    labels=['Success', 'Failed'],
                    values=[success_counts.get(True, 0), success_counts.get(False, 0)],
                    hole=0.3,
                    marker_colors=[self.colors['success'], self.colors['danger']]
                )])
                
                fig.update_layout(
                    title="Access Success Rate",
                    font=dict(size=14),
                    showlegend=True
                )
                
                return fig.to_html(include_plotlyjs='cdn')
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                labels = ['Success', 'Failed']
                sizes = [success_counts.get(True, 0), success_counts.get(False, 0)]
                colors = [self.colors['success'], self.colors['danger']]
                
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title("Access Success Rate")
                
                return self._fig_to_base64(fig)
                
        except Exception as e:
            logger.error(f"Error generating success rate chart: {e}")
            return self._generate_error_chart(f"Error: {str(e)}")
    
    def generate_person_activity_heatmap(
        self, 
        access_data: List[Dict[str, Any]],
        chart_type: str = 'plotly'
    ) -> str:
        """Generate person activity heatmap by hour and day."""
        try:
            if not access_data:
                return self._generate_empty_chart("No access data available")
            
            df = pd.DataFrame(access_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            
            # Create pivot table
            heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(day_order)
            
            if chart_type == 'plotly':
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=list(range(24)),
                    y=day_order,
                    colorscale='Blues',
                    showscale=True
                ))
                
                fig.update_layout(
                    title="Access Activity Heatmap",
                    xaxis_title="Hour of Day",
                    yaxis_title="Day of Week",
                    font=dict(size=12)
                )
                
                return fig.to_html(include_plotlyjs='cdn')
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("Access Activity Heatmap")
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel("Day of Week")
                
                return self._fig_to_base64(fig)
                
        except Exception as e:
            logger.error(f"Error generating activity heatmap: {e}")
            return self._generate_error_chart(f"Error: {str(e)}")
    
    def generate_location_usage_chart(
        self, 
        access_data: List[Dict[str, Any]],
        chart_type: str = 'plotly'
    ) -> str:
        """Generate location usage bar chart."""
        try:
            if not access_data:
                return self._generate_empty_chart("No access data available")
            
            df = pd.DataFrame(access_data)
            location_counts = df['location_name'].value_counts().head(10)
            
            if chart_type == 'plotly':
                fig = go.Figure(data=[go.Bar(
                    x=location_counts.index,
                    y=location_counts.values,
                    marker_color=self.colors['primary']
                )])
                
                fig.update_layout(
                    title="Top 10 Most Accessed Locations",
                    xaxis_title="Location",
                    yaxis_title="Access Count",
                    font=dict(size=12)
                )
                
                return fig.to_html(include_plotlyjs='cdn')
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                location_counts.plot(kind='bar', ax=ax, color=self.colors['primary'])
                ax.set_title("Top 10 Most Accessed Locations")
                ax.set_xlabel("Location")
                ax.set_ylabel("Access Count")
                ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                return self._fig_to_base64(fig)
                
        except Exception as e:
            logger.error(f"Error generating location usage chart: {e}")
            return self._generate_error_chart(f"Error: {str(e)}")
    
    def generate_recognition_confidence_distribution(
        self, 
        recognition_data: List[Dict[str, Any]],
        chart_type: str = 'plotly'
    ) -> str:
        """Generate recognition confidence score distribution."""
        try:
            if not recognition_data:
                return self._generate_empty_chart("No recognition data available")
            
            df = pd.DataFrame(recognition_data)
            confidence_scores = df['confidence_score'].dropna()
            
            if chart_type == 'plotly':
                fig = go.Figure(data=[go.Histogram(
                    x=confidence_scores,
                    nbinsx=20,
                    marker_color=self.colors['info']
                )])
                
                fig.update_layout(
                    title="Recognition Confidence Score Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Frequency",
                    font=dict(size=12)
                )
                
                return fig.to_html(include_plotlyjs='cdn')
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.hist(confidence_scores, bins=20, color=self.colors['info'], alpha=0.7)
                ax.set_title("Recognition Confidence Score Distribution")
                ax.set_xlabel("Confidence Score")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)
                
                return self._fig_to_base64(fig)
                
        except Exception as e:
            logger.error(f"Error generating confidence distribution: {e}")
            return self._generate_error_chart(f"Error: {str(e)}")
    
    def generate_system_performance_dashboard(
        self, 
        performance_data: Dict[str, Any],
        chart_type: str = 'plotly'
    ) -> str:
        """Generate comprehensive system performance dashboard."""
        try:
            if chart_type == 'plotly':
                return self._generate_plotly_dashboard(performance_data)
            else:
                return self._generate_matplotlib_dashboard(performance_data)
                
        except Exception as e:
            logger.error(f"Error generating performance dashboard: {e}")
            return self._generate_error_chart(f"Error: {str(e)}")
    
    def _generate_plotly_timeline(self, df: pd.DataFrame, title: str, freq: str) -> str:
        """Generate timeline chart using Plotly."""
        # Resample data by frequency
        df_resampled = df.set_index('timestamp').resample(freq).size()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_resampled.index,
            y=df_resampled.values,
            mode='lines+markers',
            name='Access Count',
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Access Count",
            font=dict(size=12),
            hovermode='x unified'
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def _generate_matplotlib_timeline(self, df: pd.DataFrame, title: str, freq: str) -> str:
        """Generate timeline chart using Matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Resample data by frequency
        df_resampled = df.set_index('timestamp').resample(freq).size()
        
        ax.plot(df_resampled.index, df_resampled.values, 
                color=self.colors['primary'], linewidth=2, marker='o')
        
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Access Count")
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if freq == 'H':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _generate_plotly_dashboard(self, performance_data: Dict[str, Any]) -> str:
        """Generate comprehensive dashboard using Plotly."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('System Status', 'Response Times', 'Error Rates', 'Resource Usage'),
            specs=[[{"type": "indicator"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # System status indicator
        status_value = performance_data.get('system_health', 95)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=status_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "System Health (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': self.colors['success']},
                   'steps': [{'range': [0, 50], 'color': self.colors['danger']},
                            {'range': [50, 80], 'color': self.colors['warning']},
                            {'range': [80, 100], 'color': self.colors['success']}]}
        ), row=1, col=1)
        
        # Response times
        response_times = performance_data.get('response_times', [])
        if response_times:
            timestamps = [rt['timestamp'] for rt in response_times]
            times = [rt['response_time'] for rt in response_times]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=times,
                mode='lines',
                name='Response Time',
                line=dict(color=self.colors['info'])
            ), row=1, col=2)
        
        # Error rates
        error_data = performance_data.get('error_rates', {})
        if error_data:
            fig.add_trace(go.Bar(
                x=list(error_data.keys()),
                y=list(error_data.values()),
                name='Error Rate',
                marker_color=self.colors['warning']
            ), row=2, col=1)
        
        # Resource usage
        resource_usage = performance_data.get('resource_usage', {})
        if resource_usage:
            fig.add_trace(go.Pie(
                labels=list(resource_usage.keys()),
                values=list(resource_usage.values()),
                name="Resource Usage"
            ), row=2, col=2)
        
        fig.update_layout(
            title="System Performance Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def _generate_matplotlib_dashboard(self, performance_data: Dict[str, Any]) -> str:
        """Generate comprehensive dashboard using Matplotlib."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # System health gauge (simplified as bar)
        health = performance_data.get('system_health', 95)
        ax1.barh(['System Health'], [health], color=self.colors['success'])
        ax1.set_xlim(0, 100)
        ax1.set_title('System Health (%)')
        
        # Response times
        response_times = performance_data.get('response_times', [])
        if response_times:
            timestamps = [rt['timestamp'] for rt in response_times]
            times = [rt['response_time'] for rt in response_times]
            ax2.plot(timestamps, times, color=self.colors['info'])
            ax2.set_title('Response Times')
            ax2.set_ylabel('Time (ms)')
        
        # Error rates
        error_data = performance_data.get('error_rates', {})
        if error_data:
            ax3.bar(error_data.keys(), error_data.values(), color=self.colors['warning'])
            ax3.set_title('Error Rates')
            ax3.set_ylabel('Error Count')
        
        # Resource usage
        resource_usage = performance_data.get('resource_usage', {})
        if resource_usage:
            ax4.pie(resource_usage.values(), labels=resource_usage.keys(), autopct='%1.1f%%')
            ax4.set_title('Resource Usage')
        
        plt.suptitle('System Performance Dashboard')
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig: Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_empty_chart(self, message: str) -> str:
        """Generate empty chart with message."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return self._fig_to_base64(fig)
    
    def _generate_error_chart(self, error_message: str) -> str:
        """Generate error chart."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Chart Generation Error:\n{error_message}", 
                ha='center', va='center', fontsize=14, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return self._fig_to_base64(fig)
    
    def export_chart_data(
        self, 
        data: List[Dict[str, Any]], 
        filename: str, 
        format: str = 'csv'
    ) -> str:
        """
        Export chart data to file.
        
        Args:
            data: Chart data
            filename: Output filename
            format: Export format ('csv', 'excel', 'json')
            
        Returns:
            File path of exported data
        """
        try:
            df = pd.DataFrame(data)
            
            if format == 'csv':
                filepath = f"{filename}.csv"
                df.to_csv(filepath, index=False)
            elif format == 'excel':
                filepath = f"{filename}.xlsx"
                df.to_excel(filepath, index=False)
            elif format == 'json':
                filepath = f"{filename}.json"
                df.to_json(filepath, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Chart data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting chart data: {e}")
            raise
    
    def create_custom_chart(
        self, 
        chart_config: Dict[str, Any], 
        data: List[Dict[str, Any]]
    ) -> str:
        """
        Create custom chart based on configuration.
        
        Args:
            chart_config: Chart configuration
            data: Chart data
            
        Returns:
            Generated chart (base64 or HTML)
        """
        try:
            chart_type = chart_config.get('type', 'line')
            library = chart_config.get('library', 'plotly')
            
            df = pd.DataFrame(data)
            
            if library == 'plotly':
                return self._create_plotly_custom_chart(chart_config, df)
            else:
                return self._create_matplotlib_custom_chart(chart_config, df)
                
        except Exception as e:
            logger.error(f"Error creating custom chart: {e}")
            return self._generate_error_chart(f"Custom chart error: {str(e)}")
    
    def _create_plotly_custom_chart(self, config: Dict[str, Any], df: pd.DataFrame) -> str:
        """Create custom Plotly chart."""
        chart_type = config.get('type', 'line')
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        title = config.get('title', 'Custom Chart')
        
        if chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=title)
        elif chart_type == 'bar':
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        elif chart_type == 'pie':
            fig = px.pie(df, names=x_col, values=y_col, title=title)
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title)
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_matplotlib_custom_chart(self, config: Dict[str, Any], df: pd.DataFrame) -> str:
        """Create custom Matplotlib chart."""
        chart_type = config.get('type', 'line')
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        title = config.get('title', 'Custom Chart')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == 'line':
            ax.plot(df[x_col], df[y_col])
        elif chart_type == 'bar':
            ax.bar(df[x_col], df[y_col])
        elif chart_type == 'scatter':
            ax.scatter(df[x_col], df[y_col])
        elif chart_type == 'pie':
            ax.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%')
        
        ax.set_title(title)
        if chart_type != 'pie':
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)