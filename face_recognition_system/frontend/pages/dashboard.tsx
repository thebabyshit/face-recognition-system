import { useState, useEffect } from 'react';
import { useQuery } from 'react-query';
import {
  UsersIcon,
  KeyIcon,
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  ServerIcon,
} from '@heroicons/react/24/outline';
import { withAuth } from '@/lib/auth';
import Layout from '@/components/Layout';
import StatsCard from '@/components/StatsCard';
import Chart from '@/components/Chart';
import RecentActivity from '@/components/RecentActivity';
import SystemHealth from '@/components/SystemHealth';
import { dashboardApi } from '@/lib/api';
import { DashboardData } from '@/types';

function DashboardPage() {
  const [timeRange, setTimeRange] = useState('24h');

  const {
    data: dashboardData,
    isLoading,
    error,
    refetch,
  } = useQuery<DashboardData>(
    ['dashboard', timeRange],
    () => dashboardApi.getDashboardData(timeRange).then(res => res.data),
    {
      refetchInterval: 30000, // Refetch every 30 seconds
      retry: 2,
    }
  );

  const handleTimeRangeChange = (newTimeRange: string) => {
    setTimeRange(newTimeRange);
  };

  if (isLoading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="spinner-lg"></div>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <div className="text-center py-12">
          <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">Error loading dashboard</h3>
          <p className="mt-1 text-sm text-gray-500">
            Unable to load dashboard data. Please try again.
          </p>
          <div className="mt-6">
            <button
              type="button"
              onClick={() => refetch()}
              className="btn-primary"
            >
              Retry
            </button>
          </div>
        </div>
      </Layout>
    );
  }

  const metrics = dashboardData?.metrics;
  const systemHealth = dashboardData?.system_health;
  const charts = dashboardData?.charts;
  const alerts = dashboardData?.alerts || [];

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="md:flex md:items-center md:justify-between">
          <div className="flex-1 min-w-0">
            <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
              Dashboard
            </h2>
            <p className="mt-1 text-sm text-gray-500">
              System overview and real-time monitoring
            </p>
          </div>
          <div className="mt-4 flex md:mt-0 md:ml-4">
            <select
              value={timeRange}
              onChange={(e) => handleTimeRangeChange(e.target.value)}
              className="form-select"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
          <StatsCard
            title="Total Persons"
            value={metrics?.total_persons || 0}
            icon={UsersIcon}
            color="primary"
            change={+5.2}
            changeType="increase"
          />
          <StatsCard
            title="Access Attempts"
            value={metrics?.total_access_attempts || 0}
            icon={KeyIcon}
            color="success"
            change={+12.3}
            changeType="increase"
          />
          <StatsCard
            title="Success Rate"
            value={`${metrics?.successful_access_rate?.toFixed(1) || 0}%`}
            icon={ShieldCheckIcon}
            color="warning"
            change={-2.1}
            changeType="decrease"
          />
          <StatsCard
            title="Failed Attempts"
            value={metrics?.failed_access_count || 0}
            icon={ExclamationTriangleIcon}
            color="danger"
            change={+8.7}
            changeType="increase"
          />
        </div>

        {/* System Health */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <SystemHealth data={systemHealth} />
          </div>
          <div>
            <div className="card">
              <div className="card-header">
                <h3 className="text-lg font-medium text-gray-900">System Info</h3>
              </div>
              <div className="card-body space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">Uptime</span>
                  <span className="text-sm font-medium text-gray-900">
                    {metrics?.system_uptime || 'Unknown'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">Active Sessions</span>
                  <span className="text-sm font-medium text-gray-900">
                    {metrics?.active_sessions || 0}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">Last Updated</span>
                  <span className="text-sm font-medium text-gray-900">
                    {metrics?.last_updated ? new Date(metrics.last_updated).toLocaleTimeString() : 'Unknown'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Access Timeline */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Access Activity</h3>
            </div>
            <div className="card-body">
              {charts?.timeline ? (
                <div dangerouslySetInnerHTML={{ __html: charts.timeline }} />
              ) : (
                <div className="flex items-center justify-center h-64 text-gray-500">
                  No data available
                </div>
              )}
            </div>
          </div>

          {/* Success Rate */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Success Rate</h3>
            </div>
            <div className="card-body">
              {charts?.success_rate ? (
                <div dangerouslySetInnerHTML={{ __html: charts.success_rate }} />
              ) : (
                <div className="flex items-center justify-center h-64 text-gray-500">
                  No data available
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Activity Heatmap */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Activity Heatmap</h3>
          </div>
          <div className="card-body">
            {charts?.activity_heatmap ? (
              <div dangerouslySetInnerHTML={{ __html: charts.activity_heatmap }} />
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500">
                No data available
              </div>
            )}
          </div>
        </div>

        {/* Recent Activity and Alerts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <RecentActivity />
          
          {/* Security Alerts */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Security Alerts</h3>
            </div>
            <div className="card-body">
              {alerts.length > 0 ? (
                <div className="space-y-3">
                  {alerts.slice(0, 5).map((alert, index) => (
                    <div
                      key={alert.id || index}
                      className={`p-3 rounded-md border-l-4 ${
                        alert.severity === 'critical'
                          ? 'bg-danger-50 border-danger-400'
                          : alert.severity === 'warning'
                          ? 'bg-warning-50 border-warning-400'
                          : 'bg-primary-50 border-primary-400'
                      }`}
                    >
                      <div className="flex items-start">
                        <div className="flex-shrink-0">
                          <ExclamationTriangleIcon
                            className={`h-5 w-5 ${
                              alert.severity === 'critical'
                                ? 'text-danger-400'
                                : alert.severity === 'warning'
                                ? 'text-warning-400'
                                : 'text-primary-400'
                            }`}
                          />
                        </div>
                        <div className="ml-3 flex-1">
                          <p className="text-sm font-medium text-gray-900">
                            {alert.message}
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            {new Date(alert.timestamp).toLocaleString()}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-6">
                  <ShieldCheckIcon className="mx-auto h-12 w-12 text-gray-400" />
                  <h3 className="mt-2 text-sm font-medium text-gray-900">No alerts</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    All systems are operating normally.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default withAuth(DashboardPage);