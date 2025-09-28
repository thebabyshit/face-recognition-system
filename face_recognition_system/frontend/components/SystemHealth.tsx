import { 
  CpuChipIcon, 
  CircleStackIcon, 
  ServerIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { SystemHealth as SystemHealthType } from '@/types';

interface SystemHealthProps {
  data?: SystemHealthType;
  loading?: boolean;
}

interface HealthMetricProps {
  label: string;
  value: number;
  unit: string;
  icon: React.ComponentType<any>;
  threshold?: { warning: number; critical: number };
}

function HealthMetric({ label, value, unit, icon: Icon, threshold }: HealthMetricProps) {
  const getStatusColor = () => {
    if (!threshold) return 'text-gray-500';
    
    if (value >= threshold.critical) return 'text-danger-500';
    if (value >= threshold.warning) return 'text-warning-500';
    return 'text-success-500';
  };

  const getProgressColor = () => {
    if (!threshold) return 'bg-primary-500';
    
    if (value >= threshold.critical) return 'bg-danger-500';
    if (value >= threshold.warning) return 'bg-warning-500';
    return 'bg-success-500';
  };

  return (
    <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
      <div className="flex items-center">
        <Icon className={clsx('h-6 w-6 mr-3', getStatusColor())} />
        <div>
          <p className="text-sm font-medium text-gray-900">{label}</p>
          <p className="text-xs text-gray-500">
            {value.toFixed(1)}{unit}
          </p>
        </div>
      </div>
      <div className="flex items-center">
        <div className="w-20 bg-gray-200 rounded-full h-2 mr-3">
          <div
            className={clsx('h-2 rounded-full transition-all duration-300', getProgressColor())}
            style={{ width: `${Math.min(value, 100)}%` }}
          ></div>
        </div>
        <span className={clsx('text-sm font-semibold', getStatusColor())}>
          {value.toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

interface ServiceStatusProps {
  label: string;
  status: string;
}

function ServiceStatus({ label, status }: ServiceStatusProps) {
  const getStatusIcon = () => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon className="h-5 w-5 text-success-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-5 w-5 text-warning-500" />;
      case 'error':
        return <XCircleIcon className="h-5 w-5 text-danger-500" />;
      default:
        return <ExclamationTriangleIcon className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'healthy':
        return 'Healthy';
      case 'warning':
        return 'Warning';
      case 'error':
        return 'Error';
      default:
        return 'Unknown';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'healthy':
        return 'text-success-700';
      case 'warning':
        return 'text-warning-700';
      case 'error':
        return 'text-danger-700';
      default:
        return 'text-gray-700';
    }
  };

  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-gray-600">{label}</span>
      <div className="flex items-center">
        {getStatusIcon()}
        <span className={clsx('ml-2 text-sm font-medium', getStatusColor())}>
          {getStatusText()}
        </span>
      </div>
    </div>
  );
}

export default function SystemHealth({ data, loading = false }: SystemHealthProps) {
  if (loading) {
    return (
      <div className="card animate-pulse">
        <div className="card-header">
          <div className="h-6 bg-gray-300 rounded w-1/3"></div>
        </div>
        <div className="card-body space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 bg-gray-300 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-medium text-gray-900">System Health</h3>
        </div>
        <div className="card-body">
          <div className="text-center py-6">
            <ServerIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No data available</h3>
            <p className="mt-1 text-sm text-gray-500">
              Unable to load system health data.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const overallHealthColor = () => {
    if (data.overall_health_score >= 80) return 'text-success-600';
    if (data.overall_health_score >= 60) return 'text-warning-600';
    return 'text-danger-600';
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900">System Health</h3>
          <div className="flex items-center">
            <span className="text-sm text-gray-500 mr-2">Overall Score:</span>
            <span className={clsx('text-lg font-bold', overallHealthColor())}>
              {data.overall_health_score.toFixed(0)}%
            </span>
          </div>
        </div>
      </div>
      <div className="card-body space-y-4">
        {/* Resource Usage */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-900">Resource Usage</h4>
          <HealthMetric
            label="CPU Usage"
            value={data.cpu_usage}
            unit="%"
            icon={CpuChipIcon}
            threshold={{ warning: 70, critical: 90 }}
          />
          <HealthMetric
            label="Memory Usage"
            value={data.memory_usage}
            unit="%"
            icon={CircleStackIcon}
            threshold={{ warning: 80, critical: 95 }}
          />
          <HealthMetric
            label="Disk Usage"
            value={data.disk_usage}
            unit="%"
            icon={ServerIcon}
            threshold={{ warning: 85, critical: 95 }}
          />
        </div>

        {/* Service Status */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium text-gray-900 mb-3">Service Status</h4>
          <div className="space-y-1">
            <ServiceStatus label="Database" status={data.database_status} />
            <ServiceStatus label="Camera Service" status={data.camera_status} />
            <ServiceStatus label="Recognition Service" status={data.recognition_service_status} />
          </div>
        </div>
      </div>
    </div>
  );
}