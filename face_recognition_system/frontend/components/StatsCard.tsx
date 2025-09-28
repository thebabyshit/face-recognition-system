import { ArrowUpIcon, ArrowDownIcon } from '@heroicons/react/24/solid';
import clsx from 'clsx';

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: React.ComponentType<any>;
  color: 'primary' | 'success' | 'warning' | 'danger';
  change?: number;
  changeType?: 'increase' | 'decrease';
  loading?: boolean;
}

export default function StatsCard({
  title,
  value,
  icon: Icon,
  color,
  change,
  changeType,
  loading = false,
}: StatsCardProps) {
  const colorClasses = {
    primary: 'bg-primary-500',
    success: 'bg-success-500',
    warning: 'bg-warning-500',
    danger: 'bg-danger-500',
  };

  const changeColorClasses = {
    increase: 'text-success-600',
    decrease: 'text-danger-600',
  };

  if (loading) {
    return (
      <div className="card animate-pulse">
        <div className="card-body">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="h-8 w-8 bg-gray-300 rounded-md"></div>
            </div>
            <div className="ml-5 w-0 flex-1">
              <div className="h-4 bg-gray-300 rounded w-3/4 mb-2"></div>
              <div className="h-6 bg-gray-300 rounded w-1/2"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card hover:shadow-md transition-shadow duration-200">
      <div className="card-body">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <div className={clsx('p-2 rounded-md', colorClasses[color])}>
              <Icon className="h-6 w-6 text-white" aria-hidden="true" />
            </div>
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">{title}</dt>
              <dd className="flex items-baseline">
                <div className="text-2xl font-semibold text-gray-900">
                  {typeof value === 'number' ? value.toLocaleString() : value}
                </div>
                {change !== undefined && changeType && (
                  <div className={clsx('ml-2 flex items-baseline text-sm font-semibold', changeColorClasses[changeType])}>
                    {changeType === 'increase' ? (
                      <ArrowUpIcon className="self-center flex-shrink-0 h-4 w-4" aria-hidden="true" />
                    ) : (
                      <ArrowDownIcon className="self-center flex-shrink-0 h-4 w-4" aria-hidden="true" />
                    )}
                    <span className="sr-only">
                      {changeType === 'increase' ? 'Increased' : 'Decreased'} by
                    </span>
                    {Math.abs(change)}%
                  </div>
                )}
              </dd>
            </dl>
          </div>
        </div>
      </div>
    </div>
  );
}