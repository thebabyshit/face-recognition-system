import { useQuery } from 'react-query';
import { 
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  UserIcon,
  MapPinIcon
} from '@heroicons/react/24/outline';
import { accessApi } from '@/lib/api';
import { AccessLog } from '@/types';
import clsx from 'clsx';

export default function RecentActivity() {
  const {
    data: accessLogs,
    isLoading,
    error,
  } = useQuery<AccessLog[]>(
    'recent-activity',
    () => accessApi.getAccessLogs({ limit: 10, sort_by: 'timestamp', sort_order: 'desc' }).then(res => res.data?.items || []),
    {
      refetchInterval: 30000, // Refetch every 30 seconds
    }
  );

  if (isLoading) {
    return (
      <div className="card animate-pulse">
        <div className="card-header">
          <div className="h-6 bg-gray-300 rounded w-1/3"></div>
        </div>
        <div className="card-body space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="flex items-center space-x-3">
              <div className="h-8 w-8 bg-gray-300 rounded-full"></div>
              <div className="flex-1 space-y-1">
                <div className="h-4 bg-gray-300 rounded w-3/4"></div>
                <div className="h-3 bg-gray-300 rounded w-1/2"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !accessLogs) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-medium text-gray-900">Recent Activity</h3>
        </div>
        <div className="card-body">
          <div className="text-center py-6">
            <ClockIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No activity data</h3>
            <p className="mt-1 text-sm text-gray-500">
              Unable to load recent activity.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-medium text-gray-900">Recent Activity</h3>
      </div>
      <div className="card-body">
        {accessLogs.length > 0 ? (
          <div className="flow-root">
            <ul className="-mb-8">
              {accessLogs.map((log, index) => (
                <li key={log.id}>
                  <div className="relative pb-8">
                    {index !== accessLogs.length - 1 ? (
                      <span
                        className="absolute top-4 left-4 -ml-px h-full w-0.5 bg-gray-200"
                        aria-hidden="true"
                      />
                    ) : null}
                    <div className="relative flex space-x-3">
                      <div>
                        <span
                          className={clsx(
                            log.access_granted
                              ? 'bg-success-500'
                              : 'bg-danger-500',
                            'h-8 w-8 rounded-full flex items-center justify-center ring-8 ring-white'
                          )}
                        >
                          {log.access_granted ? (
                            <CheckCircleIcon className="h-5 w-5 text-white" aria-hidden="true" />
                          ) : (
                            <XCircleIcon className="h-5 w-5 text-white" aria-hidden="true" />
                          )}
                        </span>
                      </div>
                      <div className="flex min-w-0 flex-1 justify-between space-x-4 pt-1.5">
                        <div>
                          <p className="text-sm text-gray-500">
                            <span className="font-medium text-gray-900">
                              {log.person_name || 'Unknown Person'}
                            </span>{' '}
                            {log.access_granted ? 'accessed' : 'was denied access to'}{' '}
                            <span className="font-medium text-gray-900">
                              {log.location_name}
                            </span>
                          </p>
                          <div className="mt-1 flex items-center space-x-2 text-xs text-gray-500">
                            <UserIcon className="h-3 w-3" />
                            <span>{log.access_method}</span>
                            {log.confidence_score && (
                              <>
                                <span>•</span>
                                <span>{(log.confidence_score * 100).toFixed(1)}% confidence</span>
                              </>
                            )}
                          </div>
                          {!log.access_granted && log.reason && (
                            <p className="mt-1 text-xs text-danger-600">
                              Reason: {log.reason}
                            </p>
                          )}
                        </div>
                        <div className="whitespace-nowrap text-right text-sm text-gray-500">
                          <time dateTime={log.timestamp}>
                            {new Date(log.timestamp).toLocaleTimeString()}
                          </time>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="text-center py-6">
            <ClockIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No recent activity</h3>
            <p className="mt-1 text-sm text-gray-500">
              No access attempts have been recorded recently.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}