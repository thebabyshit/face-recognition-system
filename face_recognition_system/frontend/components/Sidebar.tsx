import { useRouter } from 'next/router';
import Link from 'next/link';
import clsx from 'clsx';
import { NavItem } from '@/types';

interface SidebarProps {
  navigation: NavItem[];
}

export default function Sidebar({ navigation }: SidebarProps) {
  const router = useRouter();

  return (
    <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-white px-6 pb-4 shadow-lg">
      <div className="flex h-16 shrink-0 items-center">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <div className="h-8 w-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <svg
                className="h-5 w-5 text-white"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"
                />
              </svg>
            </div>
          </div>
          <div className="ml-3">
            <h1 className="text-lg font-semibold text-gray-900">
              Face Recognition
            </h1>
            <p className="text-xs text-gray-500">Admin Panel</p>
          </div>
        </div>
      </div>
      
      <nav className="flex flex-1 flex-col">
        <ul role="list" className="flex flex-1 flex-col gap-y-7">
          <li>
            <ul role="list" className="-mx-2 space-y-1">
              {navigation.map((item) => {
                const isActive = router.pathname === item.href || 
                  (item.href !== '/dashboard' && router.pathname.startsWith(item.href));
                
                return (
                  <li key={item.name}>
                    <Link
                      href={item.href}
                      className={clsx(
                        isActive
                          ? 'bg-primary-50 text-primary-700'
                          : 'text-gray-700 hover:text-primary-700 hover:bg-gray-50',
                        'group flex gap-x-3 rounded-md p-2 text-sm leading-6 font-semibold transition-colors duration-200'
                      )}
                    >
                      <item.icon
                        className={clsx(
                          isActive ? 'text-primary-700' : 'text-gray-400 group-hover:text-primary-700',
                          'h-6 w-6 shrink-0'
                        )}
                        aria-hidden="true"
                      />
                      {item.name}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </li>
          
          {/* System Status */}
          <li className="mt-auto">
            <div className="rounded-lg bg-gray-50 p-4">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="h-2 w-2 bg-success-400 rounded-full"></div>
                </div>
                <div className="ml-3 flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900">System Status</p>
                  <p className="text-xs text-gray-500">All systems operational</p>
                </div>
              </div>
            </div>
          </li>
        </ul>
      </nav>
    </div>
  );
}