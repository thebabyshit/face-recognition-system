import { useState } from 'react';
import { useQuery } from 'react-query';
import { 
  PlusIcon, 
  MagnifyingGlassIcon,
  FunnelIcon,
  UserIcon,
  PencilIcon,
  TrashIcon,
  EyeIcon
} from '@heroicons/react/24/outline';
import { withAuth } from '@/lib/auth';
import Layout from '@/components/Layout';
import { personApi } from '@/lib/api';
import { Person, PaginatedResponse } from '@/types';
import { toast } from 'react-hot-toast';

function PersonsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10);
  const [selectedPersons, setSelectedPersons] = useState<number[]>([]);

  const {
    data: personsData,
    isLoading,
    error,
    refetch,
  } = useQuery<PaginatedResponse<Person>>(
    ['persons', currentPage, pageSize, searchQuery],
    () => personApi.getPersons({
      page: currentPage,
      page_size: pageSize,
      search: searchQuery || undefined,
    }).then(res => res.data),
    {
      keepPreviousData: true,
    }
  );

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setCurrentPage(1);
  };

  const handleSelectPerson = (personId: number) => {
    setSelectedPersons(prev => 
      prev.includes(personId) 
        ? prev.filter(id => id !== personId)
        : [...prev, personId]
    );
  };

  const handleSelectAll = () => {
    if (!personsData?.items) return;
    
    const allIds = personsData.items.map(person => person.id);
    setSelectedPersons(
      selectedPersons.length === allIds.length ? [] : allIds
    );
  };

  const handleDeleteSelected = async () => {
    if (selectedPersons.length === 0) return;
    
    if (!confirm(`Are you sure you want to delete ${selectedPersons.length} person(s)?`)) {
      return;
    }

    try {
      await Promise.all(
        selectedPersons.map(id => personApi.deletePerson(id))
      );
      
      toast.success(`Successfully deleted ${selectedPersons.length} person(s)`);
      setSelectedPersons([]);
      refetch();
    } catch (error) {
      toast.error('Failed to delete persons');
    }
  };

  const handleDeletePerson = async (personId: number) => {
    if (!confirm('Are you sure you want to delete this person?')) {
      return;
    }

    try {
      await personApi.deletePerson(personId);
      toast.success('Person deleted successfully');
      refetch();
    } catch (error) {
      toast.error('Failed to delete person');
    }
  };

  if (error) {
    return (
      <Layout>
        <div className="text-center py-12">
          <UserIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">Error loading persons</h3>
          <p className="mt-1 text-sm text-gray-500">
            Unable to load person data. Please try again.
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

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="md:flex md:items-center md:justify-between">
          <div className="flex-1 min-w-0">
            <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
              Person Management
            </h2>
            <p className="mt-1 text-sm text-gray-500">
              Manage registered persons and their face data
            </p>
          </div>
          <div className="mt-4 flex md:mt-0 md:ml-4">
            <button
              type="button"
              className="btn-primary"
              onClick={() => {
                // TODO: Open create person modal
                toast.info('Create person feature coming soon');
              }}
            >
              <PlusIcon className="h-4 w-4 mr-2" />
              Add Person
            </button>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="card">
          <div className="card-body">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    type="text"
                    className="form-input pl-10"
                    placeholder="Search persons by name, employee ID, or department..."
                    value={searchQuery}
                    onChange={(e) => handleSearch(e.target.value)}
                  />
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  className="btn-outline"
                  onClick={() => {
                    // TODO: Open filters modal
                    toast.info('Filters feature coming soon');
                  }}
                >
                  <FunnelIcon className="h-4 w-4 mr-2" />
                  Filters
                </button>
                {selectedPersons.length > 0 && (
                  <button
                    type="button"
                    className="btn-danger"
                    onClick={handleDeleteSelected}
                  >
                    <TrashIcon className="h-4 w-4 mr-2" />
                    Delete ({selectedPersons.length})
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Persons Table */}
        <div className="card">
          <div className="card-body p-0">
            {isLoading ? (
              <div className="flex items-center justify-center h-64">
                <div className="spinner-lg"></div>
              </div>
            ) : personsData?.items && personsData.items.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="table">
                  <thead className="table-header">
                    <tr>
                      <th className="table-header-cell">
                        <input
                          type="checkbox"
                          className="form-checkbox"
                          checked={selectedPersons.length === personsData.items.length}
                          onChange={handleSelectAll}
                        />
                      </th>
                      <th className="table-header-cell">Name</th>
                      <th className="table-header-cell">Employee ID</th>
                      <th className="table-header-cell">Department</th>
                      <th className="table-header-cell">Access Level</th>
                      <th className="table-header-cell">Face Images</th>
                      <th className="table-header-cell">Status</th>
                      <th className="table-header-cell">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="table-body">
                    {personsData.items.map((person) => (
                      <tr key={person.id} className="table-row">
                        <td className="table-cell">
                          <input
                            type="checkbox"
                            className="form-checkbox"
                            checked={selectedPersons.includes(person.id)}
                            onChange={() => handleSelectPerson(person.id)}
                          />
                        </td>
                        <td className="table-cell">
                          <div className="flex items-center">
                            <div className="flex-shrink-0 h-10 w-10">
                              <div className="h-10 w-10 rounded-full bg-gray-300 flex items-center justify-center">
                                <UserIcon className="h-6 w-6 text-gray-600" />
                              </div>
                            </div>
                            <div className="ml-4">
                              <div className="text-sm font-medium text-gray-900">
                                {person.name}
                              </div>
                              {person.email && (
                                <div className="text-sm text-gray-500">
                                  {person.email}
                                </div>
                              )}
                            </div>
                          </div>
                        </td>
                        <td className="table-cell">
                          <div className="text-sm text-gray-900">
                            {person.employee_id || '-'}
                          </div>
                        </td>
                        <td className="table-cell">
                          <div className="text-sm text-gray-900">
                            {person.department || '-'}
                          </div>
                          {person.position && (
                            <div className="text-sm text-gray-500">
                              {person.position}
                            </div>
                          )}
                        </td>
                        <td className="table-cell">
                          <span className="badge badge-primary">
                            Level {person.access_level}
                          </span>
                        </td>
                        <td className="table-cell">
                          <div className="text-sm text-gray-900">
                            {person.face_count} image{person.face_count !== 1 ? 's' : ''}
                          </div>
                        </td>
                        <td className="table-cell">
                          <span className={`badge ${
                            person.is_active ? 'badge-success' : 'badge-gray'
                          }`}>
                            {person.is_active ? 'Active' : 'Inactive'}
                          </span>
                        </td>
                        <td className="table-cell">
                          <div className="flex items-center space-x-2">
                            <button
                              type="button"
                              className="text-primary-600 hover:text-primary-900"
                              onClick={() => {
                                // TODO: Navigate to person detail
                                toast.info('Person detail feature coming soon');
                              }}
                            >
                              <EyeIcon className="h-4 w-4" />
                            </button>
                            <button
                              type="button"
                              className="text-warning-600 hover:text-warning-900"
                              onClick={() => {
                                // TODO: Open edit person modal
                                toast.info('Edit person feature coming soon');
                              }}
                            >
                              <PencilIcon className="h-4 w-4" />
                            </button>
                            <button
                              type="button"
                              className="text-danger-600 hover:text-danger-900"
                              onClick={() => handleDeletePerson(person.id)}
                            >
                              <TrashIcon className="h-4 w-4" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-12">
                <UserIcon className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900">No persons found</h3>
                <p className="mt-1 text-sm text-gray-500">
                  {searchQuery 
                    ? 'No persons match your search criteria.'
                    : 'Get started by adding a new person.'
                  }
                </p>
                <div className="mt-6">
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={() => {
                      // TODO: Open create person modal
                      toast.info('Create person feature coming soon');
                    }}
                  >
                    <PlusIcon className="h-4 w-4 mr-2" />
                    Add Person
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Pagination */}
        {personsData && personsData.total_pages > 1 && (
          <div className="flex items-center justify-between">
            <div className="flex-1 flex justify-between sm:hidden">
              <button
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="btn-outline btn-disabled:opacity-50"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, personsData.total_pages))}
                disabled={currentPage === personsData.total_pages}
                className="btn-outline btn-disabled:opacity-50"
              >
                Next
              </button>
            </div>
            <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-gray-700">
                  Showing{' '}
                  <span className="font-medium">
                    {(currentPage - 1) * pageSize + 1}
                  </span>{' '}
                  to{' '}
                  <span className="font-medium">
                    {Math.min(currentPage * pageSize, personsData.total)}
                  </span>{' '}
                  of{' '}
                  <span className="font-medium">{personsData.total}</span>{' '}
                  results
                </p>
              </div>
              <div>
                <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px">
                  <button
                    onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                    disabled={currentPage === 1}
                    className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
                  >
                    Previous
                  </button>
                  {/* Page numbers */}
                  {Array.from({ length: Math.min(5, personsData.total_pages) }, (_, i) => {
                    const pageNum = i + 1;
                    return (
                      <button
                        key={pageNum}
                        onClick={() => setCurrentPage(pageNum)}
                        className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                          currentPage === pageNum
                            ? 'z-10 bg-primary-50 border-primary-500 text-primary-600'
                            : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                        }`}
                      >
                        {pageNum}
                      </button>
                    );
                  })}
                  <button
                    onClick={() => setCurrentPage(prev => Math.min(prev + 1, personsData.total_pages))}
                    disabled={currentPage === personsData.total_pages}
                    className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
                  >
                    Next
                  </button>
                </nav>
              </div>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}

export default withAuth(PersonsPage, ['person:read']);