// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

// User and Authentication Types
export interface User {
  id: number;
  username: string;
  email?: string;
  roles: string[];
  permissions: string[];
  is_active: boolean;
  created_at: string;
  last_login?: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
  remember_me?: boolean;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

// Person Management Types
export interface Person {
  id: number;
  name: string;
  employee_id?: string;
  department?: string;
  position?: string;
  email?: string;
  phone?: string;
  access_level: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  face_count: number;
}

export interface PersonCreateRequest {
  name: string;
  employee_id?: string;
  department?: string;
  position?: string;
  email?: string;
  phone?: string;
  access_level: number;
  is_active?: boolean;
}

export interface PersonUpdateRequest extends Partial<PersonCreateRequest> {
  id: number;
}

// Face Management Types
export interface Face {
  id: number;
  person_id: number;
  image_path: string;
  feature_vector?: number[];
  quality_score: number;
  is_primary: boolean;
  created_at: string;
}

export interface FaceUploadRequest {
  person_id: number;
  image: File;
  is_primary?: boolean;
}

// Location Types
export interface Location {
  id: number;
  name: string;
  description?: string;
  required_access_level: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

// Access Log Types
export interface AccessLog {
  id: number;
  person_id?: number;
  person_name?: string;
  location_id: number;
  location_name: string;
  access_granted: boolean;
  access_method: string;
  confidence_score?: number;
  reason: string;
  timestamp: string;
  additional_data?: Record<string, any>;
}

// Recognition Types
export interface RecognitionResult {
  person_id?: number;
  person_name?: string;
  confidence_score: number;
  access_granted: boolean;
  reason: string;
  timestamp: string;
}

// Dashboard Types
export interface DashboardMetrics {
  total_persons: number;
  total_access_attempts: number;
  successful_access_rate: number;
  failed_access_count: number;
  active_sessions: number;
  system_uptime: string;
  last_updated: string;
}

export interface SystemHealth {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  database_status: string;
  camera_status: string;
  recognition_service_status: string;
  overall_health_score: number;
}

export interface DashboardData {
  metrics: DashboardMetrics;
  system_health: SystemHealth;
  charts: Record<string, string>;
  alerts: SecurityAlert[];
  statistics: Record<string, any>;
  time_range: string;
  last_updated: string;
}

// Security Types
export interface SecurityAlert {
  id: string | number;
  type: string;
  message: string;
  severity: 'info' | 'warning' | 'critical';
  timestamp: string;
  additional_data?: Record<string, any>;
  acknowledged?: boolean;
}

export interface SecurityEvent {
  event_type: string;
  ip_address: string;
  user_id?: number;
  timestamp: string;
  details: Record<string, any>;
  severity: string;
}

// Report Types
export interface ReportRequest {
  report_type: 'daily' | 'weekly' | 'monthly' | 'custom';
  start_date: string;
  end_date: string;
  format: 'pdf' | 'html' | 'csv' | 'excel';
  include_charts?: boolean;
}

export interface ReportResponse {
  report_id: string;
  file_path: string;
  download_url: string;
  generated_at: string;
  expires_at: string;
}

// System Configuration Types
export interface SystemConfig {
  recognition_threshold: number;
  max_face_images_per_person: number;
  session_timeout_minutes: number;
  enable_real_time_monitoring: boolean;
  enable_access_logging: boolean;
  backup_retention_days: number;
  [key: string]: any;
}

// Pagination Types
export interface PaginationParams {
  page?: number;
  page_size?: number;
  search?: string;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

// Form Types
export interface FormField {
  name: string;
  label: string;
  type: 'text' | 'email' | 'password' | 'number' | 'select' | 'checkbox' | 'file' | 'textarea';
  required?: boolean;
  placeholder?: string;
  options?: { value: string | number; label: string }[];
  validation?: {
    min?: number;
    max?: number;
    pattern?: string;
    message?: string;
  };
}

// UI Component Types
export interface TableColumn<T = any> {
  key: keyof T | string;
  label: string;
  sortable?: boolean;
  render?: (value: any, item: T) => React.ReactNode;
  width?: string;
  align?: 'left' | 'center' | 'right';
}

export interface ActionButton {
  label: string;
  icon?: React.ComponentType<any>;
  onClick: () => void;
  variant?: 'primary' | 'secondary' | 'danger' | 'success' | 'warning';
  disabled?: boolean;
  loading?: boolean;
}

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  showCloseButton?: boolean;
}

// Chart Types
export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string | string[];
    borderWidth?: number;
  }[];
}

export interface ChartOptions {
  responsive?: boolean;
  maintainAspectRatio?: boolean;
  plugins?: {
    legend?: {
      display?: boolean;
      position?: 'top' | 'bottom' | 'left' | 'right';
    };
    title?: {
      display?: boolean;
      text?: string;
    };
  };
  scales?: {
    x?: {
      display?: boolean;
      title?: {
        display?: boolean;
        text?: string;
      };
    };
    y?: {
      display?: boolean;
      title?: {
        display?: boolean;
        text?: string;
      };
    };
  };
}

// Navigation Types
export interface NavItem {
  name: string;
  href: string;
  icon?: React.ComponentType<any>;
  current?: boolean;
  children?: NavItem[];
  permission?: string;
}

// Notification Types
export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

// File Upload Types
export interface FileUploadProps {
  accept?: string;
  multiple?: boolean;
  maxSize?: number;
  onUpload: (files: File[]) => void;
  loading?: boolean;
  error?: string;
}

// Search and Filter Types
export interface SearchFilters {
  search?: string;
  department?: string;
  access_level?: number;
  is_active?: boolean;
  date_from?: string;
  date_to?: string;
  [key: string]: any;
}

// Export Types
export interface ExportOptions {
  format: 'csv' | 'excel' | 'pdf';
  filename?: string;
  columns?: string[];
  filters?: SearchFilters;
}

// WebSocket Types
export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface RealTimeEvent {
  event_type: 'recognition' | 'access_granted' | 'access_denied' | 'system_alert';
  data: any;
  timestamp: string;
}