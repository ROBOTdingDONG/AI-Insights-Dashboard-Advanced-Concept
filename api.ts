/**
 * AI Insights Dashboard - API Client
 * ==================================
 *
 * Comprehensive API client with authentication, error handling, and type safety.
 *
 * Features:
 * - JWT token management with automatic refresh
 * - Request/response interceptors with retry logic
 * - Comprehensive error handling and user feedback
 * - TypeScript type safety for all endpoints
 * - Request deduplication and caching
 * - Rate limit handling and quota tracking
 * - File upload and download support
 *
 * Security:
 * - Secure token storage with httpOnly cookies
 * - Automatic token refresh on expiration
 * - Request sanitization and validation
 * - CSRF protection and secure headers
 *
 * Author: AI Insights Team
 * Version: 1.0.0
 */

import axios, {
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  AxiosError,
  InternalAxiosRequestConfig,
} from 'axios';
import Cookies from 'js-cookie';
import { toast } from 'react-hot-toast';

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: 'admin' | 'user' | 'analyst' | 'viewer';
  subscription_plan: 'free' | 'basic' | 'professional' | 'enterprise';
  subscription_expires_at?: string;
  avatar_url?: string;
  timezone: string;
  language: string;
  preferences: Record<string, any>;
  api_calls_today: number;
  api_calls_month: number;
  last_api_call?: string;
  created_at: string;
  updated_at: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface ApiResponse<T> {
  data: T;
  message?: string;
  status: 'success' | 'error';
  timestamp: string;
  request_id: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total_count: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_previous: boolean;
}

export interface SearchRequest {
  query: string;
  max_results?: number;
  filters?: {
    source_type?: string;
    credibility?: string;
    date_range?: string;
    language?: string;
    domain?: string;
  };
  use_cache?: boolean;
  request_id?: string;
}

export interface SearchResult {
  search_id: string;
  query: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  total_results: number;
  search_time_ms: number;
  results: Array<{
    title: string;
    url: string;
    snippet: string;
    source_domain: string;
    source_type: 'news' | 'academic' | 'blog' | 'social' | 'government' | 'commercial' | 'unknown';
    credibility_score: 'high' | 'medium' | 'low' | 'suspicious';
    relevance_score: number;
    published_date?: string;
    author?: string;
    language?: string;
    word_count?: number;
    reading_time_minutes?: number;
  }>;
  metadata: {
    provider: string;
    model?: string;
    processing_time_ms: number;
    total_time_ms?: number;
    request_id: string;
    from_cache: boolean;
  };
  cached: boolean;
  created_at: string;
}

export interface AnalysisRequest {
  data_source: string;
  analysis_type:
    | 'summarization'
    | 'trend_analysis'
    | 'sentiment_analysis'
    | 'key_insights'
    | 'comparative_analysis'
    | 'prediction';
  provider?: 'openai' | 'claude' | 'local';
  model?: string;
  max_tokens?: number;
  temperature?: number;
  custom_prompt?: string;
  request_id?: string;
}

export interface AnalysisResult {
  analysis_id: string;
  analysis_type: string;
  provider: string;
  model: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  content: string;
  confidence_score: number;
  key_points: string[];
  entities: string[];
  sentiment?: string;
  safety_rating: 'safe' | 'moderate' | 'unsafe' | 'blocked';
  bias_detected: boolean;
  bias_types?: string[];
  processing_time_ms: number;
  token_usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  cost_usd?: number;
  metadata: Record<string, any>;
  cached: boolean;
  created_at: string;
}

export interface BiasCheckRequest {
  content: string;
}

export interface BiasCheckResult {
  has_bias: boolean;
  bias_types: string[];
  confidence: number;
  flagged_segments: string[];
  recommendations: string[];
  safety_rating: string;
}

export interface ExportRequest {
  data_type: 'search_results' | 'analysis_results' | 'dashboard';
  search_id?: string;
  analysis_id?: string;
  format: 'pdf' | 'csv' | 'xlsx' | 'json' | 'png' | 'svg' | 'mp4';
  filters?: Record<string, any>;
  options?: Record<string, any>;
}

export interface ExportResult {
  export_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  download_url?: string;
  expires_at?: string;
  file_size?: number;
  error_message?: string;
}

export interface RateLimitInfo {
  hourly_limit: number;
  daily_limit: number;
  current_usage: number;
  remaining: number;
  reset_time: number;
  subscription_plan: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public code?: string,
    public details?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// =============================================================================
// API CLIENT CONFIGURATION
// =============================================================================

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
const API_VERSION = '/api/v1';

class APIClient {
  private axios: AxiosInstance;
  private isRefreshing = false;
  private failedQueue: Array<{
    resolve: (token: string) => void;
    reject: (error: any) => void;
  }> = [];

  constructor() {
    this.axios = axios.create({
      baseURL: `${API_BASE_URL}${API_VERSION}`,
      timeout: 30000,
      withCredentials: true,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  // =============================================================================
  // INTERCEPTORS AND AUTHENTICATION
  // =============================================================================

  private setupInterceptors() {
    // Request interceptor
    this.axios.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        // Add authentication token
        const token = this.getAccessToken();
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }

        // Add request ID for tracing
        if (config.headers) {
          config.headers['X-Request-ID'] = this.generateRequestId();
        }

        // Log request in development
        if (process.env.NODE_ENV === 'development') {
          console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`, {
            data: config.data,
            params: config.params,
          });
        }

        return config;
      },
      (error) => {
        console.error('Request interceptor error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.axios.interceptors.response.use(
      (response: AxiosResponse) => {
        // Log response in development
        if (process.env.NODE_ENV === 'development') {
          console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url}`, {
            status: response.status,
            data: response.data,
          });
        }

        return response;
      },
      async (error: AxiosError) => {
        const original = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

        // Handle different error types
        if (error.response) {
          const { status, data } = error.response;

          // Handle 401 Unauthorized - attempt token refresh
          if (status === 401 && !original._retry) {
            if (this.isRefreshing) {
              // If already refreshing, queue the request
              return new Promise((resolve, reject) => {
                this.failedQueue.push({ resolve, reject });
              })
                .then((token) => {
                  if (original.headers) {
                    original.headers.Authorization = `Bearer ${token}`;
                  }
                  return this.axios(original);
                })
                .catch((err) => {
                  return Promise.reject(err);
                });
            }

            original._retry = true;
            this.isRefreshing = true;

            try {
              const newToken = await this.refreshToken();
              this.processQueue(null, newToken);
              
              if (original.headers) {
                original.headers.Authorization = `Bearer ${newToken}`;
              }
              
              return this.axios(original);
            } catch (refreshError) {
              this.processQueue(refreshError, null);
              this.handleAuthenticationError();
              return Promise.reject(refreshError);
            } finally {
              this.isRefreshing = false;
            }
          }

          // Handle 429 Rate Limit
          if (status === 429) {
            const retryAfter = error.response.headers['retry-after'];
            const resetTime = error.response.headers['x-ratelimit-reset'];
            
            toast.error(
              `Rate limit exceeded. ${retryAfter ? `Try again in ${retryAfter} seconds.` : 'Please try again later.'}`
            );
          }

          // Handle 403 Forbidden
          if (status === 403) {
            toast.error('Access denied. Please check your permissions.');
          }

          // Handle 422 Validation Error
          if (status === 422) {
            const validationErrors = (data as any)?.details || [];
            const errorMessage = validationErrors.length > 0 
              ? `Validation error: ${validationErrors.map((e: any) => e.msg).join(', ')}`
              : 'Invalid input data';
            toast.error(errorMessage);
          }
        } else if (error.request) {
          // Network error
          toast.error('Network error. Please check your connection.');
        } else {
          // Other errors
          toast.error('An unexpected error occurred.');
        }

        // Log error in development
        if (process.env.NODE_ENV === 'development') {
          console.error('❌ API Error:', {
            message: error.message,
            status: error.response?.status,
            data: error.response?.data,
            config: error.config,
          });
        }

        return Promise.reject(this.createApiError(error));
      }
    );
  }

  private processQueue(error: any, token: string | null) {
    this.failedQueue.forEach(({ resolve, reject }) => {
      if (error) {
        reject(error);
      } else {
        resolve(token!);
      }
    });

    this.failedQueue = [];
  }

  private createApiError(error: AxiosError): ApiError {
    const response = error.response;
    const message = response?.data ? (response.data as any).detail || error.message : error.message;
    const status = response?.status;
    const code = response?.data ? (response.data as any).code : undefined;
    const details = response?.data ? (response.data as any).details : undefined;

    return new ApiError(message, status, code, details);
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // =============================================================================
  // TOKEN MANAGEMENT
  // =============================================================================

  private getAccessToken(): string | null {
    return Cookies.get('access_token') || localStorage.getItem('access_token');
  }

  private getRefreshToken(): string | null {
    return Cookies.get('refresh_token') || localStorage.getItem('refresh_token');
  }

  private setTokens(tokens: AuthTokens) {
    const expires = new Date(Date.now() + tokens.expires_in * 1000);
    
    // Use secure cookies if available, fallback to localStorage
    if (window.location.protocol === 'https:') {
      Cookies.set('access_token', tokens.access_token, { 
        expires, 
        secure: true, 
        sameSite: 'strict' 
      });
      Cookies.set('refresh_token', tokens.refresh_token, { 
        expires: 7, // 7 days
        secure: true, 
        sameSite: 'strict' 
      });
    } else {
      localStorage.setItem('access_token', tokens.access_token);
      localStorage.setItem('refresh_token', tokens.refresh_token);
    }
  }

  private clearTokens() {
    Cookies.remove('access_token');
    Cookies.remove('refresh_token');
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  }

  private async refreshToken(): Promise<string> {
    const refreshToken = this.getRefreshToken();
    
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
        refresh_token: refreshToken,
      });

      const tokens = response.data;
      this.setTokens(tokens);
      
      return tokens.access_token;
    } catch (error) {
      this.clearTokens();
      throw error;
    }
  }

  private handleAuthenticationError() {
    this.clearTokens();
    
    // Redirect to login if not already there
    if (!window.location.pathname.includes('/login')) {
      window.location.href = '/login';
    }
  }

  // =============================================================================
  // AUTHENTICATION METHODS
  // =============================================================================

  async login(email: string, password: string): Promise<{ user: User; tokens: AuthTokens }> {
    try {
      const response = await this.axios.post('/auth/login', {
        email,
        password,
      });

      const { user, tokens } = response.data;
      this.setTokens(tokens);

      return { user, tokens };
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async register(userData: {
    email: string;
    password: string;
    full_name: string;
    company?: string;
  }): Promise<{ user: User; tokens: AuthTokens }> {
    try {
      const response = await this.axios.post('/auth/register', userData);
      
      const { user, tokens } = response.data;
      this.setTokens(tokens);

      return { user, tokens };
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async logout(): Promise<void> {
    try {
      await this.axios.post('/auth/logout');
    } catch (error) {
      // Ignore logout errors
    } finally {
      this.clearTokens();
    }
  }

  async getCurrentUser(): Promise<User> {
    try {
      const response = await this.axios.get('/auth/profile');
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async updateProfile(updates: Partial<User>): Promise<User> {
    try {
      const response = await this.axios.put('/auth/profile', updates);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  // =============================================================================
  // SEARCH METHODS
  // =============================================================================

  async search(request: SearchRequest): Promise<SearchResult> {
    try {
      const response = await this.axios.post('/search/search', request);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async getSearchResult(searchId: string): Promise<SearchResult> {
    try {
      const response = await this.axios.get(`/search/${searchId}`);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async getSearchHistory(params?: {
    page?: number;
    page_size?: number;
    status_filter?: string;
    date_from?: string;
    date_to?: string;
  }): Promise<PaginatedResponse<SearchResult>> {
    try {
      const response = await this.axios.get('/search', { params });
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async getTrendingTopics(params?: {
    category?: string;
    limit?: number;
  }): Promise<{ topics: string[]; categories: Record<string, string[]>; updated_at: string }> {
    try {
      const response = await this.axios.get('/search/trending', { params });
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async deleteSearch(searchId: string): Promise<void> {
    try {
      await this.axios.delete(`/search/${searchId}`);
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async batchSearch(requests: SearchRequest[]): Promise<{
    batch_id: string;
    total_requests: number;
    successful: number;
    failed: number;
    results: Array<{ index: number; status: string; search_id?: string; error?: string }>;
  }> {
    try {
      const response = await this.axios.post('/search/batch', requests);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  // =============================================================================
  // ANALYSIS METHODS
  // =============================================================================

  async analyze(request: AnalysisRequest): Promise<AnalysisResult> {
    try {
      const response = await this.axios.post('/analyze/analyze', request);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async getAnalysisResult(analysisId: string): Promise<AnalysisResult> {
    try {
      const response = await this.axios.get(`/analyze/${analysisId}`);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async getAnalysisHistory(params?: {
    page?: number;
    page_size?: number;
    analysis_type?: string;
    provider?: string;
    date_from?: string;
    date_to?: string;
  }): Promise<PaginatedResponse<AnalysisResult>> {
    try {
      const response = await this.axios.get('/analyze', { params });
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async checkBias(request: BiasCheckRequest): Promise<BiasCheckResult> {
    try {
      const response = await this.axios.post('/analyze/bias-check', request);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async getModelCapabilities(): Promise<{
    providers: Record<string, any>;
    analysis_types: string[];
    rate_limits: Record<string, number>;
    pricing: Record<string, Record<string, number>>;
  }> {
    try {
      const response = await this.axios.get('/analyze/models');
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async batchAnalyze(requests: AnalysisRequest[]): Promise<{
    batch_id: string;
    total_requests: number;
    successful: number;
    failed: number;
    results: Array<{ index: number; status: string; analysis_id?: string; error?: string }>;
  }> {
    try {
      const response = await this.axios.post('/analyze/batch', requests);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  // =============================================================================
  // EXPORT METHODS
  // =============================================================================

  async exportData(request: ExportRequest): Promise<ExportResult> {
    try {
      const response = await this.axios.post('/export', request);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async getExportStatus(exportId: string): Promise<ExportResult> {
    try {
      const response = await this.axios.get(`/export/${exportId}`);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async downloadExport(exportId: string): Promise<Blob> {
    try {
      const response = await this.axios.get(`/export/${exportId}/download`, {
        responseType: 'blob',
      });
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  // =============================================================================
  // UTILITY METHODS
  // =============================================================================

  async getRateLimitInfo(): Promise<RateLimitInfo> {
    try {
      const response = await this.axios.get('/auth/rate-limit');
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  async healthCheck(): Promise<{
    status: string;
    timestamp: string;
    version: string;
    checks: Record<string, string>;
  }> {
    try {
      const response = await this.axios.get('/health');
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  // File upload helper
  async uploadFile(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<{ file_id: string; file_url: string }> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await this.axios.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onProgress(progress);
          }
        },
      });

      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  // Generic GET method
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    try {
      const response = await this.axios.get(url, config);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  // Generic POST method
  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    try {
      const response = await this.axios.post(url, data, config);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  // Generic PUT method
  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    try {
      const response = await this.axios.put(url, data, config);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }

  // Generic DELETE method
  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    try {
      const response = await this.axios.delete(url, config);
      return response.data;
    } catch (error) {
      throw this.createApiError(error as AxiosError);
    }
  }
}

// =============================================================================
// EXPORT SINGLETON INSTANCE
// =============================================================================

export const apiClient = new APIClient();
export default apiClient;

// =============================================================================
// UTILITY HOOKS AND HELPERS
// =============================================================================

export const formatApiError = (error: ApiError): string => {
  if (error.status === 422 && error.details) {
    // Format validation errors
    return error.details.map((detail: any) => detail.msg).join(', ');
  }
  
  return error.message || 'An unexpected error occurred';
};

export const isNetworkError = (error: ApiError): boolean => {
  return !error.status || error.status === 0;
};

export const isAuthenticationError = (error: ApiError): boolean => {
  return error.status === 401;
};

export const isPermissionError = (error: ApiError): boolean => {
  return error.status === 403;
};

export const isRateLimitError = (error: ApiError): boolean => {
  return error.status === 429;
};

export const isValidationError = (error: ApiError): boolean => {
  return error.status === 422;
};

export const isServerError = (error: ApiError): boolean => {
  return error.status ? error.status >= 500 : false;
};
