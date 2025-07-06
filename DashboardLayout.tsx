/**
 * AI Insights Dashboard - Main Layout Component
 * ============================================
 *
 * Modern, responsive dashboard layout with comprehensive search and analysis capabilities.
 * 
 * Features:
 * - Real-time search with Perplexity API integration
 * - Multi-provider LLM analysis (OpenAI, Claude)
 * - Interactive data visualizations with Recharts
 * - Mobile-first responsive design
 * - Advanced filtering and export capabilities
 * - Real-time performance monitoring
 * 
 * Security:
 * - JWT authentication integration
 * - Input validation and sanitization
 * - Rate limit awareness and quota display
 * - Error boundary protection
 * 
 * Author: AI Insights Team
 * Version: 1.0.0
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'react-hot-toast';
import {
  Search,
  Brain,
  BarChart3,
  Download,
  Settings,
  User,
  Bell,
  Loader2,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Globe,
  Shield,
  Zap,
  RefreshCw,
  Filter,
  X,
  ChevronDown,
  ExternalLink,
  Star,
  Clock,
  Database,
  Activity,
  PieChart,
  LineChart,
  FileText,
  Share,
  Bookmark,
  Eye,
  Calendar,
  MapPin,
  Tag,
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart as RechartsPieChart,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { format, parseISO, subDays, subHours } from 'date-fns';
import clsx from 'clsx';

import { apiClient, ApiError } from '../../services/api';
import { useAuth } from '../../hooks/useAuth';
import { useDebounce } from '../../hooks/useDebounce';
import { ErrorBoundary } from '../ErrorBoundary';
import { LoadingSkeleton } from '../LoadingSkeleton';

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

interface SearchResult {
  search_id: string;
  query: string;
  status: string;
  total_results: number;
  search_time_ms: number;
  results: Array<{
    title: string;
    url: string;
    snippet: string;
    source_domain: string;
    source_type: string;
    credibility_score: string;
    relevance_score: number;
    published_date?: string;
  }>;
  metadata: {
    provider: string;
    processing_time_ms: number;
    from_cache: boolean;
  };
  cached: boolean;
}

interface AnalysisResult {
  analysis_id: string;
  analysis_type: string;
  provider: string;
  model: string;
  status: string;
  content: string;
  confidence_score: number;
  key_points: string[];
  entities: string[];
  sentiment?: string;
  safety_rating: string;
  bias_detected: boolean;
  processing_time_ms: number;
  token_usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  cost_usd?: number;
  metadata: Record<string, any>;
}

interface UserStats {
  searches_today: number;
  analyses_today: number;
  quota_remaining: number;
  subscription_plan: string;
  total_searches: number;
  total_analyses: number;
  avg_response_time: number;
}

// =============================================================================
// VALIDATION SCHEMAS
// =============================================================================

const searchSchema = z.object({
  query: z
    .string()
    .min(2, 'Query must be at least 2 characters')
    .max(500, 'Query must be less than 500 characters')
    .refine((val) => val.trim().length > 0, 'Query cannot be empty'),
  max_results: z.number().min(1).max(50).default(10),
  filters: z
    .object({
      source_type: z.string().optional(),
      credibility: z.string().optional(),
      date_range: z.string().optional(),
    })
    .optional(),
  use_cache: z.boolean().default(true),
});

const analysisSchema = z.object({
  data_source: z.string().min(1, 'Data source is required'),
  analysis_type: z.enum([
    'summarization',
    'trend_analysis',
    'sentiment_analysis',
    'key_insights',
    'comparative_analysis',
    'prediction',
  ]),
  provider: z.enum(['openai', 'claude']).default('openai'),
  model: z.string().optional(),
  max_tokens: z.number().min(100).max(4000).default(1000),
  temperature: z.number().min(0).max(1).default(0.3),
  custom_prompt: z.string().max(2000).optional(),
});

type SearchFormData = z.infer<typeof searchSchema>;
type AnalysisFormData = z.infer<typeof analysisSchema>;

// =============================================================================
// MAIN DASHBOARD COMPONENT
// =============================================================================

export const DashboardLayout: React.FC = () => {
  const { user, logout } = useAuth();
  const queryClient = useQueryClient();

  // State Management
  const [activeTab, setActiveTab] = useState<'search' | 'analysis' | 'insights'>('search');
  const [selectedSearchId, setSelectedSearchId] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [realtimeMode, setRealtimeMode] = useState(false);

  // Search Form
  const searchForm = useForm<SearchFormData>({
    resolver: zodResolver(searchSchema),
    defaultValues: {
      query: '',
      max_results: 10,
      use_cache: true,
    },
  });

  // Analysis Form
  const analysisForm = useForm<AnalysisFormData>({
    resolver: zodResolver(analysisSchema),
    defaultValues: {
      analysis_type: 'summarization',
      provider: 'openai',
      max_tokens: 1000,
      temperature: 0.3,
    },
  });

  const debouncedQuery = useDebounce(searchForm.watch('query'), 500);

  // =============================================================================
  // API QUERIES AND MUTATIONS
  // =============================================================================

  // User Statistics
  const { data: userStats, isLoading: statsLoading } = useQuery<UserStats>({
    queryKey: ['user-stats'],
    queryFn: () => apiClient.get('/auth/profile').then((res) => res.data),
    refetchInterval: realtimeMode ? 10000 : 60000,
  });

  // Search History
  const { data: searchHistory, isLoading: historyLoading } = useQuery({
    queryKey: ['search-history'],
    queryFn: () =>
      apiClient
        .get('/search', { params: { page: 1, page_size: 10 } })
        .then((res) => res.data.searches),
    refetchInterval: realtimeMode ? 5000 : 30000,
  });

  // Analysis History
  const { data: analysisHistory, isLoading: analysisHistoryLoading } = useQuery({
    queryKey: ['analysis-history'],
    queryFn: () =>
      apiClient
        .get('/analyze', { params: { page: 1, page_size: 10 } })
        .then((res) => res.data.analyses),
    refetchInterval: realtimeMode ? 5000 : 30000,
  });

  // Search Mutation
  const searchMutation = useMutation({
    mutationFn: (data: SearchFormData) => apiClient.post('/search/search', data),
    onSuccess: (response) => {
      toast.success('Search completed successfully!');
      setSelectedSearchId(response.data.search_id);
      queryClient.invalidateQueries({ queryKey: ['search-history'] });
      queryClient.invalidateQueries({ queryKey: ['user-stats'] });
    },
    onError: (error: ApiError) => {
      toast.error(error.message || 'Search failed. Please try again.');
    },
  });

  // Analysis Mutation
  const analysisMutation = useMutation({
    mutationFn: (data: AnalysisFormData) => apiClient.post('/analyze/analyze', data),
    onSuccess: () => {
      toast.success('Analysis completed successfully!');
      queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
      queryClient.invalidateQueries({ queryKey: ['user-stats'] });
    },
    onError: (error: ApiError) => {
      toast.error(error.message || 'Analysis failed. Please try again.');
    },
  });

  // Search Results Query
  const { data: searchResults, isLoading: resultsLoading } = useQuery({
    queryKey: ['search-results', selectedSearchId],
    queryFn: () =>
      selectedSearchId
        ? apiClient.get(`/search/${selectedSearchId}`).then((res) => res.data)
        : null,
    enabled: !!selectedSearchId,
  });

  // =============================================================================
  // EVENT HANDLERS
  // =============================================================================

  const handleSearch = useCallback(
    (data: SearchFormData) => {
      if (!data.query.trim()) {
        toast.error('Please enter a search query');
        return;
      }
      searchMutation.mutate(data);
    },
    [searchMutation]
  );

  const handleAnalysis = useCallback(
    (data: AnalysisFormData) => {
      if (selectedSearchId) {
        analysisMutation.mutate({
          ...data,
          data_source: selectedSearchId,
        });
      } else {
        analysisMutation.mutate(data);
      }
    },
    [analysisMutation, selectedSearchId]
  );

  const handleExportResults = useCallback(async () => {
    if (!searchResults) return;

    try {
      const response = await apiClient.post('/export', {
        data_type: 'search_results',
        search_id: selectedSearchId,
        format: 'csv',
      });

      toast.success('Export initiated! Check your downloads.');
    } catch (error) {
      toast.error('Export failed. Please try again.');
    }
  }, [searchResults, selectedSearchId]);

  // =============================================================================
  // PERFORMANCE METRICS DATA
  // =============================================================================

  const performanceData = useMemo(() => {
    if (!searchHistory) return [];

    return searchHistory.slice(0, 7).map((search: any, index: number) => ({
      name: `Search ${index + 1}`,
      responseTime: search.search_time_ms || 0,
      results: search.total_results || 0,
      cached: search.cached ? 100 : 0,
    }));
  }, [searchHistory]);

  const credibilityData = useMemo(() => {
    if (!searchResults?.results) return [];

    const credibilityCounts = searchResults.results.reduce(
      (acc: Record<string, number>, result: any) => {
        acc[result.credibility_score] = (acc[result.credibility_score] || 0) + 1;
        return acc;
      },
      {}
    );

    return Object.entries(credibilityCounts).map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value,
      color: {
        high: '#10B981',
        medium: '#F59E0B',
        low: '#EF4444',
        suspicious: '#8B5CF6',
      }[name] || '#6B7280',
    }));
  }, [searchResults]);

  // =============================================================================
  // RENDER HELPERS
  // =============================================================================

  const renderSearchInterface = () => (
    <div className="space-y-6">
      {/* Search Form */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
      >
        <form onSubmit={searchForm.handleSubmit(handleSearch)} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Search Query
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                {...searchForm.register('query')}
                type="text"
                placeholder="Enter your search query (e.g., 'AI trends in healthcare')"
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            {searchForm.formState.errors.query && (
              <p className="mt-1 text-sm text-red-600">
                {searchForm.formState.errors.query.message}
              </p>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Results
              </label>
              <select
                {...searchForm.register('max_results', { valueAsNumber: true })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value={5}>5 results</option>
                <option value={10}>10 results</option>
                <option value={20}>20 results</option>
                <option value={50}>50 results</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Options</label>
              <label className="flex items-center space-x-2 py-2">
                <input
                  {...searchForm.register('use_cache')}
                  type="checkbox"
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-600">Use cache</span>
              </label>
            </div>

            <div className="flex items-end">
              <button
                type="button"
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
              >
                <Filter className="w-4 h-4" />
                <span>Filters</span>
                <ChevronDown
                  className={clsx('w-4 h-4 transition-transform', {
                    'rotate-180': showFilters,
                  })}
                />
              </button>
            </div>
          </div>

          <AnimatePresence>
            {showFilters && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-gray-200"
              >
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Source Type
                  </label>
                  <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    <option value="">All sources</option>
                    <option value="news">News</option>
                    <option value="academic">Academic</option>
                    <option value="blog">Blog</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Credibility
                  </label>
                  <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    <option value="">All credibility</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Date Range
                  </label>
                  <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    <option value="">Any time</option>
                    <option value="1d">Last 24 hours</option>
                    <option value="7d">Last week</option>
                    <option value="30d">Last month</option>
                  </select>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-600">
              {userStats && (
                <span>
                  Quota: {userStats.quota_remaining} remaining | Plan:{' '}
                  {userStats.subscription_plan}
                </span>
              )}
            </div>
            <button
              type="submit"
              disabled={searchMutation.isPending}
              className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {searchMutation.isPending ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Search className="w-5 h-5" />
              )}
              <span>{searchMutation.isPending ? 'Searching...' : 'Search'}</span>
            </button>
          </div>
        </form>
      </motion.div>

      {/* Search Results */}
      {selectedSearchId && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-xl shadow-sm border border-gray-200"
        >
          <div className="p-6 border-b border-gray-200">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold text-gray-900">Search Results</h3>
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleExportResults}
                  className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
                >
                  <Download className="w-4 h-4" />
                  <span>Export</span>
                </button>
              </div>
            </div>

            {resultsLoading ? (
              <LoadingSkeleton className="mt-4" />
            ) : searchResults ? (
              <div className="mt-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <Database className="w-5 h-5 text-blue-600" />
                      <span className="text-sm font-medium text-blue-900">Total Results</span>
                    </div>
                    <p className="text-2xl font-bold text-blue-600 mt-1">
                      {searchResults.total_results}
                    </p>
                  </div>

                  <div className="bg-green-50 p-4 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <Zap className="w-5 h-5 text-green-600" />
                      <span className="text-sm font-medium text-green-900">Response Time</span>
                    </div>
                    <p className="text-2xl font-bold text-green-600 mt-1">
                      {searchResults.search_time_ms}ms
                    </p>
                  </div>

                  <div className="bg-purple-50 p-4 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <Shield className="w-5 h-5 text-purple-600" />
                      <span className="text-sm font-medium text-purple-900">Source Quality</span>
                    </div>
                    <p className="text-2xl font-bold text-purple-600 mt-1">
                      {searchResults.cached ? 'Cached' : 'Fresh'}
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  {searchResults.results.map((result: any, index: number) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="p-4 border border-gray-200 rounded-lg hover:border-gray-300 transition-colors"
                    >
                      <div className="flex justify-between items-start mb-2">
                        <h4 className="text-lg font-medium text-gray-900 hover:text-blue-600">
                          <a
                            href={result.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center space-x-2"
                          >
                            <span>{result.title}</span>
                            <ExternalLink className="w-4 h-4" />
                          </a>
                        </h4>
                        <div className="flex items-center space-x-2">
                          <span
                            className={clsx('px-2 py-1 text-xs font-medium rounded-full', {
                              'bg-green-100 text-green-800':
                                result.credibility_score === 'high',
                              'bg-yellow-100 text-yellow-800':
                                result.credibility_score === 'medium',
                              'bg-red-100 text-red-800': result.credibility_score === 'low',
                            })}
                          >
                            {result.credibility_score} credibility
                          </span>
                          <span className="text-sm text-gray-500">
                            {(result.relevance_score * 100).toFixed(1)}% relevant
                          </span>
                        </div>
                      </div>

                      <p className="text-gray-600 mb-3">{result.snippet}</p>

                      <div className="flex items-center justify-between text-sm text-gray-500">
                        <div className="flex items-center space-x-4">
                          <span className="flex items-center space-x-1">
                            <Globe className="w-4 h-4" />
                            <span>{result.source_domain}</span>
                          </span>
                          <span className="flex items-center space-x-1">
                            <Tag className="w-4 h-4" />
                            <span>{result.source_type}</span>
                          </span>
                          {result.published_date && (
                            <span className="flex items-center space-x-1">
                              <Calendar className="w-4 h-4" />
                              <span>{format(parseISO(result.published_date), 'MMM d, yyyy')}</span>
                            </span>
                          )}
                        </div>
                        <div className="flex items-center space-x-2">
                          <button className="flex items-center space-x-1 text-gray-400 hover:text-blue-600 transition-colors">
                            <Bookmark className="w-4 h-4" />
                            <span>Save</span>
                          </button>
                          <button className="flex items-center space-x-1 text-gray-400 hover:text-blue-600 transition-colors">
                            <Share className="w-4 h-4" />
                            <span>Share</span>
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </motion.div>
      )}
    </div>
  );

  const renderAnalysisInterface = () => (
    <div className="space-y-6">
      {/* Analysis Form */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
      >
        <form onSubmit={analysisForm.handleSubmit(handleAnalysis)} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Data Source
            </label>
            <div className="relative">
              <Brain className="absolute left-3 top-3 text-gray-400 w-5 h-5" />
              <textarea
                {...analysisForm.register('data_source')}
                rows={4}
                placeholder={
                  selectedSearchId
                    ? `Using search results from: ${selectedSearchId}`
                    : 'Enter text to analyze or use search results from above'
                }
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={!!selectedSearchId}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Analysis Type
              </label>
              <select
                {...analysisForm.register('analysis_type')}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="summarization">Summarization</option>
                <option value="trend_analysis">Trend Analysis</option>
                <option value="sentiment_analysis">Sentiment Analysis</option>
                <option value="key_insights">Key Insights</option>
                <option value="comparative_analysis">Comparative Analysis</option>
                <option value="prediction">Prediction</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                LLM Provider
              </label>
              <select
                {...analysisForm.register('provider')}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="openai">OpenAI (GPT-4)</option>
                <option value="claude">Anthropic (Claude)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Tokens: {analysisForm.watch('max_tokens')}
              </label>
              <input
                {...analysisForm.register('max_tokens', { valueAsNumber: true })}
                type="range"
                min={100}
                max={4000}
                step={100}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Temperature: {analysisForm.watch('temperature')}
              </label>
              <input
                {...analysisForm.register('temperature', { valueAsNumber: true })}
                type="range"
                min={0}
                max={1}
                step={0.1}
                className="w-full"
              />
            </div>
          </div>

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={analysisMutation.isPending}
              className="flex items-center space-x-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {analysisMutation.isPending ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Brain className="w-5 h-5" />
              )}
              <span>{analysisMutation.isPending ? 'Analyzing...' : 'Analyze'}</span>
            </button>
          </div>
        </form>
      </motion.div>

      {/* Analysis History */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Analyses</h3>
        {analysisHistoryLoading ? (
          <LoadingSkeleton />
        ) : analysisHistory && analysisHistory.length > 0 ? (
          <div className="space-y-4">
            {analysisHistory.map((analysis: any) => (
              <div
                key={analysis.analysis_id}
                className="p-4 border border-gray-200 rounded-lg"
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="font-medium text-gray-900">{analysis.analysis_type}</h4>
                    <p className="text-sm text-gray-600">
                      {analysis.provider} • {analysis.model}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center space-x-2">
                      <span
                        className={clsx('px-2 py-1 text-xs font-medium rounded-full', {
                          'bg-green-100 text-green-800': analysis.confidence_score >= 0.8,
                          'bg-yellow-100 text-yellow-800':
                            analysis.confidence_score >= 0.6 && analysis.confidence_score < 0.8,
                          'bg-red-100 text-red-800': analysis.confidence_score < 0.6,
                        })}
                      >
                        {(analysis.confidence_score * 100).toFixed(1)}% confidence
                      </span>
                      {analysis.bias_detected && (
                        <span className="px-2 py-1 text-xs font-medium bg-orange-100 text-orange-800 rounded-full">
                          Bias detected
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-500 mt-1">
                      {analysis.processing_time_ms}ms • ${analysis.cost_usd?.toFixed(4) || '0.00'}
                    </p>
                  </div>
                </div>
                <p className="text-gray-600 text-sm mb-3">
                  {analysis.content_preview || analysis.content}
                </p>
                {analysis.key_points && analysis.key_points.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {analysis.key_points.slice(0, 3).map((point: string, index: number) => (
                      <span
                        key={index}
                        className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                      >
                        {point.substring(0, 50)}...
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <Brain className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No analyses yet. Start by analyzing some search results!</p>
          </div>
        )}
      </motion.div>
    </div>
  );

  const renderInsightsInterface = () => (
    <div className="space-y-6">
      {/* Performance Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Overview</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Response Time Chart */}
          <div>
            <h4 className="text-md font-medium text-gray-700 mb-2">Response Time Trends</h4>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="responseTime"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={{ fill: '#3B82F6' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Source Credibility */}
          <div>
            <h4 className="text-md font-medium text-gray-700 mb-2">Source Credibility</h4>
            <ResponsiveContainer width="100%" height={200}>
              <RechartsPieChart>
                <Pie
                  data={credibilityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  dataKey="value"
                >
                  {credibilityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </RechartsPieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>

      {/* User Statistics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Usage Statistics</h3>
        {userStats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="bg-blue-100 p-3 rounded-lg inline-flex">
                <Search className="w-6 h-6 text-blue-600" />
              </div>
              <p className="text-2xl font-bold text-gray-900 mt-2">{userStats.searches_today}</p>
              <p className="text-sm text-gray-600">Searches Today</p>
            </div>

            <div className="text-center">
              <div className="bg-purple-100 p-3 rounded-lg inline-flex">
                <Brain className="w-6 h-6 text-purple-600" />
              </div>
              <p className="text-2xl font-bold text-gray-900 mt-2">{userStats.analyses_today}</p>
              <p className="text-sm text-gray-600">Analyses Today</p>
            </div>

            <div className="text-center">
              <div className="bg-green-100 p-3 rounded-lg inline-flex">
                <Activity className="w-6 h-6 text-green-600" />
              </div>
              <p className="text-2xl font-bold text-gray-900 mt-2">
                {userStats.avg_response_time?.toFixed(0)}ms
              </p>
              <p className="text-sm text-gray-600">Avg Response</p>
            </div>

            <div className="text-center">
              <div className="bg-orange-100 p-3 rounded-lg inline-flex">
                <Zap className="w-6 h-6 text-orange-600" />
              </div>
              <p className="text-2xl font-bold text-gray-900 mt-2">{userStats.quota_remaining}</p>
              <p className="text-sm text-gray-600">Quota Left</p>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );

  // =============================================================================
  // MAIN RENDER
  // =============================================================================

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <h1 className="text-xl font-bold text-gray-900">AI Insights</h1>
                </div>

                <nav className="hidden md:flex space-x-6">
                  <button
                    onClick={() => setActiveTab('search')}
                    className={clsx(
                      'px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                      {
                        'bg-blue-100 text-blue-700': activeTab === 'search',
                        'text-gray-600 hover:text-gray-900': activeTab !== 'search',
                      }
                    )}
                  >
                    Search
                  </button>
                  <button
                    onClick={() => setActiveTab('analysis')}
                    className={clsx(
                      'px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                      {
                        'bg-purple-100 text-purple-700': activeTab === 'analysis',
                        'text-gray-600 hover:text-gray-900': activeTab !== 'analysis',
                      }
                    )}
                  >
                    Analysis
                  </button>
                  <button
                    onClick={() => setActiveTab('insights')}
                    className={clsx(
                      'px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                      {
                        'bg-green-100 text-green-700': activeTab === 'insights',
                        'text-gray-600 hover:text-gray-900': activeTab !== 'insights',
                      }
                    )}
                  >
                    Insights
                  </button>
                </nav>
              </div>

              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setRealtimeMode(!realtimeMode)}
                  className={clsx(
                    'flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                    {
                      'bg-green-100 text-green-700': realtimeMode,
                      'text-gray-600 hover:text-gray-900': !realtimeMode,
                    }
                  )}
                >
                  <Activity className="w-4 h-4" />
                  <span className="hidden sm:inline">
                    {realtimeMode ? 'Real-time On' : 'Real-time Off'}
                  </span>
                </button>

                <button className="relative p-2 text-gray-400 hover:text-gray-600 transition-colors">
                  <Bell className="w-5 h-5" />
                  <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
                </button>

                <div className="flex items-center space-x-3">
                  <div className="hidden sm:block text-right">
                    <p className="text-sm font-medium text-gray-900">{user?.full_name}</p>
                    <p className="text-xs text-gray-600">{user?.subscription_plan} Plan</p>
                  </div>
                  <button
                    onClick={logout}
                    className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors"
                  >
                    <User className="w-4 h-4" />
                    <span className="hidden sm:inline">Logout</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <AnimatePresence mode="wait">
            {activeTab === 'search' && (
              <motion.div
                key="search"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2 }}
              >
                {renderSearchInterface()}
              </motion.div>
            )}

            {activeTab === 'analysis' && (
              <motion.div
                key="analysis"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2 }}
              >
                {renderAnalysisInterface()}
              </motion.div>
            )}

            {activeTab === 'insights' && (
              <motion.div
                key="insights"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2 }}
              >
                {renderInsightsInterface()}
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </ErrorBoundary>
  );
};
