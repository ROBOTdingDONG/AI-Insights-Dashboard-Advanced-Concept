"""
Analytics Data Models
====================

SQLAlchemy models for AI insights dashboard with time-series optimization.

Models:
- User management and authentication
- Search queries and results tracking
- LLM analysis results and caching
- Performance metrics and monitoring
- Export tracking and management

Features:
- TimescaleDB hypertables for time-series data
- Automatic timestamp tracking
- JSON fields for flexible data storage
- Comprehensive indexing for performance
- Audit trails and soft deletes

Author: AI Insights Team
Version: 1.0.0
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum as PyEnum

import structlog
from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Text, JSON, 
    Float, ForeignKey, Index, UniqueConstraint, CheckConstraint,
    Enum, DECIMAL, BigInteger, SmallInteger
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB, INET
from sqlalchemy.orm import relationship, validates, declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declared_attr

# Import database base and utilities
from app.core.database import Base
from app.services.perplexity import SourceType, CredibilityScore
from app.services.summarizer import AnalysisType, ContentSafetyRating, LLMProvider


logger = structlog.get_logger(__name__)


# =============================================================================
# ENUMS FOR DATABASE
# =============================================================================
class UserRole(PyEnum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"
    VIEWER = "viewer"


class UserStatus(PyEnum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class SubscriptionPlan(PyEnum):
    """User subscription plans."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class QueryStatus(PyEnum):
    """Search query processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(PyEnum):
    """Export format options."""
    PDF = "pdf"
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    PNG = "png"
    SVG = "svg"
    MP4 = "mp4"


class ExportStatus(PyEnum):
    """Export processing status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# MIXINS FOR COMMON FUNCTIONALITY
# =============================================================================
class TimestampMixin:
    """Mixin for automatic timestamp tracking."""
    
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        server_default=func.now()
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        server_default=func.now(),
        onupdate=func.now()
    )


class UUIDMixin:
    """Mixin for UUID primary keys."""
    
    @declared_attr
    def id(cls):
        return Column(
            UUID(as_uuid=True),
            primary_key=True,
            default=uuid.uuid4,
            nullable=False
        )


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)
    
    def soft_delete(self):
        """Mark record as deleted."""
        self.deleted_at = datetime.now(timezone.utc)
        self.is_deleted = True
    
    def restore(self):
        """Restore soft deleted record."""
        self.deleted_at = None
        self.is_deleted = False


# =============================================================================
# USER MANAGEMENT MODELS
# =============================================================================
class User(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """
    User model with authentication and profile information.
    
    Supports multiple authentication methods, role-based access control,
    and subscription management.
    """
    
    __tablename__ = "users"
    
    # Basic Information
    email = Column(String(255), nullable=False, unique=True, index=True)
    username = Column(String(50), nullable=True, unique=True, index=True)
    full_name = Column(String(255), nullable=True)
    company = Column(String(255), nullable=True)
    
    # Authentication
    password_hash = Column(String(255), nullable=True)  # Null for OAuth users
    is_verified = Column(Boolean, nullable=False, default=False)
    verification_token = Column(String(255), nullable=True)
    
    # OAuth Information
    oauth_provider = Column(String(50), nullable=True)  # google, github, etc.
    oauth_id = Column(String(255), nullable=True)
    
    # Role and Status
    role = Column(Enum(UserRole), nullable=False, default=UserRole.USER)
    status = Column(Enum(UserStatus), nullable=False, default=UserStatus.PENDING)
    
    # Subscription and Limits
    subscription_plan = Column(Enum(SubscriptionPlan), nullable=False, default=SubscriptionPlan.FREE)
    subscription_expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Usage Tracking
    api_calls_today = Column(Integer, nullable=False, default=0)
    api_calls_month = Column(Integer, nullable=False, default=0)
    last_api_call = Column(DateTime(timezone=True), nullable=True)
    
    # Profile Information
    avatar_url = Column(String(500), nullable=True)
    timezone = Column(String(50), nullable=False, default="UTC")
    language = Column(String(10), nullable=False, default="en")
    
    # Security
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_login_ip = Column(INET, nullable=True)
    failed_login_attempts = Column(SmallInteger, nullable=False, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Two-Factor Authentication
    totp_secret = Column(String(32), nullable=True)
    backup_codes = Column(ARRAY(String), nullable=True)
    
    # Preferences
    preferences = Column(JSONB, nullable=False, default=dict)
    
    # Relationships
    search_queries = relationship("SearchQuery", back_populates="user", lazy="dynamic")
    analysis_results = relationship("AnalysisResult", back_populates="user", lazy="dynamic")
    exports = relationship("Export", back_populates="user", lazy="dynamic")
    api_keys = relationship("APIKey", back_populates="user", lazy="dynamic")
    
    # Constraints and Indexes
    __table_args__ = (
        Index("ix_users_email_status", "email", "status"),
        Index("ix_users_oauth", "oauth_provider", "oauth_id"),
        Index("ix_users_subscription", "subscription_plan", "subscription_expires_at"),
        CheckConstraint("email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'", name="ck_users_email_format"),
    )
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format."""
        import re
        if not re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$', email):
            raise ValueError("Invalid email format")
        return email.lower()
    
    @hybrid_property
    def is_active(self):
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE and not self.is_deleted
    
    @hybrid_property
    def is_premium(self):
        """Check if user has premium subscription."""
        return self.subscription_plan in [SubscriptionPlan.PROFESSIONAL, SubscriptionPlan.ENTERPRISE]
    
    def can_make_api_call(self) -> bool:
        """Check if user can make another API call based on their plan."""
        limits = {
            SubscriptionPlan.FREE: 100,
            SubscriptionPlan.BASIC: 1000,
            SubscriptionPlan.PROFESSIONAL: 10000,
            SubscriptionPlan.ENTERPRISE: 100000
        }
        
        daily_limit = limits.get(self.subscription_plan, 100)
        return self.api_calls_today < daily_limit
    
    def increment_api_usage(self):
        """Increment API usage counters."""
        self.api_calls_today += 1
        self.api_calls_month += 1
        self.last_api_call = datetime.now(timezone.utc)
    
    def reset_daily_usage(self):
        """Reset daily API usage counter."""
        self.api_calls_today = 0
    
    def __repr__(self):
        return f"<User {self.email} ({self.role.value})>"


class APIKey(Base, UUIDMixin, TimestampMixin):
    """
    API key model for programmatic access.
    
    Supports key rotation, scoping, and usage tracking.
    """
    
    __tablename__ = "api_keys"
    
    # Basic Information
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Key Data
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    key_prefix = Column(String(20), nullable=False, index=True)
    
    # Access Control
    scopes = Column(ARRAY(String), nullable=False, default=list)
    allowed_ips = Column(ARRAY(INET), nullable=True)
    
    # Status and Expiration
    is_active = Column(Boolean, nullable=False, default=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Usage Tracking
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    last_used_ip = Column(INET, nullable=True)
    usage_count = Column(BigInteger, nullable=False, default=0)
    
    # Rate Limiting
    rate_limit_per_hour = Column(Integer, nullable=True)
    
    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="api_keys")
    
    # Constraints and Indexes
    __table_args__ = (
        Index("ix_api_keys_user_active", "user_id", "is_active"),
        Index("ix_api_keys_expires", "expires_at"),
    )
    
    @hybrid_property
    def is_valid(self):
        """Check if API key is valid and not expired."""
        if not self.is_active:
            return False
        
        if self.expires_at and self.expires_at < datetime.now(timezone.utc):
            return False
        
        return True
    
    def record_usage(self, ip_address: str = None):
        """Record API key usage."""
        self.usage_count += 1
        self.last_used_at = datetime.now(timezone.utc)
        if ip_address:
            self.last_used_ip = ip_address
    
    def __repr__(self):
        return f"<APIKey {self.name} ({self.key_prefix}...)>"


# =============================================================================
# SEARCH AND ANALYSIS MODELS
# =============================================================================
class SearchQuery(Base, UUIDMixin, TimestampMixin):
    """
    Search query model with result tracking and caching.
    
    Stores user search queries, external API responses, and performance metrics.
    This table can be converted to a TimescaleDB hypertable for time-series optimization.
    """
    
    __tablename__ = "search_queries"
    
    # Query Information
    query_text = Column(String(1000), nullable=False, index=True)
    query_hash = Column(String(64), nullable=False, index=True)  # For deduplication
    
    # Processing Status
    status = Column(Enum(QueryStatus), nullable=False, default=QueryStatus.PENDING)
    error_message = Column(Text, nullable=True)
    
    # External API Information
    provider = Column(String(50), nullable=False, default="perplexity")  # perplexity, google, etc.
    model_used = Column(String(100), nullable=True)
    
    # Results Metadata
    total_results = Column(Integer, nullable=False, default=0)
    results_cached = Column(Boolean, nullable=False, default=False)
    cache_hit = Column(Boolean, nullable=False, default=False)
    
    # Performance Metrics
    search_time_ms = Column(Float, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    total_time_ms = Column(Float, nullable=True)
    
    # Cost Tracking
    estimated_cost_usd = Column(DECIMAL(10, 6), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    
    # User Context
    user_ip = Column(INET, nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Request Metadata
    request_id = Column(String(100), nullable=False, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    
    # Raw Results Storage
    raw_results = Column(JSONB, nullable=True)  # Store full API response
    processed_results = Column(JSONB, nullable=True)  # Store processed/filtered results
    
    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    user = relationship("User", back_populates="search_queries")
    
    search_results = relationship("SearchResult", back_populates="query", lazy="dynamic")
    analysis_results = relationship("AnalysisResult", back_populates="search_query", lazy="dynamic")
    
    # Constraints and Indexes
    __table_args__ = (
        Index("ix_search_queries_user_created", "user_id", "created_at"),
        Index("ix_search_queries_status_created", "status", "created_at"),
        Index("ix_search_queries_hash_created", "query_hash", "created_at"),
        Index("ix_search_queries_performance", "search_time_ms", "total_results"),
        # TimescaleDB hypertable will be created on created_at
    )
    
    @hybrid_property
    def is_successful(self):
        """Check if query was processed successfully."""
        return self.status == QueryStatus.COMPLETED and self.total_results > 0
    
    def calculate_total_time(self):
        """Calculate total processing time."""
        if self.search_time_ms and self.processing_time_ms:
            self.total_time_ms = self.search_time_ms + self.processing_time_ms
    
    def __repr__(self):
        return f"<SearchQuery '{self.query_text[:50]}...' ({self.status.value})>"


class SearchResult(Base, UUIDMixin, TimestampMixin):
    """
    Individual search result from external APIs.
    
    Stores detailed information about each search result including
    credibility scores, relevance metrics, and content analysis.
    """
    
    __tablename__ = "search_results"
    
    # Basic Information
    title = Column(String(500), nullable=False)
    url = Column(String(2000), nullable=False)
    snippet = Column(Text, nullable=True)
    
    # Source Information
    source_domain = Column(String(255), nullable=False, index=True)
    source_type = Column(Enum(SourceType), nullable=False, default=SourceType.UNKNOWN)
    
    # Content Metadata
    published_date = Column(DateTime(timezone=True), nullable=True)
    author = Column(String(255), nullable=True)
    language = Column(String(10), nullable=True)
    
    # Scoring and Analysis
    credibility_score = Column(Enum(CredibilityScore), nullable=False, default=CredibilityScore.LOW)
    relevance_score = Column(Float, nullable=False, default=0.0)
    sentiment_score = Column(Float, nullable=True)  # -1 to 1
    
    # Position and Ranking
    result_position = Column(SmallInteger, nullable=False)
    page_number = Column(SmallInteger, nullable=False, default=1)
    
    # Content Analysis
    word_count = Column(Integer, nullable=True)
    reading_time_minutes = Column(Float, nullable=True)
    content_hash = Column(String(64), nullable=True, index=True)  # For deduplication
    
    # Full Content (if fetched)
    full_content = Column(Text, nullable=True)
    content_type = Column(String(50), nullable=True)  # article, blog, news, etc.
    
    # Metadata
    raw_metadata = Column(JSONB, nullable=True)
    
    # Relationships
    query_id = Column(UUID(as_uuid=True), ForeignKey("search_queries.id"), nullable=False, index=True)
    query = relationship("SearchQuery", back_populates="search_results")
    
    # Constraints and Indexes
    __table_args__ = (
        Index("ix_search_results_query_position", "query_id", "result_position"),
        Index("ix_search_results_domain_credibility", "source_domain", "credibility_score"),
        Index("ix_search_results_relevance", "relevance_score"),
        Index("ix_search_results_published", "published_date"),
        UniqueConstraint("query_id", "url", name="uq_search_results_query_url"),
        CheckConstraint("relevance_score >= 0 AND relevance_score <= 1", name="ck_search_results_relevance"),
        CheckConstraint("sentiment_score >= -1 AND sentiment_score <= 1", name="ck_search_results_sentiment"),
    )
    
    @validates('relevance_score')
    def validate_relevance_score(self, key, score):
        """Validate relevance score is between 0 and 1."""
        if not 0 <= score <= 1:
            raise ValueError("Relevance score must be between 0 and 1")
        return score
    
    def __repr__(self):
        return f"<SearchResult '{self.title[:50]}...' ({self.relevance_score:.2f})>"


class AnalysisResult(Base, UUIDMixin, TimestampMixin):
    """
    LLM analysis results with comprehensive tracking.
    
    Stores results from various analysis types including summarization,
    trend analysis, sentiment analysis, and key insights extraction.
    """
    
    __tablename__ = "analysis_results"
    
    # Analysis Configuration
    analysis_type = Column(Enum(AnalysisType), nullable=False, index=True)
    provider = Column(Enum(LLMProvider), nullable=False, default=LLMProvider.OPENAI)
    model_used = Column(String(100), nullable=False)
    
    # Request Information
    request_id = Column(String(100), nullable=False, index=True)
    custom_prompt = Column(Text, nullable=True)
    
    # LLM Parameters
    max_tokens = Column(Integer, nullable=False)
    temperature = Column(Float, nullable=False)
    
    # Analysis Results
    content = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False)
    key_points = Column(ARRAY(String), nullable=True)
    entities = Column(ARRAY(String), nullable=True)
    sentiment = Column(String(20), nullable=True)
    
    # Quality and Safety
    safety_rating = Column(Enum(ContentSafetyRating), nullable=False, default=ContentSafetyRating.SAFE)
    bias_detected = Column(Boolean, nullable=False, default=False)
    bias_types = Column(ARRAY(String), nullable=True)
    
    # Performance Metrics
    processing_time_ms = Column(Float, nullable=False)
    token_usage_prompt = Column(Integer, nullable=True)
    token_usage_completion = Column(Integer, nullable=True)
    token_usage_total = Column(Integer, nullable=True)
    
    # Cost Tracking
    estimated_cost_usd = Column(DECIMAL(10, 6), nullable=True)
    
    # Caching
    cache_hit = Column(Boolean, nullable=False, default=False)
    cache_key = Column(String(64), nullable=True, index=True)
    
    # Source Data Reference
    source_data_hash = Column(String(64), nullable=True, index=True)
    
    # Structured Analysis Results
    analysis_data = Column(JSONB, nullable=True)  # Flexible storage for analysis-specific data
    
    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    user = relationship("User", back_populates="analysis_results")
    
    search_query_id = Column(UUID(as_uuid=True), ForeignKey("search_queries.id"), nullable=True, index=True)
    search_query = relationship("SearchQuery", back_populates="analysis_results")
    
    # Constraints and Indexes
    __table_args__ = (
        Index("ix_analysis_results_user_type_created", "user_id", "analysis_type", "created_at"),
        Index("ix_analysis_results_provider_model", "provider", "model_used"),
        Index("ix_analysis_results_confidence", "confidence_score"),
        Index("ix_analysis_results_performance", "processing_time_ms", "token_usage_total"),
        Index("ix_analysis_results_cost", "estimated_cost_usd"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="ck_analysis_results_confidence"),
        CheckConstraint("temperature >= 0 AND temperature <= 1", name="ck_analysis_results_temperature"),
    )
    
    @validates('confidence_score')
    def validate_confidence_score(self, key, score):
        """Validate confidence score is between 0 and 1."""
        if not 0 <= score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
        return score
    
    @hybrid_property
    def is_high_confidence(self):
        """Check if analysis has high confidence score."""
        return self.confidence_score >= 0.8
    
    def calculate_total_tokens(self):
        """Calculate total token usage."""
        if self.token_usage_prompt and self.token_usage_completion:
            self.token_usage_total = self.token_usage_prompt + self.token_usage_completion
    
    def __repr__(self):
        return f"<AnalysisResult {self.analysis_type.value} ({self.confidence_score:.2f})>"


# =============================================================================
# EXPORT AND SHARING MODELS
# =============================================================================
class Export(Base, UUIDMixin, TimestampMixin):
    """
    Export tracking for dashboards, reports, and visualizations.
    
    Manages various export formats including PDF, CSV, images, and videos.
    """
    
    __tablename__ = "exports"
    
    # Basic Information
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Export Configuration
    export_format = Column(Enum(ExportFormat), nullable=False)
    export_type = Column(String(50), nullable=False)  # dashboard, report, chart, etc.
    
    # Processing Status
    status = Column(Enum(ExportStatus), nullable=False, default=ExportStatus.QUEUED)
    error_message = Column(Text, nullable=True)
    
    # File Information
    file_path = Column(String(500), nullable=True)
    file_size_bytes = Column(BigInteger, nullable=True)
    file_hash = Column(String(64), nullable=True)
    
    # Export Parameters
    parameters = Column(JSONB, nullable=True)  # Export-specific parameters
    filters = Column(JSONB, nullable=True)     # Data filters applied
    
    # Processing Metrics
    processing_time_ms = Column(Float, nullable=True)
    queue_time_ms = Column(Float, nullable=True)
    
    # Access Control
    is_public = Column(Boolean, nullable=False, default=False)
    public_token = Column(String(64), nullable=True, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Download Tracking
    download_count = Column(Integer, nullable=False, default=0)
    last_downloaded_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    user = relationship("User", back_populates="exports")
    
    # Source Analysis (optional)
    analysis_result_id = Column(UUID(as_uuid=True), ForeignKey("analysis_results.id"), nullable=True, index=True)
    search_query_id = Column(UUID(as_uuid=True), ForeignKey("search_queries.id"), nullable=True, index=True)
    
    # Constraints and Indexes
    __table_args__ = (
        Index("ix_exports_user_status_created", "user_id", "status", "created_at"),
        Index("ix_exports_format_type", "export_format", "export_type"),
        Index("ix_exports_public_expires", "is_public", "expires_at"),
        Index("ix_exports_downloads", "download_count", "last_downloaded_at"),
    )
    
    @hybrid_property
    def is_accessible(self):
        """Check if export is accessible (not expired)."""
        if self.expires_at and self.expires_at < datetime.now(timezone.utc):
            return False
        return self.status == ExportStatus.COMPLETED
    
    def record_download(self):
        """Record a download event."""
        self.download_count += 1
        self.last_downloaded_at = datetime.now(timezone.utc)
    
    def generate_public_token(self):
        """Generate a public access token."""
        self.public_token = uuid.uuid4().hex
        self.is_public = True
    
    def __repr__(self):
        return f"<Export '{self.title}' ({self.export_format.value})>"


# =============================================================================
# PERFORMANCE MONITORING MODELS  
# =============================================================================
class PerformanceMetric(Base, UUIDMixin, TimestampMixin):
    """
    Performance metrics tracking for monitoring and optimization.
    
    This table should be converted to a TimescaleDB hypertable
    for efficient time-series data storage and analysis.
    """
    
    __tablename__ = "performance_metrics"
    
    # Metric Information
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)  # counter, gauge, histogram
    
    # Metric Values
    value = Column(Float, nullable=False)
    count = Column(BigInteger, nullable=True)  # For counters
    
    # Context Information
    service = Column(String(50), nullable=False, index=True)  # api, perplexity, openai, etc.
    endpoint = Column(String(200), nullable=True, index=True)
    method = Column(String(10), nullable=True)  # GET, POST, etc.
    
    # Status and Error Tracking
    status_code = Column(SmallInteger, nullable=True)
    error_type = Column(String(100), nullable=True, index=True)
    
    # User Context (optional)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    
    # Additional Metadata
    tags = Column(JSONB, nullable=True)  # Flexible tagging system
    
    # Constraints and Indexes
    __table_args__ = (
        Index("ix_perf_metrics_name_service_created", "metric_name", "service", "created_at"),
        Index("ix_perf_metrics_type_created", "metric_type", "created_at"),
        Index("ix_perf_metrics_endpoint_status", "endpoint", "status_code"),
        Index("ix_perf_metrics_user_created", "user_id", "created_at"),
        # TimescaleDB hypertable will be created on created_at
    )
    
    def __repr__(self):
        return f"<PerformanceMetric {self.metric_name}={self.value} ({self.service})>"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def create_timescale_hypertables(session):
    """
    Create TimescaleDB hypertables for time-series optimized tables.
    
    Args:
        session: SQLAlchemy session
    """
    hypertables = [
        ("search_queries", "created_at"),
        ("performance_metrics", "created_at"),
        ("analysis_results", "created_at"),
    ]
    
    for table_name, time_column in hypertables:
        try:
            # Check if table exists and is not already a hypertable
            result = session.execute(f"""
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = '{table_name}' AND table_schema = 'public'
            """).fetchone()
            
            if result:
                # Check if already a hypertable
                hypertable_check = session.execute(f"""
                    SELECT 1 FROM timescaledb_information.hypertables 
                    WHERE hypertable_name = '{table_name}'
                """).fetchone()
                
                if not hypertable_check:
                    session.execute(f"""
                        SELECT create_hypertable('{table_name}', '{time_column}')
                    """)
                    logger.info(f"Created hypertable: {table_name}")
                else:
                    logger.info(f"Hypertable already exists: {table_name}")
                    
        except Exception as e:
            logger.warning(f"Failed to create hypertable {table_name}: {str(e)}")


def get_table_stats(session) -> Dict[str, Any]:
    """
    Get comprehensive statistics about all tables.
    
    Args:
        session: SQLAlchemy session
        
    Returns:
        Dict containing table statistics
    """
    tables = [
        "users", "api_keys", "search_queries", "search_results",
        "analysis_results", "exports", "performance_metrics"
    ]
    
    stats = {}
    
    for table in tables:
        try:
            # Get row count
            count_result = session.execute(f"SELECT COUNT(*) FROM {table}").scalar()
            
            # Get table size
            size_result = session.execute(f"""
                SELECT pg_size_pretty(pg_total_relation_size('{table}'))
            """).scalar()
            
            # Get recent activity (last 24 hours)
            if table in ["search_queries", "analysis_results", "performance_metrics"]:
                recent_result = session.execute(f"""
                    SELECT COUNT(*) FROM {table} 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """).scalar()
            else:
                recent_result = 0
            
            stats[table] = {
                "total_rows": count_result,
                "table_size": size_result,
                "recent_activity_24h": recent_result
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for table {table}: {str(e)}")
            stats[table] = {"error": str(e)}
    
    return stats


# =============================================================================
# MODEL VALIDATION
# =============================================================================
def validate_models():
    """Validate all model definitions and relationships."""
    try:
        # Test model instantiation
        models = [
            User, APIKey, SearchQuery, SearchResult,
            AnalysisResult, Export, PerformanceMetric
        ]
        
        for model in models:
            # Check required columns exist
            mapper = model.__mapper__
            logger.info(f"Model {model.__name__} has {len(mapper.columns)} columns")
            
            # Check relationships
            relationships = [rel for rel in mapper.relationships]
            logger.info(f"Model {model.__name__} has {len(relationships)} relationships")
        
        logger.info("All models validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Validate models when run directly
    validate_models()
