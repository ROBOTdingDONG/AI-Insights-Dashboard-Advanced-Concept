"""
Configuration Management System
==============================

Enterprise-grade configuration management with environment-based settings,
validation, and security features.

Features:
- Environment-specific configurations (dev/staging/prod)
- Secure secret management with validation
- Database connection pooling and optimization
- API rate limiting and caching configuration
- Logging and monitoring settings
- Security headers and CORS policies

Security Features:
- Environment variable validation
- Secret rotation support
- Database connection encryption
- API key obfuscation in logs

Author: AI Insights Team
Version: 1.0.0
"""

import os
import secrets
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from functools import lru_cache

from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.networks import AnyHttpUrl, PostgresDsn
import structlog


logger = structlog.get_logger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================
class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseEngine(str, Enum):
    """Supported database engines."""
    POSTGRESQL = "postgresql"
    TIMESCALEDB = "timescaledb"


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================
class DatabaseConfig(BaseSettings):
    """Database configuration with connection pooling and security."""
    
    # Connection settings
    DB_HOST: str = Field(..., env="DB_HOST")
    DB_PORT: int = Field(5432, env="DB_PORT")
    DB_NAME: str = Field(..., env="DB_NAME")
    DB_USER: str = Field(..., env="DB_USER")
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")
    
    # Connection pool settings
    DB_POOL_SIZE: int = Field(10, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(20, env="DB_MAX_OVERFLOW")
    DB_POOL_TIMEOUT: int = Field(30, env="DB_POOL_TIMEOUT")
    DB_POOL_RECYCLE: int = Field(3600, env="DB_POOL_RECYCLE")
    
    # Security settings
    DB_SSL_MODE: str = Field("require", env="DB_SSL_MODE")
    DB_SSL_CERT: Optional[str] = Field(None, env="DB_SSL_CERT")
    DB_SSL_KEY: Optional[str] = Field(None, env="DB_SSL_KEY")
    DB_SSL_ROOT_CERT: Optional[str] = Field(None, env="DB_SSL_ROOT_CERT")
    
    # Performance settings
    DB_ECHO: bool = Field(False, env="DB_ECHO")
    DB_ECHO_POOL: bool = Field(False, env="DB_ECHO_POOL")
    DB_QUERY_TIMEOUT: int = Field(30, env="DB_QUERY_TIMEOUT")
    
    @validator('DB_PORT')
    def validate_db_port(cls, v):
        """Validate database port range."""
        if not 1024 <= v <= 65535:
            raise ValueError("Database port must be between 1024 and 65535")
        return v
    
    @validator('DB_POOL_SIZE')
    def validate_pool_size(cls, v):
        """Validate connection pool size."""
        if not 1 <= v <= 50:
            raise ValueError("Database pool size must be between 1 and 50")
        return v
    
    @property
    def database_url(self) -> str:
        """Generate database connection URL."""
        base_url = f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        
        # Add SSL parameters
        ssl_params = []
        if self.DB_SSL_MODE != "disable":
            ssl_params.append(f"sslmode={self.DB_SSL_MODE}")
        if self.DB_SSL_CERT:
            ssl_params.append(f"sslcert={self.DB_SSL_CERT}")
        if self.DB_SSL_KEY:
            ssl_params.append(f"sslkey={self.DB_SSL_KEY}")
        if self.DB_SSL_ROOT_CERT:
            ssl_params.append(f"sslrootcert={self.DB_SSL_ROOT_CERT}")
        
        if ssl_params:
            base_url += "?" + "&".join(ssl_params)
        
        return base_url
    
    @property
    def sync_database_url(self) -> str:
        """Generate synchronous database connection URL for migrations."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")


class RedisConfig(BaseSettings):
    """Redis configuration for caching and session management."""
    
    REDIS_HOST: str = Field("localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(6379, env="REDIS_PORT")
    REDIS_DB: int = Field(0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")
    REDIS_SSL: bool = Field(False, env="REDIS_SSL")
    
    # Connection pool settings
    REDIS_MAX_CONNECTIONS: int = Field(20, env="REDIS_MAX_CONNECTIONS")
    REDIS_RETRY_ON_TIMEOUT: bool = Field(True, env="REDIS_RETRY_ON_TIMEOUT")
    REDIS_SOCKET_TIMEOUT: int = Field(5, env="REDIS_SOCKET_TIMEOUT")
    REDIS_SOCKET_CONNECT_TIMEOUT: int = Field(5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    
    # Cache settings
    REDIS_DEFAULT_TTL: int = Field(3600, env="REDIS_DEFAULT_TTL")  # 1 hour
    REDIS_SESSION_TTL: int = Field(86400, env="REDIS_SESSION_TTL")  # 24 hours
    REDIS_RATE_LIMIT_TTL: int = Field(3600, env="REDIS_RATE_LIMIT_TTL")  # 1 hour
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL."""
        scheme = "rediss" if self.REDIS_SSL else "redis"
        auth_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"{scheme}://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


class SecurityConfig(BaseSettings):
    """Security configuration for authentication and encryption."""
    
    # JWT Settings
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    JWT_ALGORITHM: str = Field("HS256", env="JWT_ALGORITHM")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Password Settings
    PASSWORD_MIN_LENGTH: int = Field(8, env="PASSWORD_MIN_LENGTH")
    PASSWORD_REQUIRE_UPPERCASE: bool = Field(True, env="PASSWORD_REQUIRE_UPPERCASE")
    PASSWORD_REQUIRE_LOWERCASE: bool = Field(True, env="PASSWORD_REQUIRE_LOWERCASE")
    PASSWORD_REQUIRE_NUMBERS: bool = Field(True, env="PASSWORD_REQUIRE_NUMBERS")
    PASSWORD_REQUIRE_SPECIAL: bool = Field(True, env="PASSWORD_REQUIRE_SPECIAL")
    
    # Encryption Settings
    ENCRYPTION_KEY: Optional[str] = Field(None, env="ENCRYPTION_KEY")
    BCRYPT_ROUNDS: int = Field(12, env="BCRYPT_ROUNDS")
    
    # API Security
    API_KEY_LENGTH: int = Field(32, env="API_KEY_LENGTH")
    API_KEY_PREFIX: str = Field("ai-insights", env="API_KEY_PREFIX")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_PER_MINUTE: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_PER_HOUR: int = Field(1000, env="RATE_LIMIT_PER_HOUR")
    RATE_LIMIT_PER_DAY: int = Field(10000, env="RATE_LIMIT_PER_DAY")
    
    @validator('SECRET_KEY')
    def validate_secret_key(cls, v):
        """Validate secret key strength."""
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
    @validator('BCRYPT_ROUNDS')
    def validate_bcrypt_rounds(cls, v):
        """Validate bcrypt rounds for security."""
        if not 10 <= v <= 16:
            raise ValueError("BCRYPT_ROUNDS should be between 10 and 16 for security")
        return v
    
    @root_validator
    def generate_encryption_key_if_needed(cls, values):
        """Generate encryption key if not provided."""
        if not values.get('ENCRYPTION_KEY'):
            values['ENCRYPTION_KEY'] = secrets.token_urlsafe(32)
            logger.warning("Generated new encryption key - store securely for production")
        return values


class ExternalAPIConfig(BaseSettings):
    """Configuration for external API integrations."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_ORG_ID: Optional[str] = Field(None, env="OPENAI_ORG_ID")
    OPENAI_DEFAULT_MODEL: str = Field("gpt-3.5-turbo", env="OPENAI_DEFAULT_MODEL")
    OPENAI_MAX_TOKENS: int = Field(2000, env="OPENAI_MAX_TOKENS")
    OPENAI_TIMEOUT: int = Field(30, env="OPENAI_TIMEOUT")
    
    # Anthropic (Claude) Configuration
    ANTHROPIC_API_KEY: str = Field(..., env="ANTHROPIC_API_KEY")
    ANTHROPIC_DEFAULT_MODEL: str = Field("claude-3-sonnet", env="ANTHROPIC_DEFAULT_MODEL")
    ANTHROPIC_MAX_TOKENS: int = Field(2000, env="ANTHROPIC_MAX_TOKENS")
    ANTHROPIC_TIMEOUT: int = Field(30, env="ANTHROPIC_TIMEOUT")
    
    # Perplexity Configuration
    PERPLEXITY_API_KEY: str = Field(..., env="PERPLEXITY_API_KEY")
    PERPLEXITY_DEFAULT_MODEL: str = Field("llama-3.1-sonar-small-128k-online", env="PERPLEXITY_DEFAULT_MODEL")
    PERPLEXITY_MAX_RESULTS: int = Field(10, env="PERPLEXITY_MAX_RESULTS")
    PERPLEXITY_TIMEOUT: int = Field(30, env="PERPLEXITY_TIMEOUT")
    
    # Rate Limiting for External APIs
    OPENAI_REQUESTS_PER_MINUTE: int = Field(500, env="OPENAI_REQUESTS_PER_MINUTE")
    ANTHROPIC_REQUESTS_PER_MINUTE: int = Field(300, env="ANTHROPIC_REQUESTS_PER_MINUTE")
    PERPLEXITY_REQUESTS_PER_MINUTE: int = Field(100, env="PERPLEXITY_REQUESTS_PER_MINUTE")
    
    @validator('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'PERPLEXITY_API_KEY')
    def validate_api_keys(cls, v):
        """Validate API key format."""
        if not v or len(v) < 10:
            raise ValueError("API key must be at least 10 characters long")
        return v


class MonitoringConfig(BaseSettings):
    """Configuration for monitoring, logging, and alerting."""
    
    # Logging Configuration
    LOG_LEVEL: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    LOG_FORMAT: str = Field("json", env="LOG_FORMAT")  # json or text
    LOG_FILE_PATH: Optional[str] = Field(None, env="LOG_FILE_PATH")
    LOG_ROTATION_SIZE: str = Field("100MB", env="LOG_ROTATION_SIZE")
    LOG_RETENTION_DAYS: int = Field(30, env="LOG_RETENTION_DAYS")
    
    # Metrics Configuration
    METRICS_ENABLED: bool = Field(True, env="METRICS_ENABLED")
    METRICS_PORT: int = Field(9090, env="METRICS_PORT")
    METRICS_PATH: str = Field("/metrics", env="METRICS_PATH")
    
    # Sentry Configuration
    SENTRY_DSN: Optional[str] = Field(None, env="SENTRY_DSN")
    SENTRY_TRACES_SAMPLE_RATE: float = Field(0.1, env="SENTRY_TRACES_SAMPLE_RATE")
    SENTRY_ENVIRONMENT: Optional[str] = Field(None, env="SENTRY_ENVIRONMENT")
    
    # Health Check Configuration
    HEALTH_CHECK_ENABLED: bool = Field(True, env="HEALTH_CHECK_ENABLED")
    HEALTH_CHECK_INTERVAL: int = Field(30, env="HEALTH_CHECK_INTERVAL")  # seconds
    
    # Performance Monitoring
    SLOW_QUERY_THRESHOLD: float = Field(1.0, env="SLOW_QUERY_THRESHOLD")  # seconds
    REQUEST_TIMEOUT: int = Field(30, env="REQUEST_TIMEOUT")  # seconds


# =============================================================================
# MAIN SETTINGS CLASS
# =============================================================================
class Settings(BaseSettings):
    """
    Main application settings with environment-based configuration.
    
    Combines all configuration sections with validation and security features.
    """
    
    # Application Information
    APP_NAME: str = Field("AI Insights Dashboard", env="APP_NAME")
    APP_VERSION: str = Field("1.0.0", env="APP_VERSION")
    APP_DESCRIPTION: str = Field("Intelligent data visualization platform", env="APP_DESCRIPTION")
    
    # Environment Configuration
    ENVIRONMENT: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    DEBUG: bool = Field(False, env="DEBUG")
    TESTING: bool = Field(False, env="TESTING")
    
    # Server Configuration
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")
    WORKERS: int = Field(1, env="WORKERS")
    RELOAD: bool = Field(False, env="RELOAD")
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = Field(
        ["http://localhost:3000", "http://127.0.0.1:3000"],
        env="ALLOWED_ORIGINS"
    )
    ALLOWED_HOSTS: List[str] = Field(
        ["localhost", "127.0.0.1", "0.0.0.0"],
        env="ALLOWED_HOSTS"
    )
    
    # File Storage Configuration
    UPLOAD_MAX_SIZE: int = Field(10485760, env="UPLOAD_MAX_SIZE")  # 10MB
    UPLOAD_ALLOWED_TYPES: List[str] = Field(
        ["image/jpeg", "image/png", "application/pdf", "text/csv"],
        env="UPLOAD_ALLOWED_TYPES"
    )
    STATIC_FILES_PATH: str = Field("static", env="STATIC_FILES_PATH")
    
    # AWS Configuration (Optional)
    AWS_ACCESS_KEY_ID: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field("us-east-1", env="AWS_REGION")
    AWS_S3_BUCKET: Optional[str] = Field(None, env="AWS_S3_BUCKET")
    
    # Configuration Sections
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    security: SecurityConfig = SecurityConfig()
    external_apis: ExternalAPIConfig = ExternalAPIConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
        # Custom configuration for nested models
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )
    
    @root_validator
    def validate_environment_consistency(cls, values):
        """Ensure configuration consistency across environments."""
        env = values.get('ENVIRONMENT')
        debug = values.get('DEBUG')
        
        # Production safety checks
        if env == Environment.PRODUCTION:
            if debug:
                raise ValueError("DEBUG cannot be True in production environment")
            
            # Ensure required production settings
            if not values.get('security', {}).get('SECRET_KEY'):
                raise ValueError("SECRET_KEY is required in production")
            
            if not values.get('monitoring', {}).get('SENTRY_DSN'):
                logger.warning("SENTRY_DSN not configured for production")
        
        # Development environment optimizations
        elif env == Environment.DEVELOPMENT:
            values['DEBUG'] = True
            values['RELOAD'] = True
        
        return values
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.ENVIRONMENT == Environment.TESTING or self.TESTING
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins based on environment."""
        if self.is_production:
            # Filter out localhost origins in production
            return [origin for origin in self.ALLOWED_ORIGINS if 'localhost' not in origin]
        return self.ALLOWED_ORIGINS
    
    def get_log_level(self) -> str:
        """Get appropriate log level based on environment."""
        if self.is_production:
            return "INFO"
        elif self.DEBUG:
            return "DEBUG"
        else:
            return "INFO"
    
    def mask_sensitive_data(self) -> Dict[str, Any]:
        """Return configuration dict with sensitive data masked."""
        config_dict = self.dict()
        
        # Mask sensitive fields
        sensitive_fields = [
            'SECRET_KEY', 'DB_PASSWORD', 'REDIS_PASSWORD', 
            'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'PERPLEXITY_API_KEY',
            'AWS_SECRET_ACCESS_KEY', 'ENCRYPTION_KEY'
        ]
        
        def mask_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively mask sensitive data in nested dictionaries."""
            masked = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    masked[key] = mask_nested_dict(value)
                elif any(sensitive in key.upper() for sensitive in sensitive_fields):
                    if value:
                        masked[key] = f"{str(value)[:4]}***{str(value)[-4:]}"
                    else:
                        masked[key] = value
                else:
                    masked[key] = value
            return masked
        
        return mask_nested_dict(config_dict)


# =============================================================================
# SETTINGS FACTORY AND CACHING
# =============================================================================
@lru_cache()
def get_settings() -> Settings:
    """
    Create and cache application settings.
    
    Uses LRU cache to ensure settings are loaded only once and
    reused across the application lifecycle.
    
    Returns:
        Settings: Configured application settings
    """
    try:
        settings = Settings()
        
        # Log configuration summary (with masked sensitive data)
        masked_config = settings.mask_sensitive_data()
        logger.info(
            "Application configuration loaded",
            environment=settings.ENVIRONMENT,
            debug=settings.DEBUG,
            app_name=settings.APP_NAME,
            version=settings.APP_VERSION
        )
        
        # Validate critical dependencies
        if settings.is_production:
            _validate_production_config(settings)
        
        return settings
        
    except Exception as e:
        logger.error("Failed to load application configuration", error=str(e))
        raise


def _validate_production_config(settings: Settings) -> None:
    """Validate production-specific configuration requirements."""
    critical_checks = []
    
    # Database SSL in production
    if settings.database.DB_SSL_MODE == "disable":
        critical_checks.append("Database SSL should be enabled in production")
    
    # Redis SSL in production
    if not settings.redis.REDIS_SSL and settings.is_production:
        critical_checks.append("Redis SSL should be enabled in production")
    
    # Strong secret key
    if len(settings.security.SECRET_KEY) < 32:
        critical_checks.append("SECRET_KEY should be at least 32 characters in production")
    
    # Monitoring configured
    if not settings.monitoring.SENTRY_DSN:
        critical_checks.append("SENTRY_DSN should be configured for production monitoring")
    
    if critical_checks:
        logger.warning(
            "Production configuration warnings",
            warnings=critical_checks
        )


def reload_settings() -> Settings:
    """
    Force reload of application settings.
    
    Clears the LRU cache and reloads settings from environment.
    Useful for configuration updates during runtime.
    
    Returns:
        Settings: Freshly loaded application settings
    """
    get_settings.cache_clear()
    return get_settings()


# =============================================================================
# CONFIGURATION VALIDATION UTILITIES
# =============================================================================
def validate_environment_variables() -> Dict[str, Any]:
    """
    Validate that all required environment variables are present.
    
    Returns:
        Dict containing validation results and missing variables
    """
    required_vars = [
        "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD",
        "SECRET_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PERPLEXITY_API_KEY"
    ]
    
    missing_vars = []
    present_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            present_vars.append(var)
        else:
            missing_vars.append(var)
    
    return {
        "all_present": len(missing_vars) == 0,
        "missing_variables": missing_vars,
        "present_variables": present_vars,
        "total_required": len(required_vars)
    }


# =============================================================================
# EXPORT SETTINGS INSTANCE
# =============================================================================
# Global settings instance for easy import
settings = get_settings()


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Validate environment
    validation_result = validate_environment_variables()
    print(f"Environment validation: {validation_result}")
    
    # Load and display settings
    app_settings = get_settings()
    print(f"Application: {app_settings.APP_NAME} v{app_settings.APP_VERSION}")
    print(f"Environment: {app_settings.ENVIRONMENT}")
    print(f"Database URL: {app_settings.database.database_url}")
    print(f"Redis URL: {app_settings.redis.redis_url}")
    
    # Display masked configuration
    masked_config = app_settings.mask_sensitive_data()
    print("Masked configuration loaded successfully")
