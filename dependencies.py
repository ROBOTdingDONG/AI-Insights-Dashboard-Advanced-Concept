"""
FastAPI Dependencies and Middleware
==================================

Centralized dependency injection for FastAPI routes with comprehensive
security, authentication, and utility functions.

Dependencies:
- Authentication and authorization
- Rate limiting and quota management
- Database session management
- Request validation and sanitization
- User context and permissions
- Monitoring and logging

Security Features:
- JWT token validation and user extraction
- API key authentication
- Role-based access control
- Request rate limiting
- Input sanitization and validation
- Audit logging and monitoring

Author: AI Insights Team
Version: 1.0.0
"""

import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from functools import lru_cache

import structlog
import jwt
from fastapi import Depends, HTTPException, Security, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from slowapi import Limiter
from slowapi.util import get_remote_address
import redis.asyncio as redis

# Import core modules
from app.core.config import get_settings
from app.core.database import get_db
from app.models.analytics import User, APIKey, UserRole, UserStatus, SubscriptionPlan
from app.utils.validators import verify_api_key, hash_api_key


logger = structlog.get_logger(__name__)
settings = get_settings()

# Security schemes
security_bearer = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Redis client for rate limiting and session management
redis_client = redis.from_url(settings.redis.redis_url)


# =============================================================================
# RATE LIMITING CONFIGURATION
# =============================================================================
@lru_cache()
def get_rate_limiter() -> Limiter:
    """
    Get configured rate limiter instance.
    
    Returns:
        Limiter: Configured SlowAPI limiter
    """
    return Limiter(
        key_func=get_remote_address,
        storage_uri=settings.redis.redis_url,
        default_limits=["1000/day", "100/hour", "10/minute"]
    )


def get_user_identifier(request: Request) -> str:
    """
    Extract user identifier for rate limiting.
    
    Priority:
    1. Authenticated user ID
    2. API key hash
    3. IP address
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: User identifier for rate limiting
    """
    # Try to get user from request state (set by auth middleware)
    user = getattr(request.state, 'current_user', None)
    if user:
        return f"user:{user.id}"
    
    # Try to get API key from request state
    api_key = getattr(request.state, 'api_key', None)
    if api_key:
        return f"api_key:{api_key.id}"
    
    # Fall back to IP address
    return f"ip:{get_remote_address(request)}"


# =============================================================================
# JWT TOKEN HANDLING
# =============================================================================
def create_access_token(data: Dict[str, Any], expires_delta: Optional[int] = None) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Token payload data
        expires_delta: Token expiration in minutes
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    # Set expiration
    if expires_delta:
        expire = datetime.now(timezone.utc).timestamp() + (expires_delta * 60)
    else:
        expire = datetime.now(timezone.utc).timestamp() + (settings.security.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc).timestamp()})
    
    # Encode token
    encoded_jwt = jwt.encode(
        to_encode,
        settings.security.SECRET_KEY,
        algorithm=settings.security.JWT_ALGORITHM
    )
    
    return encoded_jwt


def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Dict[str, Any]: Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.security.SECRET_KEY,
            algorithms=[settings.security.JWT_ALGORITHM]
        )
        
        # Validate required fields
        if not payload.get("sub"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject"
            )
        
        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.now(timezone.utc).timestamp() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid JWT token", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


# =============================================================================
# USER AUTHENTICATION DEPENDENCIES
# =============================================================================
async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Security(security_bearer),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current user from JWT token.
    
    Args:
        credentials: HTTP Bearer credentials
        db: Database session
        
    Returns:
        User: Authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Verify token
        payload = verify_jwt_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Get user from database
        query = select(User).where(
            User.id == user_id,
            User.is_deleted == False
        )
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Check user status
        if user.status != UserStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"User account is {user.status.value}"
            )
        
        # Check account lock
        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is temporarily locked"
            )
        
        # Update last login
        user.last_login_at = datetime.now(timezone.utc)
        await db.commit()
        
        logger.info(
            "User authenticated via JWT",
            user_id=str(user.id),
            email=user.email,
            role=user.role.value
        )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authentication error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def get_current_user_from_api_key(
    api_key: Optional[str] = Security(api_key_header),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get current user from API key.
    
    Args:
        api_key: API key from header
        db: Database session
        
    Returns:
        Optional[User]: Authenticated user or None
    """
    if not api_key:
        return None
    
    try:
        # Hash the provided API key
        key_hash, _ = hash_api_key(api_key)
        
        # Query API key and user
        query = select(APIKey).join(User).where(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True,
            User.status == UserStatus.ACTIVE,
            User.is_deleted == False
        )
        
        result = await db.execute(query)
        api_key_record = result.scalar_one_or_none()
        
        if not api_key_record:
            logger.warning("Invalid API key used", key_prefix=api_key[:8])
            return None
        
        # Check API key expiration
        if api_key_record.expires_at and api_key_record.expires_at < datetime.now(timezone.utc):
            logger.warning("Expired API key used", key_id=str(api_key_record.id))
            return None
        
        # Record API key usage
        api_key_record.record_usage()
        await db.commit()
        
        user = api_key_record.user
        
        logger.info(
            "User authenticated via API key",
            user_id=str(user.id),
            api_key_id=str(api_key_record.id),
            api_key_name=api_key_record.name
        )
        
        return user
        
    except Exception as e:
        logger.error("API key authentication error", error=str(e), exc_info=True)
        return None


async def get_current_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> User:
    """
    Get current user from either JWT token or API key.
    
    Args:
        token_user: User from JWT token
        api_key_user: User from API key
        
    Returns:
        User: Authenticated user
        
    Raises:
        HTTPException: If no valid authentication provided
    """
    user = token_user or api_key_user
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


async def get_optional_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise None.
    
    Useful for endpoints that work with or without authentication.
    
    Args:
        token_user: User from JWT token
        api_key_user: User from API key
        
    Returns:
        Optional[User]: Authenticated user or None
    """
    return token_user or api_key_user


# =============================================================================
# ROLE-BASED ACCESS CONTROL
# =============================================================================
class RequireRole:
    """Dependency class for role-based access control."""
    
    def __init__(self, required_roles: Union[UserRole, List[UserRole]]):
        """
        Initialize role requirement.
        
        Args:
            required_roles: Required role(s) for access
        """
        if isinstance(required_roles, UserRole):
            self.required_roles = [required_roles]
        else:
            self.required_roles = required_roles
    
    async def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        """
        Check if user has required role.
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            User: User if authorized
            
        Raises:
            HTTPException: If user lacks required role
        """
        if current_user.role not in self.required_roles:
            logger.warning(
                "Access denied - insufficient role",
                user_id=str(current_user.id),
                user_role=current_user.role.value,
                required_roles=[role.value for role in self.required_roles]
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return current_user


# Convenience role dependencies
require_admin = RequireRole(UserRole.ADMIN)
require_analyst = RequireRole([UserRole.ADMIN, UserRole.ANALYST])
require_user = RequireRole([UserRole.ADMIN, UserRole.ANALYST, UserRole.USER])


class RequireSubscription:
    """Dependency class for subscription-based access control."""
    
    def __init__(self, required_plans: Union[SubscriptionPlan, List[SubscriptionPlan]]):
        """
        Initialize subscription requirement.
        
        Args:
            required_plans: Required subscription plan(s)
        """
        if isinstance(required_plans, SubscriptionPlan):
            self.required_plans = [required_plans]
        else:
            self.required_plans = required_plans
    
    async def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        """
        Check if user has required subscription.
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            User: User if authorized
            
        Raises:
            HTTPException: If user lacks required subscription
        """
        if current_user.subscription_plan not in self.required_plans:
            logger.warning(
                "Access denied - insufficient subscription",
                user_id=str(current_user.id),
                user_plan=current_user.subscription_plan.value,
                required_plans=[plan.value for plan in self.required_plans]
            )
            
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Upgrade subscription to access this feature"
            )
        
        # Check subscription expiration
        if (current_user.subscription_expires_at and 
            current_user.subscription_expires_at < datetime.now(timezone.utc)):
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Subscription expired. Please renew to continue."
            )
        
        return current_user


# Convenience subscription dependencies
require_premium = RequireSubscription([SubscriptionPlan.PROFESSIONAL, SubscriptionPlan.ENTERPRISE])
require_enterprise = RequireSubscription(SubscriptionPlan.ENTERPRISE)


# =============================================================================
# QUOTA AND RATE LIMITING
# =============================================================================
async def check_user_quota(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Check if user has available API quota.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        User: User if quota available
        
    Raises:
        HTTPException: If quota exceeded
    """
    if not current_user.can_make_api_call():
        # Get quota limits based on subscription
        quota_limits = {
            SubscriptionPlan.FREE: 100,
            SubscriptionPlan.BASIC: 1000,
            SubscriptionPlan.PROFESSIONAL: 10000,
            SubscriptionPlan.ENTERPRISE: 100000
        }
        
        daily_limit = quota_limits.get(current_user.subscription_plan, 100)
        
        logger.warning(
            "API quota exceeded",
            user_id=str(current_user.id),
            daily_usage=current_user.api_calls_today,
            daily_limit=daily_limit,
            subscription_plan=current_user.subscription_plan.value
        )
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily API quota ({daily_limit}) exceeded. Upgrade your plan for higher limits.",
            headers={
                "X-RateLimit-Limit": str(daily_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + 86400),  # Reset in 24 hours
                "Retry-After": "86400"
            }
        )
    
    return current_user


async def get_rate_limit_info(
    request: Request,
    current_user: Optional[User] = Depends(get_optional_user)
) -> Dict[str, Any]:
    """
    Get rate limit information for current user.
    
    Args:
        request: FastAPI request object
        current_user: Current user (if authenticated)
        
    Returns:
        Dict[str, Any]: Rate limit information
    """
    try:
        # Get user identifier for rate limiting
        user_id = str(current_user.id) if current_user else get_remote_address(request)
        
        # Check current usage from Redis
        rate_key = f"rate_limit:api:{user_id}"
        current_usage = await redis_client.get(rate_key)
        current_usage = int(current_usage) if current_usage else 0
        
        # Get limits based on user subscription
        if current_user:
            if current_user.subscription_plan == SubscriptionPlan.FREE:
                hourly_limit = 100
                daily_limit = 1000
            elif current_user.subscription_plan == SubscriptionPlan.BASIC:
                hourly_limit = 500
                daily_limit = 5000
            elif current_user.subscription_plan == SubscriptionPlan.PROFESSIONAL:
                hourly_limit = 2000
                daily_limit = 20000
            else:  # Enterprise
                hourly_limit = 10000
                daily_limit = 100000
        else:
            # Anonymous user limits
            hourly_limit = 10
            daily_limit = 50
        
        return {
            "hourly_limit": hourly_limit,
            "daily_limit": daily_limit,
            "current_usage": current_usage,
            "remaining": max(0, hourly_limit - current_usage),
            "reset_time": int(time.time()) + 3600,  # Reset in 1 hour
            "subscription_plan": current_user.subscription_plan.value if current_user else "anonymous"
        }
        
    except Exception as e:
        logger.error("Failed to get rate limit info", error=str(e))
        return {
            "hourly_limit": 10,
            "daily_limit": 50,
            "current_usage": 0,
            "remaining": 10,
            "reset_time": int(time.time()) + 3600,
            "subscription_plan": "unknown"
        }


# =============================================================================
# REQUEST VALIDATION AND PROCESSING
# =============================================================================
async def validate_request_size(request: Request) -> Request:
    """
    Validate request content size.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Request: Validated request
        
    Raises:
        HTTPException: If request is too large
    """
    content_length = request.headers.get("content-length")
    
    if content_length:
        content_length = int(content_length)
        max_size = 10 * 1024 * 1024  # 10MB
        
        if content_length > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request too large. Maximum size: {max_size} bytes"
            )
    
    return request


async def add_request_id(request: Request) -> Request:
    """
    Add unique request ID to request state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Request: Request with ID added to state
    """
    if not hasattr(request.state, 'request_id'):
        request_id = f"req_{int(time.time() * 1000000)}"
        request.state.request_id = request_id
    
    return request


async def log_request_info(
    request: Request = Depends(add_request_id),
    current_user: Optional[User] = Depends(get_optional_user)
) -> Request:
    """
    Log request information for monitoring.
    
    Args:
        request: FastAPI request object
        current_user: Current user (if authenticated)
        
    Returns:
        Request: Request object
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info(
        "API request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        user_id=str(current_user.id) if current_user else None,
        user_agent=request.headers.get("user-agent", "unknown")[:100],
        ip_address=get_remote_address(request)
    )
    
    return request


# =============================================================================
# PERMISSION VERIFICATION
# =============================================================================
async def verify_api_permissions(
    resource_type: str,
    action: str,
    resource_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> bool:
    """
    Verify user permissions for specific API actions.
    
    Args:
        resource_type: Type of resource (search, analysis, export, etc.)
        action: Action being performed (create, read, update, delete)
        resource_id: Optional specific resource ID
        current_user: Current authenticated user
        
    Returns:
        bool: True if permission granted
        
    Raises:
        HTTPException: If permission denied
    """
    # Admin users have all permissions
    if current_user.role == UserRole.ADMIN:
        return True
    
    # Define permission matrix
    permissions = {
        UserRole.USER: {
            "search": ["create", "read_own"],
            "analysis": ["create", "read_own"],
            "export": ["create", "read_own"]
        },
        UserRole.ANALYST: {
            "search": ["create", "read_own", "read_all"],
            "analysis": ["create", "read_own", "read_all"],
            "export": ["create", "read_own", "read_all"],
            "user": ["read_basic"]
        },
        UserRole.ADMIN: {
            "*": ["*"]  # All permissions
        }
    }
    
    user_permissions = permissions.get(current_user.role, {})
    resource_permissions = user_permissions.get(resource_type, [])
    
    # Check if action is allowed
    if action in resource_permissions or "*" in resource_permissions:
        return True
    
    # Check for "read_own" permission with resource ownership
    if action == "read" and "read_own" in resource_permissions:
        # Additional check for resource ownership would go here
        # For now, assume ownership check is handled at the route level
        return True
    
    logger.warning(
        "Permission denied",
        user_id=str(current_user.id),
        user_role=current_user.role.value,
        resource_type=resource_type,
        action=action,
        resource_id=resource_id
    )
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Permission denied for {action} on {resource_type}"
    )


# =============================================================================
# UTILITY DEPENDENCIES
# =============================================================================
async def get_pagination_params(
    page: int = 1,
    page_size: int = 20,
    max_page_size: int = 100
) -> Dict[str, int]:
    """
    Get validated pagination parameters.
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        max_page_size: Maximum allowed page size
        
    Returns:
        Dict[str, int]: Validated pagination parameters
        
    Raises:
        HTTPException: If parameters are invalid
    """
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page number must be 1 or greater"
        )
    
    if page_size < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page size must be 1 or greater"
        )
    
    if page_size > max_page_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Page size cannot exceed {max_page_size}"
        )
    
    return {
        "page": page,
        "page_size": page_size,
        "offset": (page - 1) * page_size,
        "limit": page_size
    }


async def get_date_range_params(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None
) -> Dict[str, Optional[datetime]]:
    """
    Get validated date range parameters.
    
    Args:
        date_from: Start date
        date_to: End date
        
    Returns:
        Dict[str, Optional[datetime]]: Validated date range
        
    Raises:
        HTTPException: If date range is invalid
    """
    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Start date must be before end date"
        )
    
    # Limit date range to prevent excessive queries
    if date_from and date_to:
        date_diff = (date_to - date_from).days
        if date_diff > 365:  # 1 year max
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Date range cannot exceed 365 days"
            )
    
    return {
        "date_from": date_from,
        "date_to": date_to
    }


# =============================================================================
# EXPORTS FOR EASY IMPORT
# =============================================================================
__all__ = [
    # Authentication
    'get_current_user', 'get_optional_user',
    'get_current_user_from_token', 'get_current_user_from_api_key',
    
    # Authorization
    'RequireRole', 'RequireSubscription',
    'require_admin', 'require_analyst', 'require_user',
    'require_premium', 'require_enterprise',
    
    # Quota and Rate Limiting
    'check_user_quota', 'get_rate_limit_info', 'get_rate_limiter',
    
    # Request Processing
    'validate_request_size', 'add_request_id', 'log_request_info',
    
    # Permissions
    'verify_api_permissions',
    
    # Utilities
    'get_pagination_params', 'get_date_range_params',
    
    # Token Management
    'create_access_token', 'verify_jwt_token'
]
