"""
Search API Routes
================

RESTful API endpoints for search functionality with Perplexity integration.

Endpoints:
- POST /search - Perform new search query
- GET /search/{search_id} - Retrieve search results by ID
- GET /search/trending - Get trending topics
- GET /search/history - Get user's search history
- DELETE /search/{search_id} - Delete search record
- POST /search/batch - Batch search processing

Features:
- Comprehensive input validation and sanitization
- Rate limiting and authentication
- Response caching and optimization
- Performance monitoring and logging
- Error handling with detailed responses
- Pagination and filtering support

Security:
- Request validation and sanitization
- Rate limiting per user and endpoint
- Authentication required for all endpoints
- Input size limits and content filtering
- Audit logging for all operations

Author: AI Insights Team
Version: 1.0.0
"""

import asyncio
import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Query, Path, Body
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from slowapi import Limiter
from slowapi.util import get_remote_address
import redis.asyncio as redis

# Import models and dependencies
from app.core.database import get_db
from app.core.config import get_settings
from app.models.analytics import (
    User, SearchQuery, SearchResult, QueryStatus,
    SourceType, CredibilityScore
)
from app.services.perplexity import PerplexityService, SearchResponse
from app.utils.validators import (
    SearchRequest, sanitize_text, is_malicious_input,
    generate_rate_limit_key, normalize_search_query
)
from app.api.dependencies import (
    get_current_user, verify_api_permissions,
    get_rate_limiter, check_user_quota
)


logger = structlog.get_logger(__name__)
settings = get_settings()

# Initialize router
router = APIRouter()

# Rate limiter
limiter = get_rate_limiter()

# Redis client for caching
redis_client = redis.from_url(settings.redis.redis_url)


# =============================================================================
# RESPONSE MODELS
# =============================================================================
class SearchResponseModel(BaseModel):
    """Search response model for API."""
    
    search_id: str
    query: str
    status: str
    total_results: int
    search_time_ms: float
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    pagination: Optional[Dict[str, Any]] = None
    cached: bool = False


class SearchHistoryResponse(BaseModel):
    """Search history response model."""
    
    searches: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int
    has_next: bool


class TrendingTopicsResponse(BaseModel):
    """Trending topics response model."""
    
    topics: List[str]
    categories: Dict[str, List[str]]
    updated_at: datetime


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
async def generate_search_hash(query: str, filters: Dict[str, Any] = None) -> str:
    """Generate unique hash for search deduplication."""
    search_data = {
        "query": normalize_search_query(query),
        "filters": filters or {}
    }
    
    hash_input = str(sorted(search_data.items()))
    return hashlib.sha256(hash_input.encode()).hexdigest()


async def cache_search_results(search_id: str, results: SearchResponse, ttl: int = 3600) -> None:
    """Cache search results in Redis."""
    try:
        cache_key = f"search_results:{search_id}"
        cache_data = {
            "results": results.dict(),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "ttl": ttl
        }
        
        await redis_client.setex(
            cache_key,
            ttl,
            json.dumps(cache_data, default=str)
        )
        
        logger.info("Search results cached", search_id=search_id, ttl=ttl)
        
    except Exception as e:
        logger.warning("Failed to cache search results", error=str(e), search_id=search_id)


async def get_cached_search_results(search_id: str) -> Optional[SearchResponse]:
    """Retrieve cached search results."""
    try:
        cache_key = f"search_results:{search_id}"
        cached_data = await redis_client.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            results = SearchResponse(**data["results"])
            results.cached = True
            
            logger.info("Cache hit for search results", search_id=search_id)
            return results
            
    except Exception as e:
        logger.warning("Failed to retrieve cached results", error=str(e), search_id=search_id)
    
    return None


async def save_search_to_database(
    db: AsyncSession,
    search_request: SearchRequest,
    search_response: SearchResponse,
    user: User,
    request_id: str
) -> SearchQuery:
    """Save search query and results to database."""
    
    # Generate search hash for deduplication
    search_hash = await generate_search_hash(
        search_request.query, 
        search_request.filters
    )
    
    # Create search query record
    search_query = SearchQuery(
        query_text=search_request.query,
        query_hash=search_hash,
        status=QueryStatus.COMPLETED,
        provider="perplexity",
        total_results=search_response.total_results,
        results_cached=search_response.cached,
        cache_hit=search_response.cached,
        search_time_ms=search_response.search_time_ms,
        processing_time_ms=0.0,  # Will be calculated
        request_id=request_id,
        user_id=user.id if user else None,
        raw_results=search_response.dict(),
        processed_results=None  # Will be populated by analysis
    )
    
    db.add(search_query)
    await db.flush()  # Get the ID
    
    # Save individual search results
    for i, result in enumerate(search_response.results):
        search_result = SearchResult(
            title=result.title,
            url=result.url,
            snippet=result.snippet,
            source_domain=result.source_domain,
            source_type=result.source_type,
            published_date=result.published_date,
            credibility_score=result.credibility_score,
            relevance_score=result.relevance_score,
            result_position=i + 1,
            query_id=search_query.id,
            content_hash=hashlib.sha256(result.snippet.encode()).hexdigest()[:16]
        )
        db.add(search_result)
    
    await db.commit()
    
    logger.info(
        "Search saved to database",
        search_id=str(search_query.id),
        query=search_request.query,
        results_count=search_response.total_results,
        user_id=str(user.id) if user else None
    )
    
    return search_query


# =============================================================================
# SEARCH ENDPOINTS
# =============================================================================
@router.post("/search", response_model=SearchResponseModel)
@limiter.limit("30/minute")
async def perform_search(
    request: Request,
    search_request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Perform intelligent search with external API integration.
    
    **Features:**
    - Real-time search with Perplexity API
    - Automatic result caching and deduplication
    - Source credibility scoring
    - Comprehensive result tracking
    
    **Rate Limits:**
    - 30 requests per minute per user
    - Additional quota limits based on subscription plan
    
    **Request Body:**
    - query: Search query (2-500 characters)
    - max_results: Maximum results to return (1-50)
    - filters: Optional search filters
    - use_cache: Whether to use cached results
    """
    
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', f"search_{int(time.time() * 1000)}")
    
    logger.info(
        "Search request initiated",
        query=search_request.query,
        max_results=search_request.max_results,
        user_id=str(current_user.id),
        request_id=request_id
    )
    
    try:
        # Check user quota
        if not current_user.can_make_api_call():
            raise HTTPException(
                status_code=429,
                detail="API quota exceeded for your subscription plan",
                headers={"Retry-After": "3600"}
            )
        
        # Additional input validation
        if is_malicious_input(search_request.query):
            raise HTTPException(
                status_code=400,
                detail="Invalid search query detected"
            )
        
        # Check for existing search (deduplication)
        search_hash = await generate_search_hash(
            search_request.query,
            search_request.filters
        )
        
        if search_request.use_cache:
            # Check database for recent identical search
            recent_search = await db.execute(
                select(SearchQuery)
                .where(
                    and_(
                        SearchQuery.query_hash == search_hash,
                        SearchQuery.user_id == current_user.id,
                        SearchQuery.created_at > datetime.now(timezone.utc) - timedelta(hours=1),
                        SearchQuery.status == QueryStatus.COMPLETED
                    )
                )
                .order_by(desc(SearchQuery.created_at))
                .limit(1)
            )
            
            existing_search = recent_search.scalar_one_or_none()
            if existing_search:
                # Return cached database result
                cached_results = existing_search.raw_results
                if cached_results:
                    processing_time = (time.time() - start_time) * 1000
                    
                    response = SearchResponseModel(
                        search_id=str(existing_search.id),
                        query=search_request.query,
                        status="completed",
                        total_results=existing_search.total_results,
                        search_time_ms=existing_search.search_time_ms,
                        results=cached_results.get("results", []),
                        metadata={
                            "provider": "perplexity",
                            "model": cached_results.get("model", "unknown"),
                            "processing_time_ms": processing_time,
                            "from_cache": True
                        },
                        cached=True
                    )
                    
                    logger.info(
                        "Returning cached search result",
                        search_id=str(existing_search.id),
                        query=search_request.query,
                        user_id=str(current_user.id)
                    )
                    
                    return response
        
        # Perform new search with Perplexity API
        async with PerplexityService() as perplexity:
            search_response = await perplexity.search(
                query=search_request.query,
                max_results=search_request.max_results,
                user_id=str(current_user.id),
                request_id=request_id,
                use_cache=search_request.use_cache
            )
        
        # Save to database
        search_query = await save_search_to_database(
            db, search_request, search_response, current_user, request_id
        )
        
        # Cache results
        await cache_search_results(str(search_query.id), search_response)
        
        # Update user API usage
        current_user.increment_api_usage()
        await db.commit()
        
        # Calculate total processing time
        processing_time = (time.time() - start_time) * 1000
        search_query.processing_time_ms = processing_time
        search_query.calculate_total_time()
        await db.commit()
        
        # Format response
        response = SearchResponseModel(
            search_id=str(search_query.id),
            query=search_request.query,
            status="completed",
            total_results=search_response.total_results,
            search_time_ms=search_response.search_time_ms,
            results=[
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "source_domain": result.source_domain,
                    "source_type": result.source_type.value,
                    "credibility_score": result.credibility_score.value,
                    "relevance_score": result.relevance_score,
                    "published_date": result.published_date.isoformat() if result.published_date else None
                }
                for result in search_response.results
            ],
            metadata={
                "provider": "perplexity",
                "processing_time_ms": processing_time,
                "total_time_ms": search_query.total_time_ms,
                "request_id": request_id,
                "from_cache": search_response.cached
            },
            cached=search_response.cached
        )
        
        logger.info(
            "Search completed successfully",
            search_id=str(search_query.id),
            query=search_request.query,
            results_count=search_response.total_results,
            duration_ms=processing_time,
            user_id=str(current_user.id)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Search request failed",
            query=search_request.query,
            error=str(e),
            user_id=str(current_user.id),
            request_id=request_id,
            exc_info=True
        )
        
        # Save failed search to database for tracking
        try:
            failed_search = SearchQuery(
                query_text=search_request.query,
                query_hash=await generate_search_hash(search_request.query),
                status=QueryStatus.FAILED,
                error_message=str(e),
                request_id=request_id,
                user_id=current_user.id
            )
            db.add(failed_search)
            await db.commit()
        except:
            pass  # Don't fail on database error
        
        raise HTTPException(
            status_code=500,
            detail="Search request failed. Please try again."
        )


@router.get("/search/{search_id}", response_model=SearchResponseModel)
@limiter.limit("60/minute")
async def get_search_results(
    request: Request,
    search_id: UUID = Path(..., description="Search query ID"),
    include_results: bool = Query(True, description="Include full results in response"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve search results by search ID.
    
    **Parameters:**
    - search_id: UUID of the search query
    - include_results: Whether to include full search results
    
    **Access Control:**
    - Users can only access their own search results
    - Admins can access any search results
    """
    
    try:
        # Check cache first
        cached_results = await get_cached_search_results(str(search_id))
        if cached_results and include_results:
            response = SearchResponseModel(
                search_id=str(search_id),
                query=cached_results.query,
                status="completed",
                total_results=cached_results.total_results,
                search_time_ms=cached_results.search_time_ms,
                results=[
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "source_domain": result.source_domain,
                        "source_type": result.source_type.value,
                        "credibility_score": result.credibility_score.value,
                        "relevance_score": result.relevance_score,
                        "published_date": result.published_date.isoformat() if result.published_date else None
                    }
                    for result in cached_results.results
                ],
                metadata={"from_cache": True},
                cached=True
            )
            return response
        
        # Query database
        query = select(SearchQuery).where(SearchQuery.id == search_id)
        
        # Access control: users can only see their own searches
        if current_user.role.value != "admin":
            query = query.where(SearchQuery.user_id == current_user.id)
        
        result = await db.execute(query)
        search_query = result.scalar_one_or_none()
        
        if not search_query:
            raise HTTPException(
                status_code=404,
                detail="Search not found or access denied"
            )
        
        # Get search results if requested
        results_data = []
        if include_results and search_query.status == QueryStatus.COMPLETED:
            if search_query.raw_results:
                # Use stored raw results
                raw_results = search_query.raw_results.get("results", [])
                results_data = [
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("snippet", ""),
                        "source_domain": result.get("source_domain", ""),
                        "source_type": result.get("source_type", "unknown"),
                        "credibility_score": result.get("credibility_score", "low"),
                        "relevance_score": result.get("relevance_score", 0.0),
                        "published_date": result.get("published_date")
                    }
                    for result in raw_results
                ]
            else:
                # Query individual results from database
                results_query = select(SearchResult).where(
                    SearchResult.query_id == search_id
                ).order_by(SearchResult.result_position)
                
                results_result = await db.execute(results_query)
                search_results = results_result.scalars().all()
                
                results_data = [
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "source_domain": result.source_domain,
                        "source_type": result.source_type.value,
                        "credibility_score": result.credibility_score.value,
                        "relevance_score": result.relevance_score,
                        "published_date": result.published_date.isoformat() if result.published_date else None
                    }
                    for result in search_results
                ]
        
        response = SearchResponseModel(
            search_id=str(search_query.id),
            query=search_query.query_text,
            status=search_query.status.value,
            total_results=search_query.total_results,
            search_time_ms=search_query.search_time_ms or 0.0,
            results=results_data if include_results else [],
            metadata={
                "provider": search_query.provider,
                "processing_time_ms": search_query.processing_time_ms,
                "total_time_ms": search_query.total_time_ms,
                "request_id": search_query.request_id,
                "created_at": search_query.created_at.isoformat(),
                "error_message": search_query.error_message
            },
            cached=False
        )
        
        logger.info(
            "Search results retrieved",
            search_id=str(search_id),
            user_id=str(current_user.id),
            include_results=include_results
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve search results",
            search_id=str(search_id),
            error=str(e),
            user_id=str(current_user.id),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve search results"
        )


@router.get("/search", response_model=SearchHistoryResponse)
@limiter.limit("30/minute")
async def get_search_history(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    date_from: Optional[datetime] = Query(None, description="Filter from date"),
    date_to: Optional[datetime] = Query(None, description="Filter to date"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get user's search history with pagination and filtering.
    
    **Query Parameters:**
    - page: Page number (default: 1)
    - page_size: Results per page (1-100, default: 20)
    - status_filter: Filter by search status
    - date_from: Filter searches from this date
    - date_to: Filter searches to this date
    """
    
    try:
        # Build query
        query = select(SearchQuery).where(SearchQuery.user_id == current_user.id)
        
        # Apply filters
        if status_filter:
            try:
                status_enum = QueryStatus(status_filter)
                query = query.where(SearchQuery.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status filter: {status_filter}"
                )
        
        if date_from:
            query = query.where(SearchQuery.created_at >= date_from)
        
        if date_to:
            query = query.where(SearchQuery.created_at <= date_to)
        
        # Count total results
        count_query = select(func.count(SearchQuery.id)).where(
            SearchQuery.user_id == current_user.id
        )
        if status_filter:
            count_query = count_query.where(SearchQuery.status == QueryStatus(status_filter))
        if date_from:
            count_query = count_query.where(SearchQuery.created_at >= date_from)
        if date_to:
            count_query = count_query.where(SearchQuery.created_at <= date_to)
        
        count_result = await db.execute(count_query)
        total_count = count_result.scalar()
        
        # Apply pagination and ordering
        query = query.order_by(desc(SearchQuery.created_at))
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        searches = result.scalars().all()
        
        # Format response
        searches_data = [
            {
                "search_id": str(search.id),
                "query": search.query_text,
                "status": search.status.value,
                "total_results": search.total_results,
                "search_time_ms": search.search_time_ms,
                "created_at": search.created_at.isoformat(),
                "error_message": search.error_message
            }
            for search in searches
        ]
        
        response = SearchHistoryResponse(
            searches=searches_data,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=(page * page_size) < total_count
        )
        
        logger.info(
            "Search history retrieved",
            user_id=str(current_user.id),
            page=page,
            page_size=page_size,
            total_count=total_count
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve search history",
            error=str(e),
            user_id=str(current_user.id),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve search history"
        )


@router.get("/trending", response_model=TrendingTopicsResponse)
@limiter.limit("10/minute")
async def get_trending_topics(
    request: Request,
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(10, ge=1, le=50, description="Number of topics to return")
):
    """
    Get trending search topics and categories.
    
    **Query Parameters:**
    - category: Optional category filter
    - limit: Number of topics to return (1-50)
    
    **Note:** This endpoint returns cached trending data updated periodically.
    """
    
    try:
        # Check cache for trending topics
        cache_key = f"trending_topics:{category or 'all'}:{limit}"
        cached_data = await redis_client.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            response = TrendingTopicsResponse(**data)
            
            logger.info(
                "Trending topics retrieved from cache",
                category=category,
                limit=limit
            )
            
            return response
        
        # Get trending topics from Perplexity service
        async with PerplexityService() as perplexity:
            trending_topics = await perplexity.get_trending_topics(category)
        
        # Limit results
        if len(trending_topics) > limit:
            trending_topics = trending_topics[:limit]
        
        # Categorize topics (simple implementation)
        categories = {
            "technology": [t for t in trending_topics if any(keyword in t.lower() for keyword in ["ai", "tech", "digital", "software"])],
            "health": [t for t in trending_topics if any(keyword in t.lower() for keyword in ["health", "medical", "disease", "treatment"])],
            "business": [t for t in trending_topics if any(keyword in t.lower() for keyword in ["business", "economy", "market", "finance"])],
            "science": [t for t in trending_topics if any(keyword in t.lower() for keyword in ["science", "research", "study", "discovery"])]
        }
        
        response = TrendingTopicsResponse(
            topics=trending_topics,
            categories=categories,
            updated_at=datetime.now(timezone.utc)
        )
        
        # Cache for 30 minutes
        await redis_client.setex(
            cache_key,
            1800,
            json.dumps(response.dict(), default=str)
        )
        
        logger.info(
            "Trending topics retrieved",
            category=category,
            topics_count=len(trending_topics)
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Failed to retrieve trending topics",
            error=str(e),
            category=category,
            exc_info=True
        )
        
        # Return fallback trending topics
        fallback_topics = [
            "Artificial Intelligence regulation",
            "Climate change solutions",
            "Quantum computing advances",
            "Renewable energy trends",
            "Space exploration"
        ]
        
        return TrendingTopicsResponse(
            topics=fallback_topics[:limit],
            categories={"technology": fallback_topics[:3], "science": fallback_topics[3:]},
            updated_at=datetime.now(timezone.utc)
        )


@router.delete("/search/{search_id}")
@limiter.limit("20/minute")
async def delete_search(
    request: Request,
    search_id: UUID = Path(..., description="Search query ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Soft delete a search record.
    
    **Access Control:**
    - Users can only delete their own search records
    - Admins can delete any search records
    """
    
    try:
        # Query search record
        query = select(SearchQuery).where(SearchQuery.id == search_id)
        
        # Access control
        if current_user.role.value != "admin":
            query = query.where(SearchQuery.user_id == current_user.id)
        
        result = await db.execute(query)
        search_query = result.scalar_one_or_none()
        
        if not search_query:
            raise HTTPException(
                status_code=404,
                detail="Search not found or access denied"
            )
        
        # Soft delete
        search_query.soft_delete()
        await db.commit()
        
        # Remove from cache
        cache_key = f"search_results:{search_id}"
        await redis_client.delete(cache_key)
        
        logger.info(
            "Search deleted",
            search_id=str(search_id),
            user_id=str(current_user.id)
        )
        
        return {"message": "Search deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete search",
            search_id=str(search_id),
            error=str(e),
            user_id=str(current_user.id),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail="Failed to delete search"
        )


# =============================================================================
# BATCH OPERATIONS
# =============================================================================
@router.post("/search/batch")
@limiter.limit("5/minute")
async def batch_search(
    request: Request,
    search_requests: List[SearchRequest] = Body(..., max_items=10),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Perform multiple searches in batch.
    
    **Limitations:**
    - Maximum 10 searches per batch
    - Rate limited to 5 batches per minute
    - Each search counts toward user quota
    """
    
    request_id = getattr(request.state, 'request_id', f"batch_{int(time.time() * 1000)}")
    
    if len(search_requests) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 searches allowed per batch"
        )
    
    # Check quota for all searches
    required_quota = len(search_requests)
    if current_user.api_calls_today + required_quota > current_user.can_make_api_call():
        raise HTTPException(
            status_code=429,
            detail="Insufficient API quota for batch operation"
        )
    
    logger.info(
        "Batch search initiated",
        batch_size=len(search_requests),
        user_id=str(current_user.id),
        request_id=request_id
    )
    
    results = []
    successful_searches = 0
    
    # Process searches sequentially to avoid overwhelming external APIs
    for i, search_request in enumerate(search_requests):
        try:
            # Perform individual search (reuse existing logic)
            search_response = await perform_search(
                request=request,
                search_request=search_request,
                db=db,
                current_user=current_user
            )
            
            results.append({
                "index": i,
                "status": "success",
                "search_id": search_response.search_id,
                "query": search_request.query,
                "total_results": search_response.total_results
            })
            
            successful_searches += 1
            
        except Exception as e:
            results.append({
                "index": i,
                "status": "error",
                "query": search_request.query,
                "error": str(e)
            })
            
            logger.warning(
                "Batch search item failed",
                index=i,
                query=search_request.query,
                error=str(e)
            )
    
    logger.info(
        "Batch search completed",
        total_requests=len(search_requests),
        successful=successful_searches,
        failed=len(search_requests) - successful_searches,
        user_id=str(current_user.id),
        request_id=request_id
    )
    
    return {
        "batch_id": request_id,
        "total_requests": len(search_requests),
        "successful": successful_searches,
        "failed": len(search_requests) - successful_searches,
        "results": results
    }
