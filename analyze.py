"""
Analysis API Routes
==================

RESTful API endpoints for LLM-powered analysis functionality.

Endpoints:
- POST /analyze - Perform AI analysis on data
- GET /analyze/{analysis_id} - Retrieve analysis results by ID
- GET /analyze/history - Get user's analysis history
- POST /analyze/batch - Batch analysis processing
- POST /analyze/bias-check - Detect bias in content
- GET /analyze/models - Get available models and capabilities
- POST /analyze/compare - Compare multiple analysis results

Features:
- Multi-provider LLM integration (OpenAI, Claude, local models)
- Various analysis types (summarization, trends, sentiment, insights)
- Comprehensive cost tracking and token optimization
- Content safety filtering and bias detection
- Real-time processing with caching optimization
- Detailed confidence scoring and validation

Security:
- Content safety filtering and prompt injection prevention
- Rate limiting and quota management
- Input validation and sanitization
- Audit logging and monitoring
- PII detection and masking

Author: AI Insights Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
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
import redis.asyncio as redis
from pydantic import BaseModel, Field

# Import models and dependencies
from app.core.database import get_db
from app.core.config import get_settings
from app.models.analytics import (
    User, SearchQuery, AnalysisResult, AnalysisType,
    LLMProvider, ContentSafetyRating
)
from app.services.summarizer import (
    LLMSummarizerService, AnalysisRequest as ServiceAnalysisRequest,
    BiasDetectionResult, TokenUsage
)
from app.utils.validators import (
    AnalysisRequest, sanitize_text, detect_prompt_injection,
    detect_pii, mask_pii, filter_harmful_content
)
from app.api.dependencies import (
    get_current_user, get_rate_limiter, check_user_quota
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
class AnalysisResponseModel(BaseModel):
    """Analysis response model for API."""
    
    analysis_id: str
    analysis_type: str
    provider: str
    model: str
    status: str
    content: str
    confidence_score: float
    key_points: List[str]
    entities: List[str]
    sentiment: Optional[str] = None
    safety_rating: str
    bias_detected: bool
    processing_time_ms: float
    token_usage: Optional[Dict[str, Any]] = None
    cost_usd: Optional[float] = None
    metadata: Dict[str, Any]
    cached: bool = False


class AnalysisHistoryResponse(BaseModel):
    """Analysis history response model."""
    
    analyses: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int
    has_next: bool


class ModelCapabilitiesResponse(BaseModel):
    """Model capabilities response."""
    
    providers: Dict[str, Dict[str, Any]]
    analysis_types: List[str]
    rate_limits: Dict[str, int]
    pricing: Dict[str, Dict[str, float]]


class BiasCheckResponse(BaseModel):
    """Bias detection response model."""
    
    has_bias: bool
    bias_types: List[str]
    confidence: float
    flagged_segments: List[str]
    recommendations: List[str]
    safety_rating: str


class ComparisonResponse(BaseModel):
    """Analysis comparison response."""
    
    comparison_id: str
    analyses: List[str]
    comparison_type: str
    results: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
async def generate_analysis_cache_key(
    data_hash: str,
    analysis_type: str,
    provider: str,
    model: str,
    **kwargs
) -> str:
    """Generate cache key for analysis results."""
    cache_components = {
        "data_hash": data_hash,
        "analysis_type": analysis_type,
        "provider": provider,
        "model": model,
        **kwargs
    }
    
    cache_string = json.dumps(cache_components, sort_keys=True)
    return f"analysis:{hashlib.sha256(cache_string.encode()).hexdigest()[:16]}"


async def cache_analysis_result(
    cache_key: str,
    result: AnalysisResult,
    ttl: int = 3600
) -> None:
    """Cache analysis result."""
    try:
        cache_data = {
            "result": result.dict(),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "ttl": ttl
        }
        
        await redis_client.setex(
            cache_key,
            ttl,
            json.dumps(cache_data, default=str)
        )
        
        logger.info("Analysis result cached", cache_key=cache_key[:16], ttl=ttl)
        
    except Exception as e:
        logger.warning("Failed to cache analysis result", error=str(e))


async def get_cached_analysis_result(cache_key: str) -> Optional[AnalysisResult]:
    """Retrieve cached analysis result."""
    try:
        cached_data = await redis_client.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            result = AnalysisResult(**data["result"])
            
            logger.info("Cache hit for analysis result", cache_key=cache_key[:16])
            return result
            
    except Exception as e:
        logger.warning("Failed to retrieve cached analysis", error=str(e))
    
    return None


async def validate_analysis_data(data_source: str, db: AsyncSession, user: User) -> Dict[str, Any]:
    """Validate and retrieve data for analysis."""
    
    # Check if data_source is a search query ID
    try:
        search_id = UUID(data_source)
        
        # Query search results
        query = select(SearchQuery).where(SearchQuery.id == search_id)
        
        # Access control
        if user.role.value != "admin":
            query = query.where(SearchQuery.user_id == user.id)
        
        result = await db.execute(query)
        search_query = result.scalar_one_or_none()
        
        if not search_query:
            raise HTTPException(
                status_code=404,
                detail="Search query not found or access denied"
            )
        
        if search_query.status.value != "completed":
            raise HTTPException(
                status_code=400,
                detail="Search query is not completed"
            )
        
        # Return search data for analysis
        return {
            "type": "search_results",
            "data": search_query.raw_results,
            "search_query_id": str(search_query.id),
            "query_text": search_query.query_text
        }
        
    except ValueError:
        # Not a UUID, treat as direct text data
        if len(data_source) < 10:
            raise HTTPException(
                status_code=400,
                detail="Data source must be at least 10 characters long"
            )
        
        # Validate and sanitize text
        if detect_prompt_injection(data_source):
            raise HTTPException(
                status_code=400,
                detail="Data contains potentially malicious content"
            )
        
        sanitized_data = sanitize_text(data_source, max_length=50000)
        
        return {
            "type": "text_data",
            "data": sanitized_data,
            "search_query_id": None,
            "query_text": None
        }


async def save_analysis_to_database(
    db: AsyncSession,
    analysis_request: AnalysisRequest,
    analysis_result: AnalysisResult,
    user: User,
    validated_data: Dict[str, Any],
    request_id: str
) -> AnalysisResult:
    """Save analysis result to database."""
    
    # Create database record
    db_analysis = AnalysisResult(
        analysis_type=AnalysisType(analysis_request.analysis_type),
        provider=LLMProvider(analysis_request.provider),
        model_used=analysis_request.model or "default",
        request_id=request_id,
        custom_prompt=analysis_request.custom_prompt,
        max_tokens=analysis_request.max_tokens,
        temperature=analysis_request.temperature,
        content=analysis_result.content,
        confidence_score=analysis_result.confidence_score,
        key_points=analysis_result.key_points,
        entities=analysis_result.entities,
        sentiment=analysis_result.sentiment,
        safety_rating=analysis_result.safety_rating,
        bias_detected=analysis_result.bias_detected,
        processing_time_ms=analysis_result.processing_time_ms,
        user_id=user.id,
        search_query_id=UUID(validated_data["search_query_id"]) if validated_data["search_query_id"] else None,
        source_data_hash=hashlib.sha256(str(validated_data["data"]).encode()).hexdigest()[:16],
        analysis_data=analysis_result.analysis_data
    )
    
    # Add token usage if available
    if analysis_result.token_usage:
        db_analysis.token_usage_prompt = analysis_result.token_usage.prompt_tokens
        db_analysis.token_usage_completion = analysis_result.token_usage.completion_tokens
        db_analysis.token_usage_total = analysis_result.token_usage.total_tokens
        db_analysis.estimated_cost_usd = analysis_result.token_usage.estimated_cost_usd
    
    db.add(db_analysis)
    await db.commit()
    await db.refresh(db_analysis)
    
    logger.info(
        "Analysis saved to database",
        analysis_id=str(db_analysis.id),
        analysis_type=analysis_request.analysis_type,
        provider=analysis_request.provider,
        user_id=str(user.id)
    )
    
    return db_analysis


# =============================================================================
# ANALYSIS ENDPOINTS
# =============================================================================
@router.post("/analyze", response_model=AnalysisResponseModel)
@limiter.limit("20/minute")
async def perform_analysis(
    request: Request,
    analysis_request: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Perform AI-powered analysis on data using various LLM providers.
    
    **Analysis Types:**
    - summarization: Extract key insights and create summaries
    - trend_analysis: Identify patterns and trends over time
    - sentiment_analysis: Analyze emotional tone and sentiment
    - key_insights: Extract actionable insights and recommendations
    - comparative_analysis: Compare multiple data sources
    - prediction: Generate predictions based on data patterns
    
    **Providers:**
    - openai: GPT models for high-quality analysis
    - claude: Anthropic's Claude for detailed reasoning
    - local: Local models for privacy-sensitive data
    
    **Rate Limits:**
    - 20 requests per minute per user
    - Token limits based on subscription plan
    """
    
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', f"analysis_{int(time.time() * 1000)}")
    
    logger.info(
        "Analysis request initiated",
        analysis_type=analysis_request.analysis_type,
        provider=analysis_request.provider,
        data_source=analysis_request.data_source[:100] + "..." if len(analysis_request.data_source) > 100 else analysis_request.data_source,
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
        
        # Validate and prepare data
        validated_data = await validate_analysis_data(
            analysis_request.data_source,
            db,
            current_user
        )
        
        # Generate data hash for caching
        data_hash = hashlib.sha256(str(validated_data["data"]).encode()).hexdigest()[:16]
        
        # Check cache
        cache_key = await generate_analysis_cache_key(
            data_hash=data_hash,
            analysis_type=analysis_request.analysis_type,
            provider=analysis_request.provider,
            model=analysis_request.model or "default",
            temperature=analysis_request.temperature,
            max_tokens=analysis_request.max_tokens
        )
        
        cached_result = await get_cached_analysis_result(cache_key)
        if cached_result:
            logger.info(
                "Returning cached analysis result",
                cache_key=cache_key[:16],
                analysis_type=analysis_request.analysis_type,
                user_id=str(current_user.id)
            )
            
            response = AnalysisResponseModel(
                analysis_id=str(cached_result.id),
                analysis_type=cached_result.analysis_type.value,
                provider=cached_result.provider.value,
                model=cached_result.model_used,
                status="completed",
                content=cached_result.content,
                confidence_score=cached_result.confidence_score,
                key_points=cached_result.key_points,
                entities=cached_result.entities,
                sentiment=cached_result.sentiment,
                safety_rating=cached_result.safety_rating.value,
                bias_detected=cached_result.bias_detected,
                processing_time_ms=cached_result.processing_time_ms,
                token_usage={
                    "prompt_tokens": cached_result.token_usage.prompt_tokens,
                    "completion_tokens": cached_result.token_usage.completion_tokens,
                    "total_tokens": cached_result.token_usage.total_tokens
                } if cached_result.token_usage else None,
                cost_usd=cached_result.token_usage.estimated_cost_usd if cached_result.token_usage else None,
                metadata={
                    "request_id": request_id,
                    "from_cache": True,
                    "data_type": validated_data["type"]
                },
                cached=True
            )
            return response
        
        # Prepare service request
        service_request = ServiceAnalysisRequest(
            data=validated_data["data"],
            analysis_type=AnalysisType(analysis_request.analysis_type),
            provider=LLMProvider(analysis_request.provider),
            model=analysis_request.model,
            max_tokens=analysis_request.max_tokens,
            temperature=analysis_request.temperature,
            user_id=str(current_user.id),
            request_id=request_id,
            custom_prompt=analysis_request.custom_prompt
        )
        
        # Perform analysis
        async with LLMSummarizerService() as summarizer:
            analysis_result = await summarizer.analyze(service_request, use_cache=True)
        
        # Save to database
        db_analysis = await save_analysis_to_database(
            db, analysis_request, analysis_result, current_user, validated_data, request_id
        )
        
        # Cache result
        await cache_analysis_result(cache_key, db_analysis)
        
        # Update user API usage
        current_user.increment_api_usage()
        await db.commit()
        
        # Calculate total processing time
        total_processing_time = (time.time() - start_time) * 1000
        
        # Format response
        response = AnalysisResponseModel(
            analysis_id=str(db_analysis.id),
            analysis_type=analysis_request.analysis_type,
            provider=analysis_request.provider,
            model=analysis_request.model or "default",
            status="completed",
            content=analysis_result.content,
            confidence_score=analysis_result.confidence_score,
            key_points=analysis_result.key_points,
            entities=analysis_result.entities,
            sentiment=analysis_result.sentiment,
            safety_rating=analysis_result.safety_rating.value,
            bias_detected=analysis_result.bias_detected,
            processing_time_ms=analysis_result.processing_time_ms,
            token_usage={
                "prompt_tokens": analysis_result.token_usage.prompt_tokens,
                "completion_tokens": analysis_result.token_usage.completion_tokens,
                "total_tokens": analysis_result.token_usage.total_tokens
            } if analysis_result.token_usage else None,
            cost_usd=analysis_result.token_usage.estimated_cost_usd if analysis_result.token_usage else None,
            metadata={
                "request_id": request_id,
                "total_processing_time_ms": total_processing_time,
                "data_type": validated_data["type"],
                "search_query_id": validated_data["search_query_id"],
                "from_cache": False
            },
            cached=False
        )
        
        logger.info(
            "Analysis completed successfully",
            analysis_id=str(db_analysis.id),
            analysis_type=analysis_request.analysis_type,
            provider=analysis_request.provider,
            confidence=analysis_result.confidence_score,
            duration_ms=total_processing_time,
            user_id=str(current_user.id)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Analysis request failed",
            analysis_type=analysis_request.analysis_type,
            provider=analysis_request.provider,
            error=str(e),
            user_id=str(current_user.id),
            request_id=request_id,
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail="Analysis request failed. Please try again."
        )


@router.get("/analyze/{analysis_id}", response_model=AnalysisResponseModel)
@limiter.limit("60/minute")
async def get_analysis_result(
    request: Request,
    analysis_id: UUID = Path(..., description="Analysis result ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve analysis result by ID.
    
    **Access Control:**
    - Users can only access their own analysis results
    - Admins can access any analysis results
    """
    
    try:
        # Query analysis result
        query = select(AnalysisResult).where(AnalysisResult.id == analysis_id)
        
        # Access control
        if current_user.role.value != "admin":
            query = query.where(AnalysisResult.user_id == current_user.id)
        
        result = await db.execute(query)
        analysis = result.scalar_one_or_none()
        
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail="Analysis not found or access denied"
            )
        
        # Format response
        response = AnalysisResponseModel(
            analysis_id=str(analysis.id),
            analysis_type=analysis.analysis_type.value,
            provider=analysis.provider.value,
            model=analysis.model_used,
            status="completed",
            content=analysis.content,
            confidence_score=analysis.confidence_score,
            key_points=analysis.key_points or [],
            entities=analysis.entities or [],
            sentiment=analysis.sentiment,
            safety_rating=analysis.safety_rating.value,
            bias_detected=analysis.bias_detected,
            processing_time_ms=analysis.processing_time_ms,
            token_usage={
                "prompt_tokens": analysis.token_usage_prompt,
                "completion_tokens": analysis.token_usage_completion,
                "total_tokens": analysis.token_usage_total
            } if analysis.token_usage_total else None,
            cost_usd=float(analysis.estimated_cost_usd) if analysis.estimated_cost_usd else None,
            metadata={
                "request_id": analysis.request_id,
                "created_at": analysis.created_at.isoformat(),
                "search_query_id": str(analysis.search_query_id) if analysis.search_query_id else None
            },
            cached=False
        )
        
        logger.info(
            "Analysis result retrieved",
            analysis_id=str(analysis_id),
            user_id=str(current_user.id)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve analysis result",
            analysis_id=str(analysis_id),
            error=str(e),
            user_id=str(current_user.id),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analysis result"
        )


@router.get("/analyze", response_model=AnalysisHistoryResponse)
@limiter.limit("30/minute")
async def get_analysis_history(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    analysis_type: Optional[str] = Query(None, description="Filter by analysis type"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    date_from: Optional[datetime] = Query(None, description="Filter from date"),
    date_to: Optional[datetime] = Query(None, description="Filter to date"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get user's analysis history with pagination and filtering.
    
    **Query Parameters:**
    - page: Page number (default: 1)
    - page_size: Results per page (1-100, default: 20)
    - analysis_type: Filter by analysis type
    - provider: Filter by LLM provider
    - date_from: Filter analyses from this date
    - date_to: Filter analyses to this date
    """
    
    try:
        # Build query
        query = select(AnalysisResult).where(AnalysisResult.user_id == current_user.id)
        
        # Apply filters
        if analysis_type:
            try:
                analysis_enum = AnalysisType(analysis_type)
                query = query.where(AnalysisResult.analysis_type == analysis_enum)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid analysis type: {analysis_type}"
                )
        
        if provider:
            try:
                provider_enum = LLMProvider(provider)
                query = query.where(AnalysisResult.provider == provider_enum)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid provider: {provider}"
                )
        
        if date_from:
            query = query.where(AnalysisResult.created_at >= date_from)
        
        if date_to:
            query = query.where(AnalysisResult.created_at <= date_to)
        
        # Count total results
        count_query = select(func.count(AnalysisResult.id)).where(
            AnalysisResult.user_id == current_user.id
        )
        if analysis_type:
            count_query = count_query.where(AnalysisResult.analysis_type == AnalysisType(analysis_type))
        if provider:
            count_query = count_query.where(AnalysisResult.provider == LLMProvider(provider))
        if date_from:
            count_query = count_query.where(AnalysisResult.created_at >= date_from)
        if date_to:
            count_query = count_query.where(AnalysisResult.created_at <= date_to)
        
        count_result = await db.execute(count_query)
        total_count = count_result.scalar()
        
        # Apply pagination and ordering
        query = query.order_by(desc(AnalysisResult.created_at))
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        analyses = result.scalars().all()
        
        # Format response
        analyses_data = [
            {
                "analysis_id": str(analysis.id),
                "analysis_type": analysis.analysis_type.value,
                "provider": analysis.provider.value,
                "model": analysis.model_used,
                "confidence_score": analysis.confidence_score,
                "safety_rating": analysis.safety_rating.value,
                "bias_detected": analysis.bias_detected,
                "processing_time_ms": analysis.processing_time_ms,
                "cost_usd": float(analysis.estimated_cost_usd) if analysis.estimated_cost_usd else None,
                "created_at": analysis.created_at.isoformat(),
                "content_preview": analysis.content[:200] + "..." if len(analysis.content) > 200 else analysis.content
            }
            for analysis in analyses
        ]
        
        response = AnalysisHistoryResponse(
            analyses=analyses_data,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=(page * page_size) < total_count
        )
        
        logger.info(
            "Analysis history retrieved",
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
            "Failed to retrieve analysis history",
            error=str(e),
            user_id=str(current_user.id),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analysis history"
        )


@router.post("/analyze/bias-check", response_model=BiasCheckResponse)
@limiter.limit("15/minute")
async def check_bias(
    request: Request,
    content: str = Body(..., min_length=50, max_length=10000, embed=True),
    current_user: User = Depends(get_current_user)
):
    """
    Detect potential bias in content using AI analysis.
    
    **Features:**
    - Detects various types of bias (political, confirmation, selection, etc.)
    - Provides confidence scores and specific recommendations
    - Identifies flagged segments for review
    - Content safety assessment
    
    **Rate Limits:**
    - 15 requests per minute per user
    """
    
    request_id = getattr(request.state, 'request_id', f"bias_{int(time.time() * 1000)}")
    
    logger.info(
        "Bias check initiated",
        content_length=len(content),
        user_id=str(current_user.id),
        request_id=request_id
    )
    
    try:
        # Validate content
        if detect_prompt_injection(content):
            raise HTTPException(
                status_code=400,
                detail="Content contains potentially malicious patterns"
            )
        
        # Check for PII and mask if found
        if detect_pii(content):
            content = mask_pii(content)
            logger.warning("PII detected and masked in bias check content")
        
        # Sanitize content
        sanitized_content = filter_harmful_content(content)
        
        # Perform bias detection
        async with LLMSummarizerService() as summarizer:
            bias_result = await summarizer.detect_bias(sanitized_content, request_id)
        
        # Generate recommendations based on bias types
        recommendations = []
        if bias_result.has_bias:
            if 'political' in bias_result.bias_types:
                recommendations.append("Consider presenting multiple political perspectives")
            if 'confirmation' in bias_result.bias_types:
                recommendations.append("Include contradictory evidence and viewpoints")
            if 'selection' in bias_result.bias_types:
                recommendations.append("Ensure representative sampling of sources")
            
            recommendations.append("Review source credibility and diversity")
            recommendations.append("Consider fact-checking with multiple independent sources")
        else:
            recommendations.append("Content appears to be well-balanced")
        
        # Determine safety rating
        safety_rating = "safe"
        if bias_result.has_bias and bias_result.confidence > 0.8:
            safety_rating = "moderate"
        elif len(bias_result.bias_types) > 2:
            safety_rating = "unsafe"
        
        response = BiasCheckResponse(
            has_bias=bias_result.has_bias,
            bias_types=bias_result.bias_types,
            confidence=bias_result.confidence,
            flagged_segments=bias_result.flagged_segments,
            recommendations=recommendations,
            safety_rating=safety_rating
        )
        
        logger.info(
            "Bias check completed",
            has_bias=bias_result.has_bias,
            bias_types=bias_result.bias_types,
            confidence=bias_result.confidence,
            user_id=str(current_user.id),
            request_id=request_id
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Bias check failed",
            error=str(e),
            user_id=str(current_user.id),
            request_id=request_id,
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail="Bias check failed. Please try again."
        )


@router.get("/models", response_model=ModelCapabilitiesResponse)
@limiter.limit("10/minute")
async def get_model_capabilities(request: Request):
    """
    Get available LLM models and their capabilities.
    
    **Returns:**
    - Available providers and models
    - Supported analysis types
    - Rate limits and pricing information
    - Model-specific capabilities and limitations
    """
    
    try:
        # Model capabilities data
        providers = {
            "openai": {
                "models": {
                    "gpt-4": {
                        "max_tokens": 8192,
                        "cost_per_1k_tokens": 0.03,
                        "capabilities": ["all_analysis_types"],
                        "strengths": ["reasoning", "complex_analysis", "accuracy"]
                    },
                    "gpt-3.5-turbo": {
                        "max_tokens": 4096,
                        "cost_per_1k_tokens": 0.002,
                        "capabilities": ["summarization", "sentiment_analysis", "key_insights"],
                        "strengths": ["speed", "cost_effective", "general_purpose"]
                    }
                },
                "rate_limits": {
                    "requests_per_minute": 500,
                    "tokens_per_minute": 150000
                }
            },
            "claude": {
                "models": {
                    "claude-3-opus": {
                        "max_tokens": 4096,
                        "cost_per_1k_tokens": 0.015,
                        "capabilities": ["all_analysis_types"],
                        "strengths": ["detailed_reasoning", "nuanced_analysis", "safety"]
                    },
                    "claude-3-sonnet": {
                        "max_tokens": 4096,
                        "cost_per_1k_tokens": 0.003,
                        "capabilities": ["summarization", "trend_analysis", "sentiment_analysis"],
                        "strengths": ["balanced_performance", "cost_effective", "reliable"]
                    }
                },
                "rate_limits": {
                    "requests_per_minute": 300,
                    "tokens_per_minute": 100000
                }
            }
        }
        
        analysis_types = [
            "summarization",
            "trend_analysis", 
            "sentiment_analysis",
            "key_insights",
            "comparative_analysis",
            "prediction"
        ]
        
        rate_limits = {
            "analysis_requests_per_minute": 20,
            "bias_checks_per_minute": 15,
            "batch_requests_per_minute": 5
        }
        
        pricing = {
            "openai": {
                "gpt-4": 0.03,
                "gpt-3.5-turbo": 0.002
            },
            "claude": {
                "claude-3-opus": 0.015,
                "claude-3-sonnet": 0.003
            }
        }
        
        response = ModelCapabilitiesResponse(
            providers=providers,
            analysis_types=analysis_types,
            rate_limits=rate_limits,
            pricing=pricing
        )
        
        logger.info("Model capabilities retrieved")
        
        return response
        
    except Exception as e:
        logger.error("Failed to retrieve model capabilities", error=str(e))
        
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model capabilities"
        )


@router.post("/analyze/batch")
@limiter.limit("5/minute")
async def batch_analysis(
    request: Request,
    analysis_requests: List[AnalysisRequest] = Body(..., max_items=5),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Perform multiple analyses in batch.
    
    **Limitations:**
    - Maximum 5 analyses per batch
    - Rate limited to 5 batches per minute
    - Each analysis counts toward user quota
    """
    
    request_id = getattr(request.state, 'request_id', f"batch_analysis_{int(time.time() * 1000)}")
    
    if len(analysis_requests) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 analyses allowed per batch"
        )
    
    # Check quota for all analyses
    required_quota = len(analysis_requests)
    if current_user.api_calls_today + required_quota > current_user.can_make_api_call():
        raise HTTPException(
            status_code=429,
            detail="Insufficient API quota for batch operation"
        )
    
    logger.info(
        "Batch analysis initiated",
        batch_size=len(analysis_requests),
        user_id=str(current_user.id),
        request_id=request_id
    )
    
    results = []
    successful_analyses = 0
    
    # Process analyses concurrently (limited concurrency)
    semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent analyses
    
    async def process_single_analysis(i: int, analysis_request: AnalysisRequest):
        async with semaphore:
            try:
                analysis_response = await perform_analysis(
                    request=request,
                    analysis_request=analysis_request,
                    db=db,
                    current_user=current_user
                )
                
                return {
                    "index": i,
                    "status": "success",
                    "analysis_id": analysis_response.analysis_id,
                    "analysis_type": analysis_request.analysis_type,
                    "confidence_score": analysis_response.confidence_score
                }
                
            except Exception as e:
                logger.warning(
                    "Batch analysis item failed",
                    index=i,
                    analysis_type=analysis_request.analysis_type,
                    error=str(e)
                )
                
                return {
                    "index": i,
                    "status": "error",
                    "analysis_type": analysis_request.analysis_type,
                    "error": str(e)
                }
    
    # Execute all analyses concurrently
    tasks = [
        process_single_analysis(i, req) 
        for i, req in enumerate(analysis_requests)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count successful analyses
    successful_analyses = sum(
        1 for result in results 
        if isinstance(result, dict) and result.get("status") == "success"
    )
    
    logger.info(
        "Batch analysis completed",
        total_requests=len(analysis_requests),
        successful=successful_analyses,
        failed=len(analysis_requests) - successful_analyses,
        user_id=str(current_user.id),
        request_id=request_id
    )
    
    return {
        "batch_id": request_id,
        "total_requests": len(analysis_requests),
        "successful": successful_analyses,
        "failed": len(analysis_requests) - successful_analyses,
        "results": results
    }
