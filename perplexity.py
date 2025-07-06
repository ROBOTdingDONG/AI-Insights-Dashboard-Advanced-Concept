"""
Perplexity API Service
=====================

Enterprise-grade external API integration with comprehensive error handling,
caching, rate limiting, and security features.

Features:
- Async HTTP client with retry logic and circuit breaker
- Response caching with TTL and invalidation strategies
- Rate limit management with exponential backoff
- Source credibility scoring and validation
- Content sanitization and safety filtering
- Comprehensive monitoring and alerting

Author: AI Insights Team  
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import aiohttp
import structlog
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type
)

# Import core modules (to be created)
from app.core.config import settings
from app.utils.validators import sanitize_text, validate_url
from app.utils.data_processor import extract_keywords, calculate_relevance


logger = structlog.get_logger(__name__)


# =============================================================================
# DATA MODELS AND ENUMS
# =============================================================================
class SourceType(str, Enum):
    """Source type classification for credibility scoring."""
    NEWS = "news"
    ACADEMIC = "academic" 
    BLOG = "blog"
    SOCIAL = "social"
    GOVERNMENT = "government"
    COMMERCIAL = "commercial"
    UNKNOWN = "unknown"


class CredibilityScore(str, Enum):
    """Source credibility classification."""
    HIGH = "high"           # Academic, government, established news
    MEDIUM = "medium"       # Reputable blogs, verified sources
    LOW = "low"            # Social media, unverified sources
    SUSPICIOUS = "suspicious"  # Flagged or blacklisted sources


class SearchResult(BaseModel):
    """Individual search result model with validation."""
    
    title: str = Field(..., min_length=1, max_length=500)
    url: str = Field(..., min_length=1)
    snippet: str = Field(..., max_length=2000)
    published_date: Optional[datetime] = None
    source_domain: str = Field(..., min_length=1, max_length=100)
    source_type: SourceType = SourceType.UNKNOWN
    credibility_score: CredibilityScore = CredibilityScore.LOW
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    
    @validator('url')
    def validate_url_field(cls, v):
        """Validate and sanitize URL."""
        if not validate_url(v):
            raise ValueError("Invalid URL format")
        return v
    
    @validator('title', 'snippet')
    def sanitize_text_fields(cls, v):
        """Sanitize text content."""
        return sanitize_text(v)


class SearchResponse(BaseModel):
    """Complete search response model."""
    
    query: str = Field(..., min_length=1, max_length=500)
    results: List[SearchResult] = Field(default_factory=list)
    total_results: int = Field(0, ge=0)
    search_time_ms: float = Field(0.0, ge=0.0)
    cached: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(...)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CircuitBreakerState(str, Enum):
    """Circuit breaker state management."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


# =============================================================================
# PERPLEXITY API SERVICE
# =============================================================================
class PerplexityService:
    """
    Enterprise-grade Perplexity API integration service.
    
    Provides intelligent search capabilities with caching, error handling,
    and comprehensive monitoring.
    """
    
    def __init__(self):
        self.base_url = "https://api.perplexity.ai"
        self.api_key = settings.PERPLEXITY_API_KEY
        self.redis_client = redis.from_url(settings.REDIS_URL)
        
        # Circuit breaker configuration
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_open_timeout = 60  # seconds
        
        # Rate limiting configuration
        self.rate_limit_requests = 100  # requests per minute
        self.rate_limit_window = 60     # seconds
        
        # Cache configuration
        self.cache_ttl = 3600          # 1 hour default TTL
        self.cache_prefix = "perplexity:search:"
        
        # HTTP client configuration
        self.timeout = ClientTimeout(total=30, connect=10)
        self.connector = TCPConnector(
            limit=10,
            limit_per_host=5,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Domain credibility mapping (expandable)
        self.credibility_domains = {
            # High credibility sources
            "arxiv.org": CredibilityScore.HIGH,
            "nature.com": CredibilityScore.HIGH,
            "science.org": CredibilityScore.HIGH,
            "gov.edu": CredibilityScore.HIGH,
            "gov": CredibilityScore.HIGH,
            "edu": CredibilityScore.HIGH,
            "reuters.com": CredibilityScore.HIGH,
            "bbc.com": CredibilityScore.HIGH,
            "apnews.com": CredibilityScore.HIGH,
            
            # Medium credibility sources  
            "techcrunch.com": CredibilityScore.MEDIUM,
            "wired.com": CredibilityScore.MEDIUM,
            "arstechnica.com": CredibilityScore.MEDIUM,
            "medium.com": CredibilityScore.MEDIUM,
            
            # Suspicious sources (example)
            "fakenews.com": CredibilityScore.SUSPICIOUS,
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = ClientSession(
            timeout=self.timeout,
            connector=self.connector,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "AI-Insights-Dashboard/1.0"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.session.close()
        await self.redis_client.close()
    
    def _generate_cache_key(self, query: str, **kwargs) -> str:
        """Generate deterministic cache key for query."""
        cache_data = {"query": query.strip().lower(), **kwargs}
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"{self.cache_prefix}{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[SearchResponse]:
        """Retrieve cached search result."""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                result = SearchResponse(**data)
                result.cached = True
                
                logger.info(
                    "Cache hit",
                    cache_key=cache_key[:16] + "...",
                    query=result.query
                )
                return result
        except Exception as e:
            logger.warning("Cache retrieval failed", error=str(e))
        
        return None
    
    async def _cache_result(self, cache_key: str, result: SearchResponse) -> None:
        """Cache search result with TTL."""
        try:
            cache_data = result.dict()
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_data, default=str)
            )
            
            logger.info(
                "Result cached",
                cache_key=cache_key[:16] + "...",
                ttl=self.cache_ttl
            )
        except Exception as e:
            logger.warning("Cache storage failed", error=str(e))
    
    async def _check_rate_limit(self, user_id: Optional[str] = None) -> bool:
        """Check if request is within rate limits."""
        try:
            rate_key = f"rate_limit:perplexity:{user_id or 'anonymous'}"
            current_requests = await self.redis_client.incr(rate_key)
            
            if current_requests == 1:
                await self.redis_client.expire(rate_key, self.rate_limit_window)
            
            if current_requests > self.rate_limit_requests:
                logger.warning(
                    "Rate limit exceeded",
                    user_id=user_id,
                    requests=current_requests,
                    limit=self.rate_limit_requests
                )
                return False
            
            return True
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e))
            return True  # Allow request if rate limit check fails
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_breaker_state == CircuitBreakerState.OPEN:
            if (time.time() - self.last_failure_time) > self.circuit_open_timeout:
                self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker transitioning to half-open")
                return False
            return True
        return False
    
    def _record_success(self):
        """Record successful API call."""
        if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            self.circuit_breaker_state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            logger.info("Circuit breaker closed after successful request")
    
    def _record_failure(self):
        """Record failed API call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= 5:  # Open circuit after 5 failures
            self.circuit_breaker_state = CircuitBreakerState.OPEN
            logger.warning(
                "Circuit breaker opened",
                failure_count=self.failure_count
            )
    
    def _classify_source_type(self, domain: str) -> SourceType:
        """Classify source type based on domain analysis."""
        domain = domain.lower()
        
        if any(tld in domain for tld in ['.edu', '.ac.']):
            return SourceType.ACADEMIC
        elif any(tld in domain for tld in ['.gov', '.mil']):
            return SourceType.GOVERNMENT
        elif any(word in domain for word in ['news', 'times', 'post', 'herald', 'guardian']):
            return SourceType.NEWS
        elif any(word in domain for word in ['blog', 'medium', 'substack']):
            return SourceType.BLOG
        elif any(word in domain for word in ['twitter', 'facebook', 'linkedin', 'reddit']):
            return SourceType.SOCIAL
        elif any(word in domain for word in ['shop', 'store', 'buy', 'sell']):
            return SourceType.COMMERCIAL
        else:
            return SourceType.UNKNOWN
    
    def _get_credibility_score(self, domain: str, source_type: SourceType) -> CredibilityScore:
        """Calculate credibility score for source."""
        domain = domain.lower()
        
        # Check explicit domain mapping
        for credible_domain, score in self.credibility_domains.items():
            if credible_domain in domain:
                return score
        
        # Default scoring based on source type
        type_credibility = {
            SourceType.ACADEMIC: CredibilityScore.HIGH,
            SourceType.GOVERNMENT: CredibilityScore.HIGH,
            SourceType.NEWS: CredibilityScore.MEDIUM,
            SourceType.BLOG: CredibilityScore.MEDIUM,
            SourceType.COMMERCIAL: CredibilityScore.LOW,
            SourceType.SOCIAL: CredibilityScore.LOW,
            SourceType.UNKNOWN: CredibilityScore.LOW
        }
        
        return type_credibility.get(source_type, CredibilityScore.LOW)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_api_request(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Make authenticated request to Perplexity API with retry logic."""
        
        if self._is_circuit_breaker_open():
            raise Exception("Circuit breaker is open - API temporarily unavailable")
        
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Provide comprehensive search results "
                        "with accurate sources, publication dates, and relevance scores. "
                        "Return results in structured JSON format."
                    )
                },
                {
                    "role": "user", 
                    "content": f"Search for: {query}. Provide {max_results} most relevant results."
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.2,
            "search_domain_filter": ["news", "academic", "reddit"],
            "return_citations": True
        }
        
        try:
            start_time = time.time()
            
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                
                if response.status == 429:
                    logger.warning("API rate limit exceeded")
                    raise aiohttp.ClientError("Rate limit exceeded")
                
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        "API request failed",
                        status=response.status,
                        error=error_text
                    )
                    raise aiohttp.ClientError(f"API error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Record successful request
                self._record_success()
                
                # Calculate request time
                request_time = (time.time() - start_time) * 1000
                
                logger.info(
                    "API request successful",
                    query=query[:50] + "..." if len(query) > 50 else query,
                    duration_ms=round(request_time, 2),
                    status=response.status
                )
                
                return data
                
        except Exception as e:
            self._record_failure()
            logger.error(
                "API request failed",
                query=query,
                error=str(e),
                exc_info=True
            )
            raise
    
    def _parse_api_response(self, raw_response: Dict[str, Any], query: str, request_id: str) -> SearchResponse:
        """Parse Perplexity API response into structured format."""
        
        try:
            # Extract content from API response
            content = raw_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            citations = raw_response.get("citations", [])
            
            results = []
            
            # Parse citations into search results
            for i, citation in enumerate(citations):
                try:
                    # Extract domain from URL
                    from urllib.parse import urlparse
                    parsed_url = urlparse(citation)
                    domain = parsed_url.netloc.lower()
                    
                    # Classify source
                    source_type = self._classify_source_type(domain)
                    credibility = self._get_credibility_score(domain, source_type)
                    
                    # Calculate relevance score (simplified version)
                    relevance = calculate_relevance(query, citation, content)
                    
                    result = SearchResult(
                        title=f"Source {i+1}",  # Placeholder - would extract actual title
                        url=citation,
                        snippet=content[:200] + "..." if len(content) > 200 else content,
                        published_date=None,  # Would extract from actual response
                        source_domain=domain,
                        source_type=source_type,
                        credibility_score=credibility,
                        relevance_score=relevance
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(
                        "Failed to parse citation",
                        citation=citation,
                        error=str(e)
                    )
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time_ms=0.0,  # Would be calculated from actual timing
                cached=False,
                request_id=request_id
            )
            
        except Exception as e:
            logger.error(
                "Failed to parse API response",
                error=str(e),
                exc_info=True
            )
            
            # Return empty results on parse failure
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time_ms=0.0,
                cached=False,
                request_id=request_id
            )
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        use_cache: bool = True
    ) -> SearchResponse:
        """
        Perform intelligent search with caching and error handling.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            user_id: Optional user ID for rate limiting
            request_id: Optional request ID for tracing
            use_cache: Whether to use cached results
            
        Returns:
            SearchResponse with results and metadata
            
        Raises:
            Exception: For various API and processing errors
        """
        
        # Input validation
        if not query or len(query.strip()) < 2:
            raise ValueError("Query must be at least 2 characters long")
        
        if max_results < 1 or max_results > 50:
            raise ValueError("max_results must be between 1 and 50")
        
        # Sanitize query
        query = sanitize_text(query.strip())
        request_id = request_id or f"search_{int(time.time() * 1000)}"
        
        logger.info(
            "Search request initiated",
            query=query,
            max_results=max_results,
            user_id=user_id,
            request_id=request_id
        )
        
        # Check rate limits
        if not await self._check_rate_limit(user_id):
            raise Exception("Rate limit exceeded. Please try again later.")
        
        # Check cache
        cache_key = self._generate_cache_key(query, max_results=max_results)
        
        if use_cache:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                cached_result.request_id = request_id
                return cached_result
        
        # Make API request
        start_time = time.time()
        
        try:
            raw_response = await self._make_api_request(query, max_results)
            search_time_ms = (time.time() - start_time) * 1000
            
            # Parse response
            result = self._parse_api_response(raw_response, query, request_id)
            result.search_time_ms = search_time_ms
            
            # Cache successful result
            if use_cache and result.results:
                await self._cache_result(cache_key, result)
            
            logger.info(
                "Search completed successfully",
                query=query,
                results_count=len(result.results),
                duration_ms=round(search_time_ms, 2),
                request_id=request_id
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Search failed",
                query=query,
                error=str(e),
                request_id=request_id,
                exc_info=True
            )
            raise
    
    async def get_trending_topics(self, category: Optional[str] = None) -> List[str]:
        """
        Get trending topics (placeholder for future implementation).
        
        Args:
            category: Optional category filter
            
        Returns:
            List of trending topic strings
        """
        
        # Placeholder implementation
        trending = [
            "Artificial Intelligence regulation",
            "Climate change solutions", 
            "Quantum computing advances",
            "Renewable energy trends",
            "Space exploration"
        ]
        
        logger.info("Trending topics retrieved", category=category)
        return trending


# =============================================================================
# SERVICE FACTORY
# =============================================================================
async def get_perplexity_service() -> PerplexityService:
    """Factory function to create Perplexity service instance."""
    return PerplexityService()


# =============================================================================
# USAGE EXAMPLE (for testing)
# =============================================================================
if __name__ == "__main__":
    async def test_search():
        async with PerplexityService() as service:
            result = await service.search(
                query="latest AI developments in healthcare",
                max_results=5,
                request_id="test_123"
            )
            print(f"Found {len(result.results)} results")
            for i, res in enumerate(result.results, 1):
                print(f"{i}. {res.title} ({res.credibility_score})")
    
    # asyncio.run(test_search())
