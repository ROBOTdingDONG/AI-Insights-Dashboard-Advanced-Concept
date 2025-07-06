"""
Data Processing and Transformation Utilities
============================================

Comprehensive data processing utilities for search results, content analysis,
and data transformation operations.

Features:
- Text preprocessing and normalization
- Keyword extraction and relevance scoring
- Content similarity and clustering
- Data cleaning and validation
- Statistical analysis and aggregation
- Time series data processing
- Export formatting and optimization

Processing Capabilities:
- Natural language processing (NLP)
- Content relevance calculation
- Search result optimization
- Data aggregation and summarization
- Performance metrics calculation
- Content deduplication

Author: AI Insights Team
Version: 1.0.0
"""

import re
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from urllib.parse import urlparse
import hashlib

import structlog
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob


logger = structlog.get_logger(__name__)

# Initialize NLTK components
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    logger.warning("Failed to download NLTK data")

# Initialize NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()


# =============================================================================
# TEXT PREPROCESSING UTILITIES
# =============================================================================
def clean_text(text: str, 
               remove_punctuation: bool = True,
               remove_numbers: bool = False,
               lowercase: bool = True,
               remove_extra_whitespace: bool = True) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Input text to clean
        remove_punctuation: Whether to remove punctuation
        remove_numbers: Whether to remove numbers
        lowercase: Whether to convert to lowercase
        remove_extra_whitespace: Whether to remove extra whitespace
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_keywords(text: str, 
                    max_keywords: int = 20,
                    min_word_length: int = 3,
                    use_tfidf: bool = True) -> List[Tuple[str, float]]:
    """
    Extract keywords from text with relevance scores.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        min_word_length: Minimum word length to consider
        use_tfidf: Whether to use TF-IDF scoring
        
    Returns:
        List[Tuple[str, float]]: Keywords with scores
    """
    if not text:
        return []
    
    try:
        # Clean text
        cleaned_text = clean_text(text, remove_punctuation=True, remove_numbers=True)
        
        # Tokenize
        words = word_tokenize(cleaned_text)
        
        # Filter words
        words = [
            word for word in words 
            if len(word) >= min_word_length 
            and word not in stop_words
            and word.isalpha()
        ]
        
        if not words:
            return []
        
        if use_tfidf and len(words) > 5:
            # Use TF-IDF for keyword extraction
            try:
                # Create document for TF-IDF
                doc = ' '.join(words)
                
                # Initialize TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    max_features=max_keywords,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1
                )
                
                # Fit and transform
                tfidf_matrix = vectorizer.fit_transform([doc])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                
                # Get keywords with scores
                keywords = [
                    (feature_names[i], tfidf_scores[i]) 
                    for i in range(len(feature_names))
                    if tfidf_scores[i] > 0
                ]
                
                # Sort by score
                keywords.sort(key=lambda x: x[1], reverse=True)
                
                return keywords[:max_keywords]
                
            except Exception as e:
                logger.warning("TF-IDF extraction failed, falling back to frequency", error=str(e))
        
        # Fall back to frequency-based extraction
        word_freq = Counter(words)
        total_words = len(words)
        
        # Calculate normalized frequency scores
        keywords = [
            (word, count / total_words) 
            for word, count in word_freq.most_common(max_keywords)
        ]
        
        return keywords
        
    except Exception as e:
        logger.error("Keyword extraction failed", error=str(e))
        return []


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using cosine similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        float: Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    try:
        # Clean texts
        clean1 = clean_text(text1)
        clean2 = clean_text(text2)
        
        if not clean1 or not clean2:
            return 0.0
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([clean1, clean2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
        
    except Exception as e:
        logger.error("Text similarity calculation failed", error=str(e))
        return 0.0


def calculate_relevance(query: str, title: str, content: str, url: str = "") -> float:
    """
    Calculate relevance score for search result.
    
    Args:
        query: Search query
        title: Result title
        content: Result content/snippet
        url: Result URL (optional)
        
    Returns:
        float: Relevance score (0-1)
    """
    if not query or not (title or content):
        return 0.0
    
    try:
        # Clean inputs
        query_clean = clean_text(query, remove_punctuation=True)
        title_clean = clean_text(title, remove_punctuation=True) if title else ""
        content_clean = clean_text(content, remove_punctuation=True) if content else ""
        
        # Extract query keywords
        query_words = set(word_tokenize(query_clean))
        query_words = {word for word in query_words if word not in stop_words and len(word) > 2}
        
        if not query_words:
            return 0.0
        
        # Calculate title relevance (weighted higher)
        title_words = set(word_tokenize(title_clean)) if title_clean else set()
        title_matches = len(query_words.intersection(title_words))
        title_score = title_matches / len(query_words) if query_words else 0
        
        # Calculate content relevance
        content_words = set(word_tokenize(content_clean)) if content_clean else set()
        content_matches = len(query_words.intersection(content_words))
        content_score = content_matches / len(query_words) if query_words else 0
        
        # URL bonus for exact domain matches
        url_score = 0.0
        if url:
            domain = urlparse(url).netloc.lower()
            for word in query_words:
                if word in domain:
                    url_score += 0.1
        
        # Weighted combination
        relevance = (
            title_score * 0.5 +      # Title weight: 50%
            content_score * 0.4 +    # Content weight: 40%
            min(url_score, 0.1)      # URL bonus: up to 10%
        )
        
        return min(relevance, 1.0)
        
    except Exception as e:
        logger.error("Relevance calculation failed", error=str(e))
        return 0.0


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text.
    
    Args:
        text: Input text
        
    Returns:
        Dict[str, float]: Sentiment scores (polarity, subjectivity)
    """
    if not text:
        return {"polarity": 0.0, "subjectivity": 0.0, "compound": 0.0}
    
    try:
        # Use TextBlob for sentiment analysis
        blob = TextBlob(text)
        
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Calculate compound score (normalized)
        compound = (polarity + 1) / 2  # Convert to 0-1 range
        
        return {
            "polarity": float(polarity),
            "subjectivity": float(subjectivity),
            "compound": float(compound)
        }
        
    except Exception as e:
        logger.error("Sentiment analysis failed", error=str(e))
        return {"polarity": 0.0, "subjectivity": 0.0, "compound": 0.0}


# =============================================================================
# DATA AGGREGATION AND ANALYSIS
# =============================================================================
def aggregate_search_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate and analyze search results.
    
    Args:
        results: List of search result dictionaries
        
    Returns:
        Dict[str, Any]: Aggregated analysis
    """
    if not results:
        return {}
    
    try:
        # Extract metrics
        total_results = len(results)
        relevance_scores = [r.get('relevance_score', 0) for r in results]
        credibility_scores = [r.get('credibility_score', 'low') for r in results]
        source_types = [r.get('source_type', 'unknown') for r in results]
        domains = [urlparse(r.get('url', '')).netloc for r in results if r.get('url')]
        
        # Calculate statistics
        avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0
        median_relevance = statistics.median(relevance_scores) if relevance_scores else 0
        std_relevance = statistics.stdev(relevance_scores) if len(relevance_scores) > 1 else 0
        
        # Analyze credibility distribution
        credibility_dist = Counter(credibility_scores)
        
        # Analyze source type distribution
        source_type_dist = Counter(source_types)
        
        # Analyze domain distribution
        domain_dist = Counter(domains)
        
        # Calculate diversity metrics
        unique_domains = len(set(domains))
        domain_diversity = unique_domains / max(total_results, 1)
        
        return {
            "total_results": total_results,
            "relevance_stats": {
                "average": round(avg_relevance, 3),
                "median": round(median_relevance, 3),
                "std_deviation": round(std_relevance, 3),
                "min": min(relevance_scores) if relevance_scores else 0,
                "max": max(relevance_scores) if relevance_scores else 0
            },
            "credibility_distribution": dict(credibility_dist),
            "source_type_distribution": dict(source_type_dist),
            "domain_distribution": dict(list(domain_dist.most_common(10))),
            "diversity_metrics": {
                "unique_domains": unique_domains,
                "domain_diversity": round(domain_diversity, 3)
            }
        }
        
    except Exception as e:
        logger.error("Search results aggregation failed", error=str(e))
        return {"error": str(e)}


def detect_content_duplicates(contents: List[str], similarity_threshold: float = 0.8) -> List[List[int]]:
    """
    Detect duplicate or highly similar content.
    
    Args:
        contents: List of content strings
        similarity_threshold: Similarity threshold for duplicates
        
    Returns:
        List[List[int]]: Groups of duplicate content indices
    """
    if not contents or len(contents) < 2:
        return []
    
    try:
        # Clean contents
        cleaned_contents = [clean_text(content) for content in contents]
        
        # Remove empty contents
        valid_indices = [i for i, content in enumerate(cleaned_contents) if content]
        valid_contents = [cleaned_contents[i] for i in valid_indices]
        
        if len(valid_contents) < 2:
            return []
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        tfidf_matrix = vectorizer.fit_transform(valid_contents)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicate groups
        duplicate_groups = []
        processed = set()
        
        for i in range(len(valid_contents)):
            if i in processed:
                continue
            
            # Find similar content
            similar_indices = []
            for j in range(i, len(valid_contents)):
                if similarity_matrix[i][j] >= similarity_threshold:
                    similar_indices.append(valid_indices[j])
                    processed.add(j)
            
            if len(similar_indices) > 1:
                duplicate_groups.append(similar_indices)
        
        return duplicate_groups
        
    except Exception as e:
        logger.error("Duplicate detection failed", error=str(e))
        return []


def extract_trending_topics(texts: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Extract trending topics from multiple texts.
    
    Args:
        texts: List of text documents
        top_n: Number of top topics to return
        
    Returns:
        List[Tuple[str, float]]: Topics with trend scores
    """
    if not texts:
        return []
    
    try:
        # Clean and combine texts
        cleaned_texts = [clean_text(text) for text in texts if text]
        
        if not cleaned_texts:
            return []
        
        # Extract keywords from all texts
        all_keywords = []
        for text in cleaned_texts:
            keywords = extract_keywords(text, max_keywords=50, use_tfidf=False)
            all_keywords.extend([kw[0] for kw in keywords])
        
        # Count keyword frequencies
        keyword_freq = Counter(all_keywords)
        
        # Calculate trend scores (frequency + recency weight)
        total_texts = len(cleaned_texts)
        trending_topics = []
        
        for keyword, freq in keyword_freq.most_common(top_n * 2):
            # Simple trend score: frequency normalized by total texts
            trend_score = freq / total_texts
            
            # Boost score for multi-word phrases
            if ' ' in keyword:
                trend_score *= 1.2
            
            trending_topics.append((keyword, trend_score))
        
        # Sort by trend score and return top N
        trending_topics.sort(key=lambda x: x[1], reverse=True)
        
        return trending_topics[:top_n]
        
    except Exception as e:
        logger.error("Trending topics extraction failed", error=str(e))
        return []


# =============================================================================
# PERFORMANCE METRICS CALCULATION
# =============================================================================
def calculate_search_performance_metrics(search_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate performance metrics for search operations.
    
    Args:
        search_data: List of search operation data
        
    Returns:
        Dict[str, Any]: Performance metrics
    """
    if not search_data:
        return {}
    
    try:
        # Extract timing data
        search_times = [d.get('search_time_ms', 0) for d in search_data]
        processing_times = [d.get('processing_time_ms', 0) for d in search_data]
        total_times = [d.get('total_time_ms', 0) for d in search_data]
        result_counts = [d.get('total_results', 0) for d in search_data]
        
        # Calculate timing statistics
        def calc_stats(values):
            if not values:
                return {}
            return {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "percentile_95": np.percentile(values, 95) if values else 0
            }
        
        # Cache hit rate
        cache_hits = sum(1 for d in search_data if d.get('cached', False))
        cache_hit_rate = cache_hits / len(search_data) if search_data else 0
        
        # Success rate
        successful = sum(1 for d in search_data if d.get('status') == 'completed')
        success_rate = successful / len(search_data) if search_data else 0
        
        # Results per search
        avg_results = statistics.mean(result_counts) if result_counts else 0
        
        return {
            "total_searches": len(search_data),
            "success_rate": round(success_rate, 3),
            "cache_hit_rate": round(cache_hit_rate, 3),
            "average_results_per_search": round(avg_results, 1),
            "timing_stats": {
                "search_time_ms": calc_stats(search_times),
                "processing_time_ms": calc_stats(processing_times),
                "total_time_ms": calc_stats(total_times)
            }
        }
        
    except Exception as e:
        logger.error("Performance metrics calculation failed", error=str(e))
        return {"error": str(e)}


def calculate_user_analytics(user_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate user behavior analytics.
    
    Args:
        user_data: List of user interaction data
        
    Returns:
        Dict[str, Any]: User analytics
    """
    if not user_data:
        return {}
    
    try:
        # Extract user metrics
        query_lengths = [len(d.get('query', '').split()) for d in user_data]
        session_durations = [d.get('session_duration', 0) for d in user_data if d.get('session_duration')]
        query_types = [d.get('analysis_type', 'unknown') for d in user_data]
        
        # Time-based analysis
        timestamps = [
            datetime.fromisoformat(d.get('timestamp', '').replace('Z', '+00:00'))
            for d in user_data 
            if d.get('timestamp')
        ]
        
        # Peak usage hours
        if timestamps:
            hours = [ts.hour for ts in timestamps]
            peak_hour_dist = Counter(hours)
            peak_hour = peak_hour_dist.most_common(1)[0][0] if peak_hour_dist else 0
        else:
            peak_hour = 0
            peak_hour_dist = {}
        
        # Query complexity analysis
        avg_query_length = statistics.mean(query_lengths) if query_lengths else 0
        
        # Popular query types
        query_type_dist = Counter(query_types)
        
        return {
            "total_interactions": len(user_data),
            "average_query_length": round(avg_query_length, 1),
            "peak_usage_hour": peak_hour,
            "hourly_distribution": dict(peak_hour_dist),
            "query_type_distribution": dict(query_type_dist),
            "session_stats": {
                "average_duration": statistics.mean(session_durations) if session_durations else 0,
                "total_sessions": len(session_durations)
            }
        }
        
    except Exception as e:
        logger.error("User analytics calculation failed", error=str(e))
        return {"error": str(e)}


# =============================================================================
# DATA EXPORT UTILITIES
# =============================================================================
def format_data_for_export(data: List[Dict[str, Any]], export_format: str) -> Union[str, bytes]:
    """
    Format data for various export formats.
    
    Args:
        data: Data to export
        export_format: Target format (csv, json, xlsx)
        
    Returns:
        Union[str, bytes]: Formatted data
    """
    if not data:
        return "" if export_format in ['csv', 'json'] else b""
    
    try:
        if export_format.lower() == 'csv':
            # Convert to DataFrame and export as CSV
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        
        elif export_format.lower() == 'json':
            # Export as JSON
            import json
            return json.dumps(data, indent=2, default=str)
        
        elif export_format.lower() == 'xlsx':
            # Export as Excel
            df = pd.DataFrame(data)
            
            # Create Excel file in memory
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            
            return buffer.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
            
    except Exception as e:
        logger.error("Data export formatting failed", error=str(e), format=export_format)
        raise


def create_data_summary(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create comprehensive data summary.
    
    Args:
        data: Input data to summarize
        
    Returns:
        Dict[str, Any]: Data summary
    """
    if not data:
        return {"total_records": 0}
    
    try:
        # Basic metrics
        total_records = len(data)
        
        # Analyze data structure
        all_keys = set()
        for record in data:
            all_keys.update(record.keys())
        
        # Data type analysis
        field_types = {}
        for key in all_keys:
            values = [record.get(key) for record in data if key in record]
            non_null_values = [v for v in values if v is not None]
            
            if non_null_values:
                # Determine most common type
                types = [type(v).__name__ for v in non_null_values]
                most_common_type = Counter(types).most_common(1)[0][0]
                field_types[key] = {
                    "type": most_common_type,
                    "null_count": len(values) - len(non_null_values),
                    "null_percentage": round((len(values) - len(non_null_values)) / len(values) * 100, 1)
                }
        
        # Numeric field statistics
        numeric_stats = {}
        for key, info in field_types.items():
            if info["type"] in ['int', 'float']:
                values = [
                    record.get(key) for record in data 
                    if key in record and isinstance(record[key], (int, float))
                ]
                
                if values:
                    numeric_stats[key] = {
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                        "min": min(values),
                        "max": max(values)
                    }
        
        return {
            "total_records": total_records,
            "fields": list(all_keys),
            "field_count": len(all_keys),
            "field_types": field_types,
            "numeric_statistics": numeric_stats,
            "data_quality": {
                "completeness": round(
                    sum(1 for record in data if len(record) == len(all_keys)) / total_records * 100, 1
                ),
                "avg_fields_per_record": round(
                    sum(len(record) for record in data) / total_records, 1
                )
            }
        }
        
    except Exception as e:
        logger.error("Data summary creation failed", error=str(e))
        return {"error": str(e), "total_records": len(data)}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def generate_content_hash(content: str) -> str:
    """
    Generate consistent hash for content deduplication.
    
    Args:
        content: Content to hash
        
    Returns:
        str: Content hash
    """
    if not content:
        return ""
    
    # Normalize content for consistent hashing
    normalized = clean_text(content, remove_punctuation=True, remove_numbers=True)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return hashlib.sha256(normalized.encode()).hexdigest()


def calculate_reading_time(text: str, words_per_minute: int = 200) -> float:
    """
    Calculate estimated reading time for text.
    
    Args:
        text: Input text
        words_per_minute: Average reading speed
        
    Returns:
        float: Reading time in minutes
    """
    if not text:
        return 0.0
    
    # Count words
    word_count = len(text.split())
    
    # Calculate reading time
    reading_time = word_count / words_per_minute
    
    return round(reading_time, 1)


def extract_entities(text: str) -> List[str]:
    """
    Extract named entities from text.
    
    Args:
        text: Input text
        
    Returns:
        List[str]: Extracted entities
    """
    if not text:
        return []
    
    try:
        # Simple entity extraction using patterns
        entities = []
        
        # Extract capitalized words (potential names/organizations)
        cap_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(cap_words[:10])  # Limit to top 10
        
        # Extract numbers with units
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*%|\s*million|\s*billion|\s*thousand)\b', text)
        entities.extend(numbers)
        
        # Extract dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        entities.extend(dates)
        
        # Remove duplicates and return
        return list(set(entities))
        
    except Exception as e:
        logger.error("Entity extraction failed", error=str(e))
        return []


# =============================================================================
# EXPORTS FOR EASY IMPORT
# =============================================================================
__all__ = [
    # Text Processing
    'clean_text', 'extract_keywords', 'calculate_text_similarity',
    'calculate_relevance', 'analyze_sentiment',
    
    # Data Analysis
    'aggregate_search_results', 'detect_content_duplicates',
    'extract_trending_topics',
    
    # Performance Metrics
    'calculate_search_performance_metrics', 'calculate_user_analytics',
    
    # Export Utilities
    'format_data_for_export', 'create_data_summary',
    
    # Utility Functions
    'generate_content_hash', 'calculate_reading_time', 'extract_entities'
]


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Test text processing
    sample_text = "Artificial Intelligence and Machine Learning are transforming healthcare technology."
    
    print("Testing data processing utilities...")
    
    # Test keyword extraction
    keywords = extract_keywords(sample_text)
    print(f"Keywords: {keywords}")
    
    # Test sentiment analysis
    sentiment = analyze_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")
    
    # Test relevance calculation
    relevance = calculate_relevance("AI healthcare", "AI in Healthcare", sample_text)
    print(f"Relevance: {relevance}")
    
    print("Data processing tests completed!")
