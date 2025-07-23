"""Custom exception classes for the AI Blog Generation Team."""

import time
import asyncio
from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BlogGenerationError(Exception):
    """Base exception for blog generation errors."""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize blog generation error.
        
        Args:
            message: Human-readable error message
            severity: Error severity level
            error_code: Optional error code for categorization
            context: Optional context information
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.error_code = error_code
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.value,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp,
            'original_error': str(self.original_error) if self.original_error else None
        }


class ResearchError(BlogGenerationError):
    """Research phase failed."""
    
    def __init__(
        self, 
        message: str, 
        topic: Optional[str] = None,
        search_query: Optional[str] = None,
        **kwargs
    ):
        """Initialize research error.
        
        Args:
            message: Error message
            topic: Research topic that failed
            search_query: Search query that caused the error
            **kwargs: Additional arguments for BlogGenerationError
        """
        context = kwargs.get('context', {})
        if topic:
            context['topic'] = topic
        if search_query:
            context['search_query'] = search_query
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'RESEARCH_FAILED')
        super().__init__(message, **kwargs)


class WritingError(BlogGenerationError):
    """Writing phase failed."""
    
    def __init__(
        self, 
        message: str, 
        topic: Optional[str] = None,
        draft_stage: Optional[str] = None,
        word_count: Optional[int] = None,
        **kwargs
    ):
        """Initialize writing error.
        
        Args:
            message: Error message
            topic: Topic being written about
            draft_stage: Stage of draft creation (initial, revision)
            word_count: Current word count if available
            **kwargs: Additional arguments for BlogGenerationError
        """
        context = kwargs.get('context', {})
        if topic:
            context['topic'] = topic
        if draft_stage:
            context['draft_stage'] = draft_stage
        if word_count:
            context['word_count'] = word_count
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'WRITING_FAILED')
        super().__init__(message, **kwargs)


class CritiqueError(BlogGenerationError):
    """Critique phase failed."""
    
    def __init__(
        self, 
        message: str, 
        draft_title: Optional[str] = None,
        analysis_stage: Optional[str] = None,
        **kwargs
    ):
        """Initialize critique error.
        
        Args:
            message: Error message
            draft_title: Title of draft being critiqued
            analysis_stage: Stage of analysis (clarity, facts, structure)
            **kwargs: Additional arguments for BlogGenerationError
        """
        context = kwargs.get('context', {})
        if draft_title:
            context['draft_title'] = draft_title
        if analysis_stage:
            context['analysis_stage'] = analysis_stage
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'CRITIQUE_FAILED')
        super().__init__(message, **kwargs)


class OrchestrationError(BlogGenerationError):
    """Orchestration workflow failed."""
    
    def __init__(
        self, 
        message: str, 
        workflow_stage: Optional[str] = None,
        iteration_count: Optional[int] = None,
        **kwargs
    ):
        """Initialize orchestration error.
        
        Args:
            message: Error message
            workflow_stage: Current workflow stage
            iteration_count: Current iteration number
            **kwargs: Additional arguments for BlogGenerationError
        """
        context = kwargs.get('context', {})
        if workflow_stage:
            context['workflow_stage'] = workflow_stage
        if iteration_count:
            context['iteration_count'] = iteration_count
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'ORCHESTRATION_FAILED')
        super().__init__(message, **kwargs)


class APIError(BlogGenerationError):
    """API-related errors (rate limits, timeouts, etc.)."""
    
    def __init__(
        self, 
        message: str, 
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """Initialize API error.
        
        Args:
            message: Error message
            api_name: Name of the API that failed
            status_code: HTTP status code if applicable
            retry_after: Seconds to wait before retry
            **kwargs: Additional arguments for BlogGenerationError
        """
        context = kwargs.get('context', {})
        if api_name:
            context['api_name'] = api_name
        if status_code:
            context['status_code'] = status_code
        if retry_after:
            context['retry_after'] = retry_after
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'API_ERROR')
        super().__init__(message, **kwargs)


class TimeoutError(BlogGenerationError):
    """Operation timeout error."""
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        timeout_duration: Optional[float] = None,
        **kwargs
    ):
        """Initialize timeout error.
        
        Args:
            message: Error message
            operation: Operation that timed out
            timeout_duration: Timeout duration in seconds
            **kwargs: Additional arguments for BlogGenerationError
        """
        context = kwargs.get('context', {})
        if operation:
            context['operation'] = operation
        if timeout_duration:
            context['timeout_duration'] = timeout_duration
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'TIMEOUT')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ValidationError(BlogGenerationError):
    """Data validation error."""
    
    def __init__(
        self, 
        message: str, 
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        **kwargs
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            field_name: Name of field that failed validation
            invalid_value: The invalid value
            **kwargs: Additional arguments for BlogGenerationError
        """
        context = kwargs.get('context', {})
        if field_name:
            context['field_name'] = field_name
        if invalid_value is not None:
            context['invalid_value'] = str(invalid_value)
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'VALIDATION_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        super().__init__(message, **kwargs)