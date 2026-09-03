"""Error handling and resilience patterns for audio pipeline."""

import asyncio
import logging
import time
from typing import Callable, Optional, TypeVar, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout_s: float = 30.0  # Time before half-open
    success_threshold: int = 2  # Successes to close from half-open


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics."""

    total_calls: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    state_change_time: Optional[datetime] = None


class CircuitBreaker:
    """Implements circuit breaker pattern for service calls."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.

        Args:
            name: Service name
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.state_change_time = datetime.now()

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception if circuit is OPEN or function fails
        """
        if self.state == CircuitState.OPEN:
            if self.is_recovery_timeout_exceeded():
                self.state = CircuitState.HALF_OPEN
                self.metrics.consecutive_failures = 0
                logger.info(f"Circuit {self.name}: transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit {self.name} is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def _record_success(self) -> None:
        """Record successful call."""
        self.metrics.total_calls += 1
        self.metrics.consecutive_failures = 0

        if self.state == CircuitState.HALF_OPEN:
            if self.metrics.consecutive_failures == 0:
                # Check if we've had enough successes to close
                self.state = CircuitState.CLOSED
                logger.info(f"Circuit {self.name}: CLOSED")

    def _record_failure(self) -> None:
        """Record failed call."""
        self.metrics.total_calls += 1
        self.metrics.total_failures += 1
        self.metrics.consecutive_failures += 1
        self.metrics.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # Reopen on any failure while testing
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name}: reopened (failure in HALF_OPEN)")
        elif (
            self.state == CircuitState.CLOSED
            and self.metrics.consecutive_failures >= self.config.failure_threshold
        ):
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit {self.name}: OPEN (threshold {self.config.failure_threshold} exceeded)"
            )

    def is_recovery_timeout_exceeded(self) -> bool:
        """Check if recovery timeout has elapsed."""
        if not self.metrics.last_failure_time:
            return False

        elapsed = (datetime.now() - self.metrics.last_failure_time).total_seconds()
        return elapsed >= self.config.recovery_timeout_s

    def get_metrics(self) -> dict:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.metrics.total_calls,
            "total_failures": self.metrics.total_failures,
            "consecutive_failures": self.metrics.consecutive_failures,
            "failure_rate": (
                self.metrics.total_failures / self.metrics.total_calls
                if self.metrics.total_calls > 0
                else 0.0
            ),
        }


class RetryPolicy:
    """Retry policy with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay_ms: float = 100,
        max_delay_ms: float = 5000,
        backoff_multiplier: float = 2.0,
    ):
        """Initialize retry policy.

        Args:
            max_retries: Maximum number of retries
            initial_delay_ms: Initial delay in milliseconds
            max_delay_ms: Maximum delay in milliseconds
            backoff_multiplier: Exponential backoff multiplier
        """
        self.max_retries = max_retries
        self.initial_delay_ms = initial_delay_ms
        self.max_delay_ms = max_delay_ms
        self.backoff_multiplier = backoff_multiplier

    async def execute(
        self, func: Callable[..., T], *args, retryable_exceptions: tuple = None, **kwargs
    ) -> T:
        """Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            retryable_exceptions: Tuple of exception types to retry on
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        retryable_exceptions = retryable_exceptions or (Exception,)
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{self.max_retries} after {delay}ms: {e}"
                    )
                    await asyncio.sleep(delay / 1000.0)
                else:
                    logger.error(f"Exhausted retries after {self.max_retries + 1} attempts")

        raise last_exception

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for attempt using exponential backoff."""
        delay = self.initial_delay_ms * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay_ms)


@dataclass
class PipelineError:
    """Structured error for audio pipeline."""

    error_type: str  # e.g., "stt_timeout", "llm_api_error"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict = field(default_factory=dict)
    recoverable: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "recoverable": self.recoverable,
        }


class ErrorAccumulator:
    """Accumulates and analyzes pipeline errors."""

    def __init__(self, window_size: int = 100):
        """Initialize error accumulator.

        Args:
            window_size: Number of recent errors to track
        """
        self.window_size = window_size
        self.errors: list[PipelineError] = []

    def add_error(self, error: PipelineError) -> None:
        """Add error to accumulator."""
        self.errors.append(error)
        if len(self.errors) > self.window_size:
            self.errors.pop(0)

    def get_error_rate(self, error_type: Optional[str] = None) -> float:
        """Get error rate for specific type or overall."""
        if not self.errors:
            return 0.0

        if error_type:
            matching = [e for e in self.errors if e.error_type == error_type]
            return len(matching) / len(self.errors)

        return 1.0  # All accumulated are errors

    def get_summary(self) -> dict:
        """Get error summary."""
        if not self.errors:
            return {"total_errors": 0}

        error_counts = {}
        for error in self.errors:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1

        return {
            "total_errors": len(self.errors),
            "error_types": error_counts,
            "oldest_error": self.errors[0].timestamp.isoformat(),
            "latest_error": self.errors[-1].timestamp.isoformat(),
        }
