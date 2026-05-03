"""
API Gatekeeper Service.
Simulates rate limiting and wraps data generation calls.
"""
import time

class ApiGatekeeper:
    def __init__(self, requests_per_minute: int = 60):
        self.rate_limit = requests_per_minute
        self.calls_made = 0
        self.start_time = time.time()

    def execute_call(self, func, *args, **kwargs):
        """Wraps a function call to enforce simulated rate limits."""
        self._enforce_rate_limit()
        self.calls_made += 1
        return func(*args, **kwargs)

    def _enforce_rate_limit(self):
        """Basic simulation of rate limiting."""
        elapsed = time.time() - self.start_time
        if elapsed > 60:
            # Reset every minute
            self.start_time = time.time()
            self.calls_made = 0
            
        if self.calls_made >= self.rate_limit:
            raise Exception("Rate limit exceeded. Please try again later.")