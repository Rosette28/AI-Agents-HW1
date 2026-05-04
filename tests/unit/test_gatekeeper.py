"""
Unit tests for the API Gatekeeper service.
"""
import pytest
from src.services.gatekeeper import ApiGatekeeper


def test_gatekeeper_allows_valid_calls():
    # Set limit to 2 calls per minute
    gatekeeper = ApiGatekeeper(requests_per_minute=2)

    # Create a dummy function to test
    def dummy_func(x):
        return x * 2

    # First call should pass
    result1 = gatekeeper.execute_call(dummy_func, 5)
    assert result1 == 10
    assert gatekeeper.calls_made == 1

    # Second call should pass
    result2 = gatekeeper.execute_call(dummy_func, 10)
    assert result2 == 20
    assert gatekeeper.calls_made == 2


def test_gatekeeper_blocks_excess_calls():
    gatekeeper = ApiGatekeeper(requests_per_minute=1)

    def dummy_func():
        return "success"

    # First call passes
    gatekeeper.execute_call(dummy_func)

    # Second call must raise an exception
    with pytest.raises(Exception, match="Rate limit exceeded"):
        gatekeeper.execute_call(dummy_func)