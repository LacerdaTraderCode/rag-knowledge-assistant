"""Shared pytest fixtures for the test suite."""

import httpx2
import pytest


def mock_client(handler) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


@pytest.fixture
def build_mock_client():
    return mock_client
