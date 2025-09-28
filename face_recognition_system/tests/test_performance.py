"""Performance tests for API endpoints."""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from fastapi import status

class TestPerformance:
    """Test API performance."""
    
    def test_response_time_health_check(self, client: TestClient):
        """Test health check response time."""
        start_time = time.time()
        response = client.get("/api/v1/system/health")
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second
    
    def test_response_time_authentication(self, client: TestClient):
        """Test authentication response time."""
        start_time = time.time()
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        response_time = end_time - start_time
        assert response_time < 2.0  # Should respond within 2 seconds
    
    def test_concurrent_requests(self, client: TestClient):
        """Test handling concurrent requests."""
        def make_request():
            return client.get("/api/v1/system/health")
        
        # Make 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
    
    def test_rate_limit_performance(self, client: TestClient, auth_headers_admin):
        """Test rate limiting doesn't significantly impact performance."""
        response_times = []
        
        # Make multiple requests and measure response times
        for _ in range(5):
            start_time = time.time()
            response = client.get(
                "/api/v1/auth/me",
                headers=auth_headers_admin
            )
            end_time = time.time()
            
            assert response.status_code == status.HTTP_200_OK
            response_times.append(end_time - start_time)
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 1.0  # Should average under 1 second
    
    def test_large_payload_handling(self, client: TestClient, auth_headers_admin):
        """Test handling of large payloads."""
        # Create a large but valid person data payload
        large_notes = "x" * 1000  # 1KB of text
        large_person_data = {
            "name": "Test Person with Large Data",
            "employee_id": "EMP999",
            "email": "large@example.com",
            "notes": large_notes,
            "access_level": 1
        }
        
        start_time = time.time()
        response = client.post(
            "/api/v1/persons/",
            headers=auth_headers_admin,
            json=large_person_data
        )
        end_time = time.time()
        
        # Should handle large payload reasonably quickly
        response_time = end_time - start_time
        assert response_time < 3.0  # Should respond within 3 seconds
        assert response.status_code == status.HTTP_201_CREATED

class TestLoadTesting:
    """Basic load testing."""
    
    def test_sustained_load(self, client: TestClient):
        """Test sustained load on health endpoint."""
        success_count = 0
        total_requests = 50
        
        start_time = time.time()
        
        for _ in range(total_requests):
            response = client.get("/api/v1/system/health")
            if response.status_code == status.HTTP_200_OK:
                success_count += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate requests per second
        rps = total_requests / total_time
        
        # Should handle at least 10 requests per second
        assert rps >= 10
        # Should have high success rate
        success_rate = success_count / total_requests
        assert success_rate >= 0.95  # 95% success rate
    
    def test_memory_usage_stability(self, client: TestClient, auth_headers_admin):
        """Test that memory usage remains stable under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make many requests
        for _ in range(100):
            response = client.get(
                "/api/v1/auth/me",
                headers=auth_headers_admin
            )
            assert response.status_code == status.HTTP_200_OK
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024

class TestScalability:
    """Test API scalability characteristics."""
    
    def test_response_time_with_load(self, client: TestClient):
        """Test response time degradation under load."""
        # Measure baseline response time
        start_time = time.time()
        response = client.get("/api/v1/system/health")
        baseline_time = time.time() - start_time
        
        assert response.status_code == status.HTTP_200_OK
        
        # Measure response time under load
        def make_background_requests():
            for _ in range(20):
                client.get("/api/v1/system/health")
        
        # Start background load
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.submit(make_background_requests)
            
            # Measure response time during load
            start_time = time.time()
            response = client.get("/api/v1/system/health")
            load_time = time.time() - start_time
        
        assert response.status_code == status.HTTP_200_OK
        
        # Response time under load should not be more than 3x baseline
        assert load_time <= baseline_time * 3
    
    def test_authentication_scalability(self, client: TestClient):
        """Test authentication performance under load."""
        def authenticate():
            return client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": "admin123"}
            )
        
        # Test concurrent authentication requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(authenticate) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All authentication requests should succeed
        success_count = sum(1 for r in responses if r.status_code == status.HTTP_200_OK)
        assert success_count >= 8  # At least 80% should succeed