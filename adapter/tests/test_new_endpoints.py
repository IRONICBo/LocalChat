# -*- coding: utf-8 -*-
"""
Test script for new mask/unmask API endpoints

This script tests the new standalone mask, unmask, and analyze endpoints
without requiring the full server to be running.

Usage:
    python tests/test_new_endpoints.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app


def test_mask_endpoint():
    """Test the /api/mask endpoint"""
    client = TestClient(app)

    # Test basic masking
    response = client.post("/api/mask", json={
        "text": "Contact John Smith at john.smith@email.com or call 555-123-4567",
        "enable_smart_mode": False
    })

    print("\n=== Test /api/mask ===")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Masked text: {data['masked_text']}")
        print(f"Entities count: {data['entities_count']}")
        print(f"Session ID: {data['session_id']}")
        print(f"Processing time: {data['processing_time_ms']}ms")

        # Return session_id for unmask test
        return data['session_id'], data['masked_text']
    else:
        print(f"Error: {response.json()}")
        return None, None


def test_unmask_endpoint(session_id: str, masked_text: str):
    """Test the /api/unmask endpoint"""
    client = TestClient(app)

    response = client.post("/api/unmask", json={
        "masked_text": masked_text,
        "session_id": session_id
    })

    print("\n=== Test /api/unmask ===")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Original text: {data['original_text']}")
        print(f"Entities recovered: {data['entities_recovered']}")
        print(f"Processing time: {data['processing_time_ms']}ms")
    else:
        print(f"Error: {response.json()}")


def test_analyze_endpoint():
    """Test the /api/analyze endpoint"""
    client = TestClient(app)

    # Test with financial text
    response = client.post("/api/analyze", json={
        "text": "Customer credit card 4532-1111-2222-3333 charged $150. Contact billing@bank.com.",
        "source_type": "document"
    })

    print("\n=== Test /api/analyze ===")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Analysis:")
        print(f"  Domain: {data['analysis']['domain']}")
        print(f"  Complexity: {data['analysis']['complexity_score']}")
        print(f"Recommendations:")
        print(f"  Strategy: {data['recommendations']['strategy']}")
        print(f"  Model: {data['recommendations']['model_size']}")
        print(f"Estimated PII: {data['estimated_pii_counts']}")
    else:
        print(f"Error: {response.json()}")


def test_mask_unmask_demo():
    """Test the /api/mask-unmask endpoint"""
    client = TestClient(app)

    response = client.post(
        "/api/mask-unmask",
        params={
            "text": "Hello, I am Jane Doe. Email me at jane@example.com",
            "enable_smart_mode": True
        }
    )

    print("\n=== Test /api/mask-unmask ===")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Workflow: {data['workflow']}")
        print(f"Step 1 - Mask:")
        print(f"  Input: {data['step_1_mask']['input']}")
        print(f"  Output: {data['step_1_mask']['output']}")
        print(f"  Entities: {data['step_1_mask']['entities_detected']}")
        print(f"Step 2 - LLM:")
        print(f"  Response: {data['step_2_llm']['simulated_response'][:80]}...")
        print(f"Step 3 - Unmask:")
        print(f"  Output: {data['step_3_unmask']['output'][:80]}...")
        print(f"Metadata:")
        print(f"  Strategy: {data['metadata']['detection_strategy']}")
        print(f"  Smart Mode: {data['metadata']['smart_mode_used']}")
        print(f"  Total time: {data['metadata']['total_processing_time_ms']}ms")
    else:
        print(f"Error: {response.json()}")


def test_smart_mode_mask():
    """Test masking with Smart Mode enabled"""
    client = TestClient(app)

    # Test with financial text (should use high_precision strategy)
    response = client.post("/api/mask", json={
        "text": "Payment of $500 with card 4111-1111-1111-1111. Account balance: $2,350.00",
        "enable_smart_mode": True,
        "source_type": "document"
    })

    print("\n=== Test Smart Mode Mask ===")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Smart Mode Used: {data['smart_mode_used']}")
        print(f"Recommended Strategy: {data['recommended_strategy']}")
        print(f"Entities count: {data['entities_count']}")
    else:
        print(f"Error: {response.json()}")


def test_root_endpoint():
    """Test the root endpoint to verify new features are listed"""
    client = TestClient(app)

    response = client.get("/")

    print("\n=== Test Root Endpoint ===")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Version: {data['version']}")
        print(f"Features:")
        for feature, enabled in data['configuration']['features'].items():
            print(f"  {feature}: {enabled}")
        print(f"Endpoints: {list(data['endpoints'].keys())}")
    else:
        print(f"Error: {response.json()}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing New API Endpoints")
    print("=" * 60)

    # Test root endpoint
    test_root_endpoint()

    # Test mask and unmask flow
    session_id, masked_text = test_mask_endpoint()
    if session_id and masked_text:
        test_unmask_endpoint(session_id, masked_text)

    # Test analyze endpoint
    test_analyze_endpoint()

    # Test smart mode
    test_smart_mode_mask()

    # Test complete demo
    test_mask_unmask_demo()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
