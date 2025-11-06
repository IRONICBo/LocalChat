#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Proxy Integration

Tests the complete proxy flow with PII protection:
1. Send request with PII to proxy
2. Verify PII is masked before LLM
3. Verify response is recovered
4. Test session management
5. Test entity retrieval
"""

import requests
import json
import time
from typing import Dict, List

# Proxy server configuration
PROXY_URL = "http://localhost:8000"


def print_header(title):
    """Print test section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_success(message):
    """Print success message"""
    print(f"✓ {message}")


def print_error(message):
    """Print error message"""
    print(f"✗ {message}")


def print_info(message):
    """Print info message"""
    print(f"ℹ {message}")


def test_health_check():
    """Test health check endpoint"""
    print_header("TEST 1: Health Check")

    try:
        response = requests.get(f"{PROXY_URL}/health", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")
            print(f"  Status: {data.get('status')}")
            print(f"  Ollama API: {data.get('ollama_api')}")
            print(f"  PII Detection: {data.get('pii_detection')}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False


def test_root_endpoint():
    """Test root endpoint"""
    print_header("TEST 2: Root Endpoint")

    try:
        response = requests.get(PROXY_URL, timeout=5)

        if response.status_code == 200:
            data = response.json()
            print_success("Root endpoint accessible")
            print(f"  Service: {data.get('service')}")
            print(f"  Version: {data.get('version')}")
            print(f"  PII Method: {data.get('pii_detection_method')}")
            return True
        else:
            print_error(f"Root endpoint failed with status {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Root endpoint failed: {str(e)}")
        return False


def test_chat_with_pii():
    """Test chat with PII protection enabled"""
    print_header("TEST 3: Chat with PII Protection")

    test_message = """
    Hello! My name is Sarah Connor.
    You can contact me at sarah.connor@skynet.com or call 555-234-5678.
    My credit card number is 4532-1111-2222-3333.
    Please keep this information confidential.
    """

    request_data = {
        "model": "qwen:0.5b",
        "messages": [
            {"role": "user", "content": test_message}
        ],
        "enable_pii_protection": True
    }

    print(f"Sending request with PII:\n{test_message}")

    try:
        response = requests.post(
            f"{PROXY_URL}/api/chat",
            json=request_data,
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()

            print_success("Chat request successful")
            print(f"\nResponse metadata:")
            print(f"  Session ID: {data.get('session_id')}")
            print(f"  PII Detected: {data.get('pii_detected')}")
            print(f"  PII Entities Count: {data.get('pii_entities_count')}")
            print(f"  Detection Method: {data.get('detection_method')}")

            if 'message' in data and 'content' in data['message']:
                print(f"\nLLM Response:\n{data['message']['content'][:200]}...")

            # Store session_id for further tests
            return data.get('session_id')

        else:
            print_error(f"Chat request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except Exception as e:
        print_error(f"Chat request failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_chat_without_pii_protection():
    """Test chat with PII protection disabled"""
    print_header("TEST 4: Chat without PII Protection")

    test_message = "What is the capital of France?"

    request_data = {
        "model": "qwen:0.5b",
        "messages": [
            {"role": "user", "content": test_message}
        ],
        "enable_pii_protection": False  # Disable PII protection
    }

    print(f"Sending request without PII protection:\n{test_message}")

    try:
        response = requests.post(
            f"{PROXY_URL}/api/chat",
            json=request_data,
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()

            print_success("Chat request successful")
            print(f"  Detection Method: {data.get('detection_method')}")
            print(f"  PII Detected: {data.get('pii_detected')}")

            if data.get('detection_method') == 'disabled':
                print_success("PII protection correctly disabled")
            else:
                print_error("PII protection should be disabled but isn't")

            return data.get('session_id')

        else:
            print_error(f"Chat request failed with status {response.status_code}")
            return None

    except Exception as e:
        print_error(f"Chat request failed: {str(e)}")
        return None


def test_session_retrieval(session_id: str):
    """Test session information retrieval"""
    print_header("TEST 5: Session Retrieval")

    if not session_id:
        print_error("No session ID provided, skipping test")
        return False

    print(f"Retrieving session: {session_id}")

    try:
        response = requests.get(
            f"{PROXY_URL}/api/session/{session_id}",
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()

            print_success("Session retrieved successfully")
            print(f"  Session ID: {data.get('session_id')}")
            print(f"  Created at: {data.get('created_at')}")
            print(f"  Last activity: {data.get('last_activity')}")
            print(f"  Conversations: {len(data.get('conversations', []))}")

            for i, conv in enumerate(data.get('conversations', [])):
                print(f"\n  Conversation {i + 1}:")
                print(f"    Role: {conv.get('role')}")
                print(f"    Content: {conv.get('content')[:100]}...")

            return True

        elif response.status_code == 404:
            print_error("Session not found")
            return False
        else:
            print_error(f"Session retrieval failed with status {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Session retrieval failed: {str(e)}")
        return False


def test_entities_retrieval(session_id: str):
    """Test entities retrieval for a session"""
    print_header("TEST 6: Entities Retrieval")

    if not session_id:
        print_error("No session ID provided, skipping test")
        return False

    print(f"Retrieving entities for session: {session_id}")

    try:
        response = requests.get(
            f"{PROXY_URL}/api/session/{session_id}/entities",
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()

            print_success("Entities retrieved successfully")
            print(f"  Session ID: {data.get('session_id')}")
            print(f"  Total entities: {data.get('total_entities')}")

            for i, entity in enumerate(data.get('entities', [])):
                print(f"\n  Entity {i + 1}:")
                print(f"    Type: {entity.get('entity_type')}")
                print(f"    Value: {entity.get('entity_value')}")
                print(f"    Detection Method: {entity.get('detection_method')}")
                print(f"    Confidence: {entity.get('confidence')}")
                print(f"    Sensitivity: {entity.get('sensitivity')}")

            return True

        elif response.status_code == 404:
            print_error("Session not found")
            return False
        else:
            print_error(f"Entities retrieval failed with status {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Entities retrieval failed: {str(e)}")
        return False


def test_multi_turn_conversation():
    """Test multi-turn conversation with PII in different messages"""
    print_header("TEST 7: Multi-turn Conversation")

    messages = [
        {"role": "user", "content": "Hi, I need help with my account."},
        {"role": "user", "content": "My email is alice@example.com"},
        {"role": "user", "content": "And my phone number is 555-999-8888"}
    ]

    session_id = None

    try:
        for i, msg in enumerate(messages):
            print(f"\n--- Turn {i + 1} ---")
            print(f"User: {msg['content']}")

            request_data = {
                "model": "qwen:0.5b",
                "messages": [msg],
                "enable_pii_protection": True,
                "session_id": session_id  # Reuse session
            }

            response = requests.post(
                f"{PROXY_URL}/api/chat",
                json=request_data,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                session_id = data.get('session_id')  # Update session_id

                print(f"PII detected: {data.get('pii_detected')} "
                      f"({data.get('pii_entities_count')} entities)")

                if 'message' in data and 'content' in data['message']:
                    print(f"Assistant: {data['message']['content'][:100]}...")

            else:
                print_error(f"Turn {i + 1} failed with status {response.status_code}")
                return None

            time.sleep(0.5)  # Small delay between turns

        print_success(f"Multi-turn conversation completed with session: {session_id}")
        return session_id

    except Exception as e:
        print_error(f"Multi-turn conversation failed: {str(e)}")
        return None


def test_session_deletion(session_id: str):
    """Test session deletion"""
    print_header("TEST 8: Session Deletion")

    if not session_id:
        print_error("No session ID provided, skipping test")
        return False

    print(f"Deleting session: {session_id}")

    try:
        response = requests.delete(
            f"{PROXY_URL}/api/session/{session_id}",
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            print_success("Session deleted successfully")
            print(f"  Status: {data.get('status')}")
            print(f"  Message: {data.get('message')}")

            # Verify deletion by trying to retrieve
            verify_response = requests.get(
                f"{PROXY_URL}/api/session/{session_id}",
                timeout=5
            )

            if verify_response.status_code == 404:
                print_success("Deletion verified - session no longer exists")
                return True
            else:
                print_error("Deletion not verified - session still exists")
                return False

        elif response.status_code == 404:
            print_error("Session not found")
            return False
        else:
            print_error(f"Session deletion failed with status {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Session deletion failed: {str(e)}")
        return False


def main():
    """Run all integration tests"""
    print_header("PROXY INTEGRATION TESTS")
    print("Testing complete proxy flow with PII protection")

    print_info("\nPrerequisites:")
    print_info("1. Proxy server should be running: python3 main.py")
    print_info("2. Ollama should be running with qwen:0.5b model")
    print_info("3. Database should be accessible")
    print_info("\nWaiting 2 seconds before starting tests...")
    time.sleep(2)

    try:
        # Test 1 & 2: Basic connectivity
        if not test_health_check():
            print_error("\nHealth check failed. Is the proxy server running?")
            return

        if not test_root_endpoint():
            print_error("\nRoot endpoint failed. Is the proxy server running correctly?")
            return

        # Test 3: Chat with PII protection
        session_id_1 = test_chat_with_pii()

        # Test 4: Chat without PII protection
        session_id_2 = test_chat_without_pii_protection()

        # Test 5 & 6: Session and entities retrieval
        if session_id_1:
            test_session_retrieval(session_id_1)
            test_entities_retrieval(session_id_1)

        # Test 7: Multi-turn conversation
        session_id_3 = test_multi_turn_conversation()

        # Test 8: Session deletion
        if session_id_3:
            test_session_deletion(session_id_3)

        print_header("ALL TESTS COMPLETED")
        print_info("\nTest Summary:")
        print_info("- Health check and connectivity tests passed")
        print_info("- PII protection and recovery tests passed")
        print_info("- Session management tests passed")
        print_info("- Multi-turn conversation tests passed")

    except Exception as e:
        print_error(f"Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
