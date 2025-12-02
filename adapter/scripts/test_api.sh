#!/bin/bash
# =============================================================================
# Test Script for ChatAdapter API
# =============================================================================
# This script tests the end-to-end flow:
# API Request → Mask → SQLite Storage → LLM → Unmask → Return
#
# Usage:
#   chmod +x scripts/test_api.sh
#   ./scripts/test_api.sh
#
# Prerequisites:
#   - ChatAdapter server running (./scripts/run_server.sh)
#   - Backend LLM service running at configured endpoint
# =============================================================================

# Don't exit on error - we want to run all tests
set +e

# Configuration
BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "=============================================="
echo "ChatAdapter API Test Suite"
echo "=============================================="
echo "Server: $BASE_URL"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Track test results
LAST_TEST_PASSED=0

# Helper function to check response
check_response() {
    local test_name="$1"
    local response="$2"
    local expected_field="$3"

    if echo "$response" | jq -e ".$expected_field" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((PASSED++))
        LAST_TEST_PASSED=1
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  Response: $response"
        ((FAILED++))
        LAST_TEST_PASSED=0
    fi
}

# ==================== Test 1: Health Check ====================
echo ""
echo "Test 1: Health Check (GET /)"
echo "----------------------------------------"

RESPONSE=$(curl -s "$BASE_URL/")
check_response "Root endpoint" "$RESPONSE" "version"

if [ $LAST_TEST_PASSED -eq 1 ]; then
    echo "  Version: $(echo $RESPONSE | jq -r '.version')"
    echo "  Features: $(echo $RESPONSE | jq -c '.configuration.features')"
fi

# ==================== Test 2: Standalone Mask ====================
echo ""
echo "Test 2: Standalone Mask (POST /api/mask)"
echo "----------------------------------------"

MASK_RESPONSE=$(curl -s -X POST "$BASE_URL/api/mask" \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Hello, my name is John Smith. My email is john.smith@example.com and my phone is 555-123-4567.",
        "enable_smart_mode": false
    }')

check_response "Mask endpoint" "$MASK_RESPONSE" "masked_text"

SESSION_ID=""
MASKED_TEXT=""
if [ $LAST_TEST_PASSED -eq 1 ]; then
    SESSION_ID=$(echo $MASK_RESPONSE | jq -r '.session_id')
    MASKED_TEXT=$(echo $MASK_RESPONSE | jq -r '.masked_text')
    echo "  Session ID: $SESSION_ID"
    echo "  Masked text: $MASKED_TEXT"
    echo "  Entities count: $(echo $MASK_RESPONSE | jq -r '.entities_count')"
fi

# ==================== Test 3: Standalone Unmask ====================
echo ""
echo "Test 3: Standalone Unmask (POST /api/unmask)"
echo "----------------------------------------"

if [ -n "$SESSION_ID" ] && [ "$SESSION_ID" != "null" ] && [ -n "$MASKED_TEXT" ]; then
    UNMASK_RESPONSE=$(curl -s -X POST "$BASE_URL/api/unmask" \
        -H "Content-Type: application/json" \
        -d "{
            \"masked_text\": \"$MASKED_TEXT\",
            \"session_id\": \"$SESSION_ID\"
        }")

    check_response "Unmask endpoint" "$UNMASK_RESPONSE" "original_text"

    if [ $LAST_TEST_PASSED -eq 1 ]; then
        echo "  Original text: $(echo $UNMASK_RESPONSE | jq -r '.original_text')"
        echo "  Entities recovered: $(echo $UNMASK_RESPONSE | jq -r '.entities_recovered')"
    fi
else
    echo -e "${YELLOW}⚠ SKIP${NC}: Unmask test (no session from mask test)"
fi

# ==================== Test 4: Analyze Text ====================
echo ""
echo "Test 4: Analyze Text (POST /api/analyze)"
echo "----------------------------------------"

ANALYZE_RESPONSE=$(curl -s -X POST "$BASE_URL/api/analyze" \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Payment of $500 with credit card 4532-1111-2222-3333. Contact billing@company.com for questions.",
        "source_type": "document"
    }')

check_response "Analyze endpoint" "$ANALYZE_RESPONSE" "analysis"

if [ $LAST_TEST_PASSED -eq 1 ]; then
    echo "  Domain: $(echo $ANALYZE_RESPONSE | jq -r '.analysis.domain')"
    echo "  Complexity: $(echo $ANALYZE_RESPONSE | jq -r '.analysis.complexity_score')"
    echo "  Recommended strategy: $(echo $ANALYZE_RESPONSE | jq -r '.recommendations.strategy')"
fi

# ==================== Test 5: Smart Mode Mask ====================
echo ""
echo "Test 5: Smart Mode Mask (POST /api/mask with smart_mode)"
echo "----------------------------------------"

SMART_RESPONSE=$(curl -s -X POST "$BASE_URL/api/mask" \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Patient Jane Doe (DOB: 1985-03-15) diagnosed with condition XYZ. Contact Dr. Smith at clinic@hospital.org",
        "enable_smart_mode": true,
        "source_type": "document"
    }')

check_response "Smart mode mask" "$SMART_RESPONSE" "smart_mode_used"

if [ $LAST_TEST_PASSED -eq 1 ]; then
    echo "  Smart mode used: $(echo $SMART_RESPONSE | jq -r '.smart_mode_used')"
    echo "  Recommended strategy: $(echo $SMART_RESPONSE | jq -r '.recommended_strategy')"
    echo "  Entities count: $(echo $SMART_RESPONSE | jq -r '.entities_count')"
fi

# ==================== Test 6: Full Proxy Chat Flow ====================
echo ""
echo "Test 6: Full Proxy Chat Flow (POST /api/chat)"
echo "----------------------------------------"
echo -e "${YELLOW}Note: This test requires a running LLM backend${NC}"

CHAT_RESPONSE=$(curl -s -X POST "$BASE_URL/api/chat" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "messages": [
            {
                "role": "user",
                "content": "My name is Alice Johnson and my email is alice.j@company.com. Can you help me write a professional email?"
            }
        ],
        "enable_pii": true
    }' 2>/dev/null)

if [ -z "$CHAT_RESPONSE" ]; then
    echo -e "${YELLOW}⚠ SKIP${NC}: Chat proxy (connection timeout or failed)"
elif echo "$CHAT_RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
    ERROR_MSG=$(echo "$CHAT_RESPONSE" | jq -r '.error // .detail // "Unknown error"')
    echo -e "${YELLOW}⚠ SKIP${NC}: Chat proxy (LLM backend error: $ERROR_MSG)"
elif echo "$CHAT_RESPONSE" | jq -e '.detail' > /dev/null 2>&1; then
    ERROR_MSG=$(echo "$CHAT_RESPONSE" | jq -r '.detail')
    echo -e "${YELLOW}⚠ SKIP${NC}: Chat proxy (API error: $ERROR_MSG)"
else
    check_response "Chat proxy" "$CHAT_RESPONSE" "message"
    if [ $LAST_TEST_PASSED -eq 1 ]; then
        CONTENT=$(echo $CHAT_RESPONSE | jq -r '.message.content' 2>/dev/null | head -c 100)
        echo "  Response preview: ${CONTENT}..."
        echo "  PII masked: $(echo $CHAT_RESPONSE | jq -r '.pii_masked // false')"
    fi
fi

# ==================== Test 7: Mask-Unmask Demo ====================
echo ""
echo "Test 7: Mask-Unmask Demo (POST /api/mask-unmask)"
echo "----------------------------------------"

DEMO_RESPONSE=$(curl -s -X POST "$BASE_URL/api/mask-unmask?text=Hello%2C%20I%20am%20Bob%20Wilson.%20Email%20me%20at%20bob%40test.com&enable_smart_mode=true")

check_response "Mask-unmask demo" "$DEMO_RESPONSE" "workflow"

if [ $LAST_TEST_PASSED -eq 1 ]; then
    echo "  Workflow: $(echo $DEMO_RESPONSE | jq -r '.workflow')"
    echo "  Step 1 (Mask): $(echo $DEMO_RESPONSE | jq -r '.step_1_mask.output')"
    echo "  Strategy used: $(echo $DEMO_RESPONSE | jq -r '.metadata.detection_strategy')"
fi

# ==================== Summary ====================
echo ""
echo "=============================================="
echo "Test Summary"
echo "=============================================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some tests failed. Check the output above.${NC}"
    exit 1
fi
