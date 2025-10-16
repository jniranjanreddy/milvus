#!/bin/bash

# Test Milvus Connection and Functionality

set -e

echo "=== Testing Milvus ==="
echo ""

# Port forward to Milvus
echo "1. Starting port-forward to Milvus..."
kubectl port-forward -n data svc/milvus 19530:19530 > /dev/null 2>&1 &
PF_PID=$!
echo "   Port-forward PID: $PF_PID"
sleep 5
echo ""

# Run Python test
echo "2. Running comprehensive tests..."
echo ""
python3 test-milvus.py
TEST_RESULT=$?
echo ""

# Cleanup
echo "3. Cleaning up port-forward..."
kill $PF_PID 2>/dev/null || true
wait $PF_PID 2>/dev/null || true
echo "   ✓ Port-forward stopped"
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo "=== All Tests Passed ✓ ==="
    exit 0
else
    echo "=== Tests Failed ✗ ==="
    exit 1
fi

