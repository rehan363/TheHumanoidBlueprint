"""
Test chat endpoint with minimal payload to isolate the error.
"""
import requests
import json

# Test 1: Simple health check
print("Test 1: Health Check")
print("=" * 60)
response = requests.get("http://localhost:8000/api/health")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print(f"✅ Health check passed")
    print(f"Response: {response.json()}")
else:
    print(f"❌ Health check failed")
print()

# Test 2: Check if the endpoint exists
print("Test 2: Check Endpoint Exists")
print("=" * 60)
try:
    response = requests.options("http://localhost:8000/api/chat/query")
    print(f"Status: {response.status_code}")
    print(f"Allowed methods: {response.headers.get('allow', 'N/A')}")
except Exception as e:
    print(f"Error: {e}")
print()

# Test 3: Send invalid request to see error format
print("Test 3: Invalid Request (empty query)")
print("=" * 60)
try:
    response = requests.post(
        "http://localhost:8000/api/chat/query",
        json={"query": "", "query_type": "full_text"},
        timeout=10
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
print()

# Test 4: Valid request
print("Test 4: Valid Request")
print("=" * 60)
try:
    response = requests.post(
        "http://localhost:8000/api/chat/query",
        json={"query": "What is ROS?", "query_type": "full_text"},
        timeout=60
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success!")
        print(f"Answer length: {len(data.get('answer', ''))}")
        print(f"Citations: {len(data.get('citations', []))}")
        print(f"Session ID: {data.get('session_id')}")
    else:
        print(f"❌ Failed")
        print(f"Response: {response.text[:500]}")
except requests.exceptions.Timeout:
    print("❌ Timeout")
except Exception as e:
    print(f"❌ Error: {e}")
