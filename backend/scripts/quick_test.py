"""
Quick test script for backend API endpoints.
"""
import requests
import json

def test_health():
    """Test health endpoint."""
    print("=" * 60)
    print("Testing /api/health")
    print("=" * 60)
    
    response = requests.get("http://localhost:8000/api/health")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"✅ Version: {data['version']}")
        print(f"✅ Services: {data.get('services', {})}")
    else:
        print(f"❌ Error: {response.text}")
    
    print()

def test_chat_query():
    """Test chat query endpoint."""
    print("=" * 60)
    print("Testing /api/chat/query")
    print("=" * 60)
    
    payload = {
        "query": "What is ROS 2?",
        "query_type": "full_text"
    }
    
    print(f"Query: {payload['query']}")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/chat/query",
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Answer: {data['answer'][:200]}...")
            print(f"✅ Citations: {len(data.get('citations', []))}")
            print(f"✅ Processing Time: {data.get('processing_time_ms', 0)}ms")
            print(f"✅ Session ID: {data.get('session_id', 'N/A')}")
            
            # Print citations
            if data.get('citations'):
                print("\nCitations:")
                for i, citation in enumerate(data['citations'][:3], 1):
                    print(f"  {i}. {citation.get('source', 'Unknown')} (score: {citation.get('relevance_score', 0):.2f})")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print()

if __name__ == "__main__":
    test_health()
    test_chat_query()
