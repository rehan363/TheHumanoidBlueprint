"""
Detailed test with error logging.
"""
import requests
import json
import traceback

def test_chat_detailed():
    """Test chat with detailed error output."""
    print("=" * 60)
    print("Testing /api/chat/query with detailed error logging")
    print("=" * 60)
    
    payload = {
        "query": "What is ROS 2?",
        "query_type": "full_text"
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/chat/query",
            json=payload,
            timeout=60
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"Answer: {data['answer'][:300]}...")
            print(f"Citations: {len(data.get('citations', []))}")
            print(f"Session ID: {data.get('session_id')}")
            print(f"Processing Time: {data.get('processing_time_ms')}ms")
        else:
            print(f"\n❌ ERROR!")
            print(f"Response Text: {response.text}")
            
            # Try to parse as JSON
            try:
                error_data = response.json()
                print(f"Error JSON: {json.dumps(error_data, indent=2)}")
            except:
                print("Could not parse error as JSON")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 60 seconds")
    except Exception as e:
        print(f"❌ Exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_chat_detailed()
