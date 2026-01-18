"""
Test Qdrant connection with detailed diagnostics
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

def test_qdrant():
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    
    print(f"Testing Qdrant Connection...")
    print(f"URL: {url}")
    print(f"API Key: {key[:20]}..." if key else "API Key: None")
    print()
    
    try:
        # Try with check_compatibility=False to bypass version check
        client = QdrantClient(url=url, api_key=key, timeout=10, check_compatibility=False)
        
        print("[PASS] Client created successfully")
        
        # Try to list collections
        try:
            collections = client.get_collections()
            print(f"[PASS] Connected! Found {len(collections.collections)} collections:")
            for col in collections.collections:
                print(f"   - {col.name}")
            
            # Check for our specific collection
            target = "physical_ai_textbook"  # Match actual collection name
            if any(c.name == target for c in collections.collections):
                count = client.count(target)
                print(f"\n[PASS] Collection '{target}' exists with {count.count} vectors")
                
                # Get collection info
                info = client.get_collection(target)
                print(f"   Vector size: {info.config.params.vectors.size}")
                print(f"   Distance: {info.config.params.vectors.distance}")
            else:
                print(f"\n[WARN] Collection '{target}' does not exist - needs to be created")
                
        except Exception as e:
            print(f"[FAIL] Failed to list collections: {e}")
            
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        print("\nPossible issues:")
        print("1. Invalid URL or API key")
        print("2. Qdrant cluster was deleted")
        print("3. Network/firewall issue")
        print("\nAction: Check Qdrant Cloud dashboard or create new cluster")

if __name__ == "__main__":
    test_qdrant()
