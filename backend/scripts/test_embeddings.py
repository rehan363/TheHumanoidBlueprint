"""
Test Gemini embedding generation
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def test_gemini_embeddings():
    """Test if Gemini embeddings API works"""
    key = os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY_2")
    
    if not key:
        print("[FAIL] No Gemini API keys found")
        return False
    
    try:
        genai.configure(api_key=key)
        
        # Test embedding generation
        result = genai.embed_content(
            model="models/text-embedding-004",
            content="This is a test sentence for embedding generation.",
            task_type="retrieval_document"
        )
        
        if result and 'embedding' in result:
            embedding_dim = len(result['embedding'])
            print(f"[PASS] Gemini Embeddings working!")
            print(f"  Model: text-embedding-004")
            print(f"  Dimensions: {embedding_dim}")
            print(f"  Sample values: {result['embedding'][:5]}")
            return True
        else:
            print("[FAIL] Empty embedding response")
            return False
            
    except Exception as e:
        print(f"[FAIL] Gemini Embeddings: {str(e)[:150]}")
        return False

if __name__ == "__main__":
    print("Testing Gemini Embeddings API...")
    print()
    success = test_gemini_embeddings()
    print()
    if success:
        print("[RECOMMENDATION] Embeddings working - safe to run indexing")
    else:
        print("[ERROR] Embeddings failed - indexing will fail")
