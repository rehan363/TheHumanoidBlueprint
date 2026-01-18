import os
import sys
import asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import google.generativeai as genai

# Load environment
load_dotenv()

def print_status(component, status, message=""):
    color = "\033[92m" if status == "PASS" else "\033[91m"
    reset = "\033[0m"
    print(f"[{color}{status}{reset}] {component}: {message}")

async def check_qdrant():
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    
    if not url or not key:
        print_status("Qdrant Config", "FAIL", "Missing QDRANT_URL or QDRANT_API_KEY")
        return False

    try:
        client = QdrantClient(url=url, api_key=key)
        collections = client.get_collections()
        print_status("Qdrant Connection", "PASS", f"Connected. Found {len(collections.collections)} collections.")
        
        # Check specific collection
        target_col = "physical-ai-textbook"
        exists = any(c.name == target_col for c in collections.collections)
        if exists:
            count = client.count(target_col)
            print_status("Qdrant Data", "PASS", f"Collection '{target_col}' exists with {count.count} vectors.")
            if count.count == 0:
                print_status("Qdrant Content", "WARN", "Collection is empty! Needs indexing.")
        else:
            print_status("Qdrant Data", "FAIL", f"Collection '{target_col}' NOT found.")
        return True
    except Exception as e:
        print_status("Qdrant Connection", "FAIL", str(e))
        return False

async def check_gemini():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print_status("Gemini Config", "FAIL", "Missing GEMINI_API_KEY")
        return False

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp') 
        # Using a model likely to be available, or fallback to 1.5-flash if needed
        # The user's code uses settings.gemini_model, we'll try a generic one or read from env if possible, 
        # but hardcoding a common one for connectivity check is safer than relying on complex config imports for this simple script.
        
        response = model.generate_content("Respond with 'OK'")
        if response and response.text:
            print_status("Gemini API", "PASS", "Generated response successfully.")
        else:
            print_status("Gemini API", "FAIL", "Empty response.")
        return True
    except Exception as e:
        print_status("Gemini API", "FAIL", str(e))
        if "429" in str(e):
            print("\n!!! QUOTA EXCEEDED (429) DETECTED !!!")
        return False

async def main():
    print("Starting Connectivity Diagnostics...\n")
    
    # 1. Check Env Vars Existence
    required_vars = ["GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print_status("Environment", "FAIL", f"Missing keys: {', '.join(missing)}")
    else:
        print_status("Environment", "PASS", "Critical keys present.")

    # 2. Check Qdrant
    await check_qdrant()

    # 3. Check Gemini
    await check_gemini()

if __name__ == "__main__":
    asyncio.run(main())
