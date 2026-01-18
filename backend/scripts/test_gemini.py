"""
Test Gemini API with quota detection
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def test_gemini():
    key = os.getenv("GEMINI_API_KEY")
    
    print(f"Testing Gemini API...")
    print(f"API Key: {key[:20]}..." if key else "API Key: None")
    print()
    
    if not key:
        print("[FAIL] GEMINI_API_KEY not found in .env")
        return
    
    try:
        genai.configure(api_key=key)
        
        # Try with a simple model first
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        print("Sending test request...")
        response = model.generate_content("Respond with 'OK' if you're working.")
        
        if response and response.text:
            print(f"[PASS] Gemini API working!")
            print(f"Response: {response.text.strip()}")
        else:
            print("[FAIL] Empty response from Gemini")
            
    except Exception as e:
        error_str = str(e)
        print(f"[FAIL] Gemini API error: {error_str}")
        
        if "429" in error_str or "quota" in error_str.lower():
            print("\n!!! QUOTA EXCEEDED !!!")
            print("Action: Switch to OpenRouter")
        elif "401" in error_str or "unauthorized" in error_str.lower():
            print("\n!!! INVALID API KEY !!!")
            print("Action: Check your API key")
        elif "404" in error_str:
            print("\n!!! MODEL NOT FOUND !!!")
            print("Action: Try a different model name")

if __name__ == "__main__":
    test_gemini()
