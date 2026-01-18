"""
Test all LLM providers and models configured in the backend.
"""
import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
import google.generativeai as genai

load_dotenv()

async def test_gemini_api_1():
    """Test Gemini API Key 1"""
    key = os.getenv("GEMINI_API_KEY_1")
    if not key:
        print("[SKIP] GEMINI_API_KEY_1 not set")
        return False
    
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Say 'OK'")
        if response and response.text:
            print(f"[PASS] Gemini API Key 1: {response.text.strip()}")
            return True
    except Exception as e:
        print(f"[FAIL] Gemini API Key 1: {str(e)[:100]}")
    return False

async def test_gemini_api_2():
    """Test Gemini API Key 2"""
    key = os.getenv("GEMINI_API_KEY_2")
    if not key:
        print("[SKIP] GEMINI_API_KEY_2 not set")
        return False
    
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Say 'OK'")
        if response and response.text:
            print(f"[PASS] Gemini API Key 2: {response.text.strip()}")
            return True
    except Exception as e:
        print(f"[FAIL] Gemini API Key 2: {str(e)[:100]}")
    return False

async def test_openrouter_deepseek():
    """Test OpenRouter with DeepSeek model"""
    key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL", "tngtech/deepseek-r1t2-chimera:free")
    
    if not key:
        print("[SKIP] OPENROUTER_API_KEY not set")
        return False
    
    try:
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=10
        )
        if response.choices:
            print(f"[PASS] OpenRouter DeepSeek ({model}): {response.choices[0].message.content.strip()}")
            return True
    except Exception as e:
        print(f"[FAIL] OpenRouter DeepSeek: {str(e)[:100]}")
    return False

async def test_openrouter_mistral():
    """Test OpenRouter with Mistral model"""
    key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("MISTRAL_MODEL", "mistralai/devstral-2512:free")
    
    if not key:
        print("[SKIP] OPENROUTER_API_KEY not set")
        return False
    
    try:
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=10
        )
        if response.choices:
            print(f"[PASS] OpenRouter Mistral ({model}): {response.choices[0].message.content.strip()}")
            return True
    except Exception as e:
        print(f"[FAIL] OpenRouter Mistral: {str(e)[:100]}")
    return False

async def main():
    print("="*60)
    print("Testing All LLM Providers")
    print("="*60)
    print()
    
    results = {}
    
    print("1. Testing Gemini API Keys...")
    results['gemini_1'] = await test_gemini_api_1()
    results['gemini_2'] = await test_gemini_api_2()
    print()
    
    print("2. Testing OpenRouter Models...")
    results['deepseek'] = await test_openrouter_deepseek()
    results['mistral'] = await test_openrouter_mistral()
    print()
    
    print("="*60)
    print("Summary")
    print("="*60)
    
    working_providers = [k for k, v in results.items() if v]
    failed_providers = [k for k, v in results.items() if not v]
    
    if working_providers:
        print(f"\n[PASS] Working providers: {', '.join(working_providers)}")
    
    if failed_providers:
        print(f"\n[FAIL] Failed providers: {', '.join(failed_providers)}")
    
    print()
    
    if results.get('deepseek') or results.get('mistral'):
        print("[RECOMMENDATION] OpenRouter is working - backend will use it as primary")
    elif results.get('gemini_1') or results.get('gemini_2'):
        print("[RECOMMENDATION] Only Gemini is working - backend will use it")
    else:
        print("[ERROR] No working LLM providers! Backend will fail to start.")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
