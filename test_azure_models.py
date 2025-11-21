#!/usr/bin/env python3
import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

async def test():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    print(f"Testing Azure OpenAI Models API...")
    print(f"Endpoint: {endpoint}")
    print(f"API Key: {api_key[:20] if api_key else 'None'}...")
    
    # Try different API versions and endpoints
    tests = [
        ("2024-02-01", "/openai/deployments"),
        ("2023-05-15", "/openai/deployments"),
        ("2024-02-01", "/openai/models"),
        ("2023-05-15", "/openai/models"),
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for api_version, path in tests:
            url = f"{endpoint}{path}"
            headers = {"api-key": api_key}
            params = {"api-version": api_version}
            
            print(f"\n{'='*60}")
            print(f"Trying: {url}?api-version={api_version}")
            
            try:
                response = await client.get(url, headers=headers, params=params)
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"SUCCESS! Data keys: {list(data.keys())}")
                    if "data" in data:
                        print(f"Items: {len(data['data'])}")
                        for item in data['data'][:3]:
                            print(f"  - {item}")
                else:
                    print(f"Response: {response.text[:200]}")
            except Exception as e:
                print(f"Error: {e}")

asyncio.run(test())
