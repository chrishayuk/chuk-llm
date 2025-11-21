#!/usr/bin/env python3
import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

async def test():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = "2024-02-01"
    
    print(f"Testing Azure OpenAI API...")
    print(f"Endpoint: {endpoint}")
    print(f"API Key: {api_key[:20] if api_key else 'None'}...")
    
    if not api_key or not endpoint:
        print("Missing API key or endpoint!")
        return
    
    url = f"{endpoint}/openai/deployments"
    headers = {"api-key": api_key}
    params = {"api-version": api_version}
    
    print(f"\nRequest URL: {url}")
    print(f"Params: {params}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            print(f"\nStatus: {response.status_code}")
            print(f"Response text (first 500 chars): {response.text[:500]}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nData keys: {data.keys()}")
                deployments = data.get("data", [])
                print(f"Number of deployments: {len(deployments)}")
                for dep in deployments[:5]:
                    print(f"  - {dep.get('id', 'unknown')}: {dep.get('model', 'unknown')}")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

asyncio.run(test())
