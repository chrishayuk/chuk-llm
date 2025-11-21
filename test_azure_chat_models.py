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
    
    url = f"{endpoint}/openai/models"
    headers = {"api-key": api_key}
    params = {"api-version": api_version}
    
    print(f"Fetching models...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers, params=params)
        data = response.json()
        
        chat_models = [
            m for m in data.get("data", [])
            if m.get("capabilities", {}).get("chat_completion", False)
        ]
        
        print(f"\nFound {len(chat_models)} chat completion models:")
        for model in chat_models[:20]:
            model_id = model.get("id", "unknown")
            lifecycle = model.get("lifecycle_status", "unknown")
            print(f"  - {model_id} ({lifecycle})")

asyncio.run(test())
