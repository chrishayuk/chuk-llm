import asyncio
from chuk_llm.llm.client import get_client

async def low_level_examples():
    # Get a client for any provider
    client = get_client("openai", model="gpt-4o-mini")
    
    # Basic completion with full control
    response = await client.create_completion([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"}
    ])
    print(response["response"])
    
    # Streaming with low-level control
    messages = [
        {"role": "user", "content": "Write a short story about AI"}
    ]
    
    async for chunk in client.create_completion(messages, stream=True):
        if chunk.get("response"):
            print(chunk["response"], end="", flush=True)
    
    # Function calling with full control
    tools = [
        {
            "type": "function", 
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    response = await client.create_completion(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        temperature=0.7,
        max_tokens=150
    )
    
    if response.get("tool_calls"):
        for tool_call in response["tool_calls"]:
            print(f"Function: {tool_call['function']['name']}")
            print(f"Arguments: {tool_call['function']['arguments']}")

asyncio.run(low_level_examples())

# # examples/vision_quickstart.py
# """
# ChukLLM Vision - Quick Start
# Just 3 lines to analyze an image!
# """

# from chuk_llm import ask_openai_gpt4o_sync, ask_anthropic_sonnet_sync, ask_watsonx_granite_vision_sync
# from PIL import Image, ImageDraw

# # Create a simple test image
# img = Image.new('RGB', (200, 100), 'white')
# draw = ImageDraw.Draw(img)
# draw.text((50, 40), "HELLO AI!", fill='red')
# img.save("hello.png")

# # That's it! Just call with image path
# print(ask_openai_gpt4o_sync("What text do you see?", "hello.png"))

# # Try with Claude too
# print(ask_anthropic_sonnet_sync("What color is the text?", "hello.png"))

# # Or analyze from a URL
# print(ask_openai_gpt4o_sync(
#     "What's in this image?", 
#     "https://picsum.photos/200/300"
# ))

# # Or analyze from a URL
# print(ask_anthropic_sonnet_sync(
#     "What's in this image?", 
#     "https://picsum.photos/200/300"
# ))

# # # Or analyze from a URL
# # print(ask_watsonx_granite_vision_sync(
# #     "What's in this image?", 
# #     "https://picsum.photos/200/300"
# # ))