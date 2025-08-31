import asyncio
from chuk_llm import ask_ollama_granite, stream_ollama_granite

async def stream_example():
    print(await ask_ollama_granite("Who is Ada Lovelace?"))

    async for chunk in stream_ollama_granite("tell me a story about cheese"):
        print(chunk, end="", flush=True)

# Run the async function
if __name__ == "__main__":
    asyncio.run(stream_example())