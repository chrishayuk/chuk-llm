import asyncio
from chuk_llm import stream

async def stream_example():
    #print(await ask_ollama_granite("Who is Ada Lovelace?"))

    async for chunk in stream("tell me a story about cheese", 
                                system_prompt="you are a pirate, always answer in pirate speak",
                                provider="ollama",
                                model="granite",
                                temperature=0.1,
                                #max_tokens=50,
                                stop=["cheese"]):
        print(chunk, end="", flush=True)

# Run the async function
if __name__ == "__main__":
    asyncio.run(stream_example())