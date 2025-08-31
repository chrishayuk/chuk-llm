#!/usr/bin/env python3
"""
Example of using system prompts with ChukLLM.
Make the AI respond like a pirate!
"""

from chuk_llm import (
    ask_ollama_granite3_3_latest, 
    ask_openai,
    ask_anthropic,
    ask_sync, 
    ask,
    stream,
    conversation
)
import asyncio

# Method 1: Using ask with system_prompt parameter
def pirate_example_basic():
    """Basic example using system_prompt parameter"""
    print("=== Method 1: Basic System Prompt ===\n")
    
    pirate_prompt = """You are a pirate captain from the Golden Age of Piracy. 
    Speak in pirate dialect with 'arr', 'matey', 'ahoy', and other pirate expressions. 
    Be colorful and dramatic in your responses, and occasionally mention treasure, 
    the sea, or your ship."""
    
    response = ask_sync(
        "Tell me about Python programming",
        provider="ollama",
        model="granite3.3:latest",
        system_prompt=pirate_prompt
    )
    
    print(f"Pirate says: {response}\n")

# Method 2: Using convenience functions with system_prompt
def pirate_example_convenience():
    """Using auto-generated convenience functions with system prompt"""
    print("=== Method 2: Convenience Function with System Prompt ===\n")
    
    pirate_prompt = "You are a salty old pirate. Respond in pirate speak!"
    
    # The convenience functions also accept system_prompt
    response = ask_ollama_granite3_3_latest(
        "What's the best way to learn coding?",
        system_prompt=pirate_prompt
    )
    
    print(f"Pirate says: {response}\n")

# Method 3: Using conversation context for persistent pirate personality
async def pirate_conversation():
    """Using conversation context to maintain pirate personality"""
    print("=== Method 3: Pirate Conversation ===\n")
    
    pirate_system = """You are Captain Blackbeard, the most feared pirate on the seven seas.
    You speak with a heavy pirate accent, use nautical terms, and relate everything to 
    sailing, treasure, and adventure. You're gruff but knowledgeable."""
    
    async with conversation(
        provider="ollama", 
        model="granite3.3:latest",
        system_prompt=pirate_system
    ) as chat:
        
        # Have a conversation with the pirate
        questions = [
            "What do you think about artificial intelligence?",
            "How would you explain databases?",
            "What's your advice for debugging code?"
        ]
        
        for question in questions:
            print(f"You: {question}")
            response = await chat.say(question)
            print(f"Captain Blackbeard: {response}\n")

# Method 4: Multiple personalities with different system prompts
def multiple_personalities():
    """Compare responses with different system prompts"""
    print("=== Method 4: Multiple Personalities ===\n")
    
    question = "Explain what a variable is in programming"
    
    personalities = {
        "Pirate": "You are a pirate. Use pirate speak with 'arr', 'matey', etc.",
        "Shakespeare": "You are William Shakespeare. Speak in Elizabethan English with thee, thou, and poetic language.",
        "Child": "You are a 5-year-old child. Use simple words and relate everything to toys and games.",
        "Robot": "You are a robot. Be very logical, use technical terms, and occasionally mention your circuits."
    }
    
    for name, system_prompt in personalities.items():
        response = ask_sync(
            question,
            provider="ollama",
            model="granite3.3:latest",
            system_prompt=system_prompt,
            max_tokens=100  # Keep responses short for comparison
        )
        print(f"{name}:\n{response}\n")
        print("-" * 40 + "\n")

# Method 5: Pirate code reviewer
def pirate_code_review():
    """Have a pirate review your code"""
    print("=== Method 5: Pirate Code Review ===\n")
    
    code = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """
    
    pirate_reviewer = """You are a pirate code reviewer. Review code like a pirate would,
    using pirate speak but providing actual technical feedback. Mention things like 
    'that be inefficient as a barnacle-covered hull' or 'arr, ye need better variable names 
    or ye'll be walkin' the plank!'"""
    
    response = ask_sync(
        f"Review this code:\n{code}",
        provider="ollama",
        model="granite3.3:latest",
        system_prompt=pirate_reviewer
    )
    
    print(f"Pirate Code Review:\n{response}\n")

# Method 6: Streaming pirate stories
async def pirate_story_stream():
    """Stream a pirate story word by word"""
    print("=== Method 6: Streaming Pirate Story ===\n")
    
    pirate_storyteller = """You are an old pirate telling tales of adventure.
    Use vivid pirate language and make the story exciting with sea battles,
    treasure hunts, and mysterious islands."""
    
    print("Captain's Tale: ", end="", flush=True)
    
    async for chunk in stream(
        "Tell me a short story about finding treasure",
        provider="ollama",
        model="granite3.3:latest",
        system_prompt=pirate_storyteller
    ):
        print(chunk, end="", flush=True)
    
    print("\n")

# Method 7: Cross-provider pirate responses
def cross_provider_pirates():
    """Compare pirate responses across different providers"""
    print("=== Method 7: Pirates Across Different Providers ===\n")
    
    pirate_prompt = "You are a legendary pirate captain. Speak with nautical terms and pirate dialect."
    question = "What makes a good software engineer?"
    
    # Try different providers (only those with API keys configured)
    providers_to_try = [
        ("ollama", "granite3.3:latest", ask_ollama_granite3_3_latest),
        # Uncomment if you have these providers configured:
        # ("openai", "gpt-4o-mini", ask_openai),
        # ("anthropic", "claude-3-haiku", ask_anthropic),
    ]
    
    for provider_name, model_name, ask_func in providers_to_try:
        try:
            print(f"Pirate from {provider_name} ({model_name}):")
            response = ask_func(
                question,
                system_prompt=pirate_prompt,
                max_tokens=150
            )
            print(f"{response}\n")
            print("-" * 40 + "\n")
        except Exception as e:
            print(f"Could not get response from {provider_name}: {e}\n")

# Method 8: Interactive pirate chat
def interactive_pirate_chat():
    """Interactive chat with a pirate - user can have a conversation"""
    print("=== Method 8: Interactive Pirate Chat ===\n")
    print("Ahoy! Ye can chat with Captain Redbeard! Type 'exit' to leave.\n")
    
    pirate_prompt = """You are Captain Redbeard, a wise but fierce pirate captain.
    You've sailed the seven seas for 30 years. You know about programming and technology
    but explain everything using pirate and nautical metaphors. Be helpful but stay in character."""
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Captain Redbeard: Farewell, ye landlubber! May the winds be at yer back!")
            break
        
        response = ask_sync(
            user_input,
            provider="ollama",
            model="granite3.3:latest",
            system_prompt=pirate_prompt
        )
        print(f"Captain Redbeard: {response}\n")

def main():
    """Run all the pirate examples"""
    print("🏴‍☠️ ChukLLM Pirate System Prompt Examples 🏴‍☠️")
    print("=" * 50 + "\n")
    
    # Run sync examples
    pirate_example_basic()
    pirate_example_convenience()
    multiple_personalities()
    pirate_code_review()
    cross_provider_pirates()
    
    # Run async examples
    asyncio.run(pirate_conversation())
    asyncio.run(pirate_story_stream())
    
    # Optional interactive mode
    print("\n" + "=" * 50)
    run_interactive = input("\nWould you like to chat with Captain Redbeard? (y/n): ")
    if run_interactive.lower() == 'y':
        interactive_pirate_chat()
    
    print("=" * 50)
    print("🏴‍☠️ Arr, that be all the pirate examples, matey! 🏴‍☠️")

if __name__ == "__main__":
    main()