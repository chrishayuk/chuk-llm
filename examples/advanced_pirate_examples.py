#!/usr/bin/env python3
"""
Advanced pirate system prompt examples with ChukLLM.
Demonstrates complex pirate interactions including tools, JSON mode, and more.
"""

import asyncio
import json
from typing import Dict, List, Any
from chuk_llm import ask, ask_sync, stream, conversation, ask_with_tools

# Advanced pirate with function calling/tools
async def pirate_with_tools():
    """Pirate that can use tools to help with tasks"""
    print("=== Pirate with Tools (Function Calling) ===\n")
    
    pirate_prompt = """You are Captain Codebeard, a pirate who specializes in programming.
    You speak like a pirate but can execute code and use tools. When using tools,
    describe what you're doing in pirate speak."""
    
    # Define tools the pirate can use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_treasure",
                "description": "Calculate the value of treasure in gold doubloons",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gold_coins": {"type": "integer", "description": "Number of gold coins"},
                        "silver_pieces": {"type": "integer", "description": "Number of silver pieces"},
                        "gems": {"type": "integer", "description": "Number of precious gems"}
                    },
                    "required": ["gold_coins", "silver_pieces", "gems"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_map",
                "description": "Search the treasure map for locations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Location to search for"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Tool implementations
    def calculate_treasure(gold_coins: int, silver_pieces: int, gems: int) -> Dict:
        total = gold_coins * 10 + silver_pieces * 2 + gems * 50
        return {"total_doubloons": total, "treasure_grade": "legendary" if total > 1000 else "decent"}
    
    def search_map(location: str) -> Dict:
        locations = {
            "skull island": {"found": True, "danger": "high", "treasure": "massive"},
            "tortuga": {"found": True, "danger": "medium", "treasure": "moderate"},
            "port royal": {"found": True, "danger": "low", "treasure": "small"}
        }
        return locations.get(location.lower(), {"found": False, "message": "Unknown waters, matey!"})
    
    tool_map = {
        "calculate_treasure": calculate_treasure,
        "search_map": search_map
    }
    
    try:
        result = await ask_with_tools(
            "I found 50 gold coins, 200 silver pieces, and 15 gems. How much is me treasure worth? Also, search the map for Skull Island!",
            provider="openai",
            model="gpt-4o-mini",
            tools=tools,
            tool_map=tool_map,
            system_prompt=pirate_prompt
        )
        
        print(f"Captain Codebeard says: {result['content']}")
        if result.get('tool_calls'):
            print("\nTools used by the pirate:")
            for call in result['tool_calls']:
                print(f"  - {call['function']['name']}: {call['function']['arguments']}")
    except Exception as e:
        print(f"Note: Tool calling requires OpenAI API. Error: {e}")
        print("Falling back to regular pirate response...\n")
        
        response = ask_sync(
            "Tell me about using tools as a pirate programmer",
            provider="ollama",
            model="granite3.3:latest",
            system_prompt=pirate_prompt
        )
        print(f"Captain Codebeard says: {response}")

# Pirate generating structured JSON
def pirate_json_mode():
    """Pirate that generates structured JSON responses"""
    print("\n=== Pirate JSON Mode ===\n")
    
    pirate_prompt = """You are a pirate data architect. You speak like a pirate
    but must return valid JSON when asked. Include pirate-themed field names."""
    
    providers_with_json = [
        ("ollama", "granite3.3:latest"),
        # ("openai", "gpt-4o-mini"),  # Uncomment if configured
    ]
    
    for provider, model in providers_with_json:
        try:
            print(f"Using {provider}/{model}:")
            response = ask_sync(
                "Create a JSON schema for a pirate ship with crew, cannons, and treasure",
                provider=provider,
                model=model,
                system_prompt=pirate_prompt,
                json_mode=True,
                max_tokens=300
            )
            
            # Try to parse and pretty-print the JSON
            try:
                json_data = json.loads(response)
                print(json.dumps(json_data, indent=2))
            except:
                print(response)
            print("\n" + "-" * 40 + "\n")
            break
        except Exception as e:
            print(f"Could not use {provider}: {e}")

# Pirate teaching programming concepts
async def pirate_teacher():
    """Educational pirate that teaches programming"""
    print("=== Pirate Programming Teacher ===\n")
    
    teacher_prompt = """You are Professor Blackbeard, a pirate who retired to teach programming.
    You use sailing and pirate metaphors to explain programming concepts. You're enthusiastic
    and encouraging, often saying things like 'Ye be learnin' fast as a ship in full sail!'"""
    
    topics = [
        "recursion",
        "object-oriented programming",
        "async/await"
    ]
    
    for topic in topics:
        print(f"Learning about {topic}:")
        response = ask_sync(
            f"Explain {topic} in simple terms",
            provider="ollama",
            model="granite3.3:latest",
            system_prompt=teacher_prompt,
            max_tokens=200
        )
        print(f"{response}\n")
        print("=" * 50 + "\n")

# Pirate debugging assistant
def pirate_debugger():
    """Pirate that helps debug code"""
    print("=== Pirate Debugging Assistant ===\n")
    
    debugger_prompt = """You are Debug Dan, a pirate programmer who specializes in finding bugs.
    You say things like 'Arr, I see the bug hiding in yer code like a stowaway in the hold!'
    and 'That error be as clear as a lighthouse on a foggy night!' Be helpful but colorful."""
    
    buggy_code = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # Bug: doesn't handle empty list

result = calculate_average([])
print(result)
'''
    
    response = ask_sync(
        f"Help me debug this code:\n{buggy_code}",
        provider="ollama",
        model="granite3.3:latest",
        system_prompt=debugger_prompt
    )
    
    print(f"Debug Dan says:\n{response}\n")

# Pirate pair programming session
async def pirate_pair_programming():
    """Collaborative pirate programming session"""
    print("=== Pirate Pair Programming ===\n")
    
    pair_prompt = """You are First Mate Code, a pirate programmer who does pair programming.
    You're collaborative and suggest improvements like 'What if we hoist a try-catch block 
    around that dangerous code?' and 'Let's refactor this like we're streamlining the rigging!'"""
    
    async with conversation(
        provider="ollama",
        model="granite3.3:latest",
        system_prompt=pair_prompt
    ) as chat:
        
        programming_tasks = [
            "Let's write a function to validate email addresses",
            "Now let's add error handling to it",
            "Can we make it more efficient?",
            "Let's add some unit tests"
        ]
        
        for task in programming_tasks:
            print(f"You: {task}")
            response = await chat.say(task)
            print(f"First Mate Code: {response}\n")
            await asyncio.sleep(0.5)  # Brief pause for readability

# Pirate explaining different programming paradigms
def pirate_paradigms():
    """Pirate explaining programming paradigms"""
    print("=== Pirate Programming Paradigms ===\n")
    
    paradigm_prompt = """You are Admiral Algorithm, a pirate who has mastered all programming
    paradigms. You explain each paradigm using naval and pirate analogies. For example,
    functional programming is like 'each crew member doing one job perfectly without
    changing the ship's state.'"""
    
    paradigms = {
        "Functional Programming": "pure functions and immutability",
        "Object-Oriented Programming": "classes and objects",
        "Procedural Programming": "step-by-step procedures",
        "Event-Driven Programming": "responding to events"
    }
    
    for paradigm, concept in paradigms.items():
        response = ask_sync(
            f"Explain {paradigm} focusing on {concept}",
            provider="ollama",
            model="granite3.3:latest",
            system_prompt=paradigm_prompt,
            max_tokens=150
        )
        print(f"{paradigm}:")
        print(f"{response}\n")
        print("-" * 40 + "\n")

# Main execution
async def main():
    """Run all advanced pirate examples"""
    print("🏴‍☠️ Advanced ChukLLM Pirate Examples 🏴‍☠️")
    print("=" * 60 + "\n")
    
    # Sync examples
    pirate_json_mode()
    pirate_debugger()
    pirate_paradigms()
    
    # Async examples
    await pirate_with_tools()
    await pirate_teacher()
    await pirate_pair_programming()
    
    print("=" * 60)
    print("🏴‍☠️ All hands on deck! Examples complete! 🏴‍☠️")

if __name__ == "__main__":
    asyncio.run(main())