#!/usr/bin/env python3
"""
JSON Mode and Structured Outputs
==================================

Demonstrates getting structured JSON responses from LLMs.
Great for data extraction, API responses, and structured data.
"""

import asyncio
import json
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import ask

async def basic_json_mode():
    """Basic JSON mode - LLM returns valid JSON."""
    print("=== Basic JSON Mode ===\n")

    response = await ask(
        """Extract information from this text and return as JSON:

        John Smith is 35 years old. He works as a software engineer at Google.
        His email is john.smith@google.com and he lives in Mountain View, CA.

        Return JSON with: name, age, occupation, company, email, location
        """,
        response_format={"type": "json_object"},
        model="gpt-4o-mini"
    )

    # Parse the JSON response
    data = json.loads(response)
    print("Extracted data:")
    print(json.dumps(data, indent=2))
    print()

async def structured_data_extraction():
    """Extract structured data from unstructured text."""
    print("=== Structured Data Extraction ===\n")

    text = """
    Product Review:
    The XPhone 15 Pro is amazing! It has a 6.7 inch display, 256GB storage,
    and costs $999. The camera quality is outstanding (5/5 stars).
    Battery life is good but not great (3.5/5 stars).
    Would definitely recommend to others!
    """

    response = await ask(
        f"""Extract product review information from this text as JSON:

        {text}

        Return JSON with: product_name, display_size, storage, price,
        camera_rating, battery_rating, would_recommend (boolean)
        """,
        response_format={"type": "json_object"},
        model="gpt-4o-mini"
    )

    review = json.loads(response)
    print("Extracted review:")
    print(json.dumps(review, indent=2))
    print()

async def json_schema_output():
    """Use JSON schema for precise output format."""
    print("=== JSON Schema Output ===\n")

    # Define exact schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            },
            "experience_years": {"type": "integer"},
            "is_available": {"type": "boolean"}
        },
        "required": ["name", "age", "skills"],
        "additionalProperties": False
    }

    response = await ask(
        """Create a profile for a senior Python developer named Sarah Chen
        with 8 years of experience. She knows Python, Django, React, and PostgreSQL.
        She is currently available for new opportunities.

        Return data matching the JSON schema.
        """,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "developer_profile",
                "strict": True,
                "schema": schema
            }
        },
        model="gpt-4o-mini"
    )

    profile = json.loads(response)
    print("Developer profile:")
    print(json.dumps(profile, indent=2))
    print()

async def multiple_records():
    """Extract multiple records as JSON array."""
    print("=== Multiple Records ===\n")

    text = """
    Meeting attendees:
    - Alice Johnson (alice@company.com) - Product Manager
    - Bob Smith (bob@company.com) - Engineer
    - Carol White (carol@company.com) - Designer
    - David Brown (david@company.com) - Engineer
    """

    response = await ask(
        f"""Extract attendees as JSON array:

        {text}

        Return JSON array where each item has: name, email, role
        """,
        response_format={"type": "json_object"},
        model="gpt-4o-mini"
    )

    data = json.loads(response)
    print("Meeting attendees:")
    print(json.dumps(data, indent=2))
    print()

async def api_response_format():
    """Generate API response format."""
    print("=== API Response Format ===\n")

    response = await ask(
        """Create an API response for a user query about weather.
        The weather in London is 18°C, partly cloudy, with 60% humidity.

        Return standard REST API format with:
        - success (boolean)
        - data (object with weather info)
        - timestamp (ISO format)
        - message (string)
        """,
        response_format={"type": "json_object"},
        model="gpt-4o-mini"
    )

    api_response = json.loads(response)
    print("API Response:")
    print(json.dumps(api_response, indent=2))
    print()

async def nested_json():
    """Complex nested JSON structures."""
    print("=== Nested JSON Structure ===\n")

    response = await ask(
        """Create a JSON representation of a company with:
        - Company name: TechCorp
        - Founded: 2010
        - Departments:
          * Engineering (15 people, budget: $500k)
          * Sales (8 people, budget: $300k)
          * Marketing (5 people, budget: $200k)
        - Active projects:
          * Project Alpha (status: in-progress, team: 5)
          * Project Beta (status: completed, team: 3)
        """,
        response_format={"type": "json_object"},
        model="gpt-4o-mini"
    )

    company = json.loads(response)
    print("Company data:")
    print(json.dumps(company, indent=2))
    print()

async def classification_as_json():
    """Use JSON for classification tasks."""
    print("=== Classification as JSON ===\n")

    emails = [
        "URGENT: Your account has been compromised! Click here now!",
        "Hi team, the meeting is rescheduled to 3pm tomorrow.",
        "Congratulations! You've won $1M! Send your bank details!"
    ]

    for i, email in enumerate(emails, 1):
        response = await ask(
            f"""Classify this email as JSON:

            Email: "{email}"

            Return JSON with:
            - category (spam/legitimate)
            - confidence (0-1)
            - reason (brief explanation)
            - urgency (low/medium/high)
            """,
            response_format={"type": "json_object"},
            model="gpt-4o-mini"
        )

        result = json.loads(response)
        print(f"Email {i}: {result['category']} (confidence: {result['confidence']})")
        print(f"  Reason: {result['reason']}\n")

async def json_with_different_providers():
    """JSON mode works across providers."""
    print("=== JSON Mode Across Providers ===\n")

    prompt = """Convert this to JSON:
    Product name: Laptop, Price: 999, In stock: true

    Return JSON with: name, price, in_stock"""

    providers = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-20241022"),
    ]

    for provider, model in providers:
        try:
            response = await ask(
                prompt,
                response_format={"type": "json_object"},
                provider=provider,
                model=model
            )
            data = json.loads(response)
            print(f"{provider}: {json.dumps(data)}")
        except Exception as e:
            print(f"{provider}: {e}")

    print()

if __name__ == "__main__":
    asyncio.run(basic_json_mode())
    asyncio.run(structured_data_extraction())
    asyncio.run(json_schema_output())
    asyncio.run(multiple_records())
    asyncio.run(api_response_format())
    asyncio.run(nested_json())
    asyncio.run(classification_as_json())
    asyncio.run(json_with_different_providers())

    print("="*50)
    print("✅ JSON mode makes structured outputs easy!")
    print("="*50)
