#!/usr/bin/env python3
"""
Script to fix missing commas between description and parameters.
"""

import re
from pathlib import Path


def fix_commas(content: str) -> str:
    """Fix missing commas."""
    # Pattern: "description" followed by any text, then space(s), then "parameters"
    # This regex will match and add comma
    pattern = r'("description":\s*"[^"]*")\s+("parameters":)'
    replacement = r'\1, \2'
    content = re.sub(pattern, replacement, content)
    return content

def process_file(filepath: Path):
    """Process a single test file."""
    print(f"Processing {filepath}...")

    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    # Fix commas
    content = fix_commas(content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Fixed {filepath}")

def main():
    """Main function."""
    test_dir = Path("/Users/chrishay/chris-source/chuk-ai/chuk-llm/tests/llm/providers")

    files_to_fix = [
        test_dir / "test_azure_openai_client.py",
        test_dir / "test_azure_openai_client_extended.py",
        test_dir / "test_anthropic_client.py",
        test_dir / "test_gemini_client.py",
    ]

    for filepath in files_to_fix:
        if filepath.exists():
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

    print("\nAll files processed!")

if __name__ == "__main__":
    main()
