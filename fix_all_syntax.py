#!/usr/bin/env python3
"""
Script to fix all syntax errors caused by missing commas.
"""

import re
from pathlib import Path


def fix_file(filepath: Path):
    """Fix a single file."""
    print(f"Processing {filepath}...")

    with open(filepath, 'rb') as f:
        content = f.read()

    # Decode as UTF-8 and clean any weird characters
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        # Try with latin-1 as fallback
        text = content.decode('latin-1')

    # Remove any non-printable characters except for standard whitespace
    text = ''.join(char if (char.isprintable() or char in '\n\r\t') else '' for char in text)

    # Fix the pattern: "description" followed by quoted text, then space and "parameters"
    # Use a more robust regex
    pattern = r'("description":\s*"[^"]*")\s+("parameters":)'
    text = re.sub(pattern, r'\1, \2', text)

    # Write back as clean UTF-8
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

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
            fix_file(filepath)
        else:
            print(f"File not found: {filepath}")

    print("\nAll files processed!")

if __name__ == "__main__":
    main()
