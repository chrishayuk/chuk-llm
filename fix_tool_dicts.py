#!/usr/bin/env python3
"""
Script to fix tool dict declarations to include required fields.
"""

import re
from pathlib import Path


def fix_tool_definitions(content: str) -> str:
    """Fix tool definitions to include description and parameters."""

    # Pattern 1: {"type": "function", "function": {"name": "X"}}
    # Replace with: {"type": "function", "function": {"name": "X", "description": "X desc", "parameters": {}}}
    pattern1 = r'\{"type":\s*"function",\s*"function":\s*\{"name":\s*"([^"]+)"\s*\}\}'
    replacement1 = r'{"type": "function", "function": {"name": "\1", "description": "\1 description", "parameters": {}}}'
    content = re.sub(pattern1, replacement1, content)

    # Pattern 2: Multi-line tool definitions without parameters
    # This is harder - let's use a more sophisticated approach
    lines = content.split('\n')
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for tool definitions like:  {"type": "function", "function": {"name": "tool_name"...
        if '"type": "function"' in line and '"function"' in line:
            # Check if it's missing "parameters"
            # Collect the full tool definition
            tool_lines = [line]
            j = i + 1
            bracket_count = line.count('{') - line.count('}')

            while j < len(lines) and bracket_count > 0:
                tool_lines.append(lines[j])
                bracket_count += lines[j].count('{') - lines[j].count('}')
                j += 1

            tool_def = '\n'.join(tool_lines)

            # Check if it's missing required fields
            if '"parameters"' not in tool_def and '"description"' not in tool_def:
                # Extract function name
                name_match = re.search(r'"name":\s*"([^"]+)"', tool_def)
                if name_match:
                    name = name_match.group(1)
                    # Add description and parameters before closing function bracket
                    # Find the position after the name
                    modified_def = tool_def
                    # Add description and parameters
                    modified_def = re.sub(
                        r'("name":\s*"[^"]+")(\s*\})',
                        r'\1, "description": "' + name + ' description", "parameters": {}\2',
                        modified_def
                    )
                    result_lines.extend(modified_def.split('\n'))
                    i = j
                    continue
            elif '"parameters"' not in tool_def:
                # Add parameters
                modified_def = tool_def
                modified_def = re.sub(
                    r'("description":\s*"[^"]+")(\s*\})',
                    r'\1, "parameters": {}\2',
                    modified_def
                )
                result_lines.extend(modified_def.split('\n'))
                i = j
                continue
            elif '"description"' not in tool_def:
                # Add description
                name_match = re.search(r'"name":\s*"([^"]+)"', tool_def)
                if name_match:
                    name = name_match.group(1)
                    modified_def = tool_def
                    modified_def = re.sub(
                        r'("name":\s*"[^"]+")([,\s])',
                        r'\1, "description": "' + name + ' description"\2',
                        modified_def
                    )
                    result_lines.extend(modified_def.split('\n'))
                    i = j
                    continue

        result_lines.append(line)
        i += 1

    return '\n'.join(result_lines)

def process_file(filepath: Path):
    """Process a single test file."""
    print(f"Processing {filepath}...")

    with open(filepath) as f:
        content = f.read()

    # Fix tool definitions
    content = fix_tool_definitions(content)

    with open(filepath, 'w') as f:
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
