#!/usr/bin/env python3
"""
Script to automatically add Pydantic model conversions to test files.
"""

import re
from pathlib import Path


def add_import_if_needed(content: str, provider: str) -> str:
    """Add Message and Tool imports if not present."""
    import_line = "from chuk_llm.core.models import Message, Tool"

    if import_line not in content:
        # Find where to insert the import
        # Look for the provider client import
        pattern = rf"from chuk_llm\.llm\.providers\.{provider}_client import"
        match = re.search(pattern, content)

        if match:
            # Insert after the provider client import
            end_pos = content.find('\n', match.end())
            content = content[:end_pos+1] + import_line + '\n' + content[end_pos+1:]

    return content

def fix_test_method(content: str) -> str:
    """Fix a test method to convert dicts to Pydantic models."""

    # Pattern to find message dict declarations followed by client method calls
    # We need to find lines like: messages = [{"role": ...}]
    # Then check if it's followed by calls to create_completion, _validate_request_with_config, etc.

    lines = content.split('\n')
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for messages = [{"role": ...}]
        if re.match(r'\s+messages\s*=\s*\[.*\{.*"role"', line):
            # Collect all lines of the messages assignment
            messages_lines = [line]
            indent = len(line) - len(line.lstrip())
            j = i + 1

            # Collect continuation lines
            while j < len(lines):
                if lines[j].strip().startswith(']'):
                    messages_lines.append(lines[j])
                    j += 1
                    break
                elif lines[j].strip() and not lines[j].strip().startswith('#'):
                    messages_lines.append(lines[j])
                    j += 1
                else:
                    break

            # Check if there's a client method call in the next few lines
            has_client_call = False
            for k in range(j, min(j + 20, len(lines))):
                if any(method in lines[k] for method in [
                    'create_completion',
                    '_validate_request_with_config',
                    '_regular_completion',
                    '_stream_completion_async'
                ]):
                    has_client_call = True
                    break

            if has_client_call:
                # Change `messages =` to `messages_dicts =`
                messages_lines[0] = messages_lines[0].replace('messages =', 'messages_dicts =', 1)
                result_lines.extend(messages_lines)

                # Add conversion line
                spaces = ' ' * indent
                result_lines.append(f'{spaces}# Convert dicts to Pydantic models')
                result_lines.append(f'{spaces}messages = [Message.model_validate(msg) for msg in messages_dicts]')
                result_lines.append('')

                i = j
                continue

        # Check for tools = [{"type": "function", ...}]
        if re.match(r'\s+tools\s*=\s*\[.*\{.*"type".*"function"', line):
            # Collect all lines of the tools assignment
            tools_lines = [line]
            indent = len(line) - len(line.lstrip())
            j = i + 1

            # Collect continuation lines
            while j < len(lines):
                if lines[j].strip().startswith(']'):
                    tools_lines.append(lines[j])
                    j += 1
                    break
                elif lines[j].strip() and not lines[j].strip().startswith('#'):
                    tools_lines.append(lines[j])
                    j += 1
                else:
                    break

            # Check if there's a client method call
            has_client_call = False
            for k in range(j, min(j + 20, len(lines))):
                if any(method in lines[k] for method in [
                    'create_completion',
                    '_validate_request_with_config',
                    '_regular_completion',
                    '_convert_tools'
                ]):
                    has_client_call = True
                    break

            if has_client_call:
                # Change `tools =` to `tools_dicts =`
                tools_lines[0] = tools_lines[0].replace('tools =', 'tools_dicts =', 1)
                result_lines.extend(tools_lines)

                # Add conversion line
                spaces = ' ' * indent
                result_lines.append(f'{spaces}# Convert dicts to Pydantic models')
                result_lines.append(f'{spaces}tools = [Tool.model_validate(tool) for tool in tools_dicts]')
                result_lines.append('')

                i = j
                continue

        result_lines.append(line)
        i += 1

    return '\n'.join(result_lines)

def process_file(filepath: Path, provider: str):
    """Process a single test file."""
    print(f"Processing {filepath}...")

    with open(filepath) as f:
        content = f.read()

    # Add imports
    content = add_import_if_needed(content, provider)

    # Fix test methods
    content = fix_test_method(content)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Fixed {filepath}")

def main():
    """Main function."""
    test_dir = Path("/Users/chrishay/chris-source/chuk-ai/chuk-llm/tests/llm/providers")

    files_to_fix = [
        (test_dir / "test_azure_openai_client.py", "azure_openai"),
        (test_dir / "test_azure_openai_client_extended.py", "azure_openai"),
        (test_dir / "test_anthropic_client.py", "anthropic"),
        (test_dir / "test_gemini_client.py", "gemini"),
    ]

    for filepath, provider in files_to_fix:
        if filepath.exists():
            process_file(filepath, provider)
        else:
            print(f"File not found: {filepath}")

    print("\nAll files processed!")

if __name__ == "__main__":
    main()
