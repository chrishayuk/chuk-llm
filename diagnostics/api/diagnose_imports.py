#!/usr/bin/env python3
"""
Find the exact source of the bad import error messages
"""

from pathlib import Path
import re

def find_import_messages():
    try:
        import chuk_llm
        package_root = Path(chuk_llm.__file__).parent
        print(f"📁 Searching in: {package_root}")
        print("=" * 60)
        
        # The exact error messages we're seeing
        error_patterns = [
            "Configuration module import failed",
            "LLM client import failed", 
            "API module import failed",
            "Conversation module import failed",
            "Utils module import failed"
        ]
        
        # Search for these exact strings in all Python files
        found_sources = []
        
        for py_file in package_root.rglob('*.py'):
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern in error_patterns:
                    if pattern in content:
                        rel_path = py_file.relative_to(package_root.parent)
                        found_sources.append((rel_path, pattern))
                        
                        # Find the exact line
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if pattern in line:
                                print(f"🔍 Found: {pattern}")
                                print(f"   File: {rel_path}")
                                print(f"   Line {i}: {line.strip()}")
                                
                                # Show context around this line
                                start = max(0, i-5)
                                end = min(len(lines), i+5)
                                print(f"   Context:")
                                for j in range(start, end):
                                    marker = ">>>" if j == i-1 else "   "
                                    print(f"   {marker} {j+1:3}: {lines[j]}")
                                print()
                                break
            except Exception as e:
                continue
        
        if not found_sources:
            print("❌ Could not find the source of error messages")
            print("🔍 Let's search for bad import patterns instead...")
            
            # Search for the bad import patterns
            bad_import_patterns = [
                'chuk_llm.llm.configuration',
                'chuk_llm.llm.llm',
                'chuk_llm.llm.api'
            ]
            
            for py_file in package_root.rglob('*.py'):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    
                    for pattern in bad_import_patterns:
                        if pattern in content:
                            rel_path = py_file.relative_to(package_root.parent)
                            print(f"🔍 Bad import pattern: {pattern}")
                            print(f"   File: {rel_path}")
                            
                            # Find the exact line
                            lines = content.split('\n')
                            for i, line in enumerate(lines, 1):
                                if pattern in line:
                                    print(f"   Line {i}: {line.strip()}")
                            print()
                except Exception:
                    continue
        else:
            print(f"✅ Found {len(found_sources)} sources of error messages")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    find_import_messages()