#!/usr/bin/env python3
"""
Example tools file for calculator functionality.
Usage: chuk-llm ask "Calculate 15 * 4" --tools calculator_tools.py
"""

def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

def subtract(a: float, b: float) -> float:
    """Subtract b from a"""
    return a - b

def divide(a: float, b: float) -> float:
    """Divide a by b"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression"""
    # Simple evaluation for basic arithmetic
    # In production, use a proper math parser
    try:
        # Only allow basic arithmetic operations
        allowed = set('0123456789+-*/(). ')
        if all(c in allowed for c in expression):
            return float(eval(expression, {"__builtins__": {}}))
        else:
            raise ValueError("Expression contains invalid characters")
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")

# Export functions for CLI usage
__all__ = ['add', 'multiply', 'subtract', 'divide', 'calculate']