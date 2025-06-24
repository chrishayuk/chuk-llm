# Fix for uv editable install issues

# 1. Check uv version and status
echo "=== UV Status ==="
uv --version
uv pip list | grep chuk-llm || echo "chuk-llm not found in uv pip list"

# 2. Clean any existing installation
echo "=== Cleaning existing installation ==="
uv pip uninstall chuk-llm -y || true
rm -rf .venv/lib/python*/site-packages/chuk*
rm -rf *.egg-info/ src/*.egg-info/

# 3. Install in editable mode with uv
echo "=== Installing with uv in editable mode ==="
uv pip install -e .

# 4. Check if it's properly installed
echo "=== Checking installation ==="
uv pip list | grep chuk
uv pip show chuk-llm || echo "Package not found with uv pip show"

# 5. Alternative uv installation methods
echo "=== Alternative uv installation ==="
# Try with sync
uv sync

# Or try adding to project
uv add -e .

# 6. Check Python environment
echo "=== Python environment check ==="
which python
python -c "
import sys
print('Python executable:', sys.executable)
print('Python path:')
for p in sys.path:
    print(f'  {p}')
"

# 7. Test import
echo "=== Testing import ==="
python -c "
try:
    import chuk_llm
    print('✅ Import successful')
    print('Location:', chuk_llm.__file__)
except ImportError as e:
    print('❌ Import failed:', e)
    print('Trying manual path...')
    import sys
    sys.path.insert(0, 'src')
    try:
        import chuk_llm
        print('✅ Manual path import successful')
    except ImportError as e2:
        print('❌ Manual path also failed:', e2)
"

# 8. Run tests with PYTHONPATH as fallback
echo "=== Running tests (with fallback) ==="
if python -c "import chuk_llm" 2>/dev/null; then
    echo "Running tests normally..."
    pytest tests/ -v
else
    echo "Using PYTHONPATH fallback..."
    PYTHONPATH="$(pwd)/src" pytest tests/ -v
fi