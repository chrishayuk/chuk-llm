# Contributing to ChukLLM

Thank you for your interest in contributing to ChukLLM! We welcome contributions of all kinds, from bug fixes to new features.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/chrishayuk/chuk-llm.git
   cd chuk-llm
   ```
3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   # or with uv
   uv sync --dev
   ```

## Development Setup

### Environment Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/api/test_conversation.py

# Run with coverage
make test-cov
```

### Code Quality

```bash
# Run linting
make lint

# Format code
make format

# Type checking
make typecheck

# Run all checks
make check
```

## Contribution Guidelines

### Code Style

- We use `ruff` for linting and formatting
- Follow PEP 8 with a line length of 100 characters
- Use type hints for all function signatures
- Write docstrings for all public functions and classes

### Testing

- Write tests for all new features
- Maintain or improve code coverage
- Test files should mirror the source structure in `tests/`
- Use `pytest` fixtures for common test setup

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or fixes
- `refactor`: Code refactoring
- `style`: Code style changes
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Examples:
```
feat(conversation): add branching support for conversations
fix(ollama): handle connection errors gracefully
docs(readme): simplify quick start section
```

### Pull Request Process

1. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them with clear messages

3. **Add tests** for new functionality

4. **Run checks** locally:
   ```bash
   make check
   ```

5. **Push to your fork** and create a pull request

6. **Describe your changes** in the PR:
   - What problem does it solve?
   - What is the approach?
   - Any breaking changes?

### Adding New Providers

To add a new LLM provider:

1. Create a new client in `src/chuk_llm/llm/providers/`
2. Inherit from `BaseLLMClient`
3. Implement required methods: `ask()` and `stream()`
4. Add provider configuration to `chuk_llm.yaml`
5. Add tests in `tests/llm/providers/`
6. Update documentation

Example structure:
```python
# src/chuk_llm/llm/providers/newprovider_client.py
from ..base import BaseLLMClient

class NewProviderClient(BaseLLMClient):
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(provider="newprovider", **kwargs)
        # Initialize client
    
    async def ask(self, messages, **kwargs):
        # Implement ask logic
        pass
    
    async def stream(self, messages, **kwargs):
        # Implement streaming logic
        pass
```

### Adding New Features

Before adding a major feature:

1. **Open an issue** to discuss the feature
2. **Get feedback** from maintainers
3. **Design the API** with examples
4. **Implement incrementally** with tests

## Project Structure

```
chuk-llm/
├── src/chuk_llm/
│   ├── api/           # Public API layer
│   ├── llm/           # LLM client implementations
│   ├── configuration/ # Config management
│   └── cli.py         # CLI implementation
├── tests/             # Test suite
├── examples/          # Usage examples
├── docs/              # Documentation
└── benchmarks/        # Performance benchmarks
```

## Getting Help

- 💬 [Discussions](https://github.com/chrishayuk/chuk-llm/discussions) - Ask questions
- 🐛 [Issues](https://github.com/chrishayuk/chuk-llm/issues) - Report bugs
- 📧 [Email](mailto:chrishayuk@somejunkmailbox.com) - Direct contact

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- GitHub contributors page

Thank you for helping make ChukLLM better! 🎉