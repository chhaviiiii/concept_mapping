# Contributing to PyConceptMap

Thank you for your interest in contributing to PyConceptMap! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

1. **Check existing issues** before creating a new one
2. **Use the issue template** when available
3. **Provide detailed information**:
   - Python version
   - Operating system
   - Error messages
   - Steps to reproduce

### Suggesting Features

1. **Check existing feature requests**
2. **Describe the use case** clearly
3. **Explain the expected behavior**
4. **Consider implementation complexity**

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation**
6. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- pip

### Installation

```bash
# Fork and clone the repository
git clone https://github.com/your-username/pyconceptmap.git
cd pyconceptmap

# Create a virtual environment
python -m venv pyconceptmap_env
source pyconceptmap_env/bin/activate  # Linux/Mac
# or
pyconceptmap_env\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

### Development Dependencies

```bash
# Install development dependencies
pip install pytest black flake8 mypy pytest-cov
```

## Code Style

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters
- **Import order**: Standard library, third-party, local imports
- **Docstrings**: Google style

### Formatting

```bash
# Format code with black
black pyconceptmap/

# Check style with flake8
flake8 pyconceptmap/

# Type checking with mypy
mypy pyconceptmap/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyconceptmap

# Run specific test file
pytest tests/test_core.py
```

### Writing Tests

1. **Test new functionality** thoroughly
2. **Use descriptive test names**
3. **Test edge cases** and error conditions
4. **Mock external dependencies**

### Test Structure

```python
def test_function_name():
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
```

## Documentation

### Code Documentation

- **Docstrings**: All public functions and classes
- **Type hints**: Use type annotations
- **Comments**: Explain complex logic

### User Documentation

- **README.md**: Keep updated
- **User Guide**: Add new features
- **Examples**: Provide working examples
- **API Documentation**: Document new functions

## Pull Request Process

### Before Submitting

1. **Run tests**: `pytest`
2. **Check style**: `flake8 pyconceptmap/`
3. **Format code**: `black pyconceptmap/`
4. **Update documentation**
5. **Add tests** for new features

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `setup.py`
2. **Update changelog**
3. **Run full test suite**
4. **Update documentation**
5. **Create release notes**

## Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Be patient** with newcomers
- **Be collaborative** in discussions

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code changes and reviews

## Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **Documentation** for major features

## Questions?

If you have questions about contributing:

1. **Check existing issues** and discussions
2. **Open a new issue** with the "question" label
3. **Start a discussion** for general questions

Thank you for contributing to PyConceptMap! 🎉
