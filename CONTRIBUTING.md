# Contributing to Qdrant Plugin for Dify

Thank you for your interest in contributing to the Qdrant Plugin for Dify! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs or suggest features
- Provide clear descriptions, steps to reproduce, and expected vs actual behavior
- Include relevant environment information (Dify version, Python version, etc.)

### Submitting Pull Requests

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following the coding standards
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

### Development Setup

1. Clone your fork: `git clone https://github.com/your-username/dify-plugin-qdrant.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test locally
6. Commit your changes: `git commit -m "Add: description of changes"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Coding Standards

### Python Code

- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings for functions and classes
- Keep functions focused and small
- Handle errors gracefully with clear messages

### YAML Files

- Use consistent indentation (2 spaces)
- Keep descriptions clear and concise
- Support multiple languages (en_US, zh_Hans, etc.)
- Follow Dify plugin YAML schema

### Documentation

- Write clear, concise documentation
- Use English for all documentation
- Include examples where helpful
- Keep README and SETUP_GUIDE up to date

## Commit Messages

Use clear, descriptive commit messages:

- `Add: feature description`
- `Fix: bug description`
- `Update: change description`
- `Refactor: refactoring description`
- `Docs: documentation update`

## Testing

Before submitting a PR:

- Test all affected functionality
- Test error cases
- Test with different Qdrant configurations
- Ensure no regressions

## Release Process

Releases are managed through GitHub Actions. When you push to `main`:

1. Update version in `manifest.yaml`
2. Push changes to `main` branch
3. GitHub Actions will automatically:
   - Package the plugin
   - Create a PR to dify-plugins repository
   - Update the plugin package

## Questions?

If you have questions about contributing, please:
- Open an issue for discussion
- Check existing issues and PRs
- Review the Dify plugin documentation

Thank you for contributing! 🎉
