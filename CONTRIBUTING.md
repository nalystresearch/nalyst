# Contributing to Nalyst

Thank you for considering a contribution. Nalyst is a community-friendly project focused on practical machine learning, statistics, and deep learning. Contributions of all sizes are welcome: issues, docs, examples, and code.

## Ways to contribute

- Report bugs and edge cases with clear reproduction steps
- Improve documentation and examples
- Add tests for existing functionality
- Implement well-scoped enhancements aligned with the roadmap

## Development setup

1. Ensure Python 3.10+ is available.
2. Install dependencies in editable mode:
   ```bash
   python -m pip install --upgrade pip
   pip install -e .[dev,visualization,dataframes]
   ```
3. Work on a feature branch; keep commits focused and clear.

## Testing and quality

- Run tests: `python -m pytest`
- Lint: `ruff check .`
- Type check: `mypy nalyst`
- Format (if needed): `ruff format .`

## Documentation and examples

- User guides and API docs live in `doc/` (reStructuredText).
- Examples live in `examples/` and should stay runnable with minimal dependencies.
- When adding new features, include at least one example and minimal docs.

## Pull request checklist

- [ ] Tests added/updated for new behavior
- [ ] Lint and type checks are clean
- [ ] Documentation/examples updated as needed
- [ ] Description explains the change and any trade-offs

## Code of conduct

We follow a simple respect-first approach. See CODE_OF_CONDUCT.md. Unkind or exclusionary behavior is not tolerated.
