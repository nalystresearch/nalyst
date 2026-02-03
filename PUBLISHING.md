# Publishing to PyPI

Steps to cut a release and publish Nalyst to PyPI.

## Prerequisites

- Clean working tree and tests passing
- Version bumped in `pyproject.toml` under `[project] version`
- Updated `CHANGELOG.md` with release notes

## Build artifacts

```bash
python -m pip install --upgrade pip build twine
python -m build  # creates dist/*.whl and dist/*.tar.gz
```

## Verify locally

```bash
python -m pip install dist/*.whl
python -c "import nalyst; print(nalyst.__version__)"
```

## Upload to PyPI

```bash
python -m twine upload dist/*
```

## Post-release

- Tag the release in version control (e.g., `git tag v0.1.0 && git push --tags`)
- Announce availability and update documentation links if needed
