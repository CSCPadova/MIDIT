# Contributing to MIDIT

First off, thank you for considering contributing to MIDIT! MIDIT is a research project for bidirectional symbolic MIDI infilling.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

## What kinds of contributions we are looking for

MIDIT is an open source project and we love to receive contributions from our community! There are many ways to contribute, from writing tutorials or blog posts, improving the documentation, submitting bug reports and feature requests or writing code which can be incorporated into MIDIT itself.

## Ground Rules

### Responsibilities

* Ensure that code passes all pre-commit checks before submitting a pull request
* Create issues for any major changes and enhancements that you wish to make. Discuss things transparently and get community feedback
* Keep feature versions as small as possible, preferably one new feature per version
* Be welcoming to newcomers and encourage diverse new contributors from all backgrounds
* Ensure your changes work with Python 3.12+ and are compatible with the project dependencies
* Keep the `uv.lock` file in sync when adding or updating dependencies

## Getting Started

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MIDIT.git
   cd MIDIT
   ```

3. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

4. **Install dependencies** using uv:
   ```bash
   make install
   # or manually
   uv sync
   ```

5. **Install pre-commit hooks**:
   ```bash
   uv run pre-commit install
   ```

### Development Workflow

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** and commit them using gitmoji (see below)

3. **Ensure all checks pass** before committing:
   ```bash
   # Pre-commit will automatically run on commit
   # You can also run it manually:
   uv run pre-commit run --all-files
   ```

4. **Run the relevant checks** (if applicable):
   ```bash
   # Run formatting and repository checks.
   uv run pre-commit run --all-files
   # CPU smoke tests (imports, Hydra configs, CLI --help); no data needed.
   uv run pytest
   ```

   Training requires a preprocessed dataset and configured accelerator. See the
   root README and `scripts/README.md` for supported workflows; do not treat a
   training run as the default development check.

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** to the `dev` branch (or `main` if `dev` doesn't exist yet)

## Important Guidelines

### Use Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. The hooks will automatically run when you commit, checking for:
- Trailing whitespace
- End-of-file fixes
- YAML validation
- Large files

Make sure to install pre-commit hooks as described above. If pre-commit checks fail, fix the issues before committing.

### Managing Large Files

**IMPORTANT**: Avoid committing large files, datasets, or caches to git!

- For large data files, use [Git LFS](https://git-lfs.github.com/) or [DVC](https://dvc.org/)
- Add large files to `.gitignore` if they shouldn't be tracked
- Model checkpoints, training logs, and datasets should NOT be committed to the repository
- The `.gitignore` file already excludes common large file patterns

### Use Gitmoji for Commits 🎨

We use [gitmoji](https://gitmoji.dev/) for commit messages because it's fun and enforces having a clear commit title and detailed description!

**Format**: `<gitmoji> <type>: <subject>`

Common gitmojis for this project:
- ✨ `:sparkles:` - New feature
- 🐛 `:bug:` - Bug fix
- 📝 `:memo:` - Documentation
- ♻️ `:recycle:` - Refactoring
- ⚡ `:zap:` - Performance improvement
- ✅ `:white_check_mark:` - Tests
- 🔧 `:wrench:` - Configuration files
- 🚀 `:rocket:` - Deployment

**Example**:
```bash
git commit -m "✨ feat: add top-k sampling to infilling inference

- Add --top_k option to scripts/inference/infer_from_midi.py
- Keep argmax decoding as the default
- Document the new option in README.md"
```

Install the gitmoji CLI tool (optional but helpful):
```bash
npm install -g gitmoji-cli
gitmoji -c  # Interactive commit helper
```

### Package Management with uv

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

**When adding or updating dependencies**:

1. Add the dependency to `pyproject.toml`:
   ```toml
   dependencies = [
       "new-package>=1.0.0",
   ]
   ```

2. Run `uv sync` to update `uv.lock`:
   ```bash
   uv sync
   ```

3. **Always commit both `pyproject.toml` and `uv.lock` together** to ensure reproducible builds

4. Verify the lock file is in sync:
   ```bash
   uv sync --check
   ```

## How to Report a Bug

When filing an issue, make sure to answer these questions:

1. What version of MIDIT are you using?
2. What operating system and Python version are you using?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?

Please include:
- Full error messages and stack traces
- Relevant configuration files
- Steps to reproduce the issue

## How to Suggest a Feature or Enhancement

If you find yourself wishing for a feature that doesn't exist in MIDIT, you are probably not alone. Open an issue on GitHub that describes:

1. The feature you would like to see
2. Why you need it
3. How it should work

## Code Review Process

The core team reviews Pull Requests on a regular basis. Here's what to expect:

1. **Initial Review**: We'll review your PR within a few days and provide feedback
2. **Feedback Loop**: Please respond to feedback within two weeks. We may close PRs that show no activity
3. **Approval**: Once approved, a maintainer will merge your PR into the `dev` branch
4. **Release**: Changes in `dev` will be merged to `main` during regular releases

## Development Tips

### Quick Testing

`uv run pytest` runs the CPU smoke tests in `tests/` (imports, Hydra config
composition, `--help` of the documented commands); add `--group eval` to the sync
to include the evaluation scripts. There is no data-free training test. For
changes to inference, also run the quickstart in the README and check that the
output MIDI files are unchanged.

### Configuration Override

Override Hydra configuration values without modifying files:
```bash
uv run python midit/train_phase1.py model.d_model=256 train.batch_size=16
```

This command starts training and therefore requires the configured dataset and
accelerator; it is an example of configuration composition, not a lightweight test.

### Logging

Monitor your training with TensorBoard:
```bash
tensorboard --logdir logs/
```

## Community

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: For questions and general discussion, use GitHub Discussions (if available)
- **Pull Requests**: Submit PRs for code contributions

## Code of Conduct

Be respectful, inclusive, and considerate in all interactions. We want MIDIT to be a welcoming community for everyone.

## Questions?

Don't hesitate to ask! Open an issue with your question, and we'll be happy to help.

---

Thank you for contributing to MIDIT! 🎵✨
