# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Apache 2.0 `LICENSE` + `NOTICE` with third-party data attribution.
- Community/governance files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `CODEOWNERS`.
- `pyproject.toml` packaging the `situation_testing` library with pinned
  dependencies and Black/Ruff/mypy/pytest/coverage configuration.
- `tests/` pytest suite covering `situation_testing` utilities, the kdd2011
  distance function, and an end-to-end `SituationTesting.run` smoke test.
- SPDX headers (`Copyright (c) 2026 Santander Group` / `Apache-2.0`) on all
  Python source files.
- GitHub Actions workflows (third-party actions pinned to SHA digests):
  `ci`, `codeql`, `dep-scan`, `license-check`, `pattern-check`, `cla`,
  `stale`, `release`.
- `.github/dependabot.yml` (monthly Python + GitHub Actions updates),
  issue templates (bug, feature, config) and a PR template.
- `.github/pattern-check-allowlist.txt` for the internal-pattern scan.

### Fixed
- Cross-platform data path in `src/run_exp_law_school.py` (was a Windows-only
  backslash literal; now uses `os.path.join`).

[Unreleased]: https://github.com/SantanderAI/mutatis-mutandis/commits/main
