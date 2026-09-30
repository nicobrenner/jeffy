# Jeffy v0.1.0-alpha.6 Release Notes

## What's new in alpha.6

- **Project URLs**: `pyproject.toml` now includes homepage, repository, and issue tracker links pointing to [github.com/nicobrenner/jeffy](https://github.com/nicobrenner/jeffy).
- **Demo GIF**: Animated terminal recording (`examples/demo.gif`) showing pretrained classification and custom training, embedded in README.
- **Linux verification**: Wheel and sdist tested in clean environments on Linux aarch64. Latency benchmarks updated for Linux CPU (50–80ms per prediction).
- **Corrected README**: Fixed bundled head count (was "6", now "13"). Updated deployment requirements, latency table, tested platform, and roadmap.
- **Updated demo output**: `examples/demo_output.txt` regenerated from actual Linux run with 24-example custom training.

## Artifacts

| File | SHA256 |
|------|--------|
| `jeffy_classify-0.1.0a6-py3-none-any.whl` | `552d2883226627d5c06a566fe715a208ea29a9d643b954c3c9173a4dd5a2d5f6` |
| `jeffy_classify-0.1.0a6.tar.gz` | `d3da1cc1cc8aaba4a9372c7ae38b2b4c6927b62658b9c46a76af47f4efd09a91` |

## Verification

- Wheel install (clean venv): 13 heads loaded, correct predictions, no warnings
- Sdist install (clean venv): 13 heads loaded, custom training works, no warnings
- API tests: 20/20 passed
- `warnings.filterwarnings('error')`: no warnings raised
- Git identity: all commits and tags use `296756+nicobrenner@users.noreply.github.com`

## No changes to

- Model pack artifacts (same 13 npz files as alpha.5)
- Engine, server, training, or evaluation code
- ATTRIBUTION.md or LICENSE

## Source

Built from commit `1b3ae6f`, tagged `v0.1.0-alpha.6` on branch `main`.
Alpha.5 tag is preserved at `47b113e`.
