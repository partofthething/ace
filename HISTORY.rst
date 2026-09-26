History/Changelog
=================

0.4.0
-----
- Match Friedman's FORTRAN mace/supsmu exactly: fixed-span window rounding,
  near-zero-variance windows, cross-validated residual guard, tied x values,
  and mace's initialization, update acceptance, and convergence criteria
- ``ACESolver`` takes ``delrsq``, ``maxit``, and ``nterm`` arguments; removed ``MAX_OUTERS``
- Updated second Breiman85 sample to match the paper
- Removed deprecated ``pkg_resources`` usage for the version
- matplotlib is now optional for the smoothers (``pip install ace[plot]``)
- Packaging moved to ``pyproject.toml``; removed ``setup.py`` and requirements files
- Require Python 3.11+; version is now defined only in ``pyproject.toml``
- CI moved from Travis to GitHub Actions; lint and format with ruff
- Tests no longer leave output files in the working directory

0.3-3
-----
- Fix invalid integer division (#16)

0.3-2
-----
- Fixed divide-by-zero issue (#10)

0.3
---
- Fixed too large of random seed in tests (#4)
- Fixed README instructions (#8)
