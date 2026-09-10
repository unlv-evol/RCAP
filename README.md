# RCAP — Reusable Change Adaptation Pipeline

Consumes a SAP directory produced by [SALP](https://github.com/unlv-evol/SALP) and
produces one Adaptation Package (a single LLM candidate plus provenance) per the
RCAP Implementation Reference. Stages: intake → selection → semantic evidence
reduction → payload materialization → program-context reduction → adaptation
context → request synthesis → candidate generation → adaptation package.

## Setup

```bash
uv venv .venv
uv pip install -p .venv/bin/python -e "../../SALP[structural]" -e ".[dev]"
.venv/bin/pytest
```

SALP is a pinned library dependency (clone at ../../SALP); RCAP reuses its SAP
models, schema validator, tree-sitter layer, and repo cache. Tests build their
SAP fixtures with SALP's own writer so every fixture passes `salp validate`.
