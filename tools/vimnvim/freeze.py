#!/usr/bin/env python3
"""Freeze the configuration and the case list of a results directory BEFORE any run.

  freeze.py --out DIR [--config config.json]

Writes DIR/config.json (the config with the test set's sha256 filled in) and
DIR/cases.txt (the 310 case ids in benchmark order). Refuses to overwrite a
frozen config that differs: a run is one configuration.
"""
import argparse, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import case_id  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--config", default=str(HERE / "config.json"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
cfg = json.loads(Path(a.config).read_text())
raw = Path(cfg["benchmark"]["test_set"]).read_bytes()
cfg["benchmark"]["test_set_sha256"] = hashlib.sha256(raw).hexdigest()
cases = [case_id(i, c) for i, c in enumerate(json.loads(raw))]
assert len(cases) == cfg["benchmark"]["cases"] == len(set(cases))
for k, p in cfg["stored"].items():
    if k != "note" and not Path(p).exists():
        raise SystemExit(f"stored input missing: {k} = {p}")
cfg["stored_sha256"] = {k: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for k, p in cfg["stored"].items() if k != "note" and Path(p).is_file()}
text = json.dumps(cfg, indent=2) + "\n"
f = out / "config.json"
if f.exists() and f.read_text() != text:
    raise SystemExit(f"{f} is frozen and differs from {a.config}; use a new results directory")
f.write_text(text)
(out / "cases.txt").write_text("\n".join(cases) + "\n")
print(f"frozen: {f} ({len(cases)} cases, test set {cfg['benchmark']['test_set_sha256'][:12]})")
