#!/usr/bin/env python3
"""Steps 3 and 4: re-prompting on compile failures, with rcap.reprompt unchanged.

  run_loop.py --out DIR --arm loop_feedback|loop_retry|ctx_B|ctx_C [--only SUBSTR] [--limit N] [--offline]

Subset (fixed by the config, written to DIR/subset.json on first use): the cases
the compile check can reach (the developer's own function compiles in the tree)
whose ATTEMPT 1 fails to compile. Attempt 1 is NOT regenerated: it is the stored
zero-shot baseline output (config `stored.baseline`), exactly as scored in step 1.

Every further attempt is rcap.reprompt.run_reprompt_loop: a fresh single-turn
repair request (rcap-reprompt-v2: the previous attempt + the compile errors;
ctx_C: rcap-reprompt-v3-context, the same plus what each missing name is),
temperature 0, cycle detection, echo / reversion guard, CANNOT_ADAPT decline.
The loop's only signal is the compile check. It never sees the developer's port.

loop_*: k = 10.  ctx_B / ctx_C: k = 2 (exactly one re-prompt), paired.
Resumable by case_id; writes DIR/arms/<arm>.jsonl, DIR/attempts/<arm>/ and the caches.
"""
import argparse, json, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Unit, load_cases, load_config, load_jsonl, make_backend, nl, provenance, sha  # noqa: E402
from compile_c import Compiler  # noqa: E402
from gccdiag import missing_names, parse_gcc, render  # noqa: E402
from rcap.generate import GenerationRecord  # noqa: E402
from rcap.ppathf_metrics import is_complete  # noqa: E402
from rcap.reprompt import UnitState, Validation, run_reprompt_loop  # noqa: E402
from rcap.request import synthesize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--arm", required=True)
ap.add_argument("--only"); ap.add_argument("--limit", type=int); ap.add_argument("--offline", action="store_true")
a = ap.parse_args()
OUT = Path(a.out); cfg = load_config(OUT); arm = cfg["arms"][a.arm]; assert arm["step"] in (3, 4)
cases = load_cases(cfg)
stored = load_jsonl(cfg["stored"]["baseline"])
compiler = Compiler(cfg)


def subset() -> dict:
    """Reachable cases, and those whose stored attempt 1 fails to compile."""
    f = OUT / "subset.json"
    if f.exists():
        return json.loads(f.read_text())
    reach, why, failing, no_attempt = [], {}, [], []
    for cid, case in cases.items():
        ok, w = compiler.reachable(cid, case)
        why[w] = why.get(w, 0) + 1
        if not ok:
            continue
        reach.append(cid)
        r = stored[cid]
        if r["outcome"] != "ok" or not is_complete(r.get("candidate")):
            no_attempt.append(cid)
        elif not compiler.check(cid, case, nl(r["candidate"].strip()))["ok"]:
            failing.append(cid)
    s = {"reachable": reach, "gate": why, "attempt1_fails": failing, "no_attempt1": no_attempt,
         "tree_commit": compiler.commit, "flags": compiler.extra}
    f.write_text(json.dumps(s, indent=1))
    return s


sub = subset()
backend = make_backend(cfg, live=not a.offline)
prov = provenance(cfg, backend, arm=a.arm, arm_config=arm, compile_tree=compiler.commit, compile_flags=compiler.extra)
symbols = None
if a.arm == "ctx_C":
    from symctx import SymbolContexts
    symbols = SymbolContexts(cfg)
# every request that reaches the backend is kept, also one answered by a decline
_current = {"dir": None}
_generate = backend.generate


def _generate_and_keep(request, **budget):
    if _current["dir"] is not None:
        (_current["dir"] / f"request-{sha(request.prompt)[:16]}.txt").write_text(request.prompt)
    return _generate(request, **budget)


backend.generate = _generate_and_keep
out = OUT / "arms" / f"{a.arm}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
done = load_jsonl(out)
print(f"{a.arm}: {len(sub['attempt1_fails'])} cases (reachable {len(sub['reachable'])}), {len(done)} done", flush=True)
n = 0
with out.open("a") as fh:
    for cid in sub["attempt1_fails"]:
        if cid in done or (a.only and a.only not in cid):
            continue
        if a.limit is not None and n >= a.limit:
            break
        n += 1
        case = cases[cid]; t0 = time.time()
        unit = Unit(cid, case, False)
        request = synthesize(unit.context)
        attempt1 = nl(stored[cid]["candidate"].strip())
        gen = GenerationRecord(case_id=cid, outcome="completion", candidate=attempt1,
                               request_hash=request.request_hash, exec_id="stored-baseline")
        state = UnitState(index=0, entity=unit.entity, context=unit.context, request=request, generation=gen,
                          target_original=unit.target_original, expand=unit.expand,
                          expected_placeholders=set())
        adir = OUT / "attempts" / a.arm / cid; adir.mkdir(parents=True, exist_ok=True)
        _current["dir"] = adir
        attempts: dict[int, dict] = {}

        def on_attempt(k, arts, attempts=attempts, adir=adir):
            art = arts[0]
            attempts[k] = {"attempt": k, "candidate": art["candidate_expanded"],
                           "prompt_sha256": sha(art["prompt"]) if k > 1 else None,
                           "prompt_chars": len(art["prompt"]) if k > 1 else None}
            if k > 1:
                (adir / f"a{k}.prompt.txt").write_text(art["prompt"])
            (adir / f"a{k}.candidate.c").write_text(art["candidate_expanded"])

        def validate(k, expanded, cid=cid, case=case, attempts=attempts, adir=adir):
            res = compiler.check(cid, case, expanded[0])
            if res["ok"]:
                attempts[k]["compiles"] = True
                return Validation(verdict="VALID", seconds=res["seconds"], detail={"compile": "ok"})
            errors = parse_gcc(res["log"])
            if errors:
                text, stats = render(errors, first_line=res["first_line"], last_line=res["last_line"], rel_path=res["rel"])
            else:   # a failure gcc reports in no parsable line: the end of its output
                text, stats = "The compiler failed:\n" + res["log"][-2000:], {"errors": 0}
            (adir / f"a{k}.errors.txt").write_text(text + "\n\n==== full gcc output ====\n" + res["log"])
            miss = missing_names(errors)
            attempts[k].update(compiles=False, errors=stats, missing=[m.__dict__ for m in miss])
            return Validation(verdict="BUILD_FAIL", kind="build", feedback=text, units_with_errors=[0],
                              summary=errors[0].message[:200] if errors else "compile failed", seconds=res["seconds"],
                              detail={"compile": "fail", **stats, "missing": [m.name for m in miss],
                                      "_errors": [e.__dict__ for e in errors]})

        def extras(i, validation, case=case, attempts=attempts):
            from gccdiag import Diag
            errors = [Diag(**e) for e in validation.detail.get("_errors", [])]
            miss = missing_names(errors)
            if not miss:
                return {}
            scs, omitted = symbols.build(case, miss)
            return {"symbol_context": scs, "symbol_context_omitted": omitted}

        row = {"arm": a.arm, "case_id": cid, "attempt1_source": "stored.baseline", "provenance": prov}
        try:
            rec = run_reprompt_loop([state], backend, validate, k=arm["k"],
                                    echo_policy=arm.get("echo_policy", "feedback"), on_attempt=on_attempt,
                                    repair_extras=extras if symbols is not None else None)
            for at in rec["attempts"]:
                (at.get("validation") or {}).pop("_errors", None)
                extra = attempts.get(at["attempt"], {})
                at["candidate"] = extra.get("candidate")
                at["compiles"] = extra.get("compiles")
                at["compile_errors"] = extra.get("errors")
                at["missing"] = extra.get("missing")
            row.update(rec)
        except Exception as exc:  # noqa: BLE001
            row.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-1500:])
        row["seconds"] = round(time.time() - t0, 1)
        fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
        print(f"{cid} stop={row.get('stop_reason')} attempts={row.get('attempts_run')} "
              f"verdicts={[x.get('verdict') for x in row.get('attempts', [])]} t={row['seconds']}s "
              f"{row.get('error', '')}", flush=True)
print("DONE", a.arm, dict(backend.sources))
