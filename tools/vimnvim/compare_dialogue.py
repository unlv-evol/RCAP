#!/usr/bin/env python3
"""Dialogue arm against the fresh-request loops, on the same cases.

  compare_dialogue.py --out DIR [--arms loop_feedback loop_retry dialogue]

Per case and arm: how the loop ended, after how many attempts, and how far the
delivered function is from the developers' own port (PPatHF token distance).
"Delivered" is the loop's rule: the first attempt that compiles, else the fork
unchanged. `d_first` is the distance of attempt 1 (the same stored answer in
every arm), `d_fork` that of doing nothing. Reads only; prints Markdown.
"""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_cases, load_config, load_jsonl, nl  # noqa: E402
import rcap.ppathf_metrics as pm  # noqa: E402
from rcap.reprompt import code_equal  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--arms", nargs="+", default=["loop_feedback", "loop_retry", "dialogue"])
a = ap.parse_args()
OUT = Path(a.out); cfg = load_config(OUT); cases = load_cases(cfg)
rows = {arm: {c: r for c, r in load_jsonl(OUT / "arms" / f"{arm}.jsonl").items() if "attempts" in r} for arm in a.arms}
common = [c for c in rows[a.arms[0]] if all(c in rows[x] for x in a.arms)]


def dist(cid, text):
    case = cases[cid]
    return pm.score(text, nl(case["func_after_target"]), nl(case["func_before_target"]))


def summary(cid, r):
    pre = nl(cases[cid]["func_before_target"])
    seq = [(at["attempt"], at["candidate"], at.get("verdict")) for at in r["attempts"] if at.get("candidate")]
    ok = next((x for x in seq if x[2] == "VALID"), None)
    delivered = ok[1] if ok else pre
    d = dist(cid, delivered)
    last = r["attempts"][-1]["units"][0]
    return {"stop": r.get("stop_reason"), "n": r.get("attempts_run"), "success": ok[0] if ok else None,
            "d": d["gen_distance"], "exact": d["exact_match"], "d_fork": d["src_distance"],
            "d_first": dist(cid, seq[0][1])["gen_distance"] if seq else None,
            "reverted": bool(ok) and code_equal(ok[1], pre),
            "verdicts": [x.get("verdict") for x in r["attempts"]],
            "conv_chars": last.get("conversation_chars"), "in_tok": (last.get("usage") or {}).get("input_tokens"),
            "gen_s": sum(u.get("gen_seconds") or 0 for at in r["attempts"] for u in at["units"])}


S = {arm: {c: summary(c, rows[arm][c]) for c in common} for arm in a.arms}
print(f"cases in every arm: {len(common)}\n")
print("| arm | compiles | attempts used (median / max) | closer to the developers' port than attempt 1 | further | exact | mean distance delivered | stops |")
print("|---|---|---|---|---|---|---|---|")
for arm in a.arms:
    v = list(S[arm].values()); ok = [x for x in v if x["success"]]
    ns = sorted(x["n"] for x in v); stops = {}
    for x in v:
        stops[x["stop"]] = stops.get(x["stop"], 0) + 1
    print(f"| {arm} | {len(ok)}/{len(v)} | {ns[len(ns) // 2]} / {ns[-1]} | "
          f"{sum(1 for x in ok if x['d'] < x['d_first'])} | {sum(1 for x in ok if x['d'] > x['d_first'])} | "
          f"{sum(1 for x in v if x['exact'])} | {sum(x['d'] for x in v) / len(v):.1f} | "
          f"{', '.join(f'{k} {n}' for k, n in sorted(stops.items(), key=lambda kv: -kv[1]))} |")
print("\nmean distance of attempt 1: "
      f"{sum(x['d_first'] for x in S[a.arms[0]].values()) / len(common):.1f}; of the fork unchanged: "
      f"{sum(x['d_fork'] for x in S[a.arms[0]].values()) / len(common):.1f}\n")
print("| case | d fork | d attempt 1 | " + " | ".join(f"{arm}: stop @n, d" for arm in a.arms) + " |")
print("|---|---|---|" + "---|" * len(a.arms))
for c in common:
    f = S[a.arms[0]][c]
    cells = [f"{S[arm][c]['stop']} @{S[arm][c]['n']}, {S[arm][c]['d'] if S[arm][c]['success'] else '-'}" for arm in a.arms]
    print(f"| {c} | {f['d_fork']} | {f['d_first']} | " + " | ".join(cells) + " |")
if "dialogue" in S:
    v = list(S["dialogue"].values())
    print(f"\ndialogue: longest conversation {max(x['conv_chars'] or 0 for x in v)} chars, "
          f"{max(x['in_tok'] or 0 for x in v)} input tokens (window {cfg['backend']['num_ctx']}); "
          f"generation time total {sum(x['gen_s'] for x in v) / 60:.1f} min")
    for arm in a.arms:
        if arm != "dialogue":
            g = [c for c in common if S["dialogue"][c]["success"] and not S[arm][c]["success"]]
            l = [c for c in common if S[arm][c]["success"] and not S["dialogue"][c]["success"]]
            print(f"dialogue vs {arm}: compiles only in dialogue {len(g)} {g}; only in {arm} {len(l)} {l}; "
                  f"McNemar p = {pm.mcnemar_exact([bool(S[arm][c]['success']) for c in common], [bool(S['dialogue'][c]['success']) for c in common]).get('p')}")
