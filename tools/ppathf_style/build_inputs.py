"""PPatHF-style treatment, inputs (roadmap item 8): for every function unit of a pair's SAPs, the
(source before, source after, target) triple RCAP also receives, reduced by PPatHF's reduction module
(reducer.py, Java profile) and formatted with PPatHF's porting prompt template (porting/data.py, PROMPT_TEMPLATE),
with the two repositories named in place of vim and neovim. Nothing from the SAP's evidence is used: that is
the point of the treatment.

A unit is eligible when all three payloads are exactly one function (rcap.unitkind.text_kind) in a language the
reducer has a profile for; every other unit is written with the reason it is not, never dropped.

    python build_inputs.py <SALP out dir of one pair> <source repo> <target repo> <out.jsonl> [--prs PR-1,PR-2]
                           [--direct direct.jsonl]
"""
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import reducerConfig  # noqa: E402
from rcap.unitkind import FUNCTION, text_kind  # noqa: E402

LANG_OF = {".java": "java", ".scala": "scala"}

# PPatHF porting/data.py PROMPT_TEMPLATE, with the repository names as parameters
INSTRUCTION = ("Below is a patch (including function before and function after) from {src}, paired with a "
               "corresponding function before from {tgt}. Adapt the patch from {src} to {tgt} by generating the "
               "function after based on the given function before.\n\n")
CONTEXT = ("### Function Before ({src}):\n{func_before_source}\n\n"
           "### Function After ({src}):\n{func_after_source}\n\n"
           "### Function Before ({tgt}):\n{func_before_target}\n\n"
           "### Function After ({tgt}):\n")


def prompt(src, tgt, fb, fa, ft):
    return INSTRUCTION.format(src=src, tgt=tgt) + CONTEXT.format(src=src, tgt=tgt, func_before_source=fb,
                                                                 func_after_source=fa, func_before_target=ft)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_dir"); ap.add_argument("src"); ap.add_argument("tgt"); ap.add_argument("out")
    ap.add_argument("--prs", help="comma-separated PR ids (default: all)")
    ap.add_argument("--direct", help="also write the Direct LLM inputs here: every eligible triple, unreduced")
    ap.add_argument("--pair", help="name of the repository pair, recorded in every row (needed when one input file "
                                   "holds several pairs: the same PR can be ported into two forks)")
    a = ap.parse_args()
    only = set(a.prs.split(",")) if a.prs else None
    from reducer import Reducer
    reducers = {}
    direct = open(a.direct, "w") if a.direct else None
    with open(a.out, "w") as out:
        for pr in sorted(Path(a.pair_dir).glob("PR-*"), key=lambda p: int(p.name[3:])):
            if only and pr.name not in only:
                continue
            for fd in sorted(pr.glob("sap-*/functions/*")):
                row = {**({"pair": a.pair} if a.pair else {}), "pr": pr.name, "sap": fd.parent.parent.name, "fn": fd.name,
                       "unit": f"{fd.parent.parent.name}::{fd.name}"}
                texts = {k: next(iter(sorted(fd.glob(k + ".*"))), None) for k in ("source.before", "source.after", "target")}
                tf = texts["target"]
                lang = LANG_OF.get(tf.suffix) if tf else None
                row["language"] = lang
                if any(v is None for v in texts.values()):
                    row["eligible"] = False; row["reason"] = "missing payload"; out.write(json.dumps(row) + "\n"); continue
                fb, fa, ft = (texts[k].read_text(encoding="utf-8", errors="replace") for k in
                              ("source.before", "source.after", "target"))
                kinds = [text_kind(t, tf.suffix) for t in (fb, fa, ft)]
                row["kinds"] = kinds
                raw = prompt(a.src, a.tgt, fb, fa, ft)
                row["raw_prompt_chars"] = len(raw)
                if lang not in reducerConfig.PROFILES:
                    row["eligible"] = False; row["reason"] = f"no reducer profile for {lang}"
                elif any(k != FUNCTION for k in kinds):
                    row["eligible"] = False; row["reason"] = "not a function triple: " + "/".join(kinds)
                elif fb == fa:
                    row["eligible"] = False; row["reason"] = "no change in the source function"
                else:
                    if lang not in reducers:
                        reducerConfig.set_language(lang); reducers.clear(); reducers[lang] = Reducer(tolerant=True)
                    t0 = time.time()
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            o = reducers[lang].do_reducing(fb, fa, ft)
                    except Exception as e:  # recorded, never dropped
                        row["eligible"] = False; row["reason"] = f"reducer error: {type(e).__name__}: {e}"[:300]
                    else:
                        p = prompt(a.src, a.tgt, o["func_before_reduced"], o["func_after_reduced"], o["func_mapped_reduced"])
                        row.update(eligible=True, reason=None, reduce_seconds=round(time.time() - t0, 3),
                                   placeholders=len(o["removed_pieces_mapped"]),
                                   target_chars=len(ft), target_reduced_chars=len(o["func_mapped_reduced"]),
                                   prompt=p, prompt_chars=len(p),
                                   prompt_sha256=hashlib.sha256(p.encode()).hexdigest(),
                                   removed_pieces_target={str(k): v for k, v in o["removed_pieces_mapped"].items()},
                                   target_original=ft)
                out.write(json.dumps(row) + "\n"); out.flush()
                if direct and row.get("eligible"):
                    # Direct LLM: the same triple and template, nothing removed
                    direct.write(json.dumps({**row, "treatment": "direct", "placeholders": 0, "target_reduced_chars": len(ft),
                                             "prompt": raw, "prompt_chars": len(raw),
                                             "prompt_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                                             "removed_pieces_target": {}}) + "\n")


if __name__ == "__main__":
    main()
