"""Check the port against PPatHF's published reduced test set: run the reducer in C mode on the 310 Vim->Neovim
cases and compare every reduced function and removed piece with PPatHF's own output.
    python check_c_equivalence.py <PPatHF data dir> [N: only the first N cases, file order]"""
import json, sys, collections, warnings
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import reducerConfig
reducerConfig.set_language("c")
from reducer import Reducer

raw = json.load(open(sys.argv[1] + "/vim_neovim_test_all.json"))
raw = raw[:int(sys.argv[2])] if len(sys.argv) > 2 else raw
pub = {(r["commit_id_source"], r["commit_id_target"], r["func_before_source"][:200]): r
       for r in json.load(open(sys.argv[1] + "/sliced/vim_neovim_test_sliced_all.json"))}
red = Reducer(); _dump = open(__import__('os').environ['DUMP'], 'w') if __import__('os').environ.get('DUMP') else None; c = collections.Counter(); diffs = []
for r in raw:
    p = pub.get((r["commit_id_source"], r["commit_id_target"], r["func_before_source"][:200]))
    if p is None: c["no published row"] += 1; continue
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = red.do_reducing(r["func_before_source"], r["func_after_source"], r["func_before_target"])
    except Exception as e:
        c["error"] += 1; diffs.append((r["commit_id_target"], "error", str(e)[:100])); continue
    if _dump: _dump.write(__import__('json').dumps([r['commit_id_target'], o['func_before_reduced'], o['func_after_reduced'], o['func_mapped_reduced'], {str(k): v for k, v in o['removed_pieces_before'].items()}, {str(k): v for k, v in o['removed_pieces_mapped'].items()}]) + '\n'); _dump.flush()
    same = (o["func_before_reduced"] == p["func_before_sliced_source"] and o["func_after_reduced"] == p["func_after_sliced_source"]
            and o["func_mapped_reduced"] == p["func_before_sliced_target"])
    pieces = ({str(k): v for k, v in o["removed_pieces_before"].items()} == {str(k): v for k, v in p["removed_pieces_sliced_source"].items()})
    c["identical" if same and pieces else "different"] += 1
    if not (same and pieces): diffs.append((r["commit_id_target"], len(p["func_before_sliced_target"]), len(o["func_mapped_reduced"])))
    c["published reduced anything"] += p["func_before_sliced_source"] != r["func_before_source"]
print(dict(c), flush=True); print(diffs[:10])
