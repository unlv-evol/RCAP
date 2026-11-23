"""The compile check: a candidate function spliced into Neovim's real tree.

After ppathf-memory/compile_check.py (read-only, not imported), with what the
re-prompt loop needs in addition: the full gcc diagnostics instead of the last
400 characters, a cache by content, and the tree taken from git for every
check so that a killed run cannot leave a patched file behind.

One configured build of Neovim exists (the benchmark spans a year of commits),
so every case is gated on a control: THE DEVELOPER'S OWN FUNCTION MUST COMPILE
IN THIS TREE. A case that fails the gate is unreachable and never scored.

Flags: the translation unit's own command from compile_commands.json, made
syntax-only, colours off. `-Werror=implicit-function-declaration` is ADDED
(`strict`, the default): a call of a function the fork does not declare is a
warning under the build's `-std=gnu99` and would only fail at link time, which
a syntax-only check never reaches; without the flag an answer that calls a
Vim-only function "compiles". `strict=False` gives the earlier check's flags.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

from rcap.grammars import C, function_name_node, parse

from gccdiag import parse_gcc


def _find(src: bytes, name: str) -> tuple[int, int] | None:
    tree = parse(src.decode("utf-8", "replace"), C)
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type == "function_definition":
            ident = function_name_node(n)
            if ident is not None and ident.text.decode("utf-8", "replace") == name:
                return n.start_byte, n.end_byte
            continue
        stack.extend(n.children)
    return None


def name_of(code: str) -> str | None:
    tree = parse(code, C)
    stack = [tree.root_node] if tree is not None else []
    while stack:
        n = stack.pop(0)
        if n.type == "function_definition":
            ident = function_name_node(n)
            if ident is not None:
                return ident.text.decode("utf-8", "replace")
        stack.extend(n.children)
    return None


class Compiler:
    def __init__(self, cfg: dict, *, strict: bool = True):
        out = Path(cfg["_out"])
        self.tree = out / cfg["compile"]["tree"]
        self.container = cfg["compile"]["container"]
        self.strict = strict
        self.extra = list(cfg["compile"]["flags"]) + (
            ["-Werror=implicit-function-declaration"] if strict else [])
        self.cache = out / "compile" / ("cache" if strict else "cache-lenient")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = out / "compile" / "tree.lock"
        self.commit = subprocess.run(["git", "-C", str(self.tree), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
        self.cmds: dict[str, dict] = {}
        for e in json.loads((self.tree / "build" / "compile_commands.json").read_text()):
            f = e["file"]
            if f.startswith("/src/"):
                self.cmds.setdefault(f[len("/src/"):], e)
        self._sites: dict[str, dict | None] = {}

    # ------------------------------------------------------------------ sites
    def _head(self, rel: str) -> bytes | None:
        p = subprocess.run(["git", "-C", str(self.tree), "show", f"HEAD:{rel}"], capture_output=True)
        return p.stdout if p.returncode == 0 else None

    def site(self, cid: str, case: dict) -> dict | None:
        """Where the case's function lives in the tree, or None with a reason
        in `self.why[cid]`."""
        if cid in self._sites:
            return self._sites[cid]
        self.why = getattr(self, "why", {})
        rel = case["file_path_target"]
        entry = self.cmds.get(rel)
        original = self._head(rel) if entry else None
        site = None
        if entry is None or original is None:
            self.why[cid] = "no_compile_command"
        else:
            name = name_of(case["func_after_target"]) or name_of(case["func_before_target"])
            span = _find(original, name) if name else None
            if span is None:
                self.why[cid] = "function_not_in_tree"
            else:
                site = {"rel": rel, "entry": entry, "original": original, "span": span, "name": name,
                        "first_line": original[:span[0]].count(b"\n") + 1}
        self._sites[cid] = site
        return site

    # ------------------------------------------------------------------ one check
    def _argv(self, entry: dict) -> list[str]:
        argv = entry.get("arguments") or entry["command"].split()
        out, skip = [], False
        for a in argv:
            if skip:
                skip = False
                continue
            if a in ("-o", "-MF", "-MT", "-MQ"):
                skip = True
                continue
            if a == "-c" or a.startswith("-M") or a.startswith("-fdiagnostics-color"):
                continue
            out.append(a)
        return out + self.extra

    def check(self, cid: str, case: dict, code: str) -> dict:
        """{ok, log, errors, first_line, last_line, rel, seconds, cached} for
        `code` standing where the case's function stands. ok is None when the
        case has no site."""
        site = self.site(cid, case)
        if site is None:
            return {"ok": None, "why": self.why[cid]}
        key = hashlib.sha256(f"{self.commit}\0{site['rel']}\0{site['name']}\0{' '.join(self.extra)}\0{code}".encode()).hexdigest()
        f = self.cache / f"{key}.json"
        if f.exists():
            return {**json.loads(f.read_text()), "cached": True}
        body = code.encode("utf-8")
        patched = site["original"][:site["span"][0]] + body + site["original"][site["span"][1]:]
        path = self.tree / site["rel"]
        entry = site["entry"]
        cmd = ["docker", "exec", "-e", "LC_ALL=C", "-w", entry.get("directory", "/src/build"),
               self.container] + self._argv(entry)
        t0 = time.time()
        with open(self.lock, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                path.write_bytes(patched)
                p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=600)
            finally:
                path.write_bytes(site["original"])
        log = p.stderr or ""
        first = site["first_line"]
        res = {"ok": p.returncode == 0, "returncode": p.returncode, "log": log,
               "errors": len(parse_gcc(log)), "first_line": first,
               "last_line": first + body.rstrip(b"\n").count(b"\n"), "rel": site["rel"],
               "function": site["name"], "seconds": round(time.time() - t0, 2),
               "tree_commit": self.commit, "flags": self.extra}
        f.write_text(json.dumps(res))
        return {**res, "cached": False}

    def reachable(self, cid: str, case: dict) -> tuple[bool, str]:
        """The gate: the developer's function compiles in this tree."""
        r = self.check(cid, case, case["func_after_target"])
        if r["ok"] is None:
            return False, r["why"]
        return (True, "ok") if r["ok"] else (False, "reference_does_not_compile")
