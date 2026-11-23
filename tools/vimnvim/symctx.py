"""What a name the compiler misses IS: the C counterpart of rcap.adaptctx.

For every name gcc reports (gccdiag.missing_names) one rcap.reprompt.SymbolContext,
which rcap.request renders with the v3 repair template unchanged:

* the Vim side, from the Vim repository at the case's own source commit: the
  definition of the name (a function with its body when it is short, else its
  declaration; a macro's `#define`; otherwise the first header line naming
  it), and whether the SAME commit adds it (absent from its parent) or it is
  older;
* the Neovim side: the fork's own declaration when the name exists there in
  another form (other parameters, another file), else the fork's closest
  declarations: the equivalents learned from earlier Vim -> Neovim ports
  (ppathf-memory/symbol_map.json, built from ports before the benchmark's
  cut-off and keyed on the Vim name) and the like-named prototypes.

Nothing of the developer's port is read: neither `func_after_target` nor the
target commit. The fork-side declarations come from the compile tree, which is
ONE later Neovim commit for all cases (the same tree the diagnostics come
from); that is a limitation of the benchmark set-up and is reported as such.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from rcap.feedback import rank_members
from rcap.reprompt import SymbolContext

from gccdiag import Missing

MAX_BODY_LINES = 25          # as rcap.adaptctx
MAX_SIGNATURE_CHARS = 400
MAX_SYMBOLS = 6
MAX_FORK_MEMBERS = 6
_WS = re.compile(r"\s+")
_PROTO = re.compile(r"^(?:static |DLLEXPORT |extern )*[A-Za-z_][\w \t\*]*?\b(?P<name>[A-Za-z_]\w*)\((?P<args>[^;{}]*)\)[^;{}]*;\s*$")
_ATTR = re.compile(r"\s+(?:FUNC_ATTR_\w+(?:\([^)]*\))?|REAL_FATTR_\w+(?:\([^)]*\))?)")


def _git(repo: Path, *args: str) -> str:
    cmd = ["git", "--git-dir", str(repo)] if str(repo).endswith(".git") else ["git", "-C", str(repo)]
    return subprocess.run(cmd + list(args), capture_output=True, text=True, errors="replace").stdout


class ForkIndex:
    """Declarations of the fork (the compile tree at HEAD)."""

    def __init__(self, tree: Path):
        self.tree = tree
        self.protos: dict[str, str] = {}
        for f in sorted((tree / "build" / "src" / "nvim" / "auto").rglob("*.generated.h")):
            for line in f.read_text(errors="replace").splitlines():
                m = _PROTO.match(line)
                if m and m.group("name") not in self.protos:
                    self.protos[m.group("name")] = _ATTR.sub("", line.strip())[:MAX_SIGNATURE_CHARS]
        self.members = sorted(self.protos.items())

    def declaration(self, name: str) -> tuple[str, str] | None:
        """(where, text) of the fork's declaration of `name`, or None."""
        if name in self.protos:
            return "a function", self.protos[name]
        hit = _git(self.tree, "grep", "-n", "-E", rf"^\s*#\s*define\s+{re.escape(name)}\b", "HEAD", "--", "src/nvim")
        if hit.strip():
            _, path, _, text = hit.splitlines()[0].split(":", 3)
            return f"a macro in {path}", text.strip()[:MAX_SIGNATURE_CHARS]
        hit = _git(self.tree, "grep", "-n", "-w", name, "HEAD", "--", "src/nvim/*.h")
        if hit.strip():
            _, path, _, text = hit.splitlines()[0].split(":", 3)
            return f"{path}", text.strip()[:MAX_SIGNATURE_CHARS]
        return None


class VimIndex:
    """Definitions in the Vim repository at one commit."""

    def __init__(self, repo: Path):
        self.repo = repo

    def _grep(self, sha: str, pattern: str, *paths: str, word: bool = False) -> list[tuple[str, int, str]]:
        out = _git(self.repo, "grep", "-n", "-w" if word else "-E", pattern, sha, "--", *paths)
        hits = []
        for line in out.splitlines():
            parts = line.split(":", 3)
            if len(parts) == 4:
                hits.append((parts[1], int(parts[2]), parts[3]))
        return hits

    def locate(self, sha: str, name: str) -> tuple[str, str, int, str] | None:
        """(kind, file, line, text) of the definition of `name` at `sha`."""
        esc = re.escape(name)
        hits = self._grep(sha, rf"^{esc}\(", "src/*.c")
        if hits:
            return ("function", *hits[0])
        hits = self._grep(sha, rf"^\s*#\s*define\s+{esc}\b", "src/*.h", "src/*.c")
        if hits:
            return ("macro", *hits[0])
        hits = self._grep(sha, name, "src/*.h", word=True)
        if hits:
            return ("declaration", *hits[0])
        return None

    def function_text(self, sha: str, path: str, line: int) -> tuple[str, str, int]:
        """(definition, signature, lines) of the function whose name stands at `line`."""
        src = _git(self.repo, "show", f"{sha}:{path}").splitlines()
        start = line - 1
        if start > 0 and src[start - 1].strip() and not src[start - 1].rstrip().endswith((";", "}", "*/")):
            start -= 1                       # Vim puts the return type on its own line
        end = next((n for n in range(line - 1, len(src)) if src[n].startswith("}")), len(src) - 1)
        brace = next((n for n in range(line - 1, end + 1) if src[n].startswith("{")), end)
        signature = _WS.sub(" ", " ".join(src[start:brace])).strip()[:MAX_SIGNATURE_CHARS]
        return "\n".join(src[start:end + 1]) + "\n", signature, end - start + 1


class SymbolContexts:
    def __init__(self, cfg: dict, similar_floor: float = 0.6):
        out = Path(cfg["_out"])
        self.similar_floor = similar_floor
        self.fork = ForkIndex(out / cfg["compile"]["tree"])
        self.vim = VimIndex(Path(cfg["stored"]["vim_repo"]))
        self.map = json.loads(Path(cfg["stored"]["symbol_map"]).read_text())["map"]

    def _equivalents(self, name: str) -> tuple[list[str], list[str]]:
        """(fork declarations, hints) for a Vim name the fork lacks."""
        members: list[str] = []
        hints: list[str] = []
        for e in self.map.get(name, [])[:2]:
            decl = self.fork.declaration(e["neovim"])
            if decl:
                members.append(decl[1])
                hints.append(f"In earlier ports of Vim changes, `{name}` was written `{e['neovim']}` in the fork "
                             f"({e['support']} times).")
        for sig in rank_members(name, self.fork.members, limit=4, floor=self.similar_floor):
            if sig not in members:
                members.append(sig)
        return members[:MAX_FORK_MEMBERS], hints

    def describe(self, case: dict, m: Missing) -> SymbolContext:
        sha = case["commit_id_source"]
        why: dict[str, object] = {"gcc_kind": m.kind}
        sc = SymbolContext(symbol=m.name, owner=m.owner if m.kind == "member" else None, usage=m.usage, why=why)
        found = self.vim.locate(sha, m.name)
        if found:
            kind, path, line, text = found
            before = self.vim.locate(f"{sha}~1", m.name)
            sc.upstream_file, sc.upstream_line = path, line
            sc.origin = "predates_change" if before else "added_by_change"
            why.update(vim_kind=kind, vim_before=bool(before))
            if kind == "function":
                body, sig, n = self.vim.function_text(sha, path, line)
                sc.upstream_signature, sc.upstream_lines = sig, n
                sc.upstream_text = body if n <= MAX_BODY_LINES else sig
                sc.upstream_complete = n <= MAX_BODY_LINES
            else:
                sc.upstream_signature = sc.upstream_text = text.strip()[:MAX_SIGNATURE_CHARS]
                sc.upstream_complete = kind == "macro" and not text.rstrip().endswith("\\")
                sc.upstream_lines = 1
        decl = self.fork.declaration(m.name)
        if decl:
            where, text = decl
            why["fork_has_name"] = where
            # the name exists in the fork: it is used differently there, or is not visible here
            if sc.origin == "added_by_change":
                sc.origin = "predates_change"
            if sc.origin == "unknown":
                sc.origin = "predates_change"
            if m.kind == "signature":
                sc.fork_heading = f"The fork declares `{m.name}` with other parameters"
            elif where == "a function" or where.startswith("a macro"):
                sc.fork_heading = (f"The fork declares `{m.name}` ({where}), but not where this file can see it "
                                   "or not in this form")
            else:
                sc.fork_heading = f"The fork mentions `{m.name}` in {where}"
            sc.fork_members = [text]
        else:
            members, hints = self._equivalents(m.name)
            sc.fork_members, sc.hints = members, hints
            sc.fork_heading = (f"The fork does not declare `{m.name}`. Its closest declarations"
                               if members else f"The fork declares nothing named `{m.name}` and nothing close to it.")
        return sc

    def build(self, case: dict, missing: list[Missing]) -> tuple[list[SymbolContext], int]:
        shown = missing[:MAX_SYMBOLS]
        return [self.describe(case, m) for m in shown], len(missing) - len(shown)
