"""Static name index for the no-build symbol check (symbol_check.py).

Everything here is name-level: a Java/Scala lexer (comments and literals removed), declaration extraction
from a token stream (types, methods with arity, fields, enum constants, Scala def/val/var/class/object/trait),
an index of a fork at one revision (read from a bare git clone with `git cat-file --batch`, cached as JSON),
an index of the JDK's public API (class files of the local JDK's jmods, cached as JSON), and the names a PR
adds (its changed files base..head in the source clone, plus the SAP units' source.before -> source.after).

No type resolution: a name "resolves" when a declaration (or, for library names, a use) of that name exists.
Standard library only, so it runs anywhere RCAP runs.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import struct
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

INDEX_VERSION = 5

JAVA_KEYWORDS = frozenset("""
abstract assert boolean break byte case catch char class const continue default do double else enum extends final
finally float for goto if implements import instanceof int interface long native new package private protected public
return short static strictfp super switch synchronized this throw throws transient try void volatile while true false
null var record yield sealed permits non-sealed
""".split())
SCALA_KEYWORDS = frozenset("""
abstract case catch class def do else extends false final finally for forSome if implicit import lazy match new null
object override package private protected return sealed super this throw trait try true type val var while with yield
""".split())
PRIMITIVES = frozenset("boolean byte char double float int long short void var".split())
MODIFIERS = frozenset("public private protected static final abstract native synchronized transient volatile strictfp "
                      "default sealed".split())

_TOKEN = re.compile(r'''
    (?P<ws>\s+)
  | (?P<lc>//[^\n]*)
  | (?P<bc>/\*.*?(?:\*/|\Z))
  | (?P<tb>"""(?:.|\n)*?(?:"""|\Z))
  | (?P<str>"(?:\\.|[^"\\\n])*"?)
  | (?P<chr>'(?:\\.|[^'\\\n])*'?)
  | (?P<num>\d[\w.]*|\.\d[\w]*)
  | (?P<id>[A-Za-z_$][\w$]*)
  | (?P<op>->|::|\.\.\.|[{}()\[\];,.<>@=?:!~+\-*/&|^%#])
  | (?P<other>.)
''', re.VERBOSE | re.DOTALL)


class Tok:
    __slots__ = ("kind", "text", "pos")

    def __init__(self, kind, text, pos):
        self.kind, self.text, self.pos = kind, text, pos

    def __repr__(self):
        return f"{self.kind}:{self.text}"


def lex(src: str) -> list[Tok]:
    """Identifiers, operators, literals ('str'/'num' kinds, text dropped); comments and whitespace removed."""
    out = []
    for m in _TOKEN.finditer(src):
        k = m.lastgroup
        if k in ("ws", "lc", "bc"):
            continue
        if k in ("tb", "str", "chr"):
            out.append(Tok("str", '""', m.start()))
        elif k == "num":
            out.append(Tok("num", "0", m.start()))
        elif k == "other":
            continue
        else:
            out.append(Tok(k, m.group(), m.start()))
    return out


def match_close(toks, i, open_="(", close=")"):
    """Index of the token closing toks[i] (an opener), or len(toks)."""
    d = 0
    for j in range(i, len(toks)):
        t = toks[j].text
        if toks[j].kind != "op":
            continue
        if t == open_:
            d += 1
        elif t == close:
            d -= 1
            if d == 0:
                return j
    return len(toks)


_GENERIC_OK = frozenset([",", ".", "?", "[", "]", "<", ">", "&", "@"])


def generic_close(toks, i):
    """toks[i] is '<': index of its matching '>' when the group looks like type arguments (only identifiers,
    ',', '.', '?', '[]', '&', extends/super inside), else None (a comparison)."""
    d = 0
    for j in range(i, min(len(toks), i + 200)):
        t = toks[j]
        if t.kind == "op":
            if t.text == "<":
                d += 1
            elif t.text == ">":
                d -= 1
                if d == 0:
                    return j
            elif t.text not in _GENERIC_OK:
                return None
        elif t.kind != "id":
            return None
    return None


def count_args(toks, i):
    """toks[i] is '(' of a call or declaration: (number of top-level arguments, has varargs '...')."""
    j = i + 1
    if j < len(toks) and toks[j].text == ")":
        return 0, False
    n, d, varargs = 1, 0, False
    while j < len(toks):
        t = toks[j]
        if t.kind == "op":
            if t.text in "([{":
                d += 1
            elif t.text in ")]}":
                if d == 0:
                    break
                d -= 1
            elif t.text == "<" and j > 0 and toks[j - 1].kind == "id":
                g = generic_close(toks, j)
                if g is not None:
                    j = g + 1
                    continue
            elif t.text == "," and d == 0:
                n += 1
            elif t.text == "..." and d == 0:
                varargs = True
        j += 1
    return n, varargs


def is_type_end(toks, k):
    """Can toks[k] be the last token of a type (so an identifier after it is being declared)?"""
    if k < 0:
        return False
    t = toks[k]
    if t.kind == "id":
        return t.text not in JAVA_KEYWORDS or t.text in PRIMITIVES
    return t.kind == "op" and t.text in (">", "]", "...")


def lombok_accessors(fields, has_lombok):
    if not has_lombok:
        return {}
    out = {}
    for f in fields:
        cap = f[:1].upper() + f[1:]
        out.setdefault("get" + cap, set()).add(0)
        out.setdefault("is" + cap, set()).add(0)
        out.setdefault("set" + cap, set()).add(1)
        out.setdefault(f, set()).add(0)        # @Accessors(fluent = true) / builder methods
        out.setdefault(f, set()).add(1)
    return out


def java_decls(toks):
    """Declarations of one Java compilation unit (or fragment).

    Returns dict: types (set), methods {name: set(arity; -(n) = varargs with n params)}, fields (member-level
    fields and enum constants), locals (parameters and local variables: every declared name that is not a member),
    ctors {Type: set(arity)}, extends {Type: [supertypes]}, static_imports (names), static_star (types),
    imports (simple names), pkg."""
    types, fields, locals_ = set(), set(), set()
    methods, ctors = defaultdict(set), defaultdict(set)
    extends = defaultdict(list)
    static_imports, static_star, imports = set(), set(), set()
    pkg = None
    stack = []           # brace kinds: ("class", name, is_enum) or ("block",)
    pending = None       # (name, is_enum) of a class header waiting for its '{'
    n = len(toks)
    i = 0
    cur_class = None
    while i < n:
        t = toks[i]
        tx = t.text
        if t.kind == "id" and tx in ("import", "package") and (i == 0 or toks[i - 1].text in (";", "}", "{")):
            j = i + 1
            static = j < n and toks[j].text == "static"
            if static:
                j += 1
            parts = []
            while j < n and toks[j].text != ";":
                if toks[j].kind == "id":
                    parts.append(toks[j].text)
                elif toks[j].text == "*":
                    parts.append("*")
                j += 1
            if tx == "package":
                pkg = ".".join(parts)
            elif parts:
                if static:
                    if parts[-1] == "*":
                        if len(parts) > 1:
                            static_star.add(parts[-2])
                    else:
                        static_imports.add(parts[-1])
                        if len(parts) > 1:
                            imports.add(parts[-2])
                elif parts[-1] != "*":
                    imports.add(parts[-1])
            i = j + 1
            continue
        if t.kind == "op":
            if tx == "{":
                if pending is not None:
                    stack.append(["class", pending[0], pending[1], pending[1]])   # [3]: enum constants still to come
                    pending = None
                elif i > 0 and toks[i - 1].text == ")" and _anon_class(toks, i - 1):
                    stack.append(["class", None, False, False])
                else:
                    stack.append(["block"])
                i += 1
                continue
            if tx == "}":
                if stack:
                    stack.pop()
                i += 1
                continue
            if tx == ";" and stack and stack[-1][0] == "class":
                stack[-1][3] = False
            if tx == "@" and i + 1 < n and toks[i + 1].text == "interface":
                i += 1
                continue
        in_class = bool(stack) and stack[-1][0] == "class"
        if t.kind == "id":
            if tx in ("class", "interface", "enum", "record") and i + 1 < n and toks[i + 1].kind == "id" \
                    and not (i > 0 and toks[i - 1].text == "."):
                name = toks[i + 1].text
                types.add(name)
                pending = (name, tx == "enum")
                cur_class = name
                # supertypes up to '{'
                j = i + 2
                if j < n and toks[j].text == "<":
                    g = generic_close(toks, j)
                    j = (g + 1) if g is not None else j
                if tx == "record" and j < n and toks[j].text == "(":
                    close = match_close(toks, j)
                    comps = _param_names(toks, j, close)
                    fields.update(comps)
                    for c in comps:
                        methods[c].add(0)
                    ctors[name].add(len(comps))
                    j = close + 1
                sup = []
                while j < n and toks[j].text not in ("{", ";"):
                    if toks[j].text in ("extends", "implements") or toks[j].text == ",":
                        k = j + 1
                        q = []
                        while k < n and (toks[k].kind == "id" and toks[k].text not in JAVA_KEYWORDS
                                         or toks[k].text == "."):
                            if toks[k].kind == "id":
                                q.append(toks[k].text)
                            k += 1
                        if q:
                            sup.append(q[-1])
                        j = k
                        if j < n and toks[j].text == "<":
                            g = generic_close(toks, j)
                            j = (g + 1) if g is not None else j + 1
                        continue
                    j += 1
                if sup:
                    extends[name].extend(sup)
                i += 2
                continue
            nxt = toks[i + 1].text if i + 1 < n else ""
            prev_dot = i > 0 and toks[i - 1].text in (".", "::")
            if tx not in JAVA_KEYWORDS and not prev_dot:
                # enum constants
                if in_class and stack[-1][3] and (i == 0 or toks[i - 1].text in ("{", ",", "}")
                                                               or toks[i - 1].kind == "id" and toks[i - 2].text == "@"
                                                               or toks[i - 1].text == ")") \
                        and nxt in ("(", ",", ";", "{", "}"):
                    fields.add(tx)
                    i = match_close(toks, i + 1) + 1 if nxt == "(" else i + 1
                    if i < n and toks[i].text == "{":          # constant with a class body
                        stack.append(["class", None, False, False])
                        i += 1
                    continue
                if nxt == "(":
                    close = match_close(toks, i + 1)
                    after = toks[close + 1].text if close + 1 < n else ""
                    if after in ("{", "throws", ";", "default") and \
                            (is_type_end(toks, i - 1) or _is_ctor_pos(toks, i, stack)):
                        nargs, va = count_args(toks, i + 1)
                        ar = -nargs if va else nargs
                        is_ctor = not is_type_end(toks, i - 1) or (stack and stack[-1][0] == "class"
                                                                   and stack[-1][1] == tx)
                        if is_ctor and tx[:1].isupper():
                            ctors[tx].add(ar)
                        else:
                            methods[tx].add(ar)
                        locals_.update(_param_names(toks, i + 1, close))
                        i = close + 1
                        continue
                elif is_type_end(toks, i - 1) and nxt in ("=", ";", ",", ")", ":") and \
                        not (toks[i - 1].kind == "op" and toks[i - 1].text == ">" and nxt == ")"):
                    if in_class and nxt != ")" and nxt != ":":
                        fields.add(tx)
                        if nxt in ("=", ","):
                            for extra in _more_declarators(toks, i + 1):
                                fields.add(extra)
                    else:
                        locals_.add(tx)
                elif nxt == "->" and i > 0 and toks[i - 1].text != "case":
                    locals_.add(tx)
        i += 1
    # lambda parameter lists "(a, b) ->"
    for k in range(n - 1):
        if toks[k].text == ")" and toks[k + 1].text == "->":
            d = 0
            for j in range(k, -1, -1):
                if toks[j].text == ")":
                    d += 1
                elif toks[j].text == "(":
                    d -= 1
                    if d == 0:
                        for q in range(j + 1, k):
                            if toks[q].kind == "id" and toks[q + 1].text in (",", ")"):
                                locals_.add(toks[q].text)
                        break
    return {"types": types, "methods": methods, "fields": fields, "locals": locals_, "ctors": ctors,
            "extends": extends, "static_imports": static_imports, "static_star": static_star,
            "imports": imports, "pkg": pkg}


def _anon_class(toks, close_idx):
    """toks[close_idx] is ')' followed by '{': is it 'new X(...) {' ?"""
    d = 0
    for j in range(close_idx, -1, -1):
        tx = toks[j].text
        if tx == ")":
            d += 1
        elif tx == "(":
            d -= 1
            if d == 0:
                k = j - 1
                if k >= 0 and toks[k].text == ">":       # new X<..>(
                    dd = 0
                    while k >= 0:
                        if toks[k].text == ">":
                            dd += 1
                        elif toks[k].text == "<":
                            dd -= 1
                            if dd == 0:
                                break
                        k -= 1
                    k -= 1
                while k >= 1 and toks[k].kind == "id" and toks[k - 1].text == ".":
                    k -= 2
                return k >= 1 and toks[k].kind == "id" and toks[k - 1].text == "new"
    return False


def _is_ctor_pos(toks, i, stack):
    """Name( at member level after modifiers / annotations / '}' / ';' / '{' : a constructor."""
    if not (stack and stack[-1][0] == "class"):
        return False
    k = i - 1
    if k < 0:
        return False
    p = toks[k]
    if p.kind == "id" and p.text in ("public", "private", "protected"):
        return True
    if p.kind == "op" and p.text in ("}", ";", "{"):
        return True
    if p.kind == "op" and p.text == ")":       # annotation with arguments
        return True
    if p.kind == "id" and k >= 1 and toks[k - 1].text == "@":
        return True
    return False


def _param_names(toks, open_idx, close_idx):
    names = []
    d = 0
    for j in range(open_idx + 1, close_idx):
        tx = toks[j].text
        if tx in ("(", "[", "{", "<"):
            d += 1
        elif tx in (")", "]", "}", ">"):
            d -= 1
        elif toks[j].kind == "id" and d == 0 and toks[j + 1].text in (",", ")") and is_type_end(toks, j - 1) \
                and j + 1 <= close_idx:
            names.append(tx)
    return names


def _more_declarators(toks, j):
    """After 'Type a' with next token at j being '=' or ',': names of further declarators 'b' in 'Type a = x, b'."""
    out, d, n = [], 0, len(toks)
    while j < n:
        tx = toks[j].text
        if tx in ("(", "[", "{"):
            d += 1
        elif tx in (")", "]", "}"):
            if d == 0:
                break
            d -= 1
        elif tx == "<" and toks[j - 1].kind == "id":
            g = generic_close(toks, j)
            if g is not None:
                j = g + 1
                continue
        elif tx == ";" and d == 0:
            break
        elif tx == "," and d == 0 and j + 2 < n and toks[j + 1].kind == "id" and toks[j + 2].text in ("=", ",", ";"):
            out.append(toks[j + 1].text)
        j += 1
    return out


def scala_decls(toks):
    """Scala: def/val/var names (methods/fields), class/object/trait/type names, class parameters (fields).
    Arity of a Scala def is recorded as None (any): default arguments and parameterless defs."""
    types, fields, methods, ctors = set(), set(), defaultdict(set), defaultdict(set)
    imports, static_imports = set(), set()
    n = len(toks)
    for i, t in enumerate(toks):
        if t.kind != "id" or i + 1 >= n or toks[i + 1].kind != "id":
            if t.kind == "id" and t.text == "import":
                j = i + 1
                parts = []
                while j < n and (toks[j].kind == "id" or toks[j].text in (".", "{", ",", "}", "_")) \
                        and not (toks[j].kind == "id" and toks[j].text in SCALA_KEYWORDS):
                    if toks[j].kind == "id":
                        parts.append(toks[j].text)
                    if toks[j].text == "}":
                        break
                    j += 1
                imports.update(parts)
                static_imports.update(parts)
            continue
        name = toks[i + 1].text
        if t.text in ("class", "object", "trait", "type"):
            types.add(name)
            if t.text == "class" and i + 2 < n and toks[i + 2].text in ("(", "["):
                j = i + 2
                if toks[j].text == "[":
                    j = match_close(toks, j, "[", "]") + 1
                if j < n and toks[j].text == "(":
                    close = match_close(toks, j)
                    params = [toks[q].text for q in range(j + 1, close)
                              if toks[q].kind == "id" and q + 1 < n and toks[q + 1].text == ":" and toks[q - 1].text != ":"]
                    fields.update(params)
                    for p in params:
                        methods[p].add(None)
                    ctors[name].add(None)
        elif t.text == "def":
            methods[name].add(None)
        elif t.text in ("val", "var"):
            fields.add(name)
            methods[name].add(None)
    return {"types": types, "methods": methods, "fields": fields, "locals": set(), "ctors": ctors,
            "extends": defaultdict(list), "static_imports": static_imports, "static_star": set(),
            "imports": imports, "pkg": None}


def uses(toks):
    """Names used (not declared) in a token stream: called (followed by '('), member (after '.' / '::', not a call),
    upper (capitalized identifiers), ids (every identifier)."""
    called, member, upper, ids = set(), set(), set(), set()
    n = len(toks)
    for i, t in enumerate(toks):
        if t.kind != "id":
            continue
        tx = t.text
        ids.add(tx)
        if tx[:1].isupper():
            upper.add(tx)
        nxt = toks[i + 1].text if i + 1 < n else ""
        if nxt == "(":
            called.add(tx)
        elif i > 0 and toks[i - 1].text in (".", "::"):
            member.add(tx)
            if toks[i - 1].text == "::":
                called.add(tx)
    return called, member, upper, ids


# ---------------------------------------------------------------------------------------------- fork index

def _git(git_dir, *args, input_=None):
    return subprocess.run(["git", "--git-dir", str(git_dir), *args], capture_output=True, input=input_,
                          check=True).stdout


def read_blobs(git_dir, rev, paths):
    """{path: text} of the given paths at rev (missing paths left out), one git cat-file --batch call."""
    if not paths:
        return {}
    req = "".join(f"{rev}:{p}\n" for p in paths).encode()
    out = _git(git_dir, "cat-file", "--batch", input_=req)
    res, pos = {}, 0
    for p in paths:
        nl = out.index(b"\n", pos)
        header = out[pos:nl].decode(errors="replace")
        pos = nl + 1
        if header.endswith("missing"):
            continue
        size = int(header.split()[-1])
        res[p] = out[pos:pos + size].decode("utf-8", errors="replace")
        pos += size + 1
    return res


def source_paths(git_dir, rev):
    out = _git(git_dir, "ls-tree", "-r", "--name-only", rev).decode(errors="replace").splitlines()
    return [p for p in out if p.endswith((".java", ".scala"))]


def file_record(path, text):
    toks = lex(text)
    d = scala_decls(toks) if path.endswith(".scala") else java_decls(toks)
    called, member, upper, _ = uses(toks)
    lombok = "lombok" in d["imports"] or any(x in text for x in ("@Getter", "@Setter", "@Data", "@Value"))
    has_lombok = path.endswith(".java") and lombok and ("import lombok" in text)
    for name, ars in lombok_accessors(d["fields"], has_lombok).items():
        d["methods"][name] |= ars
    return {"pkg": d["pkg"], "scala": path.endswith(".scala"),
            "types": sorted(d["types"]), "fields": sorted(d["fields"]),
            "methods": {k: sorted(v, key=lambda x: (x is None, x)) for k, v in d["methods"].items()},
            "ctors": {k: sorted(v, key=lambda x: (x is None, x)) for k, v in d["ctors"].items()},
            "extends": {k: v for k, v in d["extends"].items()},
            "static_imports": sorted(d["static_imports"]), "static_star": sorted(d["static_star"]),
            "imports": sorted(d["imports"]),
            "called": sorted(called), "member": sorted(member), "upper": sorted(upper)}


def build_repo_index(git_dir, rev):
    paths = source_paths(git_dir, rev)
    files = {}
    for k in range(0, len(paths), 2000):
        for p, text in read_blobs(git_dir, rev, paths[k:k + 2000]).items():
            files[p] = file_record(p, text)
    return {"version": INDEX_VERSION, "git_dir": str(git_dir), "rev": rev, "files": files}


def load_or_build(cache_file: Path, builder):
    if cache_file.exists():
        with gzip.open(cache_file, "rt") as f:
            d = json.load(f)
        if d.get("version") == INDEX_VERSION:
            return d
    d = builder()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(d, f)
    os.replace(tmp, cache_file)
    return d


class NameIndex:
    """Aggregated name sets over many file records (a fork, the JDK, a PR's additions)."""

    def __init__(self, files: dict | None = None):
        self.files = files or {}
        self.types, self.fields, self.called, self.member, self.upper = set(), set(), set(), set(), set()
        self.static_imports = set()
        self.methods = defaultdict(set)
        self.ctors = defaultdict(set)
        self.type_files = defaultdict(list)
        self.known_external = {"Object"}       # supertypes that are not "library": set to the JDK's types
        self.extends = defaultdict(set)
        for p, r in self.files.items():
            self.add_record(r, p)

    def add_record(self, r, path=None):
        self.types.update(r["types"])
        self.fields.update(r["fields"])
        self.called.update(r.get("called", ()))
        self.member.update(r.get("member", ()))
        self.upper.update(r.get("upper", ()))
        self.static_imports.update(r.get("static_imports", ()))
        for k, v in r["methods"].items():
            self.methods[k].update(v)
        for k, v in r.get("ctors", {}).items():
            self.ctors[k].update(v)
        for k, v in r.get("extends", {}).items():
            self.extends[k].update(v)
        if path is not None:
            for t in r["types"]:
                self.type_files[t].append(path)

    def has_external_super(self, tname, seen=None, depth=0):
        """Does the fork type (or a fork supertype) extend / implement a known external (JDK) type other than
        Object? Then bare calls may reach that type's methods."""
        seen = seen if seen is not None else set()
        if tname in seen or depth > 8:
            return False
        seen.add(tname)
        for p in self.type_files.get(tname, []):
            for sup in self.files[p].get("extends", {}).get(tname, []):
                if sup not in self.type_files:
                    if sup in self.known_external and sup != "Object":
                        return True
                elif self.has_external_super(sup, seen, depth + 1):
                    return True
        return False

    def members_of_type(self, tname, seen=None, depth=0):
        """(fields, methods, library_super) of a fork type by simple name and its fork supertypes;
        library_super is True when some supertype is not declared in the index (a library class)."""
        seen = seen if seen is not None else set()
        fields, methods, lib = set(), set(), False
        if tname in seen or depth > 8:
            return fields, methods, lib
        seen.add(tname)
        paths = self.type_files.get(tname, [])
        if not paths:
            # a supertype the fork does not declare: a library class (its members unknown), unless the JDK has it
            # (JDK members are checked by name anyway)
            return fields, methods, tname not in self.known_external
        for p in paths:
            r = self.files[p]
            fields.update(r["fields"])
            methods.update(r["methods"])
            for sup in r.get("extends", {}).get(tname, []):
                f2, m2, l2 = self.members_of_type(sup, seen, depth + 1)
                fields |= f2
                methods |= m2
                lib = lib or l2
        return fields, methods, lib


# ---------------------------------------------------------------------------------------------- JDK index

def _class_members(data):
    """(this class simple name, is public, outer-qualified name, [(field, )], [(method, arity)]) of a class file;
    None for non-public classes."""
    pos = 10
    count = struct.unpack(">H", data[8:10])[0]
    cp = [None] * count
    i = 1
    while i < count:
        tag = data[pos]
        if tag == 1:
            ln = struct.unpack(">H", data[pos + 1:pos + 3])[0]
            cp[i] = data[pos + 3:pos + 3 + ln].decode("utf-8", errors="replace")
            pos += 3 + ln
        elif tag in (3, 4):
            pos += 5
        elif tag in (5, 6):
            pos += 9
            i += 1
        elif tag == 7 or tag == 8 or tag == 16 or tag == 19 or tag == 20:
            cp[i] = ("ref", struct.unpack(">H", data[pos + 1:pos + 3])[0])
            pos += 3
        elif tag in (9, 10, 11, 12, 17, 18):
            pos += 5
        elif tag == 15:
            pos += 4
        else:
            raise ValueError(f"bad constant pool tag {tag}")
        i += 1
    access, this_cls = struct.unpack(">HH", data[pos:pos + 4])
    pos += 6
    name = cp[cp[this_cls][1]]
    icount = struct.unpack(">H", data[pos:pos + 2])[0]
    pos += 2 + 2 * icount

    def members(pos):
        cnt = struct.unpack(">H", data[pos:pos + 2])[0]
        pos += 2
        out = []
        for _ in range(cnt):
            acc, ni, di, ac = struct.unpack(">HHHH", data[pos:pos + 8])
            pos += 8
            for _ in range(ac):
                ln = struct.unpack(">I", data[pos + 2:pos + 6])[0]
                pos += 6 + ln
            if acc & 0x0005:      # public or protected
                out.append((cp[ni], cp[di], acc))
        return out, pos

    flds, pos = members(pos)
    mths, pos = members(pos)
    return name, bool(access & 0x0001), flds, mths


def _desc_arity(desc):
    s = desc[1:desc.index(")")]
    n, i = 0, 0
    while i < len(s):
        c = s[i]
        if c == "[":
            i += 1
            continue
        if c == "L":
            i = s.index(";", i) + 1
        else:
            i += 1
        n += 1
    return n


JDK_PREFIXES = ("java/", "javax/", "com/sun/source/", "com/sun/net/httpserver/", "org/w3c/", "org/xml/", "jdk/jfr/",
                "sun/misc/", "com/sun/management/", "org/ietf/")


def build_jdk_index(jmods_dir):
    types, fields, methods, ctors = set(), set(), defaultdict(set), defaultdict(set)
    for jm in sorted(Path(jmods_dir).glob("*.jmod")):
        with zipfile.ZipFile(jm) as z:
            for nm in z.namelist():
                if not nm.startswith("classes/") or not nm.endswith(".class"):
                    continue
                rel = nm[len("classes/"):]
                if not rel.startswith(JDK_PREFIXES) or rel.endswith("module-info.class"):
                    continue
                try:
                    cname, public, flds, mths = _class_members(z.read(nm))
                except Exception:
                    continue
                simple = cname.rsplit("/", 1)[-1]
                parts = simple.split("$")
                if any(p[:1].isdigit() for p in parts):
                    continue
                types.add(parts[-1])
                for f, _, _ in flds:
                    fields.add(f)
                for m, desc, acc in mths:
                    if m == "<clinit>":
                        continue
                    ar = _desc_arity(desc)
                    if acc & 0x0080:
                        ar = -ar        # varargs
                    if m == "<init>":
                        ctors[parts[-1]].add(ar)
                    else:
                        methods[m].add(ar)
    rec = {"types": sorted(types), "fields": sorted(fields),
           "methods": {k: sorted(v) for k, v in methods.items()},
           "ctors": {k: sorted(v) for k, v in ctors.items()}}
    return {"version": INDEX_VERSION, "jmods": str(jmods_dir), "files": {"<jdk>": rec}}


def default_jmods():
    for cand in [os.environ.get("JAVA_HOME"), "/usr/lib/jvm/java-11-openjdk-amd64", "/usr/lib/jvm/default-java"]:
        if cand and (Path(cand) / "jmods").is_dir():
            return Path(cand) / "jmods"
    for p in sorted(Path("/usr/lib/jvm").glob("*/jmods")) if Path("/usr/lib/jvm").is_dir() else []:
        return p
    return None


# ---------------------------------------------------------------------------------------------- PR additions

def decl_record(text, scala=False):
    toks = lex(text)
    return scala_decls(toks) if scala else java_decls(toks)


def added_names_per_file(groups):
    """groups: {key: (before_texts, after_texts)} (one changed file, or one SAP = one file): the union over groups
    of what each group adds. Per file, so a name another file already declared still counts as added here."""
    out = None
    for _, (b, a) in sorted(groups.items()):
        out = merge_added(out, added_names(b, a)) if out else added_names(b, a)
    return out or {"types": [], "fields": [], "methods": {}, "ctors": {}}


def added_names(before_texts, after_texts):
    """Declarations present after but not before (types, fields, methods with arities, ctors)."""
    def agg(texts):
        types, fields, methods, ctors = set(), set(), defaultdict(set), defaultdict(set)
        for path, text in texts:
            d = decl_record(text, scala=path.endswith(".scala"))
            types |= d["types"]
            fields |= d["fields"]
            for k, v in d["methods"].items():
                methods[k] |= v
            for k, v in d["ctors"].items():
                ctors[k] |= v
            # a Java fragment (SAP unit) holding only members is not a class body: count its locals-at-top as
            # fields too (a field unit "private final Foo bar;" lexes as a local at depth 0)
            if not path.endswith(".scala"):
                fields |= _top_level_fields(text)
        return types, fields, methods, ctors

    bt, bf, bm, bc = agg(before_texts)
    at, af, am, ac = agg(after_texts)
    methods = {k: sorted((v - bm.get(k, set())) or v, key=lambda x: (x is None, x)) for k, v in am.items()
               if k not in bm or (v - bm[k])}
    ctors = {k: sorted(v, key=lambda x: (x is None, x)) for k, v in ac.items() if k not in bc or (v - bc[k])}
    return {"types": sorted(at - bt), "fields": sorted(af - bf), "methods": methods, "ctors": ctors}


def _top_level_fields(text):
    """Fields of a member-only fragment: wrap it in a class body and read the members."""
    d = java_decls(lex("class __Frag__ {\n" + text + "\n}"))
    return d["fields"]


def merge_added(a, b):
    if not b:
        return a
    out = {"types": sorted(set(a["types"]) | set(b["types"])), "fields": sorted(set(a["fields"]) | set(b["fields"]))}
    for k in ("methods", "ctors"):
        m = {}
        for src in (a[k], b[k]):
            for name, ars in src.items():
                m.setdefault(name, set()).update(ars)
        out[k] = {name: sorted(v, key=lambda x: (x is None, x)) for name, v in m.items()}
    return out
