"""Shared build/test helpers for the validation drivers (revalidate.py, order_pr.py).

Same Gradle flags, JDK, exclusions and test-class mapping as validate.py / reprompt_pr.py; kept in
one place so the standalone and the PR-level drivers cannot drift apart. Nothing here writes
outside the fork clone's working tree.
"""
import collections, glob, os, re, subprocess, time, xml.etree.ElementTree as ET
from pathlib import Path

JAVA_HOME = os.environ.get("RCAP_JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")


def screen_unit(target_original, candidate_expanded, upstream_before="", upstream_after="", language=None):
    """Why a stored or fresh candidate must not be written into the fork, as (reason, evidence), or
    None: placeholder_leak (a marker of any spelling), duplicated_block (rcap.accept) or
    shape_mismatch (one function offered for a whole-file unit, rcap.unitkind). One rule for every
    driver, so the standalone, the re-prompt and the PR-level paths cannot drift apart."""
    from rcap.accept import rejection
    from rcap.unitkind import shape_mismatch
    rej = rejection(candidate_expanded, target_original, upstream_before or "", upstream_after or "", language=language)
    if rej: return rej
    why = shape_mismatch(target_original, candidate_expanded, language) if language else None
    return ("shape_mismatch", [why]) if why else None


def unit_sources(sap_dir, entity):
    """(upstream before, upstream after) texts of one unit of a SAP directory."""
    d = Path(sap_dir) / "functions" / (entity or "").split("/")[-1]
    get = lambda side: next((p.read_text(errors="replace") for p in sorted(d.glob(f"source.{side}.*"))), "")  # noqa: E731
    return get("before"), get("after")


def lang_of(path):
    return "scala" if str(path).endswith((".scala", ".sc")) else "java"

# What the harness excludes. checkstyle is OFF, so its UnusedImports check never runs; the fork
# builds Scala 2.12 (gradle.properties scalaVersion=2.12), for which build.gradle does NOT add
# -Xfatal-warnings (it does only when baseScala != 2.12), and javac has no unused-import lint.
# => under these flags an unused import cannot fail the build (see rcap.imports).
EXCLUDES = ["-x", "checkstyleMain", "-x", "checkstyleTest", "-x", "spotbugsMain", "-x", "spotbugsTest",
            "-x", "spotlessScala"]


class Harness:
    def __init__(self, fork, gradle_bin):
        self.fork = Path(fork).resolve(); self.gradle_bin = gradle_bin; self._ls = None

    def sh(self, cmd, timeout, stdin=None):
        t = time.time()
        try:
            r = subprocess.run(cmd, cwd=self.fork, capture_output=True, text=True, timeout=timeout, errors="replace", input=stdin)
            return r.returncode, r.stdout + r.stderr, round(time.time() - t, 1)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"") + (e.stderr or b"")
            return "timeout", out.decode(errors="replace") if isinstance(out, bytes) else out, round(time.time() - t, 1)

    def gradle(self, *tasks, timeout, keep_going=True):
        return self.sh([self.gradle_bin, "--offline", "-q", *(["--continue"] if keep_going else []),
                        f"-Dorg.gradle.java.home={JAVA_HOME}", "-PmaxParallelForks=" + os.environ.get("MAX_FORKS", "3"),
                        *tasks, *EXCLUDES], timeout)

    def build_facts(self):
        """The two facts rcap.imports.unused_import_fails_build needs, read from the build the
        harness actually runs (not assumed)."""
        props = (self.fork / "gradle.properties").read_text(errors="replace")
        m = re.search(r"^scalaVersion=(\S+)", props, re.M)
        scala = m.group(1) if m else "?"
        bg = (self.fork / "build.gradle").read_text(errors="replace")
        fatal_conditional = bool(re.search(r"baseScala != '2\.12'\)\s*\{[^}]*-Xfatal-warnings", bg, re.S))
        fatal = "-Xfatal-warnings" in bg and not (fatal_conditional and scala.startswith("2.12"))
        return {"checkstyle": not any("checkstyle" in x for x in EXCLUDES), "scala_version": scala,
                "scala_fatal_warnings": fatal}

    @staticmethod
    def project_of(path):
        return ":" + path.split("/src/")[0].replace("/", ":") if "/src/" in path else None

    def fqcn(self, path):
        try: src = (self.fork / path).read_text(errors="replace")
        except OSError: src = ""
        pk = [m.group(1) for m in re.finditer(r"^\s*package\s+([\w.]+)\s*;?\s*$", src, re.M)]
        if pk: return ".".join(pk) + "." + Path(path).stem
        m = re.search(r"/src/\w+/(?:java|scala)/(.+)\.(?:java|scala)$", path)
        return m.group(1).replace("/", ".") if m else None

    def concrete(self, rel):
        src = (self.fork / rel).read_text(errors="replace"); name = Path(rel).stem
        if not re.search(rf"abstract\s+class\s+{name}\b", src): return [self.fqcn(rel)]
        root = rel.split("/src/")[0]
        subs = [p for p in glob.glob(f"{self.fork}/{root}/src/test/**/*.*", recursive=True)
                if p.endswith((".java", ".scala")) and re.search(rf"extends\s+{name}\b", Path(p).read_text(errors="replace"))]
        return [self.fqcn(os.path.relpath(p, self.fork)) for p in subs]

    def find_test(self, path):
        """validate.py's rule: (project, [fqcn...], how) of the test classes to run for this file."""
        proj = self.project_of(path)
        if not path.endswith((".java", ".scala")): return proj, [], "not_code"
        if "/src/test/" in path: return proj, self.concrete(path), "self"
        name = Path(path).stem + "Test"; root = path.split("/src/")[0]
        same = [p for p in glob.glob(f"{self.fork}/{root}/src/test/**/{name}.*", recursive=True) if p.endswith((".java", ".scala"))]
        hits = same or [p for p in glob.glob(f"{self.fork}/**/src/test/**/{name}.*", recursive=True) if p.endswith((".java", ".scala"))]
        if len(hits) != 1: return proj, [], f"no_unique_test({len(hits)})"
        rel = os.path.relpath(hits[0], self.fork)
        return self.project_of(rel), self.concrete(rel), "same_project" if same else "other_project"

    def run_test_summary(self, proj, classes, timeout):
        """validate.py's run_test: one summary dict for a set of classes."""
        rdir = self.fork / proj.strip(":").replace(":", "/") / "build/test-results/test"
        for f in glob.glob(f"{rdir}/*.xml"): os.remove(f)
        rc, log, secs = self.gradle(f"{proj}:test", *[x for c in classes for x in ("--tests", c)], timeout=timeout, keep_going=False)
        ran, failed = 0, []
        for f in glob.glob(f"{rdir}/*.xml"):
            for tc in ET.parse(f).getroot().iter("testcase"):
                ran += 1
                if tc.find("failure") is not None or tc.find("error") is not None:
                    failed.append(f"{tc.get('classname')}.{tc.get('name')}")
        return {"rc": rc, "ran": ran, "failed": sorted(failed), "seconds": secs, "log_tail": log[-800:] if rc != 0 else ""}

    # ---------------------------------------------------------------- fork tree lookups
    def files(self):
        if self._ls is None:
            self._ls = collections.defaultdict(list)
            for p in subprocess.run(["git", "ls-files"], cwd=self.fork, capture_output=True, text=True).stdout.splitlines():
                self._ls[Path(p).name].append(p)
        return self._ls

    def tracked(self):
        return {p for ps in self.files().values() for p in ps}

    def grep(self, *args):
        r = subprocess.run(["git", "grep", *args], cwd=self.fork, capture_output=True, text=True, errors="replace")
        return r.stdout.splitlines()

    def has_word(self, name):
        """Does the identifier occur anywhere in the fork's Java/Scala sources (HEAD)?"""
        return bool(self.grep("-l", "-w", "-F", "-e", name, "HEAD", "--", "*.java", "*.scala")[:1])

    def import_exists(self, imp, near):
        """Why the target of an upstream import is available to the fork, or None (rcap.imports)."""
        fq = imp.fqn
        if fq.startswith(("java.", "javax.", "scala.")): return "JDK / Scala library"
        parts = fq.split(".")
        for n in range(len(parts), 1, -1):
            rel = "/".join(parts[:n])
            hits = [p for ext in ("java", "scala") for p in self.files().get(f"{parts[n - 1]}.{ext}", [])
                    if p.endswith(f"/{rel}.{ext}")]
            if not hits: continue
            if n == len(parts): return f"declared in the fork: {hits[0]}"
            src = subprocess.run(["git", "show", f"HEAD:{hits[0]}"], cwd=self.fork, capture_output=True, text=True, errors="replace").stdout
            if all(re.search(rf"\b{re.escape(x)}\b", src) for x in parts[n:]):
                return f"member of a fork type: {hits[0]}"
            return None
        root = near.split("/src/")[0]
        stmt = f"import {'static ' if imp.static else ''}{fq}"
        used = self.grep("-l", "-E", "-e", "^" + re.escape(stmt) + r"\s*;?\s*$", "HEAD", "--", root)
        if used: return f"the same import is used in the fork's {root}: {used[0].split(':', 1)[-1]}"
        pkg, name = fq.rsplit(".", 1)
        decl = self.grep("-l", "-E", "-e", rf"\b(class|object|trait|interface|enum|type)\s+{re.escape(name)}\b", "HEAD", "--",
                         f"*/{pkg.replace('.', '/')}/*.scala", f"*/{pkg.replace('.', '/')}/*.java")
        if decl: return f"declared in the fork: {decl[0].split(':', 1)[-1]}"
        return None


# ---------------------------------------------------------------- Maven forks (the non-Kafka pairs)
# javac through Maven prints "[ERROR] /abs/File.java:[12,5] msg"; rcap.feedback reads the Gradle/javac
# form "/abs/File.java:12:5: error: msg", so build logs are rewritten to that form before parsing.
_MVN_ERR = re.compile(r"^\[ERROR\] (/[^:\n]+\.java):\[(\d+),(\d+)\] (?:error: )?(.*)$", re.M)
_MVN_WARN = re.compile(r"^\[WARNING\] (/[^:\n]+\.java):\[(\d+),(\d+)\] (?:warning: )?(.*)$", re.M)
_MVN_CONT = re.compile(r"^\[ERROR\] (?=\s)", re.M)   # "[ERROR]   symbol: ..." detail lines of the summary copy
MVN_SKIPS = ["-Dcheckstyle.skip", "-Dlicense.skip=true", "-Denforcer.skip=true", "-Dmaven.javadoc.skip=true",
             "-Dspotbugs.skip=true", "-Dpmd.skip=true", "-Dcpd.skip=true", "-Djacoco.skip=true"]


class MavenHarness(Harness):
    """The Harness interface over Maven: a 'project' is a module directory (the part of the path before
    /src/, '.' for the root), built with -am so the modules it depends on compile from source too, as
    Gradle's :proj:testClasses does. Same exclusions in spirit as EXCLUDES: style and analysis checks off."""

    def __init__(self, fork, mvn_bin="mvn", extra=()):
        super().__init__(fork, mvn_bin); self.extra = list(extra)
        self.generated = set(self._unsourced_resources())   # build-made files (no src counterpart) at start

    def _unsourced_resources(self):
        """Non-class files in target/classes and target/test-classes with no file under src/<set>/resources.
        Maven copies resources but never deletes a copy whose source is gone (Gradle does), so a resource a
        candidate added (e.g. a database migration) would otherwise stay in every later build and test."""
        for out, src in (("classes", "main"), ("test-classes", "test")):
            for t in self.fork.glob(f"*/target/{out}"):
                res = t.parent.parent / "src" / src / "resources"
                for f in t.rglob("*"):
                    if f.is_file() and f.suffix != ".class" and not (res / f.relative_to(t)).exists():
                        yield f

    def mvn(self, *args, timeout):
        for f in self._unsourced_resources():
            if f not in self.generated: f.unlink()
        rc, log, secs = self.sh([self.gradle_bin, "-o", "-B", *MVN_SKIPS, *self.extra, *args], timeout,
                                env_java=True)
        log = _MVN_ERR.sub(lambda m: f"{m.group(1)}:{m.group(2)}:{m.group(3)}: error: {m.group(4)}", log)
        log = _MVN_WARN.sub(lambda m: f"{m.group(1)}:{m.group(2)}:{m.group(3)}: warning: {m.group(4)}", log)
        log = _MVN_CONT.sub("", log)
        log = re.sub(r"^(\[(?:ERROR|INFO|WARNING)\] \S)", r"\n\1", log, flags=re.M)   # ends an error block
        return rc, log, secs

    def sh(self, cmd, timeout, stdin=None, env_java=False):
        if not env_java: return super().sh(cmd, timeout, stdin)
        t = time.time(); env = dict(os.environ, JAVA_HOME=JAVA_HOME)
        try:
            r = subprocess.run(cmd, cwd=self.fork, capture_output=True, text=True, timeout=timeout, errors="replace", env=env)
            return r.returncode, r.stdout + r.stderr, round(time.time() - t, 1)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"") + (e.stderr or b"")
            return "timeout", out.decode(errors="replace") if isinstance(out, bytes) else out, round(time.time() - t, 1)

    def build(self, projs, timeout):
        mods = sorted({p for p in projs if p})
        return self.mvn("-q", *(["-pl", ",".join(mods), "-am"] if mods and mods != ["."] else []), "test-compile",
                        timeout=timeout)

    def test(self, proj, classes, timeout):
        return self.mvn(*(["-pl", proj, "-am"] if proj != "." else []), "test", "-Dtest=" + ",".join(classes), "-DfailIfNoTests=false",
                        "-Dsurefire.failIfNoSpecifiedTests=false", "-DskipUnitTests=false", timeout=timeout)

    def test_results_dir(self, proj):
        return self.fork / proj / "target/surefire-reports"

    def build_facts(self):
        return {"build": "maven", "checkstyle": False, "scala_version": None, "scala_fatal_warnings": False}

    def project_of(self, path):
        if path.startswith("src/"): return "."
        if "/src/" not in path: return None
        mod = path.split("/src/")[0]
        return mod if (self.fork / mod / "pom.xml").exists() else None
