"""rcap.blame: compile errors of a layer are charged to the group that caused them.

The two cases are the ones the yardstick run lost (results/yardstick-2026-10, evaluation D):
PR-15149 layer 1 and PR-12250 layer 1. The build logs are the run's own (fork path stripped);
the stub build replays them for the full layer and derives the logs of subsets from which
groups are in the state.
"""
import itertools
import random
from pathlib import Path

from rcap import blame
from rcap.feedback import parse_compile_errors

FIX = Path(__file__).parent / "fixtures" / "blame"
REST = "connect/runtime/src/main/java/org/apache/kafka/connect/runtime/rest/"


def result(log: str, applied: blame.Applied) -> blame.BuildResult:
    errors = parse_compile_errors(log)
    for e in errors:
        e.rel = e.path.lstrip("/")
    return blame.BuildResult(not errors, errors=errors, applied=applied)


def blocks(log: str) -> list[str]:
    """The compiler error blocks of a log, each with its detail lines."""
    out: list[str] = []
    for line in log.splitlines():
        if ": error:" in line or line.startswith("[Error]"):
            out.append(line + "\n")
        elif out and (line.startswith("  ") or line.strip() == "^"):
            out[-1] += line + "\n"
        elif out and not line.strip():
            continue
        elif out and line[:1] not in " \t[/":
            break
    return out


# ---------------------------------------------------------------------------- PR-15149, layer 1
L1 = "sap-LoggingResource::LoggingResource_fn1"      # adds a constructor LoggingResource(Herder): the culprit
L3 = "sap-LoggingResource::LoggingResource_fn3"      # import only
R3 = "sap-RestServer::RestServer_fn3"                # import only
R4 = "sap-RestServer::RestServer_fn4"                # the DEFAULT_REST_REQUEST_TIMEOUT_MS constant: a real port
RR = "sap-RootResource::RootResource_fn2"            # a second @Inject: its own error
W1 = "sap-Worker::Worker_fn1"                        # unrelated, fine


def pr15149_build(calls):
    log = (FIX / "PR-15149.layer-01.01.build.log").read_text()
    root_err, rest_errs = blocks(log)[0], blocks(log)[1:]
    assert "RootResource.java:35" in root_err and len(rest_errs) == 2

    def build(tag, groups):
        calls.append((tag, tuple(groups)))
        s = set(groups)
        ap = blame.Applied()
        whole = {L1: REST + "resources/LoggingResource.java", L3: REST + "resources/LoggingResource.java",
                 R3: REST + "RestServer.java", R4: REST + "RestServer.java",
                 RR: REST + "resources/RootResource.java", W1: "connect/runtime/src/main/java/W.java"}
        for g in s:
            ap.spans.setdefault(whole[g], []).append((1, 400, g))      # whole-file units: the span is the file
        if s & {L1, L3}:
            ap.changed[whole[L1]] = [(26, 26, tuple(sorted(s & {L1, L3})))]     # the shared import hunk
        if L1 in s:
            ap.changed[whole[L1]].append((169, 173, (L1,)))
        if R3 in s:
            ap.changed.setdefault(whole[R3], []).append((67, 67, (R3,)))
        if R4 in s:
            ap.changed.setdefault(whole[R4], []).append((76, 81, (R4,)))
        if RR in s:
            ap.changed[whole[RR]] = [(35, 35, (RR,))]
        ap.declares = {g: frozenset() for g in s}
        if L1 in s:
            ap.declares[L1] = frozenset({"LoggingResource"})
        if R4 in s:
            ap.declares[R4] = frozenset({"DEFAULT_REST_REQUEST_TIMEOUT_MS"})
        text = ""
        if RR in s:
            text += root_err
        if L1 in s:                       # `new LoggingResource()` in RestServer no longer compiles
            text += "".join(rest_errs)
        return result(text, ap)
    return build


def test_pr15149_constructor_changed_by_a_neighbour_is_charged_to_the_neighbour():
    calls = []
    out = blame.settle_layer([], [L1, L3, R3, R4, RR, W1], pr15149_build(calls))
    assert out.kept == sorted([L3, R3, R4, W1])
    assert set(out.failed) == {L1, RR}
    assert out.failed[L1]["how"] == "declaration" and "RestServer.java" in out.failed[L1]["error"]
    assert out.failed[L1]["names"] == ["LoggingResource"]
    assert out.failed[RR]["how"] == "changed-line"
    # the first round already charges both; RestServer's units are never suspects
    first = out.evidence[0]
    assert first["certain"] == sorted([L1, RR])
    assert not any(g in c["suspects"] for c in first["charges"] for g in (R3, R4, L3))
    # both culprits were tried again on the kept state and still do not build
    assert all("does not build" in out.failed[g]["reinstatement"] for g in (L1, RR))


def test_pr15149_layer2_both_groups_that_change_the_constructor_are_isolated():
    """ConnectorPluginsResource_fn1 and _fn3 both rewrite the constructor; the caller in RestServer is
    fork code. Two declaration changers: neither is guessed, both are confirmed by subset builds."""
    log = (FIX / "PR-15149.layer-02.01.build.log").read_text()
    sym, ctor = blocks(log)
    C1, C3, OK = "CPR_fn1", "CPR_fn3", "ConnectorsResource_fn1"
    f = REST + "resources/ConnectorPluginsResource.java"

    def build(tag, groups):
        s = set(groups); ap = blame.Applied()
        ap.declares = {g: frozenset({"ConnectorPluginsResource"}) if g in (C1, C3) else frozenset() for g in s}
        if C1 in s: ap.changed.setdefault(f, []).append((66, 67, (C1,)))
        if C3 in s: ap.changed.setdefault(f, []).append((69, 71, (C3,)))
        text = (sym if C3 in s else "") + (ctor if s & {C1, C3} else "")
        return result(text, ap)
    out = blame.settle_layer(["accepted"], [OK, C3, C1], build)
    assert out.kept == [OK] and set(out.failed) == {C1, C3}
    assert out.failed[C3]["how"] == "changed-line"            # its own `requestTimeout` error
    assert out.failed[C1]["how"] == "declaration"             # once C3 is out, C1 is the only changer left


# ---------------------------------------------------------------------------- PR-12250, layer 1
CT = "sap-ClusterTest::ClusterTest_fn1"                  # replaces `String ibp()` of the annotation
PI = "sap-ProducerIdsIntegrationTest::ProducerIdsIntegrationTest_fn1"   # ibp = "3.0-IV1": a real port
KR = "sap-KafkaRaftServer::KafkaRaftServer_fn1"


def pr12250_build(calls, *, clustertest_own_error=True, error_on_changed_line=True):
    log = (FIX / "PR-12250.layer-01.03.build.log").read_text()
    b = blocks(log)
    own, ann = "".join(b[:2]), "".join(b[2:])
    assert "ClusterTest.java:21" in own and "unknown annotation argument name: ibp" in ann
    ct = "core/src/test/java/kafka/test/annotation/ClusterTest.java"
    pi = "core/src/test/scala/integration/kafka/coordinator/transaction/ProducerIdsIntegrationTest.scala"

    def build(tag, groups):
        calls.append((tag, tuple(groups)))
        s = set(groups); ap = blame.Applied()
        ap.declares = {g: frozenset() for g in s}
        ap.sources = {pi: "\n" * 45 + '    new ClusterTest(clusterType = Type.KRAFT, brokers = 3, ibp = "3.0-IV1")\n'}
        if CT in s:
            ap.changed[ct] = [(21, 21, (CT,)), (44, 44, (CT,))]
            ap.spans[ct] = [(1, 50, CT)]
            ap.declares[CT] = frozenset({"ibp", "metadataVersion"})
        if PI in s:
            ap.changed[pi] = [(46, 46, (PI,))] if error_on_changed_line else [(47, 47, (PI,))]
            ap.spans[pi] = [(44, 60, PI)]
        text = (own if CT in s and clustertest_own_error else "") + (ann if CT in s else "")
        return result(text, ap)
    return build


def test_pr12250_annotation_argument_removed_by_a_neighbour():
    for on_changed in (True, False):
        calls = []
        out = blame.settle_layer([], [PI, CT, KR], pr12250_build(calls, error_on_changed_line=on_changed))
        assert out.kept == sorted([KR, PI]), on_changed
        assert set(out.failed) == {CT}
        assert out.failed[CT]["status"] == "BUILD_FAIL"


def test_pr12250_ambiguous_error_is_isolated_by_subset_builds_not_guessed():
    """Without ClusterTest's own error the only evidence is the error on the test's own changed line
    about a name ClusterTest un-declares: ambiguous. The subsets decide, and the record says so."""
    calls = []
    out = blame.settle_layer([], [PI, CT, KR], pr12250_build(calls, clustertest_own_error=False))
    assert out.kept == sorted([KR, PI]) and set(out.failed) == {CT}
    assert out.failed[CT]["how"].startswith("isolated") and out.failed[CT]["isolated"] is True
    first = out.evidence[0]
    assert first["certain"] == [] and first["charges"][0]["how"] == "changed-line+declaration"
    assert sorted(first["charges"][0]["suspects"]) == sorted([CT, PI])
    subsets = [e for e in out.evidence if "subset" in e]
    assert {tuple(e["subset"]): e["ok"] for e in subsets}[(CT,)] is False
    assert {tuple(e["subset"]): e["ok"] for e in subsets}[(PI,)] is True


# ---------------------------------------------------------------------------- general properties
def model_build(culprits_alone=(), breaks=None, pairs=(), own_line=True, order_log=None):
    """A stub build over abstract groups.

    culprits_alone: groups with an error on their own changed line. breaks: {D: (V, name)}: D changes
    the declaration of `name`, and the error shows in V's file (on V's changed line when own_line).
    pairs: (A, B) that build alone but not together (error on A's line, about a name B declares)."""
    breaks = breaks or {}

    def build(tag, groups):
        if order_log is not None:
            order_log.append(tuple(groups))
        s = set(groups); ap = blame.Applied(); text = ""
        ap.declares = {g: frozenset() for g in s}
        for g in groups:
            ap.changed[f"src/{g}.java"] = [(10, 12, (g,))]
            ap.spans[f"src/{g}.java"] = [(1, 100, g)]
        for g in groups:
            if g in culprits_alone:
                text += f"/src/{g}.java:11: error: incompatible types: String cannot be converted to int\n"
        for d, (v, name) in breaks.items():
            if d in s:
                ap.declares[d] = frozenset({name})
                line = 11 if own_line and v in s else 50
                text += (f"/src/{v}.java:{line}: error: method {name} in class X cannot be applied to given types;\n"
                         f"        x.{name}(1);\n         ^\n  required: no arguments\n  found: int\n")
        for x, y in pairs:
            if x in s and y in s:
                ap.declares[y] = ap.declares[y] | {"pairName"}
                text += f"/src/{x}.java:11: error: cannot find symbol\n  symbol:   method pairName()\n"
        return result(text, ap)
    return build


def test_victim_that_is_also_a_culprit_and_a_correct_neighbour():
    # D breaks V's file; V has an error of its own as well; W is fine.
    out = blame.settle_layer([], ["W", "V", "D"], model_build(culprits_alone={"V"}, breaks={"D": ("V", "frob")}))
    assert out.kept == ["W"] and set(out.failed) == {"D", "V"}
    # V only a victim: it stays, whichever line the error is on
    for own in (True, False):
        out = blame.settle_layer([], ["W", "V", "D"], model_build(breaks={"D": ("V", "frob")}, own_line=own))
        assert out.kept == ["V", "W"] and set(out.failed) == {"D"}, own


def test_a_group_is_never_lost_to_a_neighbour_even_when_the_first_charge_is_wrong():
    """Only weak evidence points at the culprit and one group of the file is in the layer: the suspects
    are verified by builds, and whatever was taken out is retried on the kept state."""
    def build(tag, groups):
        s = set(groups); ap = blame.Applied(); ap.declares = {g: frozenset() for g in s}
        for g in s: ap.spans[f"src/{g}.java"] = [(1, 100, g)]
        text = "/src/V.java:50: error: incompatible types: Foo cannot be converted to Bar\n" if "D" in s else ""
        return result(text, ap)
    out = blame.settle_layer([], ["V", "D", "W"], build)
    assert out.kept == ["V", "W"] and set(out.failed) == {"D"}
    assert out.failed["D"]["how"].startswith("isolated")


def test_two_groups_that_only_fail_together_lose_exactly_one_deterministically():
    out = blame.settle_layer([], ["B", "A", "C"], model_build(pairs=[("A", "B")]))
    # ambiguous (A's line, B's declaration); each builds alone; the minimal failing set {A, B} is taken
    # out and each is retried on the kept state in sorted order: A returns, B then does not build
    assert out.kept == ["A", "C"] and set(out.failed) == {"B"}
    assert any("jointly_inconsistent" in e for e in out.evidence)


def test_same_layer_dependents_of_a_failed_group_are_blocked_not_charged():
    deps = {"D": ["U"]}
    out = blame.settle_layer([], ["U", "D", "W"], model_build(culprits_alone={"D"}),
                             hard_dependents=lambda g: deps.get(g, []))
    assert out.kept == ["W"]
    assert out.failed["D"]["status"] == "BUILD_FAIL" and out.failed["U"] == {
        "status": "BLOCKED_BY", "blocked_by": "D", "same_layer": True}


def test_state_that_cannot_be_written_and_build_timeout():
    def build(tag, groups):
        if "M" in groups:
            return blame.BuildResult(None, unwritable={"M": "unspliceable: original text found 0 times"})
        return blame.BuildResult(True)
    out = blame.settle_layer([], ["M", "N"], build)
    assert out.kept == ["N"] and out.failed["M"]["status"] == "INTEGRATION_FAIL"
    out = blame.settle_layer([], ["A", "B"], lambda tag, groups: blame.BuildResult(False, note="build timeout"))
    assert out.kept == [] and all(f["how"] == "unattributed in its layer" for f in out.failed.values())
    assert out.builds == 1                                       # no subset builds, no retries


def test_outcome_does_not_depend_on_the_order_of_the_groups_in_the_layer():
    rng = random.Random(20261008)
    layer = [L1, L3, R3, R4, RR, W1]
    ref = blame.settle_layer([], layer, pr15149_build([]))
    scenarios = [
        dict(culprits_alone={"g3"}, breaks={"g5": ("g1", "frob"), "g2": ("g6", "nab")}, pairs=[("g4", "g7")]),
        dict(culprits_alone={"g1", "g6"}, breaks={"g1": ("g2", "frob")}, own_line=False),
        dict(pairs=[("g1", "g2"), ("g2", "g3")], culprits_alone={"g8"}),
    ]
    refs = [blame.settle_layer(["a0"], [f"g{i}" for i in range(1, 9)], model_build(**sc)) for sc in scenarios]
    for _ in range(25):
        rng.shuffle(layer)
        out = blame.settle_layer([], list(layer), pr15149_build([]))
        assert (out.kept, out.failed, out.evidence) == (ref.kept, ref.failed, ref.evidence)
        for sc, r in zip(scenarios, refs):
            gs = [f"g{i}" for i in range(1, 9)]
            rng.shuffle(gs)
            seen = []
            out = blame.settle_layer(["a0"], gs, model_build(order_log=seen, **sc))
            assert (out.kept, out.failed, out.evidence) == (r.kept, r.failed, r.evidence)
            assert all(list(s) == ["a0"] + sorted(s[1:]) for s in seen)      # every state is asked for sorted
    # and for all 24 orders of a small layer, exhaustively
    small = [blame.settle_layer([], list(p), model_build(breaks={"D": ("V", "frob")}, culprits_alone={"X"}))
             for p in itertools.permutations(["D", "V", "W", "X"])]
    assert all((o.kept, o.failed) == (small[0].kept, small[0].failed) for o in small)
    assert small[0].kept == ["V", "W"]


def test_error_symbols():
    e = parse_compile_errors((FIX / "PR-15149.layer-01.01.build.log").read_text())[1]
    strong, weak = blame.error_symbols(e)
    assert strong == {"LoggingResource"} and "Herder" in weak
    e = parse_compile_errors((FIX / "PR-12250.layer-01.03.build.log").read_text())[2]
    assert blame.error_symbols(e)[0] == {"ibp"}
