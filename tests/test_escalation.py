"""Tests du moteur d'escalation anti-motif (Lot C6, cahier §13.5)."""

from __future__ import annotations

from adan_trading_bot.future_arena import EscalationConfig, EscalationTracker


def test_grace_no_penalty_at_first():
    t = EscalationTracker(EscalationConfig(grace=3, noise_frac=0.0), seed=0)
    # pendant la grâce : aucune pénalité
    assert t.update("micro_close", sterile=True) == 0.0
    assert t.update("micro_close", sterile=True) == 0.0
    assert t.update("micro_close", sterile=True) == 0.0


def test_penalty_escalates_after_grace():
    t = EscalationTracker(EscalationConfig(grace=2, noise_frac=0.0, base_penalty=0.1), seed=0)
    penalties = [t.update("m", sterile=True) for _ in range(8)]
    # les premières (grâce) = 0, puis pénalités strictement croissantes en magnitude
    nonzero = [p for p in penalties if p < 0]
    assert len(nonzero) >= 3
    mags = [abs(p) for p in nonzero]
    assert all(b >= a for a, b in zip(mags, mags[1:])), mags  # monotone croissant


def test_penalty_is_negative_and_bounded():
    t = EscalationTracker(EscalationConfig(grace=1, noise_frac=0.0, base_penalty=1.0,
                                           max_penalty=2.0), seed=1)
    pens = [t.update("m", sterile=True) for _ in range(50)]
    assert all(p <= 0 for p in pens)
    assert min(pens) >= -2.0  # jamais sous le plafond


def test_good_outcome_resets_debt():
    cfg = EscalationConfig(grace=2, noise_frac=0.0, decay_on_good=1.0)
    t = EscalationTracker(cfg, seed=0)
    for _ in range(6):
        t.update("m", sterile=True)
    assert t.repetitions("m") > 0
    t.update("m", sterile=False)  # bon trade -> reset
    assert t.repetitions("m") == 0.0


def test_partial_decay_keeps_half():
    cfg = EscalationConfig(grace=2, noise_frac=0.0, decay_on_good=0.5)
    t = EscalationTracker(cfg, seed=0)
    for _ in range(6):
        t.update("m", sterile=True)
    before = t.repetitions("m")
    t.update("m", sterile=False)
    after = t.repetitions("m")
    assert abs(after - before * 0.5) < 1e-9


def test_severity_scales_penalty():
    cfg = EscalationConfig(grace=1, noise_frac=0.0, base_penalty=0.1, max_penalty=99)
    t1 = EscalationTracker(cfg, seed=0)
    t2 = EscalationTracker(cfg, seed=0)
    for _ in range(5):
        p_low = t1.update("m", sterile=True, severity=0.5)
        p_high = t2.update("m", sterile=True, severity=2.0)
    assert abs(p_high) > abs(p_low)  # severity plus forte -> pénalité plus forte


def test_independent_motifs():
    t = EscalationTracker(EscalationConfig(grace=1, noise_frac=0.0), seed=0)
    for _ in range(5):
        t.update("micro_close", sterile=True)
    # un autre motif reste vierge
    assert t.repetitions("hold_in_green") == 0.0
    assert t.repetitions("micro_close") > 0.0


def test_noise_breaks_predictability_but_keeps_sign():
    # avec bruit, deux trackers de seeds différentes divergent (non prévisible)
    a = EscalationTracker(EscalationConfig(grace=2, noise_frac=0.3, base_penalty=0.1), seed=1)
    b = EscalationTracker(EscalationConfig(grace=2, noise_frac=0.3, base_penalty=0.1), seed=2)
    pa = [a.update("m", sterile=True) for _ in range(10)]
    pb = [b.update("m", sterile=True) for _ in range(10)]
    assert pa != pb  # divergence stochastique
    assert all(p <= 0 for p in pa + pb)  # mais jamais de signe positif


def test_snapshot_serializable():
    t = EscalationTracker(EscalationConfig(grace=1, noise_frac=0.0), seed=0)
    for _ in range(4):
        t.update("m", sterile=True)
    snap = t.snapshot()
    assert "m" in snap
    assert set(snap["m"].keys()) == {"repetitions", "last_penalty", "total_penalty", "triggers"}


def test_passive_step_forgets():
    cfg = EscalationConfig(grace=1, noise_frac=0.0, passive_decay=0.5)
    t = EscalationTracker(cfg, seed=0)
    for _ in range(4):
        t.update("m", sterile=True)
    r0 = t.repetitions("m")
    for _ in range(3):
        t.passive_step("m")
    assert t.repetitions("m") < r0


def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passés")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
