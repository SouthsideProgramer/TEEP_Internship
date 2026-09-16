"""
Controlled separation-degradation scheme (S6-01 -- pulled forward twice:
S6 -> S5 -> S2, see BACKLOG.md for both pull-forward decisions and why).
The mechanism the charter's C2 knee-point analysis needs: a way to sweep
"how separated is this audio" *continuously*, so classification accuracy
(PROTOCOL.md Sec. 5.3) can be measured as a function of separation quality
rather than at a handful of discrete (method, SDR) points.

Design decision 1 -- the interpolation itself. For a source with ground
truth `g` and a real separation baseline's estimate `s`, define

    degraded(alpha) = (1 - alpha) * g + alpha * s = g + alpha * (s - g),   alpha in [0, 1]

alpha=0 reproduces the clean ground truth exactly; alpha=1 reproduces that
baseline's real separated output exactly. This is the same "signal + scaled
error term" shape synthetic_mix.py's own SNR_SWEEP_DB already uses
(mixed = a*(heart+lung) + noise, noise scaled to hit a target SNR) --
except the injected term here is `s - g`, a *real separation method's own
characteristic error* (its own SIR/SAR mix -- see PROTOCOL.md Sec. 2's BSS
Eval notes), not i.i.d. Gaussian noise. That is the entire point of doing
this in addition to (not instead of) the SNR sweep: SNR_SWEEP_DB degrades
the *mixture* before separation, so it stresses "how hard is the separation
problem" and the resulting post-separation SDR is measured, not chosen, and
is confounded with each method's own robustness to noise. This scheme
degrades the *already-separated output*, holding the separating method's
error signature fixed and dialing only its magnitude -- so it can hit any
target SDR *for that method's own artifact type* directly, decoupled from
mixture difficulty. Both axes are kept (see code_description.md); they
answer different questions.

Design decision 2 -- the curve's x-axis (the actual point of this memo, see
BACKLOG.md's S6-01 entries). Candidates were: (a) measured SDR from the six
real separation methods, (b) synthetic input SNR, (c) both overlaid.
Decision: **SDR (dB), never SNR, with real-method points and this
interpolation scheme's dense sweep plotted on the same shared SDR axis.**
SNR is a property of the *mixture* fed into separation, not of the
`heart_est`/`lung_est` audio the classifier actually consumes -- and it is
undefined for Yaqub et al.'s own bandpass-separated point (`[spectrotemporal]`,
their Experiment 4, PROTOCOL.md Sec. 3): they never report an SNR for their
mixtures, only that they ran their pipeline on HLS-CMDS's native mixed
recordings as-is. SDR is the one unit computable for every point we want on
the curve, including theirs (approximately, via this project's own Baseline 1
bandpass reproduction, confirmed the same method class in PROTOCOL.md Sec. 3),
so it is the only choice that lets their 89%->41% collapse be placed on our
curve as a point rather than merely cited as an anecdote. "Both overlaid"
is still the right instinct from the brief, but the two things being
overlaid on that one SDR axis are (i) the six real methods' own sparse,
naturally-occurring (SDR, accuracy) points (PROTOCOL.md Sec. 5.2, plus
their behavior across synthetic_mix.py's SNR sweep) and (ii) this scheme's
dense, controlled sweep per method -- not SDR and SNR as two competing
x-axis units.

Scope of this module: the degradation mechanism itself, verified to
actually produce a monotonic, continuous SDR sweep on real data (see
__main__ and test/test_degradation.py) -- not the accuracy-vs-SDR curve
itself, which needs Condition B (classifying separated audio,
PROTOCOL.md Sec. 5.3) to exist first. That remains open.

Usage:
    from degradation import degrade_toward_ground_truth, sdr_sweep

    degraded_heart = degrade_toward_ground_truth(heart_ref, heart_est, alpha=0.4)
    sweep = sdr_sweep(heart_ref, lung_ref, heart_est, lung_est, alphas=np.linspace(0, 1, 11))
"""
import numpy as np

from metrics import SOURCE_LABELS, evaluate_heart_lung

DEFAULT_ALPHAS = tuple(np.linspace(0.0, 1.0, 11))


def degrade_toward_ground_truth(ground_truth: np.ndarray, estimate: np.ndarray, alpha: float) -> np.ndarray:
    """
    alpha=0 -> exactly ground_truth; alpha=1 -> exactly estimate. See this
    module's docstring for why a linear amplitude-domain blend of a real
    method's own error term is the chosen scheme.
    """
    ground_truth, estimate = np.asarray(ground_truth), np.asarray(estimate)
    n = min(len(ground_truth), len(estimate))
    return (1 - alpha) * ground_truth[:n] + alpha * estimate[:n]


def degrade_row(
    heart_ref, lung_ref, heart_est, lung_est, alpha: float, source: str = "heart", compute_permutation: bool = True
) -> dict:
    """
    Degrade one source's estimate toward its own ground truth by alpha,
    holding the *other* source's estimate fixed at its real separated value,
    and score both through this project's one BSS Eval entry point
    (metrics.evaluate_heart_lung) -- degradation never invents a parallel
    metric definition, it only changes what audio gets scored.
    """
    if source not in SOURCE_LABELS:
        raise ValueError(f"source must be one of {SOURCE_LABELS}, got {source!r}")

    degraded_heart = degrade_toward_ground_truth(heart_ref, heart_est, alpha) if source == "heart" else heart_est
    degraded_lung = degrade_toward_ground_truth(lung_ref, lung_est, alpha) if source == "lung" else lung_est

    return evaluate_heart_lung(
        heart_ref, lung_ref, degraded_heart, degraded_lung, compute_permutation=compute_permutation
    )


def sdr_sweep(
    heart_ref, lung_ref, heart_est, lung_est, alphas=DEFAULT_ALPHAS, source: str = "heart", compute_permutation=True
) -> list[dict]:
    """
    Measured SDR/SIR/SAR (dB) of `source` at each alpha -- the actual x-axis
    values for the C2 curve, not alpha itself (alpha has no physical units,
    and different rows/methods reach a given alpha at different real SDRs,
    which is exactly why this returns measured metrics rather than a bare
    alpha label).

    Caveat found by validating this scheme on real data (see __main__): a
    uniform alpha grid does *not* give an even spread of measured SDR values.
    SDR falls off sharply over roughly alpha in [0, 0.1] (near-clean audio,
    where dB is most sensitive to a small absolute error) and then flattens
    out for the rest of [0.1, 1] -- on the row checked here, alpha=0.1 alone
    already accounts for most of the SDR range between "clean" and "that
    method's real output." A uniform grid therefore oversamples the flat tail
    and undersamples the steep, most-informative part of the curve. Prefer
    find_alpha_for_target_sdr() to request specific target SDR values
    directly (e.g. an evenly-spaced dB grid) rather than relying on a
    uniform alpha grid to produce one.
    """
    return [
        {"alpha": float(alpha), **degrade_row(heart_ref, lung_ref, heart_est, lung_est, alpha, source, compute_permutation)[source]}
        for alpha in alphas
    ]


def find_alpha_for_target_sdr(
    heart_ref, lung_ref, heart_est, lung_est, target_sdr_db: float, source: str = "heart",
    tol: float = 0.05, max_iter: int = 40, compute_permutation: bool = True,
) -> float:
    """
    Bisection search for the alpha in [0, 1] whose measured SDR is closest to
    target_sdr_db. Valid only because SDR(alpha) was checked (not assumed) to
    be monotonically non-increasing for the methods validated in this
    module's __main__ -- bisection would silently return a wrong answer on a
    non-monotonic curve, so a caller applying this to a new method should
    re-run that check (e.g. via sdr_sweep on a uniform grid) before trusting
    this function's output for it.

    Clamps to the nearest endpoint if target_sdr_db falls outside the
    achievable range [SDR at alpha=1 (that method's real output), SDR at
    alpha=0 (clean)].
    """
    lo, hi = 0.0, 1.0
    sdr = lambda a: degrade_row(heart_ref, lung_ref, heart_est, lung_est, a, source, compute_permutation)[source]["sdr"]
    sdr_lo, sdr_hi = sdr(lo), sdr(hi)
    if target_sdr_db >= sdr_lo:
        return lo
    if target_sdr_db <= sdr_hi:
        return hi

    mid = (lo + hi) / 2
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        sdr_mid = sdr(mid)
        if abs(sdr_mid - target_sdr_db) <= tol:
            break
        if sdr_mid > target_sdr_db:
            lo = mid
        else:
            hi = mid
    return mid


# ---------------------------------------------------------------------------
# Exact SDR(alpha) and the grid-based root finder (2026-09-13). Replaces the
# bisection above for the sweep. The blend is linear in alpha and mir_eval's
# decomposition is a fixed linear projection, so every component of the
# degraded estimate is the same convex combination of the components of g
# and of s; SDR(alpha) is then a ratio of two quadratics and can be
# evaluated exactly on a dense grid from two decompositions. The bisection
# assumed monotonicity; sdr_alpha_audit.py found 7/216 heart curves
# (all Conv-TasNet-lite) that are not, so the sweep now uses this instead.
# ---------------------------------------------------------------------------
ALPHA_GRID = np.linspace(0.0, 1.0, 1001)
ALPHA_RULE = "smallest_root_from_clean"
_FLEN = 512  # mir_eval bss_eval_sources' fixed distortion-filter length


def _target_and_error(reference_sources, estimate, j):
    from mir_eval.separation import _bss_decomp_mtifilt

    s_true, e_spat, e_interf, e_artif = _bss_decomp_mtifilt(reference_sources, estimate, j, _FLEN)
    return s_true + e_spat, e_interf + e_artif  # mir_eval's SDR numerator / denominator


def sdr_curve_closed_form(reference_sources, g, s, j, alphas=ALPHA_GRID) -> np.ndarray:
    """SDR(alpha) of (1-alpha) g + alpha s against reference j, exactly."""
    t0, e0 = _target_and_error(reference_sources, g, j)
    t1, e1 = _target_and_error(reference_sources, s, j)
    T00, T01, T11 = t0 @ t0, t0 @ t1, t1 @ t1
    E00, E01, E11 = e0 @ e0, e0 @ e1, e1 @ e1
    a = np.asarray(alphas, dtype=float)
    num = (1 - a) ** 2 * T00 + 2 * a * (1 - a) * T01 + a**2 * T11
    den = (1 - a) ** 2 * E00 + 2 * a * (1 - a) * E01 + a**2 * E11
    return 10 * np.log10(np.maximum(num, 1e-300) / np.maximum(den, 1e-300))


def grid_roots(alphas, sdr, target) -> list[float]:
    """Every alpha where the piecewise-linear interpolant of the grid crosses target."""
    roots = []
    for i in range(len(alphas) - 1):
        s0, s1 = sdr[i], sdr[i + 1]
        if s0 == target:
            roots.append(float(alphas[i]))
        elif (s0 - target) * (s1 - target) < 0:
            roots.append(float(alphas[i] + (target - s0) / (s1 - s0) * (alphas[i + 1] - alphas[i])))
    if sdr[-1] == target:
        roots.append(float(alphas[-1]))
    return roots


def _quadratic_coefficients(reference_sources, g, s, j):
    """(T00, T01, T11, E00, E01, E11): SDR(alpha) = 10 log10 N(alpha)/D(alpha) with
    N, D the quadratics in alpha built from these six inner products."""
    t0, e0 = _target_and_error(reference_sources, g, j)
    t1, e1 = _target_and_error(reference_sources, s, j)
    return t0 @ t0, t0 @ t1, t1 @ t1, e0 @ e0, e0 @ e1, e1 @ e1


def exact_roots(coeffs, target_db: float) -> list[float]:
    """Every alpha in [0, 1] with SDR(alpha) == target_db, solved exactly:
    N(alpha) - k D(alpha) = 0, k = 10^(target/10), is a quadratic in alpha."""
    T00, T01, T11, E00, E01, E11 = coeffs
    k = 10 ** (target_db / 10)
    c00, c01, c11 = T00 - k * E00, T01 - k * E01, T11 - k * E11
    # (1-a)^2 c00 + 2a(1-a) c01 + a^2 c11  =  A a^2 + B a + C
    A, B, C = c00 - 2 * c01 + c11, 2 * (c01 - c00), c00
    if abs(A) < 1e-300 * max(1.0, abs(B), abs(C)):
        roots = [-C / B] if B != 0 else []
    else:
        disc = B * B - 4 * A * C
        if disc < 0:
            return []
        sq = np.sqrt(disc)
        roots = [(-B - sq) / (2 * A), (-B + sq) / (2 * A)]
    return sorted(float(r) for r in roots if -1e-12 <= r <= 1 + 1e-12)


def find_alphas_for_target_sdrs_grid(
    heart_ref, lung_ref, heart_est, lung_est, target_sdrs, source: str = "heart", alphas=ALPHA_GRID
) -> dict:
    """
    Replacement for the bisection in find_alphas_for_target_sdrs(). The
    dense grid `alphas` is used only to audit the shape of SDR(alpha)
    (monotone? where is its minimum?); each target's alpha is solved
    EXACTLY as a root of the quadratic N(alpha) - 10^(t/10) D(alpha) = 0,
    so steepness near alpha=0 costs nothing. Rule ALPHA_RULE: the SMALLEST
    root in [0, 1] -- the crossing on the branch that starts at the ground
    truth walking toward the method's real output. A target above SDR(0)
    or with no root in [0, 1] is reported unattainable; the caller decides
    what to do (sdr_sweep.py clamps to alpha=1 and records the clamp).

    Returns {target: {"alpha": float | None, "attainable": bool, "n_roots": int}}
    plus "_curve": {"monotone_decreasing", "n_reversals", "max_reversal_db",
    "sdr_at_0", "sdr_at_1", "sdr_min"} from the grid audit.
    """
    refs = np.stack([np.asarray(heart_ref), np.asarray(lung_ref)])
    j = SOURCE_LABELS.index(source)
    g, s = (heart_ref, heart_est) if source == "heart" else (lung_ref, lung_est)
    n = min(len(g), len(s), refs.shape[1])
    refs, g, s = refs[:, :n], np.asarray(g)[:n], np.asarray(s)[:n]
    coeffs = _quadratic_coefficients(refs, g, s, j)
    T00, T01, T11, E00, E01, E11 = coeffs
    a = np.asarray(alphas, dtype=float)
    num = (1 - a) ** 2 * T00 + 2 * a * (1 - a) * T01 + a**2 * T11
    den = (1 - a) ** 2 * E00 + 2 * a * (1 - a) * E01 + a**2 * E11
    sdr = 10 * np.log10(np.maximum(num, 1e-300) / np.maximum(den, 1e-300))
    d = np.diff(sdr)
    rev = d > 1e-6
    out = {"_curve": {
        "monotone_decreasing": bool(not rev.any()), "n_reversals": int(rev.sum()),
        "max_reversal_db": float(d[rev].max()) if rev.any() else 0.0,
        "sdr_at_0": float(sdr[0]), "sdr_at_1": float(sdr[-1]), "sdr_min": float(sdr.min()),
    }}
    for target in target_sdrs:
        t = float(target)
        roots = exact_roots(coeffs, t) if t <= sdr[0] else []
        roots = [min(max(r, 0.0), 1.0) for r in roots]
        out[t] = {"alpha": (roots[0] if roots else None), "attainable": bool(roots), "n_roots": len(roots)}
    return out


def find_alphas_for_target_sdrs(
    heart_ref, lung_ref, heart_est, lung_est, target_sdrs, source: str = "heart",
    n_probe: int = 13, tol: float = 0.1, max_refine_iter: int = 8, compute_permutation: bool = True,
) -> dict:
    """
    Batched find_alpha_for_target_sdr(): finds alpha for every value in
    target_sdrs while sharing one coarse SDR(alpha) probe table across all
    of them, instead of an independent from-scratch bisection per target.

    Built for sdr_sweep.py (S6-02): generating the sweep needs several
    target SDRs *per (row, source)*, and each BSS-Eval call (mir_eval's
    permutation search over full-length audio, inside degrade_row) is
    expensive enough that running a fresh 40-iteration bisection per target
    -- as find_alpha_for_target_sdr() does alone -- multiplies that cost by
    the number of targets for no reason: the probe table only has to be
    built once per row/source, and it makes every subsequent target's
    search start from an already-narrow bracket instead of the full [0, 1]
    range.

    Probes a grid of n_probe alphas concentrated near alpha=0 (a
    log-spaced grid, since S6-01's own validation found the SDR(alpha)
    curve steepest there) to build one coarse, monotonic alpha->SDR table
    -- monotonicity is the same property find_alpha_for_target_sdr()
    already relies on, not a new assumption -- then for each target,
    brackets it between the table's two neighboring probe points and
    refines with a short bisection from that already-narrow bracket
    (typically a handful of iterations, not up to max_iter from scratch).

    Returns {target_sdr: alpha}, one entry per target_sdrs value.
    """
    alphas_probe = np.concatenate([[0.0], np.geomspace(1e-3, 1.0, n_probe - 1)])
    sdr_probe = np.array([
        degrade_row(heart_ref, lung_ref, heart_est, lung_est, a, source, compute_permutation)[source]["sdr"]
        for a in alphas_probe
    ])
    neg_sdr_probe = -sdr_probe

    results = {}
    for target in target_sdrs:
        if target >= sdr_probe[0]:
            results[float(target)] = float(alphas_probe[0])
            continue
        if target <= sdr_probe[-1]:
            results[float(target)] = float(alphas_probe[-1])
            continue

        idx = int(np.searchsorted(neg_sdr_probe, -target))
        lo, hi = float(alphas_probe[idx - 1]), float(alphas_probe[idx])

        alpha = (lo + hi) / 2
        for _ in range(max_refine_iter):
            alpha = (lo + hi) / 2
            sdr_mid = degrade_row(heart_ref, lung_ref, heart_est, lung_est, alpha, source, compute_permutation)[source]["sdr"]
            if abs(sdr_mid - target) <= tol:
                break
            if sdr_mid > target:
                lo = alpha
            else:
                hi = alpha
        results[float(target)] = alpha

    return results


if __name__ == "__main__":
    import pandas as pd

    from baseline.baseline1 import fit_bandpass_baseline
    from baseline.baseline4 import fit_ssa_baseline
    from load_dataset import load_audio, load_mix, verify_additive_triplets
    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    print("Loading a native additive row to validate the degradation scheme on real audio...")
    mix_df = load_mix()
    additivity = verify_additive_triplets(mix_df)
    valid_row = mix_df[mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].iloc[0]
    print(f"  using {valid_row['Mixed Sound ID']} ({valid_row['Heart Sound Type']} / {valid_row['Lung Sound Type']})")

    heart_ref, sr = load_audio(valid_row["heart_audio_path"], sr=None)
    lung_ref, _ = load_audio(valid_row["lung_audio_path"], sr=None)
    mixed, _ = load_audio(valid_row["mixed_audio_path"], sr=None)

    methods = {
        "Baseline 1 (bandpass)": fit_bandpass_baseline(None, None),
        "Baseline 4 (MSSA)": fit_ssa_baseline(None, None),
    }

    all_checks = []
    sections = []
    target_grid_report = []
    for label, separate_fn in methods.items():
        print(f"Sweeping alpha for {label}...")
        heart_est, lung_est = separate_fn(mixed, sr)
        sweep = sdr_sweep(heart_ref, lung_ref, heart_est, lung_est, source="heart")
        sweep_df = pd.DataFrame(sweep).set_index("alpha")

        sdrs = sweep_df["sdr"].to_numpy()
        is_monotonic = bool(np.all(np.diff(sdrs) <= 1e-6))
        matches_baseline_sdr = bool(np.isclose(sdrs[-1], evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est)["heart"]["sdr"]))
        all_checks.append({"method": label, "monotonic_non_increasing": is_monotonic, "alpha1_matches_measured_sdr": matches_baseline_sdr})

        sections.append(section(
            f"{label}: measured SDR across the alpha sweep",
            f"alpha=0 (clean) -> alpha=1 ({label}'s real separated output)",
            df_to_html(sweep_df, index_label="alpha", float_fmt="{:.2f}"),
        ))

        print(f"Requesting an evenly-spaced target-SDR grid for {label} via find_alpha_for_target_sdr()...")
        achievable_lo, achievable_hi = sdrs[-1], 30.0
        target_sdrs = np.linspace(achievable_hi, achievable_lo, 6)
        for target in target_sdrs:
            alpha = find_alpha_for_target_sdr(heart_ref, lung_ref, heart_est, lung_est, float(target), source="heart")
            achieved = degrade_row(heart_ref, lung_ref, heart_est, lung_est, alpha, "heart")["heart"]["sdr"]
            target_grid_report.append({
                "method": label, "target_sdr": float(target), "alpha_found": alpha, "achieved_sdr": achieved,
            })

    checks_df = pd.DataFrame(all_checks).set_index("method")
    all_ok = bool(checks_df.all(axis=None))
    target_grid_df = pd.DataFrame(target_grid_report).set_index(["method", "target_sdr"])

    stat_tiles = "\n".join([
        stat_tile("Methods checked", str(len(methods)), "bandpass, MSSA"),
        stat_tile("Alpha steps", str(len(DEFAULT_ALPHAS)), "0.0 to 1.0"),
        stat_tile("Monotonic on both", "yes" if all_ok else "no", "", ok=all_ok),
    ])

    body = "\n\n".join([
        section("Sanity checks", "monotonic SDR(alpha), and alpha=1 reproduces the method's own measured SDR", df_to_html(checks_df, index_label="method")),
        *sections,
        section(
            "find_alpha_for_target_sdr(): hitting a requested SDR grid directly",
            "target vs. achieved SDR, and the alpha bisection found -- the recommended way to build an evenly-spaced dB grid, per the uniform-alpha caveat above",
            df_to_html(target_grid_df, index_label="method / target_sdr", float_fmt="{:.2f}"),
        ),
    ])

    html = report_shell(
        title="Degradation Scheme Validation",
        eyebrow="HLS-CMDS · S6-01 · controlled SDR sweep",
        heading="Controlled degradation scheme: real-data validation",
        dek=(
            "Verifies (not assumes) that degrade_toward_ground_truth() produces a monotonic, "
            "continuous SDR sweep from clean ground truth (alpha=0) to a real separation "
            "method's own output (alpha=1), on one native additive mixture row, for two "
            "different baselines with different artifact signatures (bandpass ringing vs. "
            "MSSA's higher-SAR reconstruction)."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>degradation.py</code>'s module docstring for "
            "the design decision (interpolation scheme + SDR-not-SNR x-axis) and "
            "<code>PROTOCOL.md</code> Sec. 5.3.1 for the full memo.</p>"
        ),
    )

    report_path = write_report(results_dir() / "degradation_scheme_report.html", html)
    print(f"\nAll checks passed: {all_ok}")
    print(f"Report written to {report_path}")
