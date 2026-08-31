"""
MACs (multiply-accumulate operations) and parameter counts per method --
the computational-cost dimension no table in this project has reported
yet. PROTOCOL.md's own Edge-Enabled-paper discussion (Sec. 2/5.2) flags
deployment hardware constraints as a real concern for this problem
(PYNQ-ZU FPGA, ~15 W), and Yaqub et al.'s own Table 5 benchmarks their
classifier candidates on exactly this axis (Params, GFLOPs, model size,
inference time) -- this module does the same for this project's six
separation baselines plus the Condition A classifier, using this
dataset's own real signal length (60,000 samples / 15 s @ 4000 Hz) rather
than a generic benchmark input.

Cost is reported per real *inference* call (separating one mixture, or
classifying one recording) -- the deployment-relevant number -- with
one-time fitting/training cost (dictionary learning, model training)
called out separately where it exists, since it is amortized across many
inferences rather than paid every time.

Precision tiers, stated explicitly per method rather than implied by a
single confident-looking number:
  - "exact"   -- literal matrix-multiplication / conv-layer shapes taken
                 directly from the method's own real code (import the
                 actual constants, don't duplicate/guess them), or
                 measured directly from a real forward pass (Baseline 6).
  - "estimate"-- standard textbook complexity-class formulas (SVD, FFT)
                 applied to this method's real problem size. Order of
                 magnitude, not an op-for-op count -- LAPACK's actual
                 constant factors vary by algorithm/implementation.

See code_description.md for the full per-method derivation notes.

Usage:
    python compute_cost.py
"""
import numpy as np

N_SAMPLES = 60000
SAMPLE_RATE = 4000


def _stft_frame_count(n_samples: int, n_fft: int, hop_length: int) -> int:
    """Real librosa.stft() output frame count (center=True padding) --
    measured once via an actual call rather than hand-derived, to avoid an
    off-by-one in the padding convention feeding every NMF MAC formula below."""
    import librosa

    return librosa.stft(np.zeros(n_samples), n_fft=n_fft, hop_length=hop_length).shape[1]


def bandpass_macs(n_samples: int = N_SAMPLES) -> dict:
    """
    Zero-phase Butterworth bandpass (baseline.common._bandpass), applied
    once per band (heart, lung) via scipy.signal.sosfiltfilt.

    order=4 -> ceil(4/2)=2 second-order (biquad) sections. Each biquad
    section costs 5 multiplies (b0,b1,b2,-a1,-a2) per output sample --
    counted as 5 MACs/sample, the standard convention for direct-form-II
    biquads. filtfilt runs the filter forward then backward (2x) for zero
    phase. Applied to both heart and lung bands (2x) per mixture.
    """
    from baseline.common import HEART_BAND, LUNG_BAND

    order = 4
    n_sections = -(-order // 2)
    macs_per_band = n_samples * 5 * n_sections * 2
    n_bands = len([HEART_BAND, LUNG_BAND])
    return {
        "method": "Baseline 1 (bandpass)",
        "params": 0,
        "macs_per_inference": macs_per_band * n_bands,
        "precision": "exact",
        "notes": f"{n_sections} biquad sections/band x {n_bands} bands x fwd+back (filtfilt)",
    }


def _nmf_iteration_macs(f: int, k: int, t: int, supervised: bool) -> int:
    """
    One KL-MU-NMF iteration (baseline.baseline2._nmf_kl), counted directly
    from its real matrix shapes:
      WH = W @ H                    -> F x K times K x T = F*K*T MACs
      H *= (W.T @ (V/WH)) / ...     -> K x F times F x T = K*F*T MACs
    supervised (W frozen, activation-solving only): 2 matmuls/iteration.
    unsupervised (W also updates, same shapes again for the W step):
    4 matmuls/iteration.
    """
    per_matmul = f * k * t
    return per_matmul * (2 if supervised else 4)


def supervised_nmf_macs(n_samples: int = N_SAMPLES, sr: int = SAMPLE_RATE) -> dict:
    """Baseline 2: per-mixture cost is activation-solving only (dictionary
    already fit, frozen) -- ACTIVATION_ITERS supervised iterations at rank
    K_HEART+K_LUNG, plus the final mask reconstruction matmul."""
    from baseline.baseline2 import ACTIVATION_ITERS, HOP_LENGTH, K_HEART, K_LUNG, N_FFT

    f = N_FFT // 2 + 1
    t = _stft_frame_count(n_samples, N_FFT, HOP_LENGTH)
    k = K_HEART + K_LUNG

    activation_macs = ACTIVATION_ITERS * _nmf_iteration_macs(f, k, t, supervised=True)
    reconstruction_macs = f * k * t
    params = k * f

    return {
        "method": "Baseline 2 (supervised NMF)",
        "params": params,
        "macs_per_inference": activation_macs + reconstruction_macs,
        "precision": "exact",
        "notes": f"F={f}, K={k}, T={t}, {ACTIVATION_ITERS} activation-only MU iters/mixture "
                 f"(one-time dictionary fit: {_fit_dict_macs(f, K_HEART, K_LUNG)} MACs/fold, not counted per-mixture)",
    }


def _fit_dict_macs(f: int, k_heart: int, k_lung: int, t_pool: int | None = None) -> str:
    """One-time dictionary-fitting cost (per fold, not per mixture) --
    T_pool (total concatenated training frames) varies by fold's allowed
    pool size, so this is reported as a formula, not a single number."""
    from baseline.baseline2 import DICT_ITERS

    return f"{DICT_ITERS} unsupervised MU iters x 4*F*K*T_pool, T_pool = pool's total STFT frames (varies per fold)"


def standard_nmf_macs(n_samples: int = N_SAMPLES, sr: int = SAMPLE_RATE) -> dict:
    """Baseline 3: fully transductive -- every one of STANDARD_NMF_ITERS
    iterations updates both W and H fresh per mixture (unsupervised
    branch), so there is no separate one-time fitting cost to subtract
    out; the whole thing is per-mixture inference cost."""
    from baseline.baseline2 import HOP_LENGTH, K_HEART, K_LUNG, N_FFT
    from baseline.baseline3 import STANDARD_NMF_ITERS

    f = N_FFT // 2 + 1
    t = _stft_frame_count(n_samples, N_FFT, HOP_LENGTH)
    k = K_HEART + K_LUNG

    nmf_macs = STANDARD_NMF_ITERS * _nmf_iteration_macs(f, k, t, supervised=False)
    reconstruction_macs = f * k * t

    return {
        "method": "Baseline 3 (standard NMF)",
        "params": 0,
        "macs_per_inference": nmf_macs + reconstruction_macs,
        "precision": "exact",
        "notes": f"F={f}, K={k}, T={t}, {STANDARD_NMF_ITERS} full (W+H) MU iters/mixture, "
                 "no persisted dictionary -- entirely transductive, see BACKLOG.md's B2-vs-B3 fairness note",
    }


def mssa_macs(n_samples: int = N_SAMPLES) -> dict:
    """
    Two SSA decompositions (stage 1 on the mixture, stage 2 on the
    residual, baseline.baseline4.mssa_separate), each an L x K trajectory
    matrix (L=50, K=N-L+1) reduced SVD via numpy.linalg.svd (LAPACK
    gesdd). Using the standard reduced-SVD FLOP estimate for an L x K
    matrix with L << K: ~4*L^2*K + 8*L^3 (Golub & Van Loan's bidiagonalization
    + divide-and-conquer estimate; the L^3 term is negligible here since
    L=50 << K~60000). This is an order-of-magnitude estimate, not an
    op-for-op count -- LAPACK's real constant depends on the exact
    algorithm path chosen internally.
    """
    from baseline.baseline4 import SSA_WINDOW_LENGTH

    L = SSA_WINDOW_LENGTH
    K = n_samples - L + 1
    svd_macs = 4 * L * L * K + 8 * L**3
    total = 2 * svd_macs

    return {
        "method": "Baseline 4 (MSSA)",
        "params": 0,
        "macs_per_inference": total,
        "precision": "estimate",
        "notes": f"L={L}, K={K}, 2x reduced SVD (stage 1 + stage 2), ~4*L^2*K FLOP estimate",
    }


def evmd_macs(n_samples: int = N_SAMPLES) -> dict:
    """
    VMD's ADMM loop (baseline.baseline5._vmd) has no per-iteration matrix
    multiply -- its cost is K modes x max_iter iterations of O(T_ext)
    elementwise complex arithmetic (a handful of complex multiply/divide/
    subtract ops per sample; ~8 real-MAC-equivalents per (mode, sample,
    iteration) is used here as a round, order-of-magnitude count for that
    handful of ops, not a literal op-for-op tally) plus one forward + one
    inverse FFT per call (O(T_ext log T_ext), negligible next to the ADMM
    term at these K/max_iter values).

    The real per-mixture cost depends on where each attempted K's ADMM
    loop actually stops (the `change < tol` early-exit, data-dependent) --
    this reports the *worst-common-case* estimate: the full K=2..10 sweep
    (9 values, EVMD_K_MIN..EVMD_K_MAX) each running the full EVMD_MAX_ITER
    ADMM iterations, since BACKLOG.md's own K-selection convergence check
    found only 2/9 sampled mixtures converge before exhausting the sweep
    -- i.e. the worst case is the realistic common case for this method on
    this dataset, not a pessimistic outlier.
    """
    from baseline.baseline5 import EVMD_K_MAX, EVMD_K_MIN, EVMD_MAX_ITER

    half = n_samples // 2
    t_ext = half + n_samples + half
    admm_ops_per_mode_per_sample = 8

    total = 0
    for k in range(EVMD_K_MIN, EVMD_K_MAX + 1):
        total += EVMD_MAX_ITER * k * t_ext * admm_ops_per_mode_per_sample
    fft_macs = 2 * (t_ext * np.log2(t_ext))
    total += (EVMD_K_MAX - EVMD_K_MIN + 1) * fft_macs

    return {
        "method": "Baseline 5 (EVMD)",
        "params": 0,
        "macs_per_inference": int(total),
        "precision": "estimate",
        "notes": f"K={EVMD_K_MIN}..{EVMD_K_MAX} full sweep (worst-common-case per BACKLOG.md's "
                 f"K-selection finding), {EVMD_MAX_ITER} ADMM iters/K, T_ext={t_ext}",
    }


def convtasnet_params_and_macs(n_samples: int = N_SAMPLES) -> dict:
    """
    Real PyTorch model, so params and MACs are *measured* directly rather
    than hand-derived: params via model.parameters(), MACs via forward
    hooks on every Conv1d/ConvTranspose1d that record real output shapes
    from an actual forward pass (not analytically re-derived stride/
    padding/dilation arithmetic, which is exactly the kind of place an
    off-by-one bug hides).
    """
    import torch

    from convtasnet import ConvTasNetLite

    model = ConvTasNetLite()
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    macs = 0
    hooks = []

    def make_hook():
        def hook(module, inputs, output):
            nonlocal macs
            out_length = output.shape[-1]
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups
            kernel_size = module.kernel_size[0]
            macs += out_channels * (in_channels // groups) * kernel_size * out_length
        return hook

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            hooks.append(module.register_forward_hook(make_hook()))

    with torch.no_grad():
        model(torch.zeros(1, n_samples))
    for h in hooks:
        h.remove()

    return {
        "method": "Baseline 6 (Conv-TasNet-lite)",
        "params": params,
        "macs_per_inference": macs,
        "precision": "exact",
        "notes": "measured via forward hooks on every Conv1d/ConvTranspose1d, real 60,000-sample input",
    }


def classifier_params_and_macs(n_samples: int = N_SAMPLES, sr: int = SAMPLE_RATE) -> dict:
    """
    Real sklearn SVC, so support-vector count (its actual "parameter"
    footprint at inference time -- an RBF-SVM's decision function needs
    every support vector, not a fixed-size weight matrix) is measured by
    actually fitting one fold's classifier (heart_classifier.py), not
    guessed. MFCC extraction cost is counted analytically from n_mfcc/
    n_fft/hop_length; RBF-kernel evaluation cost is
    n_support_vectors x n_features x n_classes-worth of decision functions
    (one-vs-one for a 4-class SVC -> C(4,2)=6 binary classifiers).
    """
    from heart_classifier import HOP_LENGTH, N_FFT, N_MFCC, assign_classifier_folds, train_fold_classifiers

    hs_df = assign_classifier_folds(n_folds=5, seed=0)
    classifiers = train_fold_classifiers(hs_df, n_folds=5)
    svm = classifiers[0].named_steps["svm"]
    n_support = int(svm.support_vectors_.shape[0])
    n_features = int(svm.support_vectors_.shape[1])

    t = _stft_frame_count(n_samples, N_FFT, HOP_LENGTH)
    n_mels = 128
    f = N_FFT // 2 + 1
    mfcc_macs = n_mels * f * t + N_MFCC * n_mels * t

    n_ovo_classifiers = 6
    rbf_macs = n_support * n_features * n_ovo_classifiers

    return {
        "method": "Condition A classifier (MFCC + RBF-SVM)",
        "params": n_support,
        "macs_per_inference": mfcc_macs + rbf_macs,
        "precision": "exact (fold 0's real fitted model)",
        "notes": f"{n_support} support vectors x {n_features} features, {n_ovo_classifiers} one-vs-one pairs "
                 "(support-vector count varies slightly per fold -- this is fold 0's)",
    }


def all_methods() -> list[dict]:
    return [
        bandpass_macs(),
        supervised_nmf_macs(),
        standard_nmf_macs(),
        mssa_macs(),
        evmd_macs(),
        convtasnet_params_and_macs(),
        classifier_params_and_macs(),
    ]


if __name__ == "__main__":
    import pandas as pd

    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    print(f"Computing MACs/params per method (real {N_SAMPLES}-sample / 15s @ {SAMPLE_RATE} Hz inference)...")
    rows = []
    for fn in (bandpass_macs, supervised_nmf_macs, standard_nmf_macs, mssa_macs, evmd_macs):
        print(f"  {fn.__name__}...")
        rows.append(fn())
    print("  convtasnet_params_and_macs (instantiating + real forward pass)...")
    rows.append(convtasnet_params_and_macs())
    print("  classifier_params_and_macs (fitting fold 0's real classifier)...")
    rows.append(classifier_params_and_macs())

    df = pd.DataFrame(rows).set_index("method")
    df["gmacs_per_inference"] = df["macs_per_inference"] / 1e9

    exact_n = (df["precision"].str.startswith("exact")).sum()

    stat_tiles = "\n".join([
        stat_tile("Methods", str(len(df)), "6 separation baselines + Condition A classifier"),
        stat_tile("Exact counts", f"{exact_n}/{len(df)}", "literal matmul/conv shapes or measured forward pass"),
        stat_tile("Widest range", f"{df['macs_per_inference'].max() / df['macs_per_inference'].min():.0f}x", "most vs. least expensive method"),
    ])

    display_df = df[["params", "macs_per_inference", "gmacs_per_inference", "precision", "notes"]]
    body = "\n\n".join([
        section(
            "MACs and parameters per method",
            f"per real inference call on this dataset's {N_SAMPLES}-sample (15s @ {SAMPLE_RATE} Hz) recordings",
            df_to_html(display_df, index_label="method", float_fmt="{:.3f}"),
        ),
    ])

    html = report_shell(
        title="MACs and Parameter Counts",
        eyebrow="HLS-CMDS · compute cost per method",
        heading="MACs and parameter counts, per separation baseline + classifier",
        dek=(
            "The computational-cost dimension this project hadn't reported yet -- matching Yaqub et "
            "al.'s own Table 5 (Params/GFLOPs/model size) and PROTOCOL.md's Edge-Enabled-paper "
            "deployment-hardware discussion, applied to this project's own six separation baselines "
            "and Condition A classifier, on this dataset's real 15s/4000Hz signal length. 'exact' rows "
            "are literal matrix-multiplication/conv-layer shapes from the method's own real code (or a "
            "measured real forward pass); 'estimate' rows use standard textbook complexity formulas "
            "(SVD, ADMM) and are order-of-magnitude, not op-for-op counts."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>compute_cost.py</code>'s module and per-function "
            "docstrings for the exact derivation (or estimate formula + its stated approximation) "
            "behind each row.</p>"
        ),
    )

    report_path = write_report(results_dir() / "compute_cost_report.html", html)
    print(f"\nReport written to {report_path}")
    print(df[["params", "macs_per_inference", "precision"]].to_string())
