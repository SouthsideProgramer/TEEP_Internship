"""
Baseline 6: Conv-TasNet-lite -- the first neural (learned end-to-end)
separation baseline for the heart/lung mix set. See code_description.md
for the full writeup (architecture derivation, training-loop design,
honest caveats).

SAMPLE-RATE DECISION (made before any training code was written): this
dataset is natively 4000 Hz (see load_dataset.py / README.md's Dataset
section), so the public Conv-TasNet/Sepformer checkpoints -- trained at
8-16 kHz -- do not apply; loading them would require resampling this
dataset up, which invents no real information and would need declaring in
the paper as a resampling artifact. Decision: train from scratch at the
native 4000 Hz rather than resample. This costs nothing physiologically --
heart energy sits at 20-200 Hz and lung at 100-1000 Hz (Baseline 1's own
Welch-PSD survey), comfortably under a 2000 Hz Nyquist at 4 kHz -- so the
native rate is not a resolution limitation for this task, just a mismatch
with speech-separation pretraining conventions. No pretrained weights are
loaded anywhere in this module.

Architecture: encoder/TCN-separator/decoder, the same three-stage design
as Conv-TasNet (Luo & Mesgarani, IEEE/ACM TASLP 2019,
papers/Conv-TasNet_Surpassing_Ideal_Time-Frequency.pdf) and NeoSSNet (Poh
et al., IEEE OJEMB 2024,
papers/NeoSSNet_Real-Time_Neonatal_Chest_Sound_Separation_Using_Deep_Learning.pdf
-- the closest prior work: also a masked Conv-TasNet-style model
separating heart/lung sound from a single chest channel at 4 kHz).
"Lite" here specifically means: no transformer mask generator (NeoSSNet's
addition over vanilla Conv-TasNet) and a small stacked-dilated-TCN
separator sized for this dataset's ~50 recordings/class, not NeoSSNet's
8.4M-parameter configuration tuned on 71 recordings with heavier
augmentation -- see MODEL PARAMETER CHOICES below for the arithmetic.
Fixed source order (channel 0 = heart, channel 1 = lung) is used instead
of Conv-TasNet's permutation-invariant training: heart and lung are
distinguishable classes here, not interchangeable speakers, matching
NeoSSNet's own (non-PIT) choice for the same reason.

Usage:
    from convtasnet import make_convtasnet_baseline
    from eval_harness import cross_validate

    results_df, fold_summary, cv_summary = cross_validate(make_convtasnet_baseline(seed=0), n_folds=5)

    # or, the primary evaluation substrate for this project (see synthetic_mix.py):
    from synthetic_mix import build_synthetic_set, evaluate_synthetic
    synthetic_df = build_synthetic_set(seed=0)
    results_df, fold_summary, cv_summary = evaluate_synthetic(make_convtasnet_baseline(seed=0), synthetic_df)
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from load_dataset import load_audio

# --- Encoder/decoder ---------------------------------------------------

ENCODER_N = 64        # number of encoder/decoder basis filters
ENCODER_L = 20         # encoder kernel length, samples (5 ms at 4 kHz)
ENCODER_STRIDE = ENCODER_L // 2  # 50% overlap, Conv-TasNet Sec. III-B convention

# --- TCN separator (MODEL PARAMETER CHOICES) ----------------------------
# "Lite" relative to both Conv-TasNet's own speech-separation config
# (N=512, B=128, H=512, P=3, X=8, R=3, ~5M params) and NeoSSNet's
# transformer-augmented 8.4M-param model: this project's training pool is
# ~40-50 recordings/class *per fold* (fewer once a fold's own recordings
# are excluded, see split.py/synthetic_mix.py's leakage-safe pools), so a
# multi-million-parameter network would have far more capacity than this
# baseline's augmented-but-still-small training signal can constrain.
TCN_B = 64      # bottleneck / residual-path channels
TCN_H = 128     # depthwise conv-block channels
TCN_SC = 64     # skip-connection channels
TCN_P = 3       # depthwise conv kernel size
TCN_X = 6       # conv blocks per repeat, dilation 2**0 .. 2**(X-1)
TCN_R = 2       # number of repeats
# Receptive field: R * sum_{i=0}^{X-1} (P-1)*2**i encoder frames + 1
#                = 2 * (2*63) + 1 = 253 encoder frames
#                = 253 * ENCODER_STRIDE / SAMPLE_RATE ~= 1.27s of audio,
# comfortably covering one heartbeat cycle (~0.6-1s) and a breath phase.

N_SOURCES = 2  # heart (channel 0), lung (channel 1) -- fixed order, see module docstring

# --- Training loop --------------------------------------------------------

CROP_SECONDS = 4.0      # random-crop length during training (recordings are 15s uniformly, per BACKLOG.md)
BATCH_SIZE = 8
STEPS_PER_EPOCH = 40
MAX_EPOCHS = 25
INIT_LR = 1e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 5.0   # matches Conv-TasNet/NeoSSNet's own gradient-clipping choice
LR_DECAY = 0.5         # matches NeoSSNet's "scaled by 0.5 when val doesn't improve"
LR_PATIENCE = 4        # epochs of no val improvement before halving LR (NeoSSNet: 4)
EARLY_STOP_PATIENCE = 10  # epochs of no val improvement before stopping training entirely
VAL_FRACTION = 0.2     # fraction of the allowed pool's recordings (by file, not by mixture) held out for internal validation
VAL_PAIRS = 24         # fixed validation batch size (pairs), resampled once per fit, not every epoch


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def _global_layer_norm(channels: int) -> nn.GroupNorm:
    """
    Conv-TasNet's gLN (Eq. 9-11): normalize jointly over the channel AND
    time dimensions of a (batch, channels, time) feature, with per-channel
    affine parameters broadcast across time. Mathematically identical to
    torch.nn.GroupNorm(num_groups=1, ...) -- used directly instead of a
    hand-rolled reimplementation.
    """
    return nn.GroupNorm(1, channels)


class _TCNBlock(nn.Module):
    """One dilated depthwise-separable 1-D conv block, Conv-TasNet Fig. 1(C):
    1x1-conv -> PReLU -> norm -> depthwise D-conv (dilated) -> PReLU -> norm
    -> {1x1-conv residual path, 1x1-conv skip path}."""

    def __init__(self, b_channels: int, h_channels: int, sc_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.in_conv = nn.Conv1d(b_channels, h_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = _global_layer_norm(h_channels)
        pad = (kernel_size - 1) * dilation // 2  # symmetric "same" padding -- kernel_size is odd, so this is exact
        self.depthwise = nn.Conv1d(h_channels, h_channels, kernel_size, padding=pad, dilation=dilation, groups=h_channels)
        self.prelu2 = nn.PReLU()
        self.norm2 = _global_layer_norm(h_channels)
        self.res_conv = nn.Conv1d(h_channels, b_channels, 1)
        self.skip_conv = nn.Conv1d(h_channels, sc_channels, 1)

    def forward(self, x: torch.Tensor):
        h = self.norm1(self.prelu1(self.in_conv(x)))
        h = self.norm2(self.prelu2(self.depthwise(h)))
        return x + self.res_conv(h), self.skip_conv(h)


class _TemporalConvNet(nn.Module):
    """Stacked dilated TCN blocks (X per repeat, R repeats, dilation resets
    to 1 at the start of each repeat) estimating C*N mask logits from the
    N-channel encoder representation, per Conv-TasNet Fig. 1(B)."""

    def __init__(self, n_filters: int, b_channels: int, h_channels: int, sc_channels: int, kernel_size: int, x_blocks: int, r_repeats: int, n_sources: int):
        super().__init__()
        self.input_norm = _global_layer_norm(n_filters)
        self.bottleneck = nn.Conv1d(n_filters, b_channels, 1)
        self.blocks = nn.ModuleList([
            _TCNBlock(b_channels, h_channels, sc_channels, kernel_size, dilation=2**x)
            for _r in range(r_repeats)
            for x in range(x_blocks)
        ])
        self.output_prelu = nn.PReLU()
        self.output_conv = nn.Conv1d(sc_channels, n_sources * n_filters, 1)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck(self.input_norm(w))
        skip_sum = 0.0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        return self.output_conv(self.output_prelu(skip_sum))


class ConvTasNetLite(nn.Module):
    """
    Full encoder -> TCN separator -> decoder model, Conv-TasNet's own
    architecture (Fig. 1(A)/(B)) with the "lite" sizing above. Mask
    activation is an independent sigmoid per source (no unit-sum
    constraint) -- Conv-TasNet's own ablation (Sec. IV-A) found sigmoid
    masks performed at least as well as a softmax unit-sum constraint, and
    heart+lung need not sum back to the exact mixture here (there is also
    a noise/residual component in this dataset's mixtures, see
    load_dataset.verify_additive_triplets).
    """

    def __init__(
        self,
        n_filters: int = ENCODER_N,
        kernel_length: int = ENCODER_L,
        stride: int = ENCODER_STRIDE,
        b_channels: int = TCN_B,
        h_channels: int = TCN_H,
        sc_channels: int = TCN_SC,
        tcn_kernel_size: int = TCN_P,
        x_blocks: int = TCN_X,
        r_repeats: int = TCN_R,
        n_sources: int = N_SOURCES,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.n_sources = n_sources
        self.encoder = nn.Conv1d(1, n_filters, kernel_length, stride=stride, bias=False)
        self.separator = _TemporalConvNet(n_filters, b_channels, h_channels, sc_channels, tcn_kernel_size, x_blocks, r_repeats, n_sources)
        self.decoder = nn.ConvTranspose1d(n_filters, 1, kernel_length, stride=stride, bias=False)

    def forward(self, mixed: torch.Tensor) -> torch.Tensor:
        """mixed: (batch, T) -> (batch, n_sources, T), channel 0 = heart, 1 = lung."""
        batch, length = mixed.shape
        w = F.relu(self.encoder(mixed.unsqueeze(1)))  # (batch, N, T')
        mask_logits = self.separator(w)  # (batch, C*N, T')
        t_frames = w.shape[-1]
        masks = torch.sigmoid(mask_logits.view(batch, self.n_sources, self.n_filters, t_frames))
        d = w.unsqueeze(1) * masks  # (batch, C, N, T')
        d = d.reshape(batch * self.n_sources, self.n_filters, t_frames)
        est = self.decoder(d).squeeze(1).view(batch, self.n_sources, -1)  # (batch, C, T'')

        # The encoder/decoder pair is not an exact inverse for arbitrary T
        # (Conv-TasNet's own "valid"-convolution encoder), so trim/pad the
        # reconstruction back to the input length rather than assume they match.
        if est.shape[-1] > length:
            est = est[..., :length]
        elif est.shape[-1] < length:
            est = F.pad(est, (0, length - est.shape[-1]))
        return est


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-invariant SDR (Conv-TasNet Eq. 15 / NeoSSNet Eq. 1), batched over
    the leading dimension(s). Both signals are zero-mean centered first
    (this dataset's recordings, unlike the speech Conv-TasNet was
    developed on, are not guaranteed DC-free).

    estimate, target: (..., T) -> (...) SI-SDR in dB, one value per
    leading-dimension slice.
    """
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    alpha = (estimate * target).sum(dim=-1, keepdim=True) / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
    s_target = alpha * target
    e_noise = estimate - s_target
    ratio = s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def _crop_or_pad(y: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(y) >= length:
        start = int(rng.integers(0, len(y) - length + 1))
        return y[start : start + length]
    return np.pad(y, (0, length - len(y)))


def _sample_batch(
    heart_ids: list,
    lung_ids: list,
    heart_cache: dict,
    lung_cache: dict,
    batch_size: int,
    crop_len: int,
    rng: np.random.Generator,
    gain_range: tuple[float, float],
    snr_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One augmented training/validation batch: random (heart, lung) pairing
    with replacement, random crop offset per recording, and the same
    mixed = a*(heart+lung) + noise recipe synthetic_mix.py uses for
    evaluation -- a is log-uniform over the native-additive-row gain range,
    noise is drawn to hit a uniformly-sampled target SNR. Reusing this
    exact recipe (not inventing a separate one) keeps train-time mixtures
    drawn from the same distribution the model is evaluated against.
    """
    log_a_min, log_a_max = np.log(gain_range[0]), np.log(gain_range[1])
    mixed_batch, heart_batch, lung_batch = [], [], []
    for _ in range(batch_size):
        heart_y, _sr = heart_cache[heart_ids[rng.integers(len(heart_ids))]]
        lung_y, _sr = lung_cache[lung_ids[rng.integers(len(lung_ids))]]
        heart_c = _crop_or_pad(heart_y, crop_len, rng)
        lung_c = _crop_or_pad(lung_y, crop_len, rng)

        gain = float(np.exp(rng.uniform(log_a_min, log_a_max)))
        base = gain * (heart_c + lung_c)
        snr_db = float(rng.uniform(*snr_range))
        noise_rms = _rms(base) / (10 ** (snr_db / 20.0))
        mixed = base + rng.normal(0.0, noise_rms, size=crop_len)

        mixed_batch.append(mixed)
        heart_batch.append(heart_c)
        lung_batch.append(lung_c)

    return np.stack(mixed_batch), np.stack(heart_batch), np.stack(lung_batch)


def _load_pool_cache(df, id_col: str) -> dict:
    return {row[id_col]: load_audio(row["audio_path"], sr=None) for _, row in df.iterrows()}


def _train_convtasnet(
    hs_allowed,
    ls_allowed,
    seed: int = 0,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    batch_size: int = BATCH_SIZE,
    crop_seconds: float = CROP_SECONDS,
):
    """
    Train one ConvTasNetLite from scratch on hs_allowed/ls_allowed's
    isolated recordings (this fold's leakage-safe dictionary pool, in the
    same hs_allowed/ls_allowed format every other baseline's fit_fn
    receives -- see baselines.py/eval_harness.py). Since Conv-TasNet needs
    paired (mixed, heart, lung) examples rather than a spectral dictionary,
    training mixtures are synthesized on the fly (see _sample_batch)
    instead of relying on native Mix.csv pairs, which are both too few
    (145, only 36 additive) and would leak the eval fold's own mixtures.

    A file-level 80/20 train/val split is carved out of hs_allowed/
    ls_allowed themselves (never touching the outer CV fold's held-out
    data) for early stopping and LR scheduling on validation SI-SDR, per
    NeoSSNet's own training recipe (AdamW, LR halved after
    LR_PATIENCE epochs without improvement, best-checkpoint restore rather
    than returning the final epoch's weights).

    Returns (model in eval() mode, sample_rate, train_info dict).
    """
    from synthetic_mix import SNR_SWEEP_DB, _native_gain_range

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    heart_id_col = "Heart Sound ID" if "Heart Sound ID" in hs_allowed.columns else hs_allowed.columns[0]
    lung_id_col = "Lung Sound ID" if "Lung Sound ID" in ls_allowed.columns else ls_allowed.columns[0]
    heart_cache = _load_pool_cache(hs_allowed, heart_id_col)
    lung_cache = _load_pool_cache(ls_allowed, lung_id_col)

    heart_ids = list(heart_cache)
    lung_ids = list(lung_cache)
    rng.shuffle(heart_ids)
    rng.shuffle(lung_ids)
    n_val_heart = max(1, int(round(len(heart_ids) * VAL_FRACTION)))
    n_val_lung = max(1, int(round(len(lung_ids) * VAL_FRACTION)))
    val_heart_ids, train_heart_ids = heart_ids[:n_val_heart], heart_ids[n_val_heart:]
    val_lung_ids, train_lung_ids = lung_ids[:n_val_lung], lung_ids[n_val_lung:]
    if not train_heart_ids or not train_lung_ids:
        raise ValueError("Allowed pool too small to carve out a train/val split for Baseline 6 training.")

    sr = next(iter(heart_cache.values()))[1]
    crop_len = int(crop_seconds * sr)
    gain_range = _native_gain_range()
    snr_range = (min(SNR_SWEEP_DB), max(SNR_SWEEP_DB))

    val_rng = np.random.default_rng(seed + 1)  # fixed validation batch, drawn once (not resampled every epoch)
    val_mixed, val_heart, val_lung = _sample_batch(
        val_heart_ids, val_lung_ids, heart_cache, lung_cache, VAL_PAIRS, crop_len, val_rng, gain_range, snr_range
    )
    val_mixed_t = torch.as_tensor(val_mixed, dtype=torch.float32, device=device)
    val_heart_t = torch.as_tensor(val_heart, dtype=torch.float32, device=device)
    val_lung_t = torch.as_tensor(val_lung, dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    model = ConvTasNetLite().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)

    best_val_si_sdr = float("-inf")
    best_state = None
    epochs_no_improve = 0
    epoch = 0
    for epoch in range(max_epochs):
        model.train()
        for _step in range(steps_per_epoch):
            mixed, heart, lung = _sample_batch(
                train_heart_ids, train_lung_ids, heart_cache, lung_cache, batch_size, crop_len, rng, gain_range, snr_range
            )
            mixed_t = torch.as_tensor(mixed, dtype=torch.float32, device=device)
            heart_t = torch.as_tensor(heart, dtype=torch.float32, device=device)
            lung_t = torch.as_tensor(lung, dtype=torch.float32, device=device)

            est = model(mixed_t)
            loss = -0.5 * (si_sdr(est[:, 0], heart_t).mean() + si_sdr(est[:, 1], lung_t).mean())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_est = model(val_mixed_t)
            val_si_sdr = 0.5 * (si_sdr(val_est[:, 0], val_heart_t).mean() + si_sdr(val_est[:, 1], val_lung_t).mean())
        val_si_sdr = float(val_si_sdr.item())

        if val_si_sdr > best_val_si_sdr:
            best_val_si_sdr = val_si_sdr
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve % LR_PATIENCE == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= LR_DECAY
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    train_info = {
        "epochs_trained": epoch + 1,
        "best_val_si_sdr": best_val_si_sdr,
        "n_train_heart": len(train_heart_ids),
        "n_train_lung": len(train_lung_ids),
        "n_val_heart": len(val_heart_ids),
        "n_val_lung": len(val_lung_ids),
        "device": str(device),
    }
    return model, sr, train_info


def make_convtasnet_baseline(
    seed: int = 0,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    batch_size: int = BATCH_SIZE,
):
    """
    fit_and_separate_fn factory for eval_harness.cross_validate /
    synthetic_mix.evaluate_synthetic, matching baselines.py's contract.
    Unlike Baselines 1-5, this baseline is not zero-training and not a
    fixed-dictionary fit: it trains a fresh ConvTasNetLite from scratch on
    each fold's hs_allowed/ls_allowed pool (see _train_convtasnet), so
    running a full k-fold CV retrains the network k times -- this is
    deliberate (matches every other baseline's leakage-safe-per-fold
    contract) but is the dominant cost of running this baseline; see
    baseline6_report.py for measured wall-clock.

    The returned separate_fn stashes the training run's diagnostics
    (final epoch count, best validation SI-SDR, train/val pool sizes) on
    `separate_fn.train_info` for reporting, since "how well did training
    itself go" is a materially different question from "how good is the
    trained model's separation" and both are worth keeping.
    """

    def fit_and_separate(hs_allowed, ls_allowed):
        model, _sr, train_info = _train_convtasnet(
            hs_allowed, ls_allowed, seed=seed, device=device, max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch, batch_size=batch_size,
        )
        model_device = next(model.parameters()).device

        def separate(mixed: np.ndarray, _sr: int):
            with torch.no_grad():
                x = torch.as_tensor(mixed, dtype=torch.float32, device=model_device).unsqueeze(0)
                est = model(x).cpu().numpy()[0]
            n = len(mixed)
            heart_est = est[0][:n] if est.shape[-1] >= n else np.pad(est[0], (0, n - est.shape[-1]))
            lung_est = est[1][:n] if est.shape[-1] >= n else np.pad(est[1], (0, n - est.shape[-1]))
            return heart_est, lung_est

        separate.train_info = train_info
        return separate

    return fit_and_separate
