"""
Leakage-safe, triplet-level train/eval split for the HLS-CMDS mix set.

See code_description.md for why a naive split leaks ground truth into
training and how "leak groups" are computed and assigned to folds.

Usage:
    from split import assign_folds, dictionary_pool

    mix_df = assign_folds(n_folds=5, seed=0)   # adds 'leak_group' and 'fold' columns
    for k in range(5):
        eval_rows = mix_df[mix_df["fold"] == k]
        hs_allowed, ls_allowed = dictionary_pool(hs_df, ls_df, mix_df, held_out_fold=k)
        # fit a dictionary on hs_allowed/ls_allowed only, then evaluate on eval_rows
"""
import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd
import soundfile as sf

from load_dataset import load_hs, load_ls, load_mix


@lru_cache(maxsize=None)
def _content_hash(path: str) -> str:
    """Hash raw PCM samples (not the file bytes) so header/metadata differences don't matter."""
    data, _ = sf.read(path, dtype="int16", always_2d=True)
    return hashlib.md5(data.tobytes()).hexdigest()


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _compute_leak_groups(mix_df: pd.DataFrame) -> np.ndarray:
    """
    Connected-component id per Mix.csv row: two rows are in the same group iff
    they share a heart_hash or a lung_hash (directly or transitively).
    """
    uf = _UnionFind(len(mix_df))
    by_hash = {}
    for col in ("heart_hash", "lung_hash"):
        for i, h in enumerate(mix_df[col]):
            if h in by_hash:
                uf.union(by_hash[h], i)
            else:
                by_hash[h] = i
        by_hash.clear()  # heart/lung hash namespaces don't collide with each other

    roots = [uf.find(i) for i in range(len(mix_df))]
    renumber = {root: gid for gid, root in enumerate(dict.fromkeys(roots))}
    return np.array([renumber[r] for r in roots])


def with_content_hashes(mix_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """load_mix()'s DataFrame, plus heart_hash/lung_hash columns (content hash of the audio)."""
    mix_df = (mix_df if mix_df is not None else load_mix()).copy()
    mix_df["heart_hash"] = mix_df["heart_audio_path"].map(_content_hash)
    mix_df["lung_hash"] = mix_df["lung_audio_path"].map(_content_hash)
    return mix_df


def assign_folds(mix_df: pd.DataFrame | None = None, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """
    Partition Mix.csv rows into n_folds folds at the leak-group level (every
    row in the same leak group gets the same fold, so no shared recording
    crosses a fold boundary), balanced across folds via greedy
    largest-group-to-smallest-fold assignment.

    Returns mix_df with 'heart_hash', 'lung_hash', 'leak_group', and 'fold' columns added.
    """
    mix_df = with_content_hashes(mix_df)
    mix_df["leak_group"] = _compute_leak_groups(mix_df)

    group_sizes = mix_df.groupby("leak_group").size()
    group_ids = group_sizes.index.to_numpy().copy()
    np.random.default_rng(seed).shuffle(group_ids)  # randomize tie order among equal-size groups
    group_ids = sorted(group_ids, key=lambda g: -group_sizes[g])

    fold_totals = [0] * n_folds
    group_to_fold = {}
    for g in group_ids:
        k = int(np.argmin(fold_totals))
        group_to_fold[g] = k
        fold_totals[k] += group_sizes[g]

    mix_df["fold"] = mix_df["leak_group"].map(group_to_fold)
    return mix_df


def dictionary_pool(
    hs_df: pd.DataFrame, ls_df: pd.DataFrame, mix_df_with_folds: pd.DataFrame, held_out_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    HS.csv / LS.csv rows safe to fit a dictionary on while evaluating `held_out_fold`:
    every row whose audio content also appears (by content hash) in that fold's
    mix rows is excluded. Recordings never reused in any mixture are unaffected
    and remain available in every fold.
    """
    held_out = mix_df_with_folds[mix_df_with_folds["fold"] == held_out_fold]
    excluded_heart = set(held_out["heart_hash"])
    excluded_lung = set(held_out["lung_hash"])

    hs_hashes = hs_df["audio_path"].map(_content_hash)
    ls_hashes = ls_df["audio_path"].map(_content_hash)

    hs_allowed = hs_df[~hs_hashes.isin(excluded_heart)].reset_index(drop=True)
    ls_allowed = ls_df[~ls_hashes.isin(excluded_lung)].reset_index(drop=True)
    return hs_allowed, ls_allowed


if __name__ == "__main__":
    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    print("Assigning leakage-safe folds...")
    mix_df = assign_folds(n_folds=5, seed=0)
    hs_df, ls_df = load_hs(), load_ls()

    n_groups = mix_df["leak_group"].nunique()
    group_sizes = sorted(mix_df.groupby("leak_group").size().tolist(), reverse=True)
    fold_sizes = mix_df.groupby("fold").size().rename("mix_rows").to_frame()

    print("Computing per-fold dictionary pools...")
    pool_rows = []
    for k in sorted(mix_df["fold"].unique()):
        hs_allowed, ls_allowed = dictionary_pool(hs_df, ls_df, mix_df, held_out_fold=k)
        pool_rows.append({
            "fold": k,
            "eval_rows": int((mix_df["fold"] == k).sum()),
            "hs_allowed": len(hs_allowed),
            "hs_total": len(hs_df),
            "ls_allowed": len(ls_allowed),
            "ls_total": len(ls_df),
        })
    pool_df = pd.DataFrame(pool_rows).set_index("fold")

    stat_tiles = "\n".join([
        stat_tile("Leak groups", str(n_groups), f"over {len(mix_df)} mix rows"),
        stat_tile("Largest group", str(group_sizes[0]), "mix rows"),
        stat_tile("Folds", str(mix_df["fold"].nunique()), "folds"),
    ])

    body = "\n\n".join([
        section("Fold sizes", "mix rows per fold", df_to_html(fold_sizes, index_label="fold")),
        section(
            "Leak group sizes",
            f"{n_groups} groups, largest first",
            f'<p class="mono-block">{", ".join(map(str, group_sizes))}</p>',
        ),
        section(
            "Dictionary pool per fold",
            "HS/LS recordings left after excluding the held-out fold's content",
            df_to_html(pool_df, index_label="fold"),
        ),
    ])

    html = report_shell(
        title="Fold Split Report",
        eyebrow="HLS-CMDS · leakage-safe split",
        heading="Leak-group fold assignment",
        dek=(
            "Mix.csv rows grouped into leak-safe folds (connected components by shared "
            "heart/lung recording); per-fold dictionary pool shows how many HS/LS "
            "recordings remain once anything reused in the held-out fold is excluded."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer="<p><strong>Method.</strong> See <code>split.py</code> / <code>code_description.md</code>.</p>",
    )

    report_path = write_report(results_dir() / "split_report.html", html)
    print(f"Report written to {report_path}")
