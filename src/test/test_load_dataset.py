"""
Unit tests for load_dataset.py's verify_additive_triplets(): the check that
a Mix.csv row's mixed recording is actually mixed ~= a*(heart+lung), not
just that the three files exist and share format (verify_mix_triplets()
already covers that). See TEEP2026_Sprint0_Review: on the GitHub copy of
this dataset, 109/145 rows failed this test despite passing every check
verify_mix_triplets() runs.
"""
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from load_dataset import load_mix, verify_additive_triplets


class TestSyntheticClosedForm:
    """Construct heart/lung/mixed signals with a known-true answer, independent of any real dataset copy."""

    def _write_row(self, tmp_path, name, heart, lung, mixed, sr=4000):
        # subtype="FLOAT": these signals aren't scaled to fit [-1, 1] (mixed
        # in particular can exceed it), and this test is about verify_additive_triplets()'s
        # math, not 16-bit PCM quantization/clipping behavior.
        heart_path, lung_path, mixed_path = (tmp_path / f"{name}_{part}.wav" for part in ("h", "l", "m"))
        sf.write(heart_path, heart.astype(np.float32), sr, subtype="FLOAT")
        sf.write(lung_path, lung.astype(np.float32), sr, subtype="FLOAT")
        sf.write(mixed_path, mixed.astype(np.float32), sr, subtype="FLOAT")
        return {
            "Mixed Sound ID": name,
            "heart_audio_path": str(heart_path),
            "lung_audio_path": str(lung_path),
            "mixed_audio_path": str(mixed_path),
        }

    def test_exact_additive_mixture_is_flagged_additive(self, tmp_path):
        rng = np.random.default_rng(0)
        heart = rng.standard_normal(4000) * 0.1
        lung = rng.standard_normal(4000) * 0.1
        mixed = 2.5 * (heart + lung)  # a genuine scaled sum, gain a=2.5

        row = self._write_row(tmp_path, "M_good", heart, lung, mixed)
        result = verify_additive_triplets(pd.DataFrame([row]))

        assert result["rows"][0]["additive"] is True
        assert result["rows"][0]["relative_residual"] < 1e-3
        assert result["rows"][0]["gain"] == pytest.approx(2.5, rel=1e-2)
        assert "M_good" in result["valid_ids"]

    def test_unrelated_mixture_is_flagged_non_additive(self, tmp_path):
        rng = np.random.default_rng(1)
        heart = rng.standard_normal(4000) * 0.1
        lung = rng.standard_normal(4000) * 0.1
        mixed = rng.standard_normal(4000) * 0.1  # independent signal, not derived from heart/lung at all

        row = self._write_row(tmp_path, "M_bad", heart, lung, mixed)
        result = verify_additive_triplets(pd.DataFrame([row]))

        assert result["rows"][0]["additive"] is False
        assert result["rows"][0]["relative_residual"] > 0.5
        assert "M_bad" not in result["valid_ids"]

    def test_threshold_is_configurable(self, tmp_path):
        rng = np.random.default_rng(2)
        heart = rng.standard_normal(4000) * 0.1
        lung = rng.standard_normal(4000) * 0.1
        noise = rng.standard_normal(4000) * 0.01  # small perturbation, not exact
        mixed = (heart + lung) + noise

        row = pd.DataFrame([self._write_row(tmp_path, "M_noisy", heart, lung, mixed)])

        strict = verify_additive_triplets(row, residual_threshold=1e-6)
        loose = verify_additive_triplets(row, residual_threshold=0.5)

        assert strict["rows"][0]["additive"] is False
        assert loose["rows"][0]["additive"] is True


class TestRealData:
    """
    Structural checks against whatever dataset copy is actually configured,
    without hardcoding row counts or IDs tied to one specific copy (those
    are expected to change once the Mendeley re-download replaces the
    current GitHub copy -- see TEEP2026_Sprint0_Review action 1/2).
    """

    @pytest.fixture(scope="class")
    @classmethod
    def additivity(cls):
        return verify_additive_triplets(load_mix())

    def test_every_row_gets_scored(self, additivity):
        mix_df = load_mix()
        assert len(additivity["rows"]) == len(mix_df)
        assert additivity["valid_ids"] <= set(mix_df["Mixed Sound ID"])

    def test_residuals_are_well_separated(self, additivity):
        """
        Regardless of which dataset copy is loaded, additive and
        non-additive rows should not sit ambiguously close to the
        threshold -- if they did, residual_threshold would be a real
        judgment call rather than a robust cutoff (see the docstring's
        sensitivity note).
        """
        residuals = [r["relative_residual"] for r in additivity["rows"]]
        additive_residuals = [r for r, row in zip(residuals, additivity["rows"]) if row["additive"]]
        non_additive_residuals = [r for r, row in zip(residuals, additivity["rows"]) if not row["additive"]]

        if additive_residuals and non_additive_residuals:
            assert max(additive_residuals) < 0.1 * min(non_additive_residuals)

    def test_gain_is_finite_and_positive_for_additive_rows(self, additivity):
        for row in additivity["rows"]:
            if row["additive"]:
                assert row["gain"] > 0
                assert np.isfinite(row["gain"])
