"""
Unit tests for heart_classifier_cnn.py (S7-06, the second architecture for
the C2 knee-point robustness check). Uses max_epochs=2-3 throughout
(train_fold_classifiers'/make_classifier's override, mirroring
convtasnet.py's own testability pattern) so these run fast -- the
properties checked here (shapes, the backend contract, reproducibility)
don't depend on how well-trained the model actually is.
"""
import numpy as np
import pytest

from heart_classifier import CLASS_GROUPS, assign_classifier_folds
from heart_classifier_cnn import (
    CNNClassifier,
    N_MELS,
    cross_validate_classifier,
    extract_features,
    make_classifier,
    predict_one,
    train_fold_classifiers,
)
from load_dataset import load_audio, load_hs

FAST_EPOCHS = 2


class TestExtractFeatures:
    def test_feature_shape(self):
        hs_df = load_hs()
        y, sr = load_audio(hs_df.loc[0, "audio_path"], sr=None)
        features = extract_features(y, sr)
        assert features.shape[0] == N_MELS
        assert features.ndim == 2
        assert np.all(np.isfinite(features))

    def test_all_recordings_produce_the_same_shape(self):
        """Every HS.csv recording is exactly 60,000 samples (module
        docstring's claim) -- if that ever stops being true, feature
        shapes would stop matching and CNNClassifier.fit would break on a
        ragged stack, so this is worth checking directly."""
        hs_df = load_hs()
        shapes = set()
        for path in hs_df["audio_path"].iloc[:5]:
            y, sr = load_audio(path, sr=None)
            shapes.add(extract_features(y, sr).shape)
        assert len(shapes) == 1


class TestCNNClassifier:
    def test_fit_predict_roundtrip_on_synthetic_data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, N_MELS, 20)).astype(np.float32)
        y = np.array(["Normal", "Murmur", "Extra Sound", "Rhythm Disorder"] * 2)

        clf = CNNClassifier(max_epochs=FAST_EPOCHS)
        clf.fit(X, y)
        preds = clf.predict(X)

        assert len(preds) == len(X)
        assert set(preds) <= set(CLASS_GROUPS)

    def test_predict_before_fit_raises(self):
        clf = CNNClassifier(max_epochs=FAST_EPOCHS)
        with pytest.raises(AssertionError):
            clf.predict(np.zeros((1, N_MELS, 20), dtype=np.float32))

    def test_seed_gives_reproducible_predictions(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, N_MELS, 20)).astype(np.float32)
        y = np.array(["Normal", "Murmur", "Extra Sound", "Rhythm Disorder"] * 2)

        clf_a = CNNClassifier(max_epochs=FAST_EPOCHS, seed=42).fit(X, y)
        clf_b = CNNClassifier(max_epochs=FAST_EPOCHS, seed=42).fit(X, y)
        np.testing.assert_array_equal(clf_a.predict(X), clf_b.predict(X))


class TestBackendContract:
    """heart_classifier_cnn.py must satisfy the same train_fold_classifiers/
    predict_one contract heart_classifier.py does, since condition_b.py/
    sdr_accuracy_curve.py call either backend generically."""

    def test_train_fold_classifiers_returns_one_classifier_per_fold(self):
        hs_df = assign_classifier_folds(n_folds=2, seed=0)
        classifiers = train_fold_classifiers(hs_df, n_folds=2, max_epochs=FAST_EPOCHS)
        assert set(classifiers.keys()) == {0, 1}
        assert all(isinstance(clf, CNNClassifier) for clf in classifiers.values())

    def test_predict_one_returns_a_valid_class_group(self):
        hs_df = assign_classifier_folds(n_folds=2, seed=0)
        classifiers = train_fold_classifiers(hs_df, n_folds=2, max_epochs=FAST_EPOCHS)
        y, sr = load_audio(load_hs().loc[0, "audio_path"], sr=None)
        pred = predict_one(classifiers[0], y, sr)
        assert pred in CLASS_GROUPS


class TestCrossValidateClassifier:
    def test_every_recording_predicted_exactly_once(self):
        predictions_df, fold_summary, cv_summary = cross_validate_classifier(
            n_folds=2, seed=0, max_epochs=FAST_EPOCHS
        )

        assert len(predictions_df) == len(load_hs())
        assert predictions_df["Heart Sound ID"].is_unique
        assert fold_summary["accuracy"].between(0, 1).all()
        assert set(cv_summary.index) == {"accuracy", "macro_f1"}
