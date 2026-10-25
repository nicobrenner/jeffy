"""Tests for the binary-flag tuning options added after external feedback on
rare-positive flags (class_weight, positive-class F1 scoring, CV fold count,
and per-flag threshold tuning).

These exercise the tuning logic directly against precomputed embeddings so the
suite does not need to download an encoder.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _rare_flag_data(n=400, n_pos=20, offset=0.8, dims=4, seed=0):
    """Embedding-like features where the positive class is a small, *overlapping*
    cluster.

    The overlap matters: with a cleanly separable positive class every option
    scores perfectly and the trade-offs these tests check disappear. At this
    offset the shape matches what was reported on real data, where balanced
    weighting raises many more flags than the unweighted fit.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 24))
    y = ["no"] * n
    for i in range(n_pos):
        X[i, :dims] += offset
        y[i] = "yes"
    return X, y


class TestFoldCount:
    """The old code used cv=min(3, len(set(labels))), which for a binary flag is
    always 2 because it counts classes, not folds."""

    def test_old_expression_never_yielded_three_folds(self):
        for labels in (["yes", "no"], ["yes"] * 50 + ["no"] * 50):
            assert min(3, len(set(labels))) == 2

    def test_new_expression_respects_the_rarer_class(self):
        def n_folds(n_pos, n_neg):
            return max(2, min(3, n_pos, n_neg))

        assert n_folds(50, 50) == 3
        assert n_folds(2, 300) == 2
        assert n_folds(1, 300) == 2  # clamped; caller gates on n_pos >= 2

    def test_three_folds_with_two_positives_starves_a_fold(self):
        """Why the clamp matters: 3 folds over 2 positives leaves a fold with none."""
        X, y = _rare_flag_data(n=60, n_pos=2)
        skf = StratifiedKFold(n_splits=3)
        with pytest.warns(UserWarning):
            folds = list(skf.split(X, y))
        starved = [
            i for i, (_, test_idx) in enumerate(folds)
            if "yes" not in [y[j] for j in test_idx]
        ]
        assert starved, "expected at least one fold with no positives"

    def test_clamped_folds_all_contain_positives(self):
        X, y = _rare_flag_data(n=60, n_pos=2)
        n_pos, n_neg = y.count("yes"), y.count("no")
        skf = StratifiedKFold(n_splits=max(2, min(3, n_pos, n_neg)))
        for _, test_idx in skf.split(X, y):
            assert "yes" in [y[j] for j in test_idx]


class TestScorer:
    def test_macro_f1_is_inflated_by_the_easy_negative_class(self):
        X, y = _rare_flag_data()
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
        preds = clf.predict(X)
        macro = f1_score(y, preds, average="macro", zero_division=0)
        positive = f1_score(y, preds, pos_label="yes", zero_division=0)
        assert macro > positive, "macro should sit above positive-class F1 here"

    def test_positive_scorer_is_usable_with_string_labels(self):
        """sklearn's built-in 'f1' scorer assumes pos_label=1 and raises on
        yes/no labels, which is why this needs make_scorer."""
        X, y = _rare_flag_data()
        scorer = make_scorer(f1_score, pos_label="yes", zero_division=0)
        scores = cross_val_score(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            X, y, cv=3, scoring=scorer)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_builtin_f1_scorer_rejects_string_labels(self):
        X, y = _rare_flag_data()
        with pytest.raises(ValueError):
            cross_val_score(
                LogisticRegression(max_iter=2000), X, y, cv=3, scoring="f1",
                error_score="raise")


class TestClassWeight:
    def test_balanced_raises_more_flags_than_unweighted(self):
        """The trade-off the class_weight parameter exists to expose."""
        X, y = _rare_flag_data(n=400, n_pos=20)
        balanced = LogisticRegression(
            max_iter=3000, C=0.01, class_weight="balanced").fit(X, y)
        unweighted = LogisticRegression(max_iter=3000, C=0.01).fit(X, y)

        n_balanced = list(balanced.predict(X)).count("yes")
        n_unweighted = list(unweighted.predict(X)).count("yes")
        assert n_balanced > n_unweighted

    def test_none_is_a_distinct_fit(self):
        X, y = _rare_flag_data()
        a = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced").fit(X, y)
        b = LogisticRegression(max_iter=3000, C=0.1, class_weight=None).fit(X, y)
        assert not np.allclose(a.coef_, b.coef_)


class TestThresholdFolding:
    """The tuned threshold is folded into the intercept so it survives save()."""

    @pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.65, 0.9, 0.95])
    def test_shifted_intercept_matches_probability_threshold(self, threshold):
        import copy

        X, y = _rare_flag_data()
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X, y)
        yes_col = list(clf.classes_).index("yes")
        p_yes = clf.predict_proba(X)[:, yes_col]
        by_threshold = np.array(["yes" if p >= threshold else "no" for p in p_yes])

        shifted = copy.deepcopy(clf)
        shifted.intercept_ = clf.intercept_ - np.log(threshold / (1 - threshold))

        assert (by_threshold == shifted.predict(X)).all()

    def test_positive_class_is_second_in_classes(self):
        """The fold assumes binary coef_/intercept_ describe classes_[1]."""
        X, y = _rare_flag_data()
        clf = LogisticRegression(max_iter=2000).fit(X, y)
        assert list(clf.classes_) == ["no", "yes"]
        assert clf.coef_.shape[0] == 1

    def test_higher_threshold_raises_fewer_flags(self):
        import copy

        X, y = _rare_flag_data()
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X, y)
        counts = []
        for t in (0.2, 0.5, 0.8):
            s = copy.deepcopy(clf)
            s.intercept_ = clf.intercept_ - np.log(t / (1 - t))
            counts.append(list(s.predict(X)).count("yes"))
        assert counts[0] >= counts[1] >= counts[2]


class TestSignature:
    def test_new_parameters_exist_with_documented_defaults(self):
        import inspect

        from jeffy.train import train_binary_flags

        params = inspect.signature(train_binary_flags).parameters
        assert params["class_weight"].default == "balanced"
        assert params["tune_threshold"].default is False
        assert params["min_threshold_positives"].default == 8

    def test_docstring_explains_the_scorer_choice(self):
        from jeffy.train import train_binary_flags

        doc = train_binary_flags.__doc__
        assert "positive-class f1" in doc.lower()


def _flag_corpus():
    """A rare flag with enough positives to allow threshold tuning."""
    urgent = [
        "I need this resolved immediately",
        "this is an emergency, please call me",
        "urgent: my account is locked and I cannot pay rent",
        "critical issue, I have been waiting all day",
        "please escalate this right now",
        "this cannot wait another day",
        "I need someone to call me urgently",
        "emergency, my card is blocked abroad",
        "urgent help needed with my payment",
        "this is time critical, please respond",
        "I am in a real hurry with this one",
        "please treat this as top priority",
    ]
    routine = [
        "what are your opening hours",
        "how do I change my profile picture",
        "I was wondering about your fees",
        "can you explain how transfers work",
        "just checking my balance",
        "where can I find my statements",
        "is there a mobile app",
        "how long do transfers usually take",
        "I would like to know about savings",
        "can I add a second card",
        "what currencies do you support",
        "how do I update my address",
        "do you have a referral scheme",
        "what is the interest rate",
        "can I see last month's transactions",
        "how do I contact support",
        "is there a fee for withdrawals",
        "I want to read the terms",
        "how do I set up notifications",
        "can I change my statement date",
        "what documents do you need",
        "how do I close a joint account",
        "is my deposit insured",
        "can I use the card overseas",
        "how do I order a replacement",
        "what is my account number",
        "do you offer business accounts",
        "how do I enable dark mode",
    ]
    texts = urgent + routine
    labels = ["yes"] * len(urgent) + ["no"] * len(routine)
    return texts, labels


def _big_flag_corpus():
    """Same shape as _flag_corpus but large enough that the dev slice clears the
    default min_threshold_positives of 8, which needs roughly 32 positives in
    train because dev is a quarter of it."""
    urgent_seeds = [
        "I need this resolved immediately",
        "this is an emergency, please call me",
        "urgent, my account is locked",
        "critical issue, I have waited all day",
        "please escalate this right now",
        "this cannot wait another day",
        "I need someone to call me urgently",
        "emergency, my card is blocked abroad",
    ]
    routine_seeds = [
        "what are your opening hours",
        "how do I change my profile picture",
        "I was wondering about your fees",
        "can you explain how transfers work",
        "just checking my balance",
        "where can I find my statements",
        "is there a mobile app",
        "how long do transfers usually take",
    ]
    # Suffixes keep the texts distinct without changing their class.
    suffixes = ["", " please", " thanks", " today", " again", " for me",
                " as soon as possible", " if you can"]
    urgent = [s + suf for s in urgent_seeds for suf in suffixes[:6]]
    routine = [s + suf for s in routine_seeds for suf in suffixes]
    texts = urgent + routine
    labels = ["yes"] * len(urgent) + ["no"] * len(routine)
    assert labels.count("yes") >= 32, labels.count("yes")
    return texts, labels


class TestEndToEnd:
    """Exercises the real function with a small encoder (same one the other
    training tests use, so it is already cached)."""

    ENCODER = "all-MiniLM-L6-v2"

    def test_class_weight_none_raises_fewer_flags(self):
        from jeffy.train import train_binary_flags

        texts, labels = _flag_corpus()
        probes = ["this is an emergency", "what are your opening hours",
                  "I would like some information", "please help me quickly"]

        counts = {}
        for weight in ("balanced", None):
            res = train_binary_flags(
                texts, {"urgent": labels}, task_id="t", test_size=0,
                verbose=False, encoder=self.ENCODER, class_weight=weight)
            tc = res["urgent"]
            counts[str(weight)] = sum(
                1 for p in probes if tc.predict(p)["label"] == "yes")

        assert counts["balanced"] >= counts["None"]

    def test_tune_threshold_runs_and_shifts_the_intercept(self, capsys):
        from jeffy.train import train_binary_flags

        texts, labels = _big_flag_corpus()
        plain = train_binary_flags(
            texts, {"urgent": labels}, task_id="t", test_size=0, verbose=False,
            encoder=self.ENCODER, C=0.1)["urgent"]
        tuned = train_binary_flags(
            texts, {"urgent": labels}, task_id="t", test_size=0, verbose=True,
            encoder=self.ENCODER, C=0.1, tune_threshold=True,
            min_threshold_positives=8)["urgent"]
        out = capsys.readouterr().out

        assert "dev positives" in out, out
        assert "left at 0.5" not in out, out

        # Same C and same data, so the coefficients must match exactly and the
        # only difference is the folded threshold in the intercept.
        assert np.allclose(plain.clf.coef_, tuned.clf.coef_)
        assert not np.allclose(plain.clf.intercept_, tuned.clf.intercept_), (
            "tuning reported a threshold but the intercept was not shifted")

    def test_intercept_shift_corresponds_to_a_grid_threshold(self):
        """The shift must be exactly the logit of one of the searched thresholds.

        Note this does not assert that predictions change: on a cleanly separable
        corpus every threshold in the grid yields the same decisions. Whether a
        shift flips decisions on overlapping data is covered by
        TestThresholdFolding.
        """
        from jeffy.train import train_binary_flags

        texts, labels = _big_flag_corpus()
        kw = dict(task_id="t", test_size=0, verbose=False,
                  encoder=self.ENCODER, C=0.1)
        plain = train_binary_flags(texts, {"urgent": labels}, **kw)["urgent"]
        tuned = train_binary_flags(
            texts, {"urgent": labels}, tune_threshold=True,
            min_threshold_positives=8, **kw)["urgent"]

        shift = float(plain.clf.intercept_[0] - tuned.clf.intercept_[0])
        grid = [round(0.05 * i, 2) for i in range(2, 19)]
        implied = 1.0 / (1.0 + np.exp(-shift))
        assert any(abs(implied - t) < 1e-6 for t in grid), (
            f"shift {shift} implies threshold {implied}, not in the search grid")

    def test_threshold_tuning_skipped_without_enough_positives(self, capsys):
        from jeffy.train import train_binary_flags

        texts, labels = _flag_corpus()
        train_binary_flags(
            texts, {"urgent": labels}, task_id="t", test_size=0, verbose=True,
            encoder=self.ENCODER, tune_threshold=True,
            min_threshold_positives=99)
        out = capsys.readouterr().out
        assert "left at 0.5" in out

    def test_tune_C_reports_fold_count(self, capsys):
        from jeffy.train import train_binary_flags

        texts, labels = _flag_corpus()
        train_binary_flags(
            texts, {"urgent": labels}, task_id="t", test_size=0, verbose=True,
            encoder=self.ENCODER, tune_C=True)
        out = capsys.readouterr().out
        assert "3-fold CV F1+" in out, out

    def test_tune_C_skipped_when_a_class_is_too_small(self, capsys):
        from jeffy.train import train_binary_flags

        texts, labels = _flag_corpus()
        labels = ["yes"] + ["no"] * (len(texts) - 1)
        train_binary_flags(
            texts, {"urgent": labels}, task_id="t", test_size=0, verbose=True,
            encoder=self.ENCODER, tune_C=True)
        out = capsys.readouterr().out
        assert "tune_C: skipped" in out, out

    def test_folded_threshold_survives_save_and_load(self, tmp_path):
        from jeffy.model_pack import load_artifact
        from jeffy.train import train_binary_flags

        texts, labels = _flag_corpus()
        tc = train_binary_flags(
            texts, {"urgent": labels}, task_id="saveme", test_size=0,
            verbose=False, encoder=self.ENCODER, C=0.1, tune_threshold=True,
            min_threshold_positives=3)["urgent"]
        tc.save(str(tmp_path))

        clf, scaler, _ = load_artifact(tmp_path / "saveme_flag_urgent")
        assert np.allclose(clf.intercept_, tc.clf.intercept_)
        assert np.allclose(clf.coef_, tc.clf.coef_)
