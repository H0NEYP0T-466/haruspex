"""Integration test — full HARUSPEX pipeline on a small BODMAS subset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@pytest.mark.timeout(120)
class TestFullPipeline:
    """Integration test: load → train → predict → explain → correlate → report."""

    def test_full_pipeline(self):
        """Run the complete HARUSPEX pipeline on a small data subset."""
        from haruspex.config import get_feature_names, ensure_output_dirs
        from haruspex.correlation.correlator import Correlator
        from haruspex.data.bodmas import load_bodmas
        from haruspex.data.splits import random_split
        from haruspex.forensics.ioc_extractor import IOCExtractor
        from haruspex.forensics.static_analysis import StaticAnalyzer
        from haruspex.models.xgboost_model import XGBoostDetector
        from haruspex.xai.shap_explainer import SHAPExplainer

        ensure_output_dirs()

        # --- Step 1: Load data (small subset) ---
        dataset = load_bodmas()
        rng = np.random.RandomState(42)
        subset_idx = rng.choice(dataset.n_samples, 2000, replace=False)

        X = dataset.X[subset_idx]
        y = dataset.y[subset_idx]
        metadata = dataset.metadata.iloc[subset_idx].reset_index(drop=True)
        feature_names = dataset.feature_names

        assert X.shape[0] == 2000
        assert X.shape[1] == 2381

        # --- Step 2: Train ---
        split = random_split(X, y, metadata, test_size=0.2)
        model = XGBoostDetector(n_estimators=30, max_depth=4)
        metrics = model.fit(split.X_train, split.y_train, split.X_test, split.y_test)

        assert model.is_fitted
        assert "train_accuracy" in metrics

        # --- Step 3: Predict ---
        test_sample = split.X_test[0]
        pred = model.predict(test_sample.reshape(1, -1))
        proba = model.predict_proba(test_sample.reshape(1, -1))

        assert pred.shape == (1,)
        assert 0.0 <= proba[0] <= 1.0

        # --- Step 4: Explain (SHAP) ---
        explainer = SHAPExplainer(model, feature_names=feature_names)
        shap_exp = explainer.explain_sample(test_sample)

        assert shap_exp.shap_values.shape == (2381,)
        assert np.any(shap_exp.shap_values != 0)

        top_features = shap_exp.top_features(k=10)
        assert len(top_features) == 10

        # --- Step 5: Forensic analysis ---
        analyzer = StaticAnalyzer()
        forensic_report = analyzer.analyze_feature_vector(test_sample, feature_names)

        assert forensic_report.is_pe
        assert len(forensic_report.feature_indicators) > 0

        # --- Step 6: IOC extraction ---
        ioc_extractor = IOCExtractor()
        iocs = ioc_extractor.extract_from_analysis(forensic_report)

        assert isinstance(iocs, list)

        # --- Step 7: Correlate ---
        correlator = Correlator()
        evidence = correlator.correlate(shap_exp, forensic_report, iocs, top_k=15)

        assert len(evidence) > 0
        # Should have at least some AI explanation items
        ai_items = [e for e in evidence if e.source == "ai_explanation"]
        assert len(ai_items) > 0

        # --- Step 8: Generate report ---
        prediction = "malware" if pred[0] == 1 else "benign"
        confidence = float(proba[0]) if pred[0] == 1 else float(1.0 - proba[0])

        report = correlator.generate_report(
            sample_id="integration_test",
            prediction=prediction,
            confidence=confidence,
            model_name="xgboost",
            evidence_items=evidence,
            shap_explanation=shap_exp,
            forensic_report=forensic_report,
            iocs=iocs,
        )

        # Verify report
        assert report.prediction in ("malware", "benign")
        assert report.total_evidence > 0

        report_dict = report.to_dict()
        assert "prediction" in report_dict
        assert "evidence_items" in report_dict
        assert len(report_dict["evidence_items"]) > 0

        report_text = report.to_text()
        assert "HARUSPEX" in report_text
        assert "VERDICT" in report_text

        print(f"\n{'='*60}")
        print(f"Integration test passed!")
        print(f"  Prediction: {prediction} ({confidence:.1%})")
        print(f"  Evidence items: {len(evidence)}")
        print(f"  Correlated: {report.correlated_evidence}")
        print(f"  AI-only: {report.ai_only_evidence}")
        print(f"  Forensic-only: {report.forensic_only_evidence}")
        print(f"{'='*60}")
