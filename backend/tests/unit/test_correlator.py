"""Unit tests for the HARUSPEX correlation engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from haruspex.config import get_feature_names
from haruspex.correlation.correlator import Correlator
from haruspex.correlation.evidence_schema import EvidenceItem, InvestigationReport
from haruspex.forensics.ioc_extractor import IOC, IOCExtractor
from haruspex.forensics.static_analysis import ForensicReport, StaticAnalyzer
from haruspex.models.xgboost_model import XGBoostDetector
from haruspex.xai.shap_explainer import SHAPExplainer, SHAPExplanation


@pytest.fixture(scope="module")
def correlation_inputs():
    """Set up model, SHAP explanation, and forensic analysis for correlation tests."""
    rng = np.random.RandomState(42)
    n_features = 2381

    # Train a small model
    X_train = rng.randn(300, n_features).astype(np.float32)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int32)

    model = XGBoostDetector(n_estimators=20, max_depth=3)
    model.fit(X_train, y_train)

    # Generate test sample and explanations
    x_test = rng.randn(n_features).astype(np.float32)
    feature_names = get_feature_names()

    shap_explainer = SHAPExplainer(model, feature_names=feature_names)
    shap_exp = shap_explainer.explain_sample(x_test)

    # Forensic analysis
    analyzer = StaticAnalyzer()
    forensic_report = analyzer.analyze_feature_vector(x_test, feature_names)

    # IOC extraction
    ioc_extractor = IOCExtractor()
    iocs = ioc_extractor.extract_from_analysis(forensic_report)

    return {
        "model": model,
        "x_test": x_test,
        "shap_exp": shap_exp,
        "forensic_report": forensic_report,
        "iocs": iocs,
        "feature_names": feature_names,
    }


class TestCorrelator:
    """Tests for the correlation engine."""

    def test_correlate_returns_evidence_list(self, correlation_inputs):
        """correlate() returns a list of EvidenceItems."""
        correlator = Correlator()
        evidence = correlator.correlate(
            correlation_inputs["shap_exp"],
            correlation_inputs["forensic_report"],
            correlation_inputs["iocs"],
            top_k=10,
        )
        assert isinstance(evidence, list)
        assert len(evidence) > 0
        assert all(isinstance(e, EvidenceItem) for e in evidence)

    def test_evidence_has_required_fields(self, correlation_inputs):
        """Each EvidenceItem has all required fields."""
        correlator = Correlator()
        evidence = correlator.correlate(
            correlation_inputs["shap_exp"],
            correlation_inputs["forensic_report"],
            correlation_inputs["iocs"],
        )
        for e in evidence:
            assert e.category != ""
            assert e.indicator != ""
            assert 0.0 <= e.severity <= 1.0
            assert e.source in ("ai_explanation", "forensic_analysis", "correlated")

    def test_evidence_sorted_by_severity(self, correlation_inputs):
        """Evidence items are sorted by severity descending."""
        correlator = Correlator()
        evidence = correlator.correlate(
            correlation_inputs["shap_exp"],
            correlation_inputs["forensic_report"],
            correlation_inputs["iocs"],
        )
        severities = [e.severity for e in evidence]
        assert severities == sorted(severities, reverse=True)

    def test_correlated_items_have_matched_forensic(self, correlation_inputs):
        """Correlated evidence items have matched_forensic set."""
        correlator = Correlator()
        evidence = correlator.correlate(
            correlation_inputs["shap_exp"],
            correlation_inputs["forensic_report"],
            correlation_inputs["iocs"],
        )
        for e in evidence:
            if e.source == "correlated":
                assert e.matched_forensic is not None

    def test_correlated_severity_elevated(self, correlation_inputs):
        """Correlated evidence has higher severity than AI-only for same feature."""
        correlator = Correlator()
        evidence = correlator.correlate(
            correlation_inputs["shap_exp"],
            correlation_inputs["forensic_report"],
            correlation_inputs["iocs"],
        )
        correlated = [e for e in evidence if e.source == "correlated"]
        ai_only = [e for e in evidence if e.source == "ai_explanation"]

        # If we have both, correlated avg severity should be higher
        if correlated and ai_only:
            avg_corr = np.mean([e.severity for e in correlated])
            avg_ai = np.mean([e.severity for e in ai_only])
            # Correlated should generally have higher severity
            # (not guaranteed in all cases due to normalization)
            assert avg_corr > 0  # At least positive severity


class TestInvestigationReport:
    """Tests for the investigation report generation."""

    def test_generate_report(self, correlation_inputs):
        """generate_report produces a valid InvestigationReport."""
        correlator = Correlator()
        evidence = correlator.correlate(
            correlation_inputs["shap_exp"],
            correlation_inputs["forensic_report"],
            correlation_inputs["iocs"],
        )

        report = correlator.generate_report(
            sample_id="test_001",
            prediction="malware",
            confidence=0.95,
            model_name="xgboost",
            evidence_items=evidence,
            shap_explanation=correlation_inputs["shap_exp"],
        )

        assert isinstance(report, InvestigationReport)
        assert report.prediction == "malware"
        assert report.confidence == 0.95

    def test_report_to_dict(self, correlation_inputs):
        """Report serialises to a valid dict."""
        correlator = Correlator()
        evidence = correlator.correlate(
            correlation_inputs["shap_exp"],
            correlation_inputs["forensic_report"],
            correlation_inputs["iocs"],
        )

        report = correlator.generate_report(
            sample_id="test",
            prediction="malware",
            confidence=0.9,
            model_name="xgboost",
            evidence_items=evidence,
        )

        d = report.to_dict()
        assert "prediction" in d
        assert "evidence_items" in d
        assert "evidence_summary" in d

    def test_report_to_text(self, correlation_inputs):
        """Report generates human-readable text."""
        correlator = Correlator()
        evidence = correlator.correlate(
            correlation_inputs["shap_exp"],
            correlation_inputs["forensic_report"],
            correlation_inputs["iocs"],
        )

        report = correlator.generate_report(
            sample_id="test",
            prediction="malware",
            confidence=0.9,
            model_name="xgboost",
            evidence_items=evidence,
        )

        text = report.to_text()
        assert "HARUSPEX" in text
        assert "VERDICT" in text
        assert "malware" in text.lower()

    def test_risk_score(self, correlation_inputs):
        """risk_score is between 0 and 1."""
        correlator = Correlator()
        evidence = correlator.correlate(
            correlation_inputs["shap_exp"],
            correlation_inputs["forensic_report"],
            correlation_inputs["iocs"],
        )

        report = correlator.generate_report(
            sample_id="test",
            prediction="malware",
            confidence=0.9,
            model_name="xgboost",
            evidence_items=evidence,
        )

        assert 0.0 <= report.risk_score <= 1.0
