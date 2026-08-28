"""Unit tests for forensic static analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from haruspex.config import get_feature_names
from haruspex.forensics.static_analysis import ForensicReport, StaticAnalyzer
from haruspex.forensics.ioc_extractor import IOC, IOCExtractor


class TestStaticAnalyzer:
    """Tests for the feature vector forensic analysis."""

    def test_analyze_feature_vector_returns_report(self):
        """analyze_feature_vector returns a ForensicReport."""
        analyzer = StaticAnalyzer()
        features = np.random.RandomState(42).randn(2381).astype(np.float32)
        feature_names = get_feature_names()

        report = analyzer.analyze_feature_vector(features, feature_names)
        assert isinstance(report, ForensicReport)

    def test_report_has_indicators(self):
        """Report should contain at least one indicator."""
        analyzer = StaticAnalyzer()
        features = np.random.RandomState(42).randn(2381).astype(np.float32)
        feature_names = get_feature_names()

        report = analyzer.analyze_feature_vector(features, feature_names)
        assert len(report.feature_indicators) > 0

    def test_report_to_dict(self):
        """ForensicReport.to_dict() produces a valid dict."""
        analyzer = StaticAnalyzer()
        features = np.zeros(2381, dtype=np.float32)
        feature_names = get_feature_names()

        report = analyzer.analyze_feature_vector(features, feature_names)
        d = report.to_dict()

        assert isinstance(d, dict)
        assert "is_pe" in d
        assert "sections" in d
        assert "feature_indicators" in d

    def test_high_entropy_detection(self):
        """High entropy features should trigger packing indicator."""
        analyzer = StaticAnalyzer()
        features = np.zeros(2381, dtype=np.float32)
        # Set high values in byte entropy histogram (indices 256-511)
        features[456:512] = 0.1  # Upper entropy bins (indices 200+ within group)
        feature_names = get_feature_names()

        report = analyzer.analyze_feature_vector(features, feature_names)
        # Should detect high entropy
        entropy_indicators = [
            ind for ind in report.feature_indicators
            if ind.get("category") == "entropy"
        ]
        assert len(entropy_indicators) > 0


class TestIOCExtractor:
    """Tests for IOC extraction from forensic reports."""

    def test_extract_from_analysis(self):
        """IOCExtractor produces IOCs from a forensic report."""
        analyzer = StaticAnalyzer()
        extractor = IOCExtractor()

        features = np.random.RandomState(42).randn(2381).astype(np.float32)
        feature_names = get_feature_names()
        report = analyzer.analyze_feature_vector(features, feature_names)

        iocs = extractor.extract_from_analysis(report)
        assert isinstance(iocs, list)
        # Should produce at least some IOCs from feature indicators
        assert all(isinstance(ioc, IOC) for ioc in iocs)

    def test_ioc_has_required_fields(self):
        """Each IOC has all required fields."""
        analyzer = StaticAnalyzer()
        extractor = IOCExtractor()

        features = np.random.RandomState(42).randn(2381).astype(np.float32)
        feature_names = get_feature_names()
        report = analyzer.analyze_feature_vector(features, feature_names)

        iocs = extractor.extract_from_analysis(report)
        for ioc in iocs:
            assert hasattr(ioc, "ioc_type")
            assert hasattr(ioc, "description")
            assert 0.0 <= ioc.severity <= 1.0
            assert 0.0 <= ioc.confidence <= 1.0

    def test_iocs_sorted_by_severity(self):
        """IOCs are returned sorted by severity descending."""
        analyzer = StaticAnalyzer()
        extractor = IOCExtractor()

        features = np.random.RandomState(42).randn(2381).astype(np.float32)
        feature_names = get_feature_names()
        report = analyzer.analyze_feature_vector(features, feature_names)

        iocs = extractor.extract_from_analysis(report)
        if len(iocs) >= 2:
            severities = [ioc.severity for ioc in iocs]
            assert severities == sorted(severities, reverse=True)

    def test_ioc_to_dict(self):
        """IOC.to_dict() produces a valid dict."""
        ioc = IOC(
            ioc_type="suspicious_api",
            description="Test IOC",
            severity=0.8,
            confidence=0.9,
        )
        d = ioc.to_dict()
        assert d["ioc_type"] == "suspicious_api"
        assert d["severity"] == 0.8

    def test_extract_from_pe_with_yara_match(self, tmp_path):
        """extract_from_pe processes YARA matches without raising TypeError."""
        pe_file = tmp_path / "test.exe"
        pe_file.write_bytes(b"MZ" + b"\x00" * 50 + b"IsDebuggerPresent\x00CheckRemoteDebuggerPresent\x00")
        extractor = IOCExtractor()
        iocs = extractor.extract_from_pe(pe_file)
        yara_iocs = [ioc for ioc in iocs if ioc.ioc_type == "yara_match"]
        assert len(yara_iocs) > 0
        assert yara_iocs[0].details["rule"] == "anti_debug_api"
        assert len(yara_iocs[0].details["strings"]) > 0
