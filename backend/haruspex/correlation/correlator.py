"""HARUSPEX Correlation Engine — the core research contribution.

Rule-based mapping that correlates AI model explanations (SHAP feature
attributions) with forensic evidence (PE analysis, IOCs) to produce
unified investigation evidence.

The correlation works by:
  1. Taking top-K SHAP features (by absolute attribution weight)
  2. Mapping each feature to a forensic evidence category via a
     predefined feature→forensic mapping dictionary
  3. Checking if the forensic analysis found a matching indicator
  4. Creating correlated EvidenceItems when both sources agree
  5. Creating standalone items for unmatched findings

This produces a unified evidence list where "correlated" items carry
elevated severity because both the ML model and forensic analysis
independently identified the same concern.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from haruspex.config import FEATURE_GROUPS, get_feature_group
from haruspex.correlation.evidence_schema import EvidenceItem, InvestigationReport
from haruspex.forensics.ioc_extractor import IOC
from haruspex.forensics.static_analysis import ForensicReport
from haruspex.xai.shap_explainer import SHAPExplanation
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature group → forensic category mapping
# ---------------------------------------------------------------------------
# Maps EMBER feature groups to forensic evidence categories.
# This is the explicit, rule-based mapping that enables correlation.

FEATURE_GROUP_TO_FORENSIC_CATEGORY: dict[str, str] = {
    "byte_hist": "entropy",
    "byte_entropy_hist": "entropy",
    "string_info": "strings",
    "general_info": "pe_structure",
    "header_info": "header_anomaly",
    "section_info": "sections",
    "import_info": "imports",
    "export_info": "pe_structure",
    "data_dir_info": "pe_structure",
}

# Detailed feature-to-forensic indicator mapping for specific feature ranges
# This provides more granular correlation within each feature group.
FEATURE_INDEX_FORENSIC_MAP: dict[tuple[int, int], dict] = {
    # Byte histogram: high values in certain byte ranges indicate packing
    (0, 256): {
        "category": "entropy",
        "forensic_pattern": "byte distribution anomaly",
        "description": "Unusual byte distribution — may indicate packing or encryption",
    },
    # Byte entropy histogram: high entropy bins → packed/encrypted content
    (256, 512): {
        "category": "entropy",
        "forensic_pattern": "high entropy",
        "description": "High entropy regions detected — possible packing or encryption",
    },
    # String info: very few strings → packed; many suspicious strings → malicious
    (512, 616): {
        "category": "strings",
        "forensic_pattern": "string anomaly",
        "description": "Anomalous string characteristics in binary",
    },
    # General info: file size, has_signature, etc.
    (616, 626): {
        "category": "pe_structure",
        "forensic_pattern": "PE structure anomaly",
        "description": "Unusual PE general properties",
    },
    # Header info: DOS/PE/Optional header fields
    (626, 688): {
        "category": "header_anomaly",
        "forensic_pattern": "header anomaly",
        "description": "PE header field anomaly — possible manipulation",
    },
    # Section info: section count, sizes, entropy, characteristics
    (688, 943): {
        "category": "sections",
        "forensic_pattern": "section anomaly",
        "description": "Suspicious PE section properties",
    },
    # Import info: imported DLLs/functions (hashed features)
    (943, 2223): {
        "category": "imports",
        "forensic_pattern": "suspicious import",
        "description": "Import-related feature contributes to malware classification",
    },
    # Export info
    (2223, 2351): {
        "category": "pe_structure",
        "forensic_pattern": "export anomaly",
        "description": "Unusual export characteristics",
    },
    # Data directory info
    (2351, 2381): {
        "category": "pe_structure",
        "forensic_pattern": "data directory anomaly",
        "description": "Unusual PE data directory configuration",
    },
}


class Correlator:
    """Rule-based correlation engine — the core HARUSPEX contribution.

    Bridges AI explanations (SHAP/LIME) with forensic evidence to produce
    a unified investigation output.
    """

    def __init__(self) -> None:
        self._category_map = FEATURE_GROUP_TO_FORENSIC_CATEGORY
        self._index_map = FEATURE_INDEX_FORENSIC_MAP

    def correlate(
        self,
        shap_explanation: SHAPExplanation,
        forensic_report: ForensicReport,
        iocs: list[IOC],
        top_k: int = 20,
    ) -> list[EvidenceItem]:
        """Correlate SHAP explanation with forensic evidence.

        This is the core algorithm:
          1. Extract top-K SHAP features
          2. Map each to a forensic category
          3. Check for matching forensic findings
          4. Create correlated or standalone evidence items
          5. Add remaining forensic findings as standalone items

        Args:
            shap_explanation: SHAP explanation for the sample.
            forensic_report: Forensic analysis results.
            iocs: IOCs extracted from the analysis.
            top_k: Number of top SHAP features to consider.

        Returns:
            List of EvidenceItems sorted by severity (descending).
        """
        evidence: list[EvidenceItem] = []
        matched_ioc_indices: set[int] = set()
        matched_forensic_indicators: set[int] = set()

        # --- Step 1: Process top SHAP features ---
        top_features = shap_explanation.top_features(k=top_k)

        for feat in top_features:
            feat_idx = feat["index"]
            feat_name = feat["feature_name"]
            shap_value = feat["shap_value"]

            # Determine forensic category for this feature
            group = get_feature_group(feat_idx)
            forensic_category = self._category_map.get(group, "unknown")

            # Get detailed mapping
            detail = self._get_feature_detail(feat_idx)

            # --- Step 2: Check for matching forensic findings ---
            matched_forensic = self._find_forensic_match(
                forensic_category, forensic_report, iocs, matched_ioc_indices
            )

            if matched_forensic:
                # CORRELATED: Both AI and forensic analysis agree
                base_severity = abs(shap_value) / (abs(shap_value) + 1.0)  # Normalise
                # Elevate severity for correlated evidence
                severity = min(base_severity * 1.3 + 0.2, 1.0)

                evidence.append(EvidenceItem(
                    category=forensic_category,
                    indicator=(
                        f"AI feature '{feat_name}' (SHAP={shap_value:+.4f}) "
                        f"correlates with forensic finding: {matched_forensic}"
                    ),
                    severity=severity,
                    source="correlated",
                    feature_name=feat_name,
                    shap_value=shap_value,
                    matched_forensic=matched_forensic,
                    confidence=0.9,
                ))
            else:
                # AI-only evidence
                severity = abs(shap_value) / (abs(shap_value) + 1.0)
                direction = "malware" if shap_value > 0 else "benign"

                evidence.append(EvidenceItem(
                    category=forensic_category,
                    indicator=(
                        f"Feature '{feat_name}' pushes prediction toward {direction} "
                        f"(SHAP={shap_value:+.4f}) — {detail.get('description', '')}"
                    ),
                    severity=severity,
                    source="ai_explanation",
                    feature_name=feat_name,
                    shap_value=shap_value,
                    confidence=0.7,
                ))

        # --- Step 3: Add unmatched forensic IOCs ---
        for i, ioc in enumerate(iocs):
            if i not in matched_ioc_indices:
                evidence.append(EvidenceItem(
                    category=ioc.ioc_type,
                    indicator=ioc.description,
                    severity=ioc.severity,
                    source="forensic_analysis",
                    confidence=ioc.confidence,
                ))

        # --- Step 4: Add unmatched forensic indicators ---
        for i, indicator in enumerate(forensic_report.feature_indicators):
            if i not in matched_forensic_indicators:
                # Skip if already captured as IOC
                desc = indicator.get("indicator", "")
                if not any(e.indicator == desc for e in evidence):
                    evidence.append(EvidenceItem(
                        category=indicator.get("category", "unknown"),
                        indicator=desc,
                        severity=indicator.get("severity", 0.3),
                        source="forensic_analysis",
                        confidence=0.6,
                    ))

        # Sort by severity descending
        evidence.sort(key=lambda e: e.severity, reverse=True)

        n_correlated = sum(1 for e in evidence if e.source == "correlated")
        n_ai = sum(1 for e in evidence if e.source == "ai_explanation")
        n_forensic = sum(1 for e in evidence if e.source == "forensic_analysis")

        logger.info(
            "Correlation complete: %d evidence items "
            "(%d correlated, %d AI-only, %d forensic-only)",
            len(evidence),
            n_correlated,
            n_ai,
            n_forensic,
        )

        return evidence

    def generate_report(
        self,
        sample_id: str,
        prediction: str,
        confidence: float,
        model_name: str,
        evidence_items: list[EvidenceItem],
        shap_explanation: Optional[SHAPExplanation] = None,
        lime_features: Optional[list[dict]] = None,
        forensic_report: Optional[ForensicReport] = None,
        iocs: Optional[list[IOC]] = None,
        sample_sha256: str = "",
        sample_family: str = "",
        sample_category: str = "",
    ) -> InvestigationReport:
        """Generate a complete investigation report.

        Args:
            sample_id: Unique identifier for the sample.
            prediction: Model verdict ("malware" or "benign").
            confidence: Model confidence probability.
            model_name: Name of the model used.
            evidence_items: Correlated evidence list.
            shap_explanation: Optional SHAP explanation for XAI details.
            lime_features: Optional LIME feature contributions.
            forensic_report: Optional forensic analysis for summary.
            iocs: Optional IOC list for count.
            sample_sha256: SHA-256 hash if known.
            sample_family: Malware family if known.
            sample_category: Malware category if known.

        Returns:
            Complete InvestigationReport.
        """
        report = InvestigationReport(
            sample_id=sample_id,
            sample_sha256=sample_sha256,
            sample_family=sample_family,
            sample_category=sample_category,
            prediction=prediction,
            confidence=confidence,
            model_name=model_name,
            evidence_items=evidence_items,
        )

        # Add SHAP details
        if shap_explanation:
            report.top_shap_features = shap_explanation.top_features(k=10)

        # Add LIME details
        if lime_features:
            report.lime_features = lime_features

        # Add forensic summary
        if forensic_report:
            report.forensic_summary = forensic_report.to_dict()

        # IOC count
        if iocs:
            report.ioc_count = len(iocs)

        report.compute_summary_stats()

        logger.info(
            "Investigation report generated: %s → %s (%.1f%% confidence, "
            "%d evidence items, %d correlated)",
            sample_id,
            prediction,
            confidence * 100,
            report.total_evidence,
            report.correlated_evidence,
        )

        return report

    def _get_feature_detail(self, feature_index: int) -> dict:
        """Get detailed forensic description for a feature index."""
        for (start, end), detail in self._index_map.items():
            if start <= feature_index < end:
                return detail
        return {"description": "Unknown feature group", "category": "unknown"}

    def _find_forensic_match(
        self,
        forensic_category: str,
        forensic_report: ForensicReport,
        iocs: list[IOC],
        matched_indices: set[int],
    ) -> Optional[str]:
        """Try to find a matching forensic indicator for a given category.

        Returns a description of the matching forensic finding, or None.
        """
        # Check IOCs first (higher specificity)
        for i, ioc in enumerate(iocs):
            if i in matched_indices:
                continue

            # Category matching (flexible)
            if self._categories_match(forensic_category, ioc.ioc_type):
                matched_indices.add(i)
                return ioc.description

        # Check forensic report indicators
        if forensic_category == "entropy" and forensic_report.is_likely_packed:
            return f"Binary likely packed (max entropy: {forensic_report.max_section_entropy:.2f})"

        if forensic_category == "sections":
            for section in forensic_report.sections:
                if section.is_suspicious:
                    return f"Suspicious section: {section.name} ({section.reason})"

        if forensic_category == "imports" and forensic_report.suspicious_imports:
            imp = forensic_report.suspicious_imports[0]
            return f"Suspicious API: {imp}"

        if forensic_category == "header_anomaly" and forensic_report.anomalies:
            return forensic_report.anomalies[0]

        if forensic_category == "strings":
            for indicator in forensic_report.feature_indicators:
                if indicator.get("category") == "strings":
                    return indicator.get("indicator", "String anomaly detected")

        return None

    @staticmethod
    def _categories_match(ai_category: str, forensic_category: str) -> bool:
        """Check if an AI category and forensic IOC type are semantically related."""
        # Direct match
        if ai_category == forensic_category:
            return True

        # Flexible matching table
        match_table = {
            "entropy": {"entropy_anomaly", "packed_section", "entropy"},
            "imports": {"suspicious_api", "suspicious_imports", "imports"},
            "sections": {"packed_section", "section_anomaly", "sections"},
            "header_anomaly": {"compiler_anomaly", "header_anomaly"},
            "pe_structure": {"compiler_anomaly", "pe_structure", "general"},
            "strings": {"strings"},
        }

        related = match_table.get(ai_category, set())
        return forensic_category in related
