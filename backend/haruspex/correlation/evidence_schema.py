"""Unified evidence schema for HARUSPEX investigation reports.

Defines the data structures that bridge AI explanations and forensic evidence
into a single coherent output — the core research contribution of HARUSPEX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class EvidenceItem:
    """A single piece of evidence from the HARUSPEX investigation.

    Evidence can originate from:
      - AI explanation (SHAP/LIME feature attribution)
      - Forensic analysis (PE structure, imports, YARA)
      - Correlated (both AI and forensic sources agree)

    Correlated evidence has elevated severity because both the model's
    reasoning and the forensic evidence support the same conclusion.
    """

    category: str              # "pe_structure" | "imports" | "sections" | "entropy"
                               # | "header_anomaly" | "ioc" | "strings" | "general"
    indicator: str             # Human-readable description of the evidence
    severity: float            # 0.0 – 1.0 (higher = more severe)
    source: str                # "ai_explanation" | "forensic_analysis" | "correlated"
    feature_name: str = ""     # BODMAS/EMBER feature name (for AI source)
    shap_value: float = 0.0    # SHAP attribution value (for AI source)
    matched_forensic: Optional[str] = None  # If correlated: which forensic indicator matched
    confidence: float = 0.0    # Confidence in this evidence (0.0 – 1.0)

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dict."""
        d = {
            "category": self.category,
            "indicator": self.indicator,
            "severity": round(self.severity, 4),
            "source": self.source,
            "confidence": round(self.confidence, 4),
        }
        if self.feature_name:
            d["feature_name"] = self.feature_name
        if self.shap_value != 0.0:
            d["shap_value"] = round(self.shap_value, 6)
        if self.matched_forensic:
            d["matched_forensic"] = self.matched_forensic
        return d


@dataclass
class InvestigationReport:
    """Complete HARUSPEX investigation report for a single sample.

    Combines model prediction, AI explanation, forensic analysis, and
    correlated evidence into a unified output.
    """

    # Sample identification
    sample_id: str = ""
    sample_sha256: str = ""
    sample_family: str = ""
    sample_category: str = ""

    # Model prediction
    prediction: str = ""           # "malware" | "benign"
    confidence: float = 0.0        # Model confidence (probability)
    model_name: str = ""           # Which model made the prediction

    # Evidence (the core output)
    evidence_items: list[EvidenceItem] = field(default_factory=list)

    # Summary statistics
    total_evidence: int = 0
    correlated_evidence: int = 0
    ai_only_evidence: int = 0
    forensic_only_evidence: int = 0

    # XAI details
    top_shap_features: list[dict] = field(default_factory=list)
    lime_features: list[dict] = field(default_factory=list)

    # Forensic details
    forensic_summary: dict = field(default_factory=dict)
    ioc_count: int = 0

    # Metadata
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    haruspex_version: str = "0.1.0"

    def compute_summary_stats(self) -> None:
        """Compute summary statistics from evidence items."""
        self.total_evidence = len(self.evidence_items)
        self.correlated_evidence = sum(
            1 for e in self.evidence_items if e.source == "correlated"
        )
        self.ai_only_evidence = sum(
            1 for e in self.evidence_items if e.source == "ai_explanation"
        )
        self.forensic_only_evidence = sum(
            1 for e in self.evidence_items if e.source == "forensic_analysis"
        )

    @property
    def risk_score(self) -> float:
        """Compute an overall risk score from evidence severity.

        Returns:
            Float between 0.0 and 1.0 representing overall risk.
        """
        if not self.evidence_items:
            return self.confidence

        # Weighted average: correlated evidence counts more
        weights = []
        values = []
        for e in self.evidence_items:
            w = 2.0 if e.source == "correlated" else 1.0
            weights.append(w)
            values.append(e.severity)

        weighted_sum = sum(w * v for w, v in zip(weights, values))
        total_weight = sum(weights)

        evidence_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        # Blend model confidence with evidence score
        return 0.6 * self.confidence + 0.4 * evidence_score

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dict."""
        self.compute_summary_stats()
        return {
            "sample": {
                "id": self.sample_id,
                "sha256": self.sample_sha256,
                "family": self.sample_family,
                "category": self.sample_category,
            },
            "prediction": {
                "verdict": self.prediction,
                "confidence": round(self.confidence, 4),
                "risk_score": round(self.risk_score, 4),
                "model": self.model_name,
            },
            "evidence_summary": {
                "total": self.total_evidence,
                "correlated": self.correlated_evidence,
                "ai_only": self.ai_only_evidence,
                "forensic_only": self.forensic_only_evidence,
            },
            "evidence_items": [e.to_dict() for e in self.evidence_items],
            "xai": {
                "top_shap_features": self.top_shap_features,
                "lime_features": self.lime_features,
            },
            "forensics": self.forensic_summary,
            "ioc_count": self.ioc_count,
            "metadata": {
                "timestamp": self.timestamp,
                "haruspex_version": self.haruspex_version,
            },
        }

    def to_text(self) -> str:
        """Generate a human-readable text report."""
        self.compute_summary_stats()
        lines = [
            "=" * 72,
            "  HARUSPEX INVESTIGATION REPORT",
            "=" * 72,
            "",
            f"  Sample:     {self.sample_sha256 or self.sample_id}",
            f"  Family:     {self.sample_family or 'Unknown'}",
            f"  Category:   {self.sample_category or 'Unknown'}",
            f"  Timestamp:  {self.timestamp}",
            "",
            "-" * 72,
            "  VERDICT",
            "-" * 72,
            f"  Prediction: {self.prediction.upper()}",
            f"  Confidence: {self.confidence:.1%}",
            f"  Risk Score: {self.risk_score:.1%}",
            f"  Model:      {self.model_name}",
            "",
            "-" * 72,
            "  EVIDENCE SUMMARY",
            "-" * 72,
            f"  Total evidence items:    {self.total_evidence}",
            f"  Correlated (AI+Forensic): {self.correlated_evidence}",
            f"  AI-only:                 {self.ai_only_evidence}",
            f"  Forensic-only:           {self.forensic_only_evidence}",
            "",
        ]

        if self.evidence_items:
            lines.extend([
                "-" * 72,
                "  EVIDENCE DETAILS",
                "-" * 72,
            ])
            for i, e in enumerate(self.evidence_items, 1):
                source_tag = {
                    "correlated": "★ CORRELATED",
                    "ai_explanation": "🤖 AI",
                    "forensic_analysis": "🔍 FORENSIC",
                }.get(e.source, e.source)
                lines.append(
                    f"  [{i:2d}] [{source_tag}] [{e.category}] "
                    f"(severity: {e.severity:.2f})"
                )
                lines.append(f"       {e.indicator}")
                if e.matched_forensic:
                    lines.append(f"       ↔ Forensic match: {e.matched_forensic}")
                lines.append("")

        lines.extend([
            "=" * 72,
            f"  Generated by HARUSPEX v{self.haruspex_version}",
            "=" * 72,
        ])

        return "\n".join(lines)
