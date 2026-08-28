"""IOC (Indicator of Compromise) extraction from PE analysis and YARA matches."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from haruspex.forensics.static_analysis import ForensicReport, SUSPICIOUS_APIS
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)

# Path to YARA rules directory
YARA_RULES_DIR = Path(__file__).parent / "yara_rules"


@dataclass
class IOC:
    """An Indicator of Compromise."""

    ioc_type: str          # "suspicious_api" | "packed_section" | "section_anomaly"
                           # | "yara_match" | "compiler_anomaly" | "entropy_anomaly"
    description: str       # Human-readable description
    severity: float        # 0.0 – 1.0
    confidence: float      # 0.0 – 1.0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dict."""
        return {
            "ioc_type": self.ioc_type,
            "description": self.description,
            "severity": self.severity,
            "confidence": self.confidence,
            "details": self.details,
        }


class IOCExtractor:
    """Extract IOCs from static analysis results and/or raw PE files."""

    def __init__(self, yara_rules_dir: Optional[Path] = None) -> None:
        """Initialise the IOC extractor.

        Args:
            yara_rules_dir: Directory containing YARA rule files.
                           Defaults to haruspex/forensics/yara_rules/.
        """
        self._yara_rules_dir = yara_rules_dir or YARA_RULES_DIR
        self._yara_rules = None
        self._load_yara_rules()

    def _load_yara_rules(self) -> None:
        """Compile all YARA rule files from the rules directory."""
        try:
            import yara

            if not self._yara_rules_dir.exists():
                logger.warning("YARA rules directory not found: %s", self._yara_rules_dir)
                return

            rule_files = list(self._yara_rules_dir.glob("*.yar"))
            if not rule_files:
                logger.warning("No YARA rule files found in %s", self._yara_rules_dir)
                return

            # Compile all rules
            filepaths = {
                f.stem: str(f) for f in rule_files
            }
            self._yara_rules = yara.compile(filepaths=filepaths)
            logger.info("Loaded %d YARA rule files", len(rule_files))

        except ImportError:
            logger.warning("yara-python not installed — YARA matching disabled")
        except Exception as e:
            logger.warning("Failed to compile YARA rules: %s", e)

    def extract_from_analysis(self, report: ForensicReport) -> list[IOC]:
        """Extract IOCs from a ForensicReport (static analysis output).

        Args:
            report: ForensicReport from StaticAnalyzer.

        Returns:
            List of IOC objects sorted by severity (descending).
        """
        iocs: list[IOC] = []

        # --- Suspicious API imports ---
        for imp in report.suspicious_imports:
            iocs.append(IOC(
                ioc_type="suspicious_api",
                description=imp,
                severity=0.8,
                confidence=0.9,
                details={"source": "import_analysis"},
            ))

        # --- Packed/encrypted sections ---
        for section in report.sections:
            if section.is_suspicious and section.entropy > 7.0:
                iocs.append(IOC(
                    ioc_type="packed_section",
                    description=(
                        f"Section '{section.name}' has high entropy ({section.entropy:.2f})"
                        " — likely packed or encrypted"
                    ),
                    severity=0.7,
                    confidence=0.85,
                    details={
                        "section_name": section.name,
                        "entropy": section.entropy,
                    },
                ))

        # --- Unusual section names ---
        for section in report.sections:
            if section.is_suspicious and section.entropy <= 7.0:
                iocs.append(IOC(
                    ioc_type="section_anomaly",
                    description=f"Unusual section name: '{section.name}'",
                    severity=0.4,
                    confidence=0.7,
                    details={"section_name": section.name, "reason": section.reason},
                ))

        # --- Header anomalies ---
        for anomaly in report.anomalies:
            severity = 0.5
            if "future" in anomaly.lower():
                severity = 0.6
            elif "zero" in anomaly.lower() and "timestamp" in anomaly.lower():
                severity = 0.5
            iocs.append(IOC(
                ioc_type="compiler_anomaly",
                description=anomaly,
                severity=severity,
                confidence=0.75,
                details={"source": "header_analysis"},
            ))

        # --- Packing indicator ---
        if report.is_likely_packed:
            iocs.append(IOC(
                ioc_type="entropy_anomaly",
                description=(
                    f"Binary is likely packed (max section entropy: "
                    f"{report.max_section_entropy:.2f})"
                ),
                severity=0.75,
                confidence=0.8,
                details={
                    "max_section_entropy": report.max_section_entropy,
                    "overall_entropy": report.overall_entropy,
                },
            ))

        # --- Feature-derived indicators ---
        for indicator in report.feature_indicators:
            iocs.append(IOC(
                ioc_type=indicator.get("category", "unknown"),
                description=indicator.get("indicator", ""),
                severity=indicator.get("severity", 0.3),
                confidence=0.6,  # Lower confidence for feature-derived
                details={"source": "feature_vector_analysis"},
            ))

        # Sort by severity descending
        iocs.sort(key=lambda x: x.severity, reverse=True)

        logger.info("Extracted %d IOCs from forensic analysis", len(iocs))
        return iocs

    def extract_from_pe(self, pe_path: str | Path) -> list[IOC]:
        """Extract IOCs from a raw PE file, including YARA matches.

        Args:
            pe_path: Path to the PE file.

        Returns:
            List of IOC objects.
        """
        pe_path = Path(pe_path)
        iocs: list[IOC] = []

        # Run YARA matching
        if self._yara_rules and pe_path.exists():
            try:
                matches = self._yara_rules.match(str(pe_path))
                for match in matches:
                    iocs.append(IOC(
                        ioc_type="yara_match",
                        description=f"YARA rule matched: {match.rule}",
                        severity=0.85,
                        confidence=0.95,
                        details={
                            "rule": match.rule,
                            "tags": list(match.tags) if match.tags else [],
                            "namespace": match.namespace,
                            "strings": self._format_yara_strings(match),
                        },
                    ))
                logger.info("YARA: %d rule matches on %s", len(matches), pe_path.name)
            except Exception as e:
                logger.warning("YARA matching failed: %s", e)

        return iocs

    @staticmethod
    def _format_yara_strings(match) -> list[dict]:
        """Extract YARA match string instances safely across yara-python versions."""
        if not hasattr(match, "strings") or not match.strings:
            return []

        string_details = []
        for s in match.strings[:5]:
            if isinstance(s, (tuple, list)):
                offset = s[0]
                identifier = str(s[1])
                raw_data = s[2] if isinstance(s[2], bytes) else str(s[2]).encode()
                string_details.append({
                    "offset": offset,
                    "identifier": identifier,
                    "data": raw_data.hex()[:64],
                })
            elif hasattr(s, "identifier") and hasattr(s, "instances"):
                identifier = str(s.identifier)
                for inst in getattr(s, "instances", []):
                    offset = getattr(inst, "offset", 0)
                    matched_data = getattr(inst, "matched_data", b"")
                    if isinstance(matched_data, str):
                        matched_data = matched_data.encode()
                    string_details.append({
                        "offset": offset,
                        "identifier": identifier,
                        "data": matched_data.hex()[:64],
                    })
        return string_details[:5]

    def extract_all(
        self, report: ForensicReport, pe_path: Optional[str | Path] = None
    ) -> list[IOC]:
        """Extract IOCs from both analysis and PE file (if available).

        Args:
            report: ForensicReport from static analysis.
            pe_path: Optional path to the raw PE file.

        Returns:
            Combined list of IOCs sorted by severity.
        """
        iocs = self.extract_from_analysis(report)

        if pe_path:
            yara_iocs = self.extract_from_pe(pe_path)
            iocs.extend(yara_iocs)

        iocs.sort(key=lambda x: x.severity, reverse=True)
        return iocs
