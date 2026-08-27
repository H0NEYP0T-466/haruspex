"""Static PE analysis via LIEF — forensic indicator extraction.

Operates in two modes:
  1. PE file mode — parse a real Windows PE binary using LIEF
  2. Feature vector mode — interpret BODMAS/EMBER feature vectors to infer
     forensic indicators from the pre-extracted numeric values

Mode 2 is the primary mode for BODMAS-based analysis since we have feature
vectors, not raw PE files. Mode 1 is used when a PE file is uploaded via
the API.

Phase 2 placeholder: behavioral analysis (CAPE/Cuckoo sandbox) is NOT
implemented. See ``behavioral_analysis.py`` TODO below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from haruspex.config import FEATURE_GROUPS, get_feature_group, get_feature_names
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SectionInfo:
    """Forensic details for a single PE section."""

    name: str = ""
    entropy: float = 0.0
    size: int = 0
    virtual_size: int = 0
    characteristics: list[str] = field(default_factory=list)
    is_suspicious: bool = False
    reason: str = ""


@dataclass
class ForensicReport:
    """Structured output of static PE analysis."""

    # PE header info
    is_pe: bool = True
    machine_type: str = ""
    compilation_timestamp: str = ""
    entry_point: int = 0
    has_digital_signature: bool = False

    # Sections
    sections: list[SectionInfo] = field(default_factory=list)
    num_sections: int = 0

    # Imports
    imported_dlls: list[str] = field(default_factory=list)
    imported_functions: list[str] = field(default_factory=list)
    suspicious_imports: list[str] = field(default_factory=list)

    # Exports
    exported_functions: list[str] = field(default_factory=list)
    num_exports: int = 0

    # Entropy analysis
    overall_entropy: float = 0.0
    max_section_entropy: float = 0.0
    is_likely_packed: bool = False

    # Header anomalies
    anomalies: list[str] = field(default_factory=list)

    # Feature-derived indicators (from BODMAS feature vector)
    feature_indicators: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dict."""
        return {
            "is_pe": self.is_pe,
            "machine_type": self.machine_type,
            "compilation_timestamp": self.compilation_timestamp,
            "entry_point": self.entry_point,
            "has_digital_signature": self.has_digital_signature,
            "sections": [
                {
                    "name": s.name,
                    "entropy": s.entropy,
                    "size": s.size,
                    "virtual_size": s.virtual_size,
                    "characteristics": s.characteristics,
                    "is_suspicious": s.is_suspicious,
                    "reason": s.reason,
                }
                for s in self.sections
            ],
            "num_sections": self.num_sections,
            "imported_dlls": self.imported_dlls,
            "suspicious_imports": self.suspicious_imports,
            "num_imports": len(self.imported_functions),
            "num_exports": self.num_exports,
            "overall_entropy": self.overall_entropy,
            "max_section_entropy": self.max_section_entropy,
            "is_likely_packed": self.is_likely_packed,
            "anomalies": self.anomalies,
            "feature_indicators": self.feature_indicators,
        }


# Suspicious Windows API calls that indicate malicious behavior
SUSPICIOUS_APIS = {
    "WriteProcessMemory": "Process injection — writes to another process's memory",
    "VirtualAllocEx": "Remote memory allocation — allocates memory in another process",
    "CreateRemoteThread": "Remote code execution — creates a thread in another process",
    "NtUnmapViewOfSection": "Process hollowing — unmaps legitimate code from memory",
    "SetWindowsHookEx": "Keylogging/hooking — installs a system-wide hook",
    "GetAsyncKeyState": "Keylogging — monitors keyboard state",
    "InternetOpenUrl": "Network activity — downloads data from the internet",
    "URLDownloadToFile": "File download — downloads a file from a URL",
    "ShellExecute": "Command execution — runs programs or scripts",
    "WinExec": "Command execution — runs a program",
    "CreateService": "Persistence — installs a Windows service",
    "RegSetValueEx": "Registry modification — modifies Windows registry",
    "CryptEncrypt": "Encryption — may indicate ransomware or data exfiltration",
    "IsDebuggerPresent": "Anti-debugging — checks if debugger is attached",
    "CheckRemoteDebuggerPresent": "Anti-debugging — checks for remote debugger",
    "NtQueryInformationProcess": "Anti-analysis — queries process information",
}


class StaticAnalyzer:
    """Static PE analysis engine.

    Provides both PE-file analysis (via LIEF) and feature-vector analysis
    (interpreting BODMAS/EMBER numeric features as forensic indicators).
    """

    def __init__(self) -> None:
        self._feature_names = get_feature_names()

    def analyze_pe(self, pe_path: str | Path) -> ForensicReport:
        """Analyse a real PE binary file using LIEF.

        Args:
            pe_path: Path to the PE file on disk.

        Returns:
            ForensicReport with full static analysis.

        Raises:
            FileNotFoundError: If the PE file doesn't exist.
            ValueError: If the file is not a valid PE.
        """
        import lief

        pe_path = Path(pe_path)
        if not pe_path.exists():
            raise FileNotFoundError(f"PE file not found: {pe_path}")

        logger.info("Analyzing PE file: %s", pe_path)
        binary = lief.parse(str(pe_path))

        if binary is None:
            raise ValueError(f"Not a valid PE file: {pe_path}")

        report = ForensicReport(is_pe=True)

        # --- Header info ---
        if hasattr(binary, "header"):
            header = binary.header
            report.machine_type = str(header.machine)
            report.compilation_timestamp = str(header.time_date_stamps)

        if hasattr(binary, "optional_header"):
            report.entry_point = binary.optional_header.addressof_entrypoint

        # --- Digital signature ---
        report.has_digital_signature = bool(
            getattr(binary, "has_signatures", False)
            or getattr(binary, "has_signature", False)
            or (len(getattr(binary, "signatures", [])) > 0)
        )

        # --- Sections ---
        for section in binary.sections:
            name = section.name.rstrip("\x00")
            entropy = section.entropy
            sec_info = SectionInfo(
                name=name,
                entropy=entropy,
                size=section.size,
                virtual_size=section.virtual_size,
            )

            # Check for suspicious sections
            if entropy > 7.0:
                sec_info.is_suspicious = True
                sec_info.reason = f"High entropy ({entropy:.2f}) — likely packed or encrypted"
            if name not in (".text", ".data", ".rdata", ".rsrc", ".reloc", ".bss", ".idata", ".edata"):
                sec_info.is_suspicious = True
                sec_info.reason = f"Unusual section name: {name}"

            report.sections.append(sec_info)

        report.num_sections = len(report.sections)
        if report.sections:
            report.max_section_entropy = max(s.entropy for s in report.sections)
            report.is_likely_packed = report.max_section_entropy > 7.0

        # --- Imports ---
        if hasattr(binary, "imports"):
            for imp in binary.imports:
                dll_name = imp.name
                report.imported_dlls.append(dll_name)
                for entry in imp.entries:
                    func_name = entry.name if entry.name else f"ordinal_{entry.data}"
                    report.imported_functions.append(f"{dll_name}:{func_name}")
                    if func_name in SUSPICIOUS_APIS:
                        report.suspicious_imports.append(
                            f"{func_name} ({SUSPICIOUS_APIS[func_name]})"
                        )

        # --- Exports ---
        if hasattr(binary, "exported_functions"):
            for exp in binary.exported_functions:
                report.exported_functions.append(exp.name)
            report.num_exports = len(report.exported_functions)

        # --- Header anomalies ---
        self._check_anomalies(binary, report)

        # --- Overall entropy ---
        try:
            content = pe_path.read_bytes()
            report.overall_entropy = self._compute_entropy(content)
        except Exception:
            report.overall_entropy = 0.0

        logger.info(
            "PE analysis complete: %d sections, %d imports, %d suspicious APIs, "
            "packed=%s, %d anomalies",
            report.num_sections,
            len(report.imported_functions),
            len(report.suspicious_imports),
            report.is_likely_packed,
            len(report.anomalies),
        )

        return report

    def analyze_feature_vector(
        self, features: np.ndarray, feature_names: Optional[list[str]] = None
    ) -> ForensicReport:
        """Analyse a BODMAS/EMBER feature vector to infer forensic indicators.

        Maps numeric feature values back to forensic categories based on
        the EMBER feature schema (byte histogram, entropy histogram, sections,
        imports, etc.).

        Args:
            features: Feature vector (2381,).
            feature_names: Optional custom feature names.

        Returns:
            ForensicReport with feature-derived indicators.
        """
        names = feature_names or self._feature_names
        report = ForensicReport(is_pe=True)
        indicators: list[dict] = []

        # --- Byte entropy histogram → entropy analysis ---
        entropy_start, entropy_count = 256, 256
        entropy_features = features[entropy_start:entropy_start + entropy_count]
        report.overall_entropy = float(np.mean(entropy_features)) * 8.0  # Rough estimate

        # High entropy in upper bins suggests packing
        high_entropy_bins = np.sum(entropy_features[200:] > 0.01)
        if high_entropy_bins > 10:
            report.is_likely_packed = True
            indicators.append({
                "category": "entropy",
                "indicator": f"High concentration in upper entropy bins ({high_entropy_bins}/56)",
                "severity": min(high_entropy_bins / 56.0, 1.0),
            })

        # --- Section info features ---
        section_start = 688
        section_count = 255
        section_features = features[section_start:section_start + section_count]

        # In EMBER schema: 5 properties per section [size, entropy, vsize, entry, char]
        section_entropies = section_features[1:250:5]  # exact entropy field
        max_sec_entropy = float(np.max(section_entropies)) if len(section_entropies) > 0 else 0.0
        report.max_section_entropy = max_sec_entropy
        if max_sec_entropy > 7.0:
            indicators.append({
                "category": "sections",
                "indicator": f"Section with very high entropy ({max_sec_entropy:.2f}/8.00) — likely packed",
                "severity": min(max_sec_entropy / 8.0, 1.0),
            })

        # --- Import features → suspicious API analysis ---
        import_start = 943
        import_count = 1280
        import_features = features[import_start:import_start + import_count]

        # Non-zero import features indicate imported libraries/functions
        active_imports = int(np.sum(import_features > 0))
        indicators.append({
            "category": "imports",
            "indicator": f"{active_imports} active import features detected",
            "severity": 0.05,
        })

        # Check for patterns indicating suspicious imports
        import_nonzero = np.where(import_features > 0)[0]
        if len(import_nonzero) > 600:
            indicators.append({
                "category": "imports",
                "indicator": "Unusually large number of import features active",
                "severity": 0.3,
            })

        # --- General info features ---
        general_start = 616
        general_count = 10
        general_features = features[general_start:general_start + general_count]

        if general_features[0] > 20_000_000:  # > 20MB file size indicator
            indicators.append({
                "category": "general",
                "indicator": f"Large file size indicator ({general_features[0]:.0f} bytes)",
                "severity": 0.1,
            })

        # --- Header features ---
        header_start = 626
        header_count = 62
        header_features = features[header_start:header_start + header_count]

        header_zeros = int(np.sum(header_features == 0))
        if header_zeros > 58:
            indicators.append({
                "category": "header_anomaly",
                "indicator": f"Highly stripped PE header features ({header_zeros}/62 zeros)",
                "severity": 0.3,
            })
            report.anomalies.append("Highly stripped PE header features")

        # --- String features ---
        string_start = 512
        string_count = 104
        string_features = features[string_start:string_start + string_count]
        avg_string_val = float(np.mean(string_features))
        if avg_string_val < 0.01:
            indicators.append({
                "category": "strings",
                "indicator": "Very few printable strings (possible packing)",
                "severity": 0.7,
            })

        report.feature_indicators = indicators

        logger.info(
            "Feature vector analysis: %d indicators, packed=%s, entropy=%.2f",
            len(indicators),
            report.is_likely_packed,
            report.overall_entropy,
        )

        return report

    @staticmethod
    def _check_anomalies(binary: "lief.PE.Binary", report: ForensicReport) -> None:
        """Check for PE header anomalies."""
        try:
            # Timestamp anomaly
            ts = binary.header.time_date_stamps
            if ts == 0:
                report.anomalies.append("Compilation timestamp is zero (stripped)")
            elif ts > 2_000_000_000:
                report.anomalies.append("Compilation timestamp is in the future")

            # Section count anomaly
            if report.num_sections > 10:
                report.anomalies.append(f"Unusual number of sections: {report.num_sections}")
            elif report.num_sections == 0:
                report.anomalies.append("No sections found (malformed PE)")

            # Size of optional header
            if hasattr(binary.header, "sizeof_optional_header"):
                if binary.header.sizeof_optional_header == 0:
                    report.anomalies.append("Optional header size is zero")

            # Checksum
            if hasattr(binary.optional_header, "checksum"):
                if binary.optional_header.checksum == 0:
                    report.anomalies.append("PE checksum is zero (not verified)")
        except Exception as e:
            logger.warning("Anomaly check error: %s", e)

    @staticmethod
    def _compute_entropy(data: bytes) -> float:
        """Compute Shannon entropy of raw bytes."""
        if not data:
            return 0.0
        byte_counts = np.zeros(256)
        for b in data:
            byte_counts[b] += 1
        probs = byte_counts / len(data)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))


# ---------------------------------------------------------------------------
# Phase 2 placeholder: Behavioral analysis
# ---------------------------------------------------------------------------
# TODO: Implement behavioral analysis module (Phase 2)
# This would integrate with CAPE/Cuckoo sandbox to analyse:
#   - Runtime API call sequences
#   - Network connections
#   - File system modifications
#   - Registry changes
#   - Process injection techniques
# See: haruspex/forensics/behavioral_analysis.py (future)
