"""PE Feature Extractor — Extracts 2,381 EMBER-compatible features from raw PE bytes.

Enables end-to-end ML prediction and SHAP explainability for real Windows .exe/.dll files
(e.g., 7zip.exe, calc.exe, custom malware samples).

Feature groups (total = 2,381):
  1. Byte histogram (256)
  2. Byte-entropy histogram (256)
  3. String info (104)
  4. General file info (10)
  5. Header file info (62)
  6. Section info (255)
  7. Import info (1280)
  8. Export info (128)
  9. Data directories (30)
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

from haruspex.config import NUM_FEATURES
from haruspex.utils.logging import get_logger

from sklearn.feature_extraction import FeatureHasher

logger = get_logger(__name__)


class PEFeatureExtractor:
    """Extracts 2,381 EMBER-compatible numeric features from a raw PE file."""

    def __init__(self) -> None:
        self._lib_hasher = FeatureHasher(256, input_type="string")
        self._imp_hasher = FeatureHasher(1024, input_type="string")
        self._exp_hasher = FeatureHasher(128, input_type="string")
        self._sec_pair_hasher = FeatureHasher(50, input_type="pair")
        self._sec_str_hasher = FeatureHasher(50, input_type="string")

    def extract(self, pe_data_or_path: Union[bytes, str, Path]) -> np.ndarray:
        """Extract 2,381 feature vector from raw PE bytes or file path.

        Args:
            pe_data_or_path: Raw bytes or Path/str to a PE file.

        Returns:
            (2381,) float32 numpy array ready for XGBoost/LightGBM prediction.
        """
        if isinstance(pe_data_or_path, (str, Path)):
            bytez = Path(pe_data_or_path).read_bytes()
        else:
            bytez = bytes(pe_data_or_path)

        features = np.zeros(NUM_FEATURES, dtype=np.float32)

        # 1. Byte histogram (0 - 255)
        byte_hist = self._extract_byte_histogram(bytez)
        features[0:256] = byte_hist

        # 2. Byte entropy histogram (256 - 511)
        entropy_hist = self._extract_byte_entropy_histogram(bytez)
        features[256:512] = entropy_hist

        # 3. String info (512 - 615)
        string_features = self._extract_string_features(bytez)
        features[512:616] = string_features

        # LIEF PE parsing for structure features (616 - 2380)
        try:
            import lief

            binary = lief.parse(list(bytez))
            if binary is not None:
                # 4. General info (616 - 625)
                features[616:626] = self._extract_general_info(binary, len(bytez))

                # 5. Header info (626 - 687)
                features[626:688] = self._extract_header_info(binary)

                # 6. Section info (688 - 942)
                features[688:943] = self._extract_section_info(binary)

                # 7. Import info (943 - 2222)
                features[943:2223] = self._extract_import_info(binary)

                # 8. Export info (2223 - 2350)
                features[2223:2351] = self._extract_export_info(binary)

                # 9. Data directories (2351 - 2380)
                features[2351:2381] = self._extract_data_directories(binary)

        except Exception as e:
            logger.warning("LIEF structure extraction partial failure: %s", e)

        return features

    @staticmethod
    def _extract_byte_histogram(bytez: bytes) -> np.ndarray:
        """Normalized 256-bin byte count histogram."""
        if not bytez:
            return np.zeros(256, dtype=np.float32)
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return (counts / len(bytez)).astype(np.float32)

    @staticmethod
    def _extract_byte_entropy_histogram(bytez: bytes, window_size: int = 2048, step: int = 1024) -> np.ndarray:
        """2D Joint entropy histogram (16 byte bins x 16 entropy bins = 256 values)."""
        output = np.zeros(256, dtype=np.float32)
        if len(bytez) < window_size:
            return output

        # Sliding window entropy computation
        n_windows = max(1, (len(bytez) - window_size) // step + 1)
        for i in range(n_windows):
            start = i * step
            window = bytez[start : start + window_size]
            counts = np.bincount(np.frombuffer(window, dtype=np.uint8), minlength=256)
            probs = counts[counts > 0] / window_size
            ent = -np.sum(probs * np.log2(probs))  # 0.0 to 8.0
            ent_bin = min(15, int(ent * 2))  # 16 bins

            # Quantize bytes to 16 bins
            for b_idx in range(16):
                byte_count = np.sum(counts[b_idx * 16 : (b_idx + 1) * 16])
                output[b_idx * 16 + ent_bin] += byte_count

        if output.sum() > 0:
            output /= output.sum()
        return output.astype(np.float32)

    @staticmethod
    def _extract_string_features(bytez: bytes) -> np.ndarray:
        """Extract 104 string statistics features."""
        feat = np.zeros(104, dtype=np.float32)
        # Find printable ASCII strings of length >= 5
        strings = re.findall(b"[\x20-\x7e]{5,}", bytez)
        if not strings:
            return feat

        lengths = [len(s) for s in strings]
        feat[0] = len(strings)                           # total string count
        feat[1] = float(np.mean(lengths))               # avg string length
        feat[2] = float(np.max(lengths))                # max string length
        feat[3] = float(np.min(lengths))                # min string length

        # Character type counts
        all_chars = b"".join(strings)
        feat[4] = sum(1 for c in all_chars if 0x30 <= c <= 0x39)  # digits
        feat[5] = sum(1 for c in all_chars if 0x41 <= c <= 0x5A or 0x61 <= c <= 0x7A)  # letters
        feat[6] = len(all_chars) - (feat[4] + feat[5])  # special chars

        # Category occurrences
        feat[7] = sum(1 for s in strings if b"http://" in s.lower() or b"https://" in s.lower())  # URLs
        feat[8] = sum(1 for s in strings if b"software\\" in s.lower() or b"system\\" in s.lower())  # Reg keys
        feat[9] = sum(1 for s in strings if b".dll" in s.lower() or b".exe" in s.lower())  # File extensions

        # Normalized character histogram (94 printable ASCII chars -> positions 10..103)
        for c in all_chars:
            if 32 <= c <= 126 and (c - 32 + 10) < 104:
                feat[c - 32 + 10] += 1
        if len(all_chars) > 0:
            feat[10:104] /= len(all_chars)

        return feat

    @staticmethod
    def _extract_general_info(binary, file_size: int) -> np.ndarray:
        """10 general info features."""
        feat = np.zeros(10, dtype=np.float32)
        feat[0] = float(file_size)
        feat[1] = float(getattr(binary, "has_debug", False))
        feat[2] = float(len(getattr(binary, "exports", [])))
        feat[3] = float(len(getattr(binary, "imports", [])))
        feat[4] = float(getattr(binary, "has_relocations", False))
        feat[5] = float(getattr(binary, "has_resources", False))
        has_sig = (
            getattr(binary, "has_signatures", False)
            or getattr(binary, "has_signature", False)
            or (len(getattr(binary, "signatures", [])) > 0)
        )
        feat[6] = float(has_sig)
        feat[7] = float(getattr(binary, "has_tls", False))
        feat[8] = float(getattr(binary, "is_pie", False))
        feat[9] = float(getattr(binary, "entrypoint", 0))
        return feat

    @staticmethod
    def _extract_header_info(binary) -> np.ndarray:
        """62 header features."""
        feat = np.zeros(62, dtype=np.float32)
        if hasattr(binary, "header"):
            hdr = binary.header
            feat[0] = float(hdr.machine.value if hasattr(hdr.machine, "value") else int(hdr.machine))
            feat[1] = float(hdr.number_of_sections)
            feat[2] = float(hdr.time_date_stamps)
            feat[3] = float(hdr.pointerto_symbol_table)
            feat[4] = float(hdr.number_of_symbols)
            feat[5] = float(hdr.sizeof_optional_header)
            feat[6] = float(hdr.characteristics)

        if hasattr(binary, "optional_header"):
            opthdr = binary.optional_header
            feat[7] = float(opthdr.magic.value if hasattr(opthdr.magic, "value") else int(opthdr.magic))
            feat[8] = float(opthdr.major_linker_version)
            feat[9] = float(opthdr.minor_linker_version)
            feat[10] = float(opthdr.sizeof_code)
            feat[11] = float(opthdr.sizeof_initialized_data)
            feat[12] = float(opthdr.sizeof_uninitialized_data)
            feat[13] = float(opthdr.addressof_entrypoint)
            feat[14] = float(opthdr.base_of_code)
            feat[15] = float(opthdr.imagebase)
            feat[16] = float(opthdr.section_alignment)
            feat[17] = float(opthdr.file_alignment)
            feat[18] = float(opthdr.major_operating_system_version)
            feat[19] = float(opthdr.minor_operating_system_version)
            feat[20] = float(opthdr.major_image_version)
            feat[21] = float(opthdr.minor_image_version)
            feat[22] = float(opthdr.major_subsystem_version)
            feat[23] = float(opthdr.minor_subsystem_version)
            feat[24] = float(opthdr.sizeof_image)
            feat[25] = float(opthdr.sizeof_headers)
            feat[26] = float(opthdr.checksum)
            feat[27] = float(opthdr.subsystem.value if hasattr(opthdr.subsystem, "value") else int(opthdr.subsystem))
            feat[28] = float(opthdr.dll_characteristics)
            feat[29] = float(opthdr.sizeof_stack_reserve)
            feat[30] = float(opthdr.sizeof_stack_commit)
            feat[31] = float(opthdr.sizeof_heap_reserve)
            feat[32] = float(opthdr.sizeof_heap_commit)

        return feat

    def _extract_section_info(self, binary) -> np.ndarray:
        """255 deterministic section properties features matching EMBER layout."""
        sections = getattr(binary, "sections", [])
        general = [
            len(sections),
            sum(1 for s in sections if s.size == 0),
            sum(1 for s in sections if not s.name or s.name == ""),
            sum(1 for s in sections if hasattr(s, "characteristics") and s.characteristics & 0x20000000 and s.characteristics & 0x40000000),
            sum(1 for s in sections if hasattr(s, "characteristics") and s.characteristics & 0x80000000),
        ]

        section_sizes = [(s.name, s.size) for s in sections]
        section_entropy = [(s.name, s.entropy) for s in sections]
        section_vsize = [(s.name, s.virtual_size) for s in sections]

        entry_section = ""
        if hasattr(binary, "entrypoint"):
            try:
                sec = binary.section_from_rva(binary.entrypoint - getattr(binary.optional_header, "imagebase", 0))
                if sec:
                    entry_section = sec.name
            except Exception:
                pass

        chars = [str(sec.characteristics) for sec in sections]

        sizes_h = self._sec_pair_hasher.transform([section_sizes]).toarray()[0]
        entropy_h = self._sec_pair_hasher.transform([section_entropy]).toarray()[0]
        vsize_h = self._sec_pair_hasher.transform([section_vsize]).toarray()[0]
        entry_h = self._sec_str_hasher.transform([[entry_section]]).toarray()[0]
        chars_h = self._sec_str_hasher.transform([chars]).toarray()[0]

        return np.hstack([general, sizes_h, entropy_h, vsize_h, entry_h, chars_h]).astype(np.float32)

    def _extract_import_info(self, binary) -> np.ndarray:
        """1280 deterministic hashed import features (DLLs and function names)."""
        imports = getattr(binary, "imports", [])
        libraries = list(set([imp.name.lower() for imp in imports if imp.name]))
        lib_hashed = self._lib_hasher.transform([libraries]).toarray()[0]

        import_list = []
        for imp in imports:
            dll_name = imp.name.lower() if imp.name else ""
            for entry in getattr(imp, "entries", []):
                func_name = entry.name.lower() if entry.name else f"ordinal_{entry.data}"
                import_list.append(f"{dll_name}:{func_name}")

        imp_hashed = self._imp_hasher.transform([import_list]).toarray()[0]
        return np.hstack([lib_hashed, imp_hashed]).astype(np.float32)

    def _extract_export_info(self, binary) -> np.ndarray:
        """128 deterministic hashed export features."""
        exports = getattr(binary, "exported_functions", [])
        export_list = [exp.name.lower() if exp.name else str(exp.ordinal) for exp in exports]
        exp_hashed = self._exp_hasher.transform([export_list]).toarray()[0]
        return exp_hashed.astype(np.float32)

    @staticmethod
    def _extract_data_directories(binary) -> np.ndarray:
        """30 data directory features (15 directories x 2: VirtualAddress & Size)."""
        feat = np.zeros(30, dtype=np.float32)
        data_dirs = getattr(binary, "data_directories", [])

        for i, d in enumerate(data_dirs[:15]):
            feat[i * 2] = float(d.rva)
            feat[i * 2 + 1] = float(d.size)

        return feat
