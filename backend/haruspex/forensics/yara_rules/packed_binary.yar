/*
    HARUSPEX YARA Rule: Packed Binary Detection
    Detects common packers: UPX, ASPack, Themida, PECompact, MPRESS
*/

rule upx_packed {
    meta:
        description = "Detects UPX-packed binaries"
        severity = "medium"
        category = "packer"
    strings:
        $upx0 = "UPX0" ascii
        $upx1 = "UPX1" ascii
        $upx2 = "UPX2" ascii
        $upx_sig = { 55 50 58 21 }
    condition:
        uint16(0) == 0x5A4D and ($upx0 or $upx1 or $upx2 or $upx_sig)
}

rule aspack_packed {
    meta:
        description = "Detects ASPack-packed binaries"
        severity = "medium"
        category = "packer"
    strings:
        $aspack = ".aspack" ascii nocase
        $adata = ".adata" ascii nocase
    condition:
        uint16(0) == 0x5A4D and ($aspack or $adata)
}

rule themida_packed {
    meta:
        description = "Detects Themida/WinLicense protected binaries"
        severity = "high"
        category = "packer"
    strings:
        $themida1 = ".themida" ascii nocase
        $themida2 = ".winlice" ascii nocase
        $themida3 = "THEMIDA" ascii
    condition:
        uint16(0) == 0x5A4D and any of ($themida*)
}

rule pecompact_packed {
    meta:
        description = "Detects PECompact-packed binaries"
        severity = "medium"
        category = "packer"
    strings:
        $pec = "PEC2" ascii
        $pec2 = "PECompact2" ascii
    condition:
        uint16(0) == 0x5A4D and ($pec or $pec2)
}

rule mpress_packed {
    meta:
        description = "Detects MPRESS-packed binaries"
        severity = "medium"
        category = "packer"
    strings:
        $mpress1 = ".MPRESS1" ascii
        $mpress2 = ".MPRESS2" ascii
    condition:
        uint16(0) == 0x5A4D and ($mpress1 or $mpress2)
}
