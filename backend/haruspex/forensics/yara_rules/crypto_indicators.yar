/*
    HARUSPEX YARA Rule: Cryptographic Indicators
    Detects use of cryptographic APIs and known crypto constants
    that may indicate ransomware or data exfiltration.
*/

rule windows_crypto_api {
    meta:
        description = "Detects Windows Crypto API usage"
        severity = "medium"
        category = "crypto"
    strings:
        $crypt_enc = "CryptEncrypt" ascii
        $crypt_dec = "CryptDecrypt" ascii
        $crypt_key = "CryptGenKey" ascii
        $crypt_imp = "CryptImportKey" ascii
        $crypt_hash = "CryptHashData" ascii
        $crypt_ctx = "CryptAcquireContextA" ascii
    condition:
        uint16(0) == 0x5A4D and 3 of them
}

rule aes_constants {
    meta:
        description = "Detects AES S-box constants (encryption)"
        severity = "medium"
        category = "crypto"
    strings:
        $aes_sbox = { 63 7c 77 7b f2 6b 6f c5 30 01 67 2b fe d7 ab 76 }
    condition:
        uint16(0) == 0x5A4D and $aes_sbox
}

rule ransomware_extensions {
    meta:
        description = "Detects common ransomware file extension strings"
        severity = "high"
        category = "ransomware"
    strings:
        $ext1 = ".encrypted" ascii nocase
        $ext2 = ".locked" ascii nocase
        $ext3 = ".crypto" ascii nocase
        $ext4 = ".crypt" ascii nocase
        $ransom1 = "YOUR FILES HAVE BEEN ENCRYPTED" ascii nocase
        $ransom2 = "pay the ransom" ascii nocase
        $ransom3 = "bitcoin" ascii nocase
    condition:
        uint16(0) == 0x5A4D and (2 of ($ext*) or any of ($ransom*))
}
