/*
    HARUSPEX YARA Rule: Suspicious Imports
    Detects Windows API calls commonly used by malware for process injection,
    code execution, and privilege escalation.
*/

rule process_injection_imports {
    meta:
        description = "Detects imports used for process injection"
        severity = "high"
        category = "suspicious_imports"
    strings:
        $wpm = "WriteProcessMemory" ascii
        $vae = "VirtualAllocEx" ascii
        $crt = "CreateRemoteThread" ascii
        $nuvs = "NtUnmapViewOfSection" ascii
    condition:
        uint16(0) == 0x5A4D and 2 of ($wpm, $vae, $crt, $nuvs)
}

rule code_execution_imports {
    meta:
        description = "Detects imports used for arbitrary code execution"
        severity = "high"
        category = "suspicious_imports"
    strings:
        $winexec = "WinExec" ascii
        $shellexec = "ShellExecuteA" ascii
        $shellexecw = "ShellExecuteW" ascii
        $createproc = "CreateProcessA" ascii
        $createprocw = "CreateProcessW" ascii
    condition:
        uint16(0) == 0x5A4D and 2 of them
}

rule network_download_imports {
    meta:
        description = "Detects imports used for downloading files from the internet"
        severity = "medium"
        category = "suspicious_imports"
    strings:
        $urldl = "URLDownloadToFileA" ascii
        $urldlw = "URLDownloadToFileW" ascii
        $inetopen = "InternetOpenA" ascii
        $inetopenw = "InternetOpenW" ascii
        $inetopenurl = "InternetOpenUrlA" ascii
        $httpopen = "HttpOpenRequestA" ascii
    condition:
        uint16(0) == 0x5A4D and 2 of them
}

rule persistence_imports {
    meta:
        description = "Detects imports used for establishing persistence"
        severity = "medium"
        category = "suspicious_imports"
    strings:
        $createsvc = "CreateServiceA" ascii
        $createsvcw = "CreateServiceW" ascii
        $regsetval = "RegSetValueExA" ascii
        $regsetvalw = "RegSetValueExW" ascii
        $regcreate = "RegCreateKeyExA" ascii
    condition:
        uint16(0) == 0x5A4D and 2 of them
}
