/*
    HARUSPEX YARA Rule: Anti-Debug Techniques
    Detects API calls and techniques used to evade debuggers and analysis tools.
*/

rule anti_debug_api {
    meta:
        description = "Detects anti-debugging API calls"
        severity = "medium"
        category = "anti_debug"
    strings:
        $isdbg = "IsDebuggerPresent" ascii
        $chkremdbg = "CheckRemoteDebuggerPresent" ascii
        $ntqip = "NtQueryInformationProcess" ascii
        $outdbg = "OutputDebugStringA" ascii
        $outdbgw = "OutputDebugStringW" ascii
    condition:
        uint16(0) == 0x5A4D and 2 of them
}

rule anti_vm_techniques {
    meta:
        description = "Detects anti-VM/sandbox evasion strings"
        severity = "medium"
        category = "anti_analysis"
    strings:
        $vmware1 = "VMware" ascii nocase
        $vbox1 = "VBoxService" ascii nocase
        $vbox2 = "VBoxTray" ascii nocase
        $sandboxie = "SbieDll" ascii nocase
        $wine = "wine_get_unix_file_name" ascii
        $qemu = "QEMU" ascii
    condition:
        uint16(0) == 0x5A4D and 2 of them
}

rule timing_evasion {
    meta:
        description = "Detects timing-based anti-analysis techniques"
        severity = "low"
        category = "anti_analysis"
    strings:
        $gettick = "GetTickCount" ascii
        $querypc = "QueryPerformanceCounter" ascii
        $sleep = "Sleep" ascii
        $ntdelay = "NtDelayExecution" ascii
    condition:
        uint16(0) == 0x5A4D and 3 of them
}
