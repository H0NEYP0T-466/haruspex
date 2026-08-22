/*
    HARUSPEX YARA Rule: Dropper/Downloader Indicators
    Detects patterns commonly used by droppers and downloaders
    to fetch and execute additional malware payloads.
*/

rule dropper_file_operations {
    meta:
        description = "Detects file operations typical of droppers"
        severity = "high"
        category = "dropper"
    strings:
        $urldownload = "URLDownloadToFile" ascii
        $winexec = "WinExec" ascii
        $shellexec = "ShellExecute" ascii
        $tmppath = "GetTempPath" ascii
        $writefile = "WriteFile" ascii
        $createfile = "CreateFileA" ascii
    condition:
        uint16(0) == 0x5A4D and ($urldownload or $shellexec or $winexec) and ($tmppath or $writefile or $createfile)
}

rule embedded_executable {
    meta:
        description = "Detects embedded PE executable within the binary"
        severity = "high"
        category = "dropper"
    strings:
        $mz = { 4D 5A 90 00 }
        $pe_msg = "This program cannot be run in DOS mode" ascii
    condition:
        uint16(0) == 0x5A4D and #mz > 1 and $pe_msg
}

rule powershell_execution {
    meta:
        description = "Detects PowerShell execution strings"
        severity = "high"
        category = "dropper"
    strings:
        $ps1 = "powershell" ascii nocase
        $ps2 = "powershell.exe" ascii nocase
        $ps3 = "-encodedcommand" ascii nocase
        $ps4 = "-exec bypass" ascii nocase
        $ps5 = "Invoke-Expression" ascii nocase
        $ps6 = "IEX" ascii
    condition:
        uint16(0) == 0x5A4D and 2 of them
}

rule suspicious_url_patterns {
    meta:
        description = "Detects suspicious URL patterns in binary"
        severity = "medium"
        category = "network"
    strings:
        $http = "http://" ascii nocase
        $https = "https://" ascii nocase
        $pastebin = "pastebin.com" ascii nocase
        $discord = "discord.com/api/webhooks" ascii nocase
        $telegram = "api.telegram.org" ascii nocase
    condition:
        uint16(0) == 0x5A4D and ($pastebin or $discord or $telegram) and ($http or $https)
}
