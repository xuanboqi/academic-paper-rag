$ErrorActionPreference = "SilentlyContinue"

$Ports = @(5173, 5174, 8001, 8002)

foreach ($Port in $Ports) {
    $Connections = Get-NetTCPConnection -LocalPort $Port -State Listen
    foreach ($Connection in $Connections) {
        $ProcessId = $Connection.OwningProcess
        $Process = Get-Process -Id $ProcessId
        if ($Process) {
            Write-Host "Stopping $($Process.ProcessName) PID $ProcessId on port $Port"
            Stop-Process -Id $ProcessId -Force
        }
    }
}

Write-Host "academic-paper-rag web services stopped."
