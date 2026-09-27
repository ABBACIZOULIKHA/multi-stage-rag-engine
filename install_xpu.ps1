$argsArray = @(
    "install",
    "torch",
    "--index-url",
    "https://download.pytorch.org/whl/xpu",
    "--no-cache-dir",
    "--timeout",
    "600",
    "--retries",
    "10"
)
Start-Process -FilePath "C:\Users\DELL\Documents\RAG_Project\.venv\Scripts\pip.exe" -ArgumentList $argsArray -WorkingDirectory "C:\Users\DELL\Documents\RAG_Project" -NoNewWindow -RedirectStandardOutput "C:\Users\DELL\Documents\RAG_Project\install_out.txt" -RedirectStandardError "C:\Users\DELL\Documents\RAG_Project\install_err.txt"
Write-Host "pip install started in background"
