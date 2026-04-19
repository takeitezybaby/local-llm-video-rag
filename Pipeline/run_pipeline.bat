@echo off
setlocal

set PIPELINE_DIR=%~dp0

echo ===============================
echo   Local LLM Video RAG Pipeline
echo ===============================

echo.
echo [1/5] Converting video to MP3...
python "%PIPELINE_DIR%mp3_Conversion.py"
if %errorlevel% neq 0 ( echo ERROR: mp3_Conversion.py failed & pause & exit /b 1 )

echo.
echo [2/5] Transcribing audio to text...
python "%PIPELINE_DIR%Speech_To_Text.py"
if %errorlevel% neq 0 ( echo ERROR: Speech_To_Text.py failed & pause & exit /b 1 )

echo.
echo [3/5] Creating chunks...
python "%PIPELINE_DIR%new_Chunks.py"
if %errorlevel% neq 0 ( echo ERROR: new_Chunks.py failed & pause & exit /b 1 )

echo.
echo [4/5] Reading and embedding chunks...
python "%PIPELINE_DIR%readChunks.py"
if %errorlevel% neq 0 ( echo ERROR: readChunks.py failed & pause & exit /b 1 )

echo.
echo [5/5] Starting query interface...
python "%PIPELINE_DIR%processing_Query.py"
if %errorlevel% neq 0 ( echo ERROR: processing_Query.py failed & pause & exit /b 1 )

echo.
echo ===============================
echo   Pipeline complete!
echo ===============================
pause
