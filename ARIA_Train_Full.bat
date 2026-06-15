@echo off
chcp 65001 >nul
title ARIA Full Training - 14.6M sequences / 1M vocab
cd /D "C:\Users\lold\Documents\GitHub\ARIA"
set ARIA_MAX_SEQS=14600000
set ARIA_VOCAB_LINES=1000000
echo =========================================
echo   ARIA Full Training
echo   MAX_SEQS  = %ARIA_MAX_SEQS%
echo   VOCAB     = %ARIA_VOCAB_LINES%
echo   Working dir: %CD%
echo =========================================
echo.
cargo run --release --bin train_fresh
pause
