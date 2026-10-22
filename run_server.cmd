@echo off
rem Starts the local analysis server. Works from Command Prompt or by double-click.
rem
rem   run_server.cmd         only this PC can connect (Chrome, Android emulator)
rem   run_server.cmd -Lan    phones on the same Wi-Fi can connect too
rem
rem This just runs run_server.ps1 in PowerShell for this one command, bypassing the
rem script-blocking policy without changing any Windows setting.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_server.ps1" %*
