@echo off

REM If python.exe is not in the path, set it's location explicitly
set PYTHONDIR="C:\Program Files\Python\Python311"

REM Run the script
%PYTHONDIR%\python.exe transient_grating_artifact_filter.py
