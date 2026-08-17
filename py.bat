@echo off
rem Wrapper to run Python inside venv, avoiding Smart App Control blocks.
rem Usage: py.bat script.py [args]
call "%~dp0.venv\Scripts\activate.bat"
python %*
