set __VSCMD_script_err_count=0
if "%VSCMD_TEST%" NEQ "" goto :test
if "%VSCMD_ARG_CLEAN_ENV%" NEQ "" goto :clean_env

@REM ------------------------------------------------------------------------
:start

@REM ******************************************************************
@REM Adding CLI Path to the Environment Variable Path.
@REM ******************************************************************
if EXIST "%DEVENVDIR%\Extensions\Microsoft\IntelliCode\CLI" set "PATH=%DEVENVDIR%\Extensions\Microsoft\IntelliCode\CLI;%PATH%"

goto :end


@REM ------------------------------------------------------------------------
:test

set __VSCMD_TEST_FailCount=0

setlocal

@REM ******************************************************************
@REM Basic validation. Check if intellicode.exe file exists in the path.
@REM ******************************************************************
if NOT EXIST "%DEVENVDIR%\Extensions\Microsoft\IntelliCode\CLI" goto :test_end
@echo [TEST:%~nx0] Checking for intellicode.exe...
where intellicode.exe > NUL 2>&1
if "%ERRORLEVEL%" NEQ "0" (
    @echo [ERROR:%~nx0] 'where intellicode.exe' failed
    set /A __VSCMD_TEST_FailCount=__VSCMD_TEST_FailCount+1
)

endlocal & set __VSCMD_TEST_FailCount=%__VSCMD_TEST_FailCount%
 
:test_end

@REM return value other than 0 if tests failed.
if "%__VSCMD_TEST_FailCount%" NEQ "0" (
   set __VSCMD_Test_FailCount=
   exit /B 1
)

set __VSCMD_Test_FailCount=
exit /B 0

@REM ------------------------------------------------------------------------
:clean_env
@REM ******************************************************************
@REM Nothing to do on Environment cleanup. Go to end.
@REM ******************************************************************

goto :end
:end

exit /B 0