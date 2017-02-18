@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
@ECHO OFF
SET IP="10.6.12.2"
SET MAX_ATTEMPTS=4

echo Attempting connection to Jetson...

set /A failcount=0
:checkloop
	PING %IP%
	IF %errorlevel% NEQ 1 GOTO success
		SET /A failcount+=1
		IF %failcount% == %MAX_ATTEMPTS% GOTO failure
		PING 1.1.1.1 -n 1 -w 3000 >NUL
		echo Connection failed %failcount% time(s)


:success
	echo SSHing in and running the vision program
	FOR /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
	FOR /f "tokens=1-2 delims=/:" %%a in ("%TIME%") do (set mytime=%%a%%b)
	plink.exe -ssh ubuntu@10.6.12.21 -pw ubuntu run_streamer.sh > logs/stream/%mydate%_%mytime%.txt
	plink.exe -ssh ubuntu@10.6.12.21 -pw ubuntu run_vision.sh > logs/vision/%mydate%_%mytime%.txt
	ECHO Vision started
	GOTO end
	
:failure
	echo Failed to connect
	GOTO end

:end
	PAUSE