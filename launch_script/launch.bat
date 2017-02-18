@ECHO OFF

echo Checking if logs folder exists...

IF EXIST logs (
	echo File exists. 
) ELSE (
	echo Folder does not exist, creating...
	mkdir logs
	cd logs
	mkdir stream
	mkdir vision
	cd ..
)

SET IP="10.6.12.21"
SET MAX_ATTEMPTS=4

echo Attempting connection to Jetson...

set /A failcount=0
:checkloop
	PING %IP% -n 1 -w 5000 >NUL
	IF %errorlevel% NEQ 1 GOTO success
		SET /A failcount+=1
		IF %failcount% == %MAX_ATTEMPTS% GOTO failure
		echo Connection failed %failcount% time(s)
		GOTO checkloop


:success
	ECHO SSHing in and running the vision program
	FOR /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
	FOR /f "tokens=1-2 delims=/:" %%a in ("%TIME%") do (set mytime=%%a%%b)
	START cmd /c plink.exe -ssh ubuntu@10.6.12.21 -pw ubuntu -m run_streamer.sh^>^>logs/stream/%mydate%_%mytime%.txt 2^>^>^&^1
	START cmd /c plink.exe -ssh ubuntu@10.6.12.21 -pw ubuntu -m run_vision.sh^>^>logs/vision/%mydate%_%mytime%.txt 2^>^>^&^1
	ECHO Vision started
	GOTO end
	
:failure
	echo Failed to connect
	GOTO end

:end
	PAUSE