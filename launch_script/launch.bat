@ECHO OFF

SET JETSON_IP="10.6.12.21"
SET RIO_IP="10.6.12.2"
SET MAX_ATTEMPTS=4

ECHO Checking if logs folder exists...

IF EXIST logs (
	ECHO File exists. 
) ELSE (
	ECHO Folder does not exist, creating...
	MKDIR logs
	CD logs
	MKDIR stream
	MKDIR vision
	MKDIR networktables
	MKDIR smarterdashboard
	CD ..
)

ECHO Retrieving date and time for logging purposes...
FOR /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
FOR /f "tokens=1-2 delims=/:" %%a in ("%TIME%") do (set mytime=%%a%%b)

ECHO Attempting connection to RoboRio...
SET IP=%RIO_IP%

CALL :tester
if %errorlevel% NEQ 1 (
	ECHO Starting driver station...
	START DriverStation.exe
	ECHO Starting network tables server...
	START cmd /c py -3 -m pynetworktables2js --robot roborio-612-frc.local ^>^>logs/networktables/%mydate%_%mytime%.txt 2^>^>^&^1
	ECHO Starting EvenSmarterDashboard...
	START cmd /c npm start ^>^>logs/smarterdashboard/%mydate%_%mytime%.txt 2^>^>^&^1
)

ECHO Attempting connection to Jetson...
SET IP=%JETSON_IP%

CALL :tester
if %errorlevel% NEQ 1 (
	ECHO SSHing in and running the vision program
	START cmd /c plink.exe -ssh ubuntu@10.6.12.21 -pw ubuntu -m run_streamer.sh^>^>logs/stream/%mydate%_%mytime%.txt 2^>^>^&^1
	START cmd /c plink.exe -ssh ubuntu@10.6.12.21 -pw ubuntu -m run_vision.sh^>^>logs/vision/%mydate%_%mytime%.txt 2^>^>^&^1
	ECHO Vision started
)
	
GOTO end
	
:tester
	SET /A failcount=0
	:checkloop
		PING %IP% -n 1 -w 5000 >NUL
		IF %errorlevel% NEQ 1 GOTO :eof
			SET /A failcount+=1
			IF %failcount% == %MAX_ATTEMPTS% GOTO failure
			echo Connection failed %failcount% time(s)
			CALL :checkloop
			GOTO :eof 
			:failure
				echo Failed to connect
				SET errorlevel=1
				GOTO :eof
:end
	PAUSE
