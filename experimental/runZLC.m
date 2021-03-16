%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Imperial College London, United Kingdom
% Multifunctional Nanomaterials Laboratory
%
% Project:  ERASE
% Year:     2021
% MATLAB:   R2020a
% Authors:  Hassan Azzan (HA)
%           Ashwin Kumar Rajagopalan (AK)
%
% Purpose: 
% Runs the ZLC setup. This function will provide set points to the 
% controllers, will read flow data.
%
% Last modified:
% - 2021-03-16, AK: Add valve switch times
% - 2021-03-15, AK: Bug fixes
% - 2021-03-12, AK: Add set point to zero at the end of the experiment
% - 2021-03-12, AK: Add auto detection of ports and change structure
% - 2021-03-11, HA: Add data logger, set points, and refine code
% - 2021-03-10, HA: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function runZLC(varargin)
    if(nargin<1)
        % Display default value being used
        % Get the date/time
        currentDateTime = datestr(now,'yyyymmdd_HHMMSS');
        disp([currentDateTime,'-> Default experimental settings are used!!'])
        % Experiment name
        expInfo.expName = 'ZLC';
        % Maximum time of the experiment
        expInfo.maxTime = 300;
        % Sampling time for the device
        expInfo.samplingTime = 2;
        % Define gas for MFM
        expInfo.gasName_MFM = 'He';
        % Define gas for MFC1
        expInfo.gasName_MFC1 = 'He';
        % Define gas for MFC2
        expInfo.gasName_MFC2 = 'CO2';
        % Define set point for MFC1
        expInfo.MFC1_SP = 5.0;
        % Define gas for MFC2
        expInfo.MFC2_SP = 5.0;
        % Calibrate meters flag
        expInfo.calibrateMeters = False;
    else
        % Use the value passed to the function
        currentDateTime = datestr(now,'yyyymmdd_HHMMSS');
        disp([currentDateTime,'-> Experimental settings passed to the function are used!!'])
        expInfo = varargin{1};
    end
    % Find COM Ports
    % Initatlize ports
    portMFM = []; portMFC1 = []; portMFC2 = []; portUMFM = [];
    % Find COM port for MFM
    portText = matchUSBport({'FT1EQDD6A'});
    if ~isempty(portText{1})
        portMFM = ['COM',portText{1}(regexp(portText{1},'COM[123456789] - FTDI')+3)];
    end
    % Find COM port for MFC1
    portText = matchUSBport({'FT1EU0ACA'});
    if ~isempty(portText{1})
        portMFC1 = ['COM',portText{1}(regexp(portText{1},'COM[123456789] - FTDI')+3)];
    end
    % Find COM port for UMFM
    portText = matchUSBport({'3065335A3235'});
    if ~isempty(portText{1})
        portUMFM = ['COM',portText{1}(regexp(portText{1},'COM[1234567891011]')+3)];
    end
    % Comm setup for the flow meter and controller
    serialObj.MFM = struct('portName',portMFM,'baudRate',19200,'terminator','CR');
    serialObj.MFC1 = struct('portName',portMFC1,'baudRate',19200,'terminator','CR');
    serialObj.MFC2 = struct('portName',portMFC2,'baudRate',19200,'terminator','CR');
    serialObj.UMFM = struct('portName',portUMFM,'baudRate',9600);

    % Generate serial command for polling data
    serialObj.cmdPollData = generateSerialCommand('pollData',1);

    %% Initialize timer
    timerDevice = timer;
    timerDevice.ExecutionMode = 'fixedRate';
    timerDevice.BusyMode = 'drop';
    timerDevice.Period = expInfo.samplingTime; % [s]
    timerDevice.StartDelay = 5; % [s]
    timerDevice.TasksToExecute = floor((expInfo.maxTime)/expInfo.samplingTime);

    % Specify timer callbacks
    timerDevice.StartFcn = {@initializeTimerDevice,expInfo,serialObj};
    timerDevice.TimerFcn = {@executeTimerDevice, expInfo, serialObj};
    timerDevice.StopFcn = {@stopTimerDevice, expInfo, serialObj};
    
    % Start the experiment
    % Get the date/time
    currentDateTime = datestr(now,'yyyymmdd_HHMMSS');
    disp([currentDateTime,'-> Starting the experiment!!'])
    % Start the timer
    start(timerDevice)
    % Block the command line
    wait(timerDevice)

    % Load the experimental data and add a few more things
    % Load the output .mat file
    load(['experimentalData',filesep,expInfo.expName])
    % Get the git commit ID
    gitCommitID = getGitCommit;
    % Load the output .mat file
    save(['experimentalData',filesep,expInfo.expName],...
        'gitCommitID','outputStruct')
end

%% initializeTimerDevice: Initialisation of timer device
function initializeTimerDevice(~, thisEvent, expInfo, serialObj)
    % Get the event date/time
    currentDateTime = datestr(thisEvent.Data.time,'yyyymmdd_HHMMSS');
    disp([currentDateTime,'-> Initializing the experiment!!'])
    % Parse out gas name from expInfo
    gasName_MFM = expInfo.gasName_MFM;
    gasName_MFC1 = expInfo.gasName_MFC1;
    gasName_MFC2 = expInfo.gasName_MFC2;  
    % Generate Gas ID for Alicat devices
    gasID_MFM = checkGasName(gasName_MFM);
    gasID_MFC1 = checkGasName(gasName_MFC1);
    gasID_MFC2 = checkGasName(gasName_MFC2);
    % Initialize the gas for the meter and the controller
    % MFM
    if ~isempty(serialObj.MFM.portName)
        [~] = controlAuxiliaryEquipments(serialObj.MFM, gasID_MFM,1);   % Set gas for MFM
    end
    % MFC1
    if ~isempty(serialObj.MFC1.portName)
        [~] = controlAuxiliaryEquipments(serialObj.MFC1, gasID_MFC1,1); % Set gas for MFC1
        % Generate serial command for volumteric flow rate set point
        cmdSetPt = generateSerialCommand('setPoint',1,expInfo.MFC1_SP); % Same units as device
        [~] = controlAuxiliaryEquipments(serialObj.MFC1, cmdSetPt,1); % Set gas for MFC1
        % Check if the set point was sent to the controller
        outputMFC1 = controlAuxiliaryEquipments(serialObj.MFC1, serialObj.cmdPollData,1);
        outputMFC1Temp = strsplit(outputMFC1,' '); % Split the output string
        if str2double(outputMFC1Temp(6)) ~= expInfo.MFC1_SP
            error("You should not be here!!!")
        end
    end
    % MFC2
    if ~isempty(serialObj.MFC2.portName)
        [~] = controlAuxiliaryEquipments(serialObj.MFC2, gasID_MFC2,1); % Set gas for MFC2
        % Generate serial command for volumteric flow rate set point
        cmdSetPt = generateSerialCommand('setPoint',1,expInfo.MFC2_SP); % Same units as device
        [~] = controlAuxiliaryEquipments(serialObj.MFC2, cmdSetPt,1); % Set gas for MFC1
        % Check if the set point was sent to the controller
        outputMFC2 = controlAuxiliaryEquipments(serialObj.MFC2, serialObj.cmdPollData,1);
        outputMFC2Temp = strsplit(outputMFC2,' '); % Split the output string
        if str2double(outputMFC2Temp(6)) ~= expInfo.MFC2_SP
            error("You should not be here!!!")
        end
    end
    % Get the event date/time
    currentDateTime = datestr(now,'yyyymmdd_HHMMSS');
    disp([currentDateTime,'-> Initialization complete!!'])
end
%% executeTimerDevice: Execute function for the timer at each instant
function executeTimerDevice(timerObj, thisEvent, expInfo, serialObj)
    % Initialize outputs
    MFM = []; MFC1 = []; MFC2 = []; UMFM = [];
    % Get user input to indicate switching of the valve
    if timerObj.tasksExecuted == 1 && ~expInfo.calibrateMeters
        % Waiting for user to switch the valve
        promptUser = 'Switch asap! When you press Y, the gas switches (you wish)! [Y/N]: ';
        userInput = input(promptUser,'s');
        % This is recorded as the time of switch
        % Empty readings (just for analysis purpose)
        if strcmp(userInput,'Y') || strcmp(userInput,'y')
            currentDateTime = datestr(now,'yyyymmdd_HHMMSS');
            dataLogger(timerObj,expInfo,currentDateTime,[],...
                [],[],[]);
        end
    end
    % Get the sampling date/time
    currentDateTime = datestr(now,'yyyymmdd_HHMMSS');
    disp([currentDateTime,'-> Performing task #', num2str(timerObj.tasksExecuted)])
    % Get the current state of the flow meter
    if ~isempty(serialObj.MFM.portName)
        outputMFM = controlAuxiliaryEquipments(serialObj.MFM, serialObj.cmdPollData,1);
        outputMFMTemp = strsplit(outputMFM,' '); % Split the output string
        MFM.pressure = str2double(outputMFMTemp(2)); % [bar]
        MFM.temperature = str2double(outputMFMTemp(3)); % [C]
        MFM.volFlow = str2double(outputMFMTemp(4)); % device units [ml/min]
        MFM.massFlow = str2double(outputMFMTemp(5)); % standard units [sccm]
        MFM.gas = outputMFMTemp(6); % gas in the meter
    end
    % Get the current state of the flow controller 1
    if ~isempty(serialObj.MFC1.portName)
        outputMFC1 = controlAuxiliaryEquipments(serialObj.MFC1, serialObj.cmdPollData,1);
        outputMFC1Temp = strsplit(outputMFC1,' '); % Split the output string
        MFC1.pressure = str2double(outputMFC1Temp(2)); % [bar]
        MFC1.temperature = str2double(outputMFC1Temp(3)); % [C]
        MFC1.volFlow = str2double(outputMFC1Temp(4)); % device units [ml/min]
        MFC1.massFlow = str2double(outputMFC1Temp(5)); % standard units [sccm]
        MFC1.setpoint = str2double(outputMFC1Temp(6)); % device units [ml/min]
        MFC1.gas = outputMFC1Temp(7); % gas in the controller
    end
    % Get the current state of the flow controller 2
    if ~isempty(serialObj.MFC2.portName)
        outputMFC2 = controlAuxiliaryEquipments(serialObj.MFC2, serialObj.cmdPollData,1);
        outputMFC2Temp = strsplit(outputMFC2,' '); % Split the output string
        MFC2.pressure = str2double(outputMFC2Temp(2)); % [bar]
        MFC2.temperature = str2double(outputMFC2Temp(3)); % [C]
        MFC2.volFlow = str2double(outputMFC2Temp(4)); % device units [ml/min]
        MFC2.massFlow = str2double(outputMFC2Temp(5)); % standard units [sccm]
        MFC2.setpoint = outputMFC2Temp(6); % device units [ml/min]
        MFC2.gas = outputMFC2Temp(7); % gas in the controller
    end
    % Get the current state of the universal flow controller
    if ~isempty(serialObj.UMFM.portName)
        outputUMFM = controlAuxiliaryEquipments(serialObj.UMFM, "UMFM");
        UMFM.volFlow = str2double(outputUMFM);
    end
    % Call the data logger function
    dataLogger(timerObj,expInfo,currentDateTime,MFM,...
        MFC1,MFC2,UMFM);
end
%% stopTimerDevice: Stop timer device
function stopTimerDevice(~, thisEvent, expInfo, serialObj)
    % Get the event date/time
    currentDateTime = datestr(thisEvent.Data.time,'yyyymmdd_HHMMSS');
    disp([currentDateTime,'-> And its over babyyyyyy!!'])
    % Generate serial command for volumteric flow rate set point to zero
    % MFC1
    if ~isempty(serialObj.MFC1.portName)
        cmdSetPt = generateSerialCommand('setPoint',1,0); % Same units as device
        [~] = controlAuxiliaryEquipments(serialObj.MFC1, cmdSetPt,1); % Set gas for MFC1
        % Check if the set point was sent to the controller
        outputMFC1 = controlAuxiliaryEquipments(serialObj.MFC1, serialObj.cmdPollData,1);
        outputMFC1Temp = strsplit(outputMFC1,' '); % Split the output string
        if str2double(outputMFC1Temp(6)) ~= 0
            error("You should not be here!!!")
        end
    end
    % Generate serial command for volumteric flow rate set point to zero
    % MFC2
    if ~isempty(serialObj.MFC2.portName)
        cmdSetPt = generateSerialCommand('setPoint',1,0); % Same units as device
        [~] = controlAuxiliaryEquipments(serialObj.MFC2, cmdSetPt,1); % Set gas for MFC1
        % Check if the set point was sent to the controller
        outputMFC2 = controlAuxiliaryEquipments(serialObj.MFC2, serialObj.cmdPollData,1);
        outputMFC2Temp = strsplit(outputMFC2,' '); % Split the output string
        if str2double(outputMFC2Temp(6)) ~= 0
            error("You should not be here!!!")
        end
    end
end
%% dataLogger: Function to log data into a .mat file
function dataLogger(~, expInfo, currentDateTime, ...
    MFM, MFC1, MFC2, UMFM)
% Check if the file exists
if exist(['experimentalData',filesep,expInfo.expName,'.mat'])==2
    load(['experimentalData',filesep,expInfo.expName])
    % Initialize the counter to existing size plus 1
    nCount = size(outputStruct,2);
    % Load the output .mat file
    % Save the data into the structure
    outputStruct(nCount+1).samplingDateTime = currentDateTime;
    outputStruct(nCount+1).timeElapsed = seconds(datetime(currentDateTime,...
                                        'InputFormat','yyyyMMdd_HHmmss')...
                                        -datetime(outputStruct(1).samplingDateTime,...
                                        'InputFormat','yyyyMMdd_HHmmss')); % Time elapsed [s]
    outputStruct(nCount+1).MFM = MFM;
    outputStruct(nCount+1).MFC1= MFC1;
    outputStruct(nCount+1).MFC2 = MFC2;
    outputStruct(nCount+1).UMFM = UMFM;
    % Save the output into a .mat file
    save(['experimentalData',filesep,expInfo.expName],'outputStruct')
% First initiaization 
else
    nCount = 1;
    outputStruct(nCount).samplingDateTime = currentDateTime;
    outputStruct(nCount).timeElapsed = 0;
    outputStruct(nCount).MFM = MFM;
    outputStruct(nCount).MFC1= MFC1;
    outputStruct(nCount).MFC2 = MFC2;
    outputStruct(nCount).UMFM = UMFM;
    % Create an experimental data folder if it doesnt exost
    if ~exist(['experimentalData'],'dir')
        mkdir(['experimentalData']);
    % Save the output into a .mat file
    else
        save(['experimentalData',filesep,expInfo.expName],'outputStruct')
    end
end
end