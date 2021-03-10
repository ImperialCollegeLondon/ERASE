%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Imperial College London, United Kingdom
% Multifunctional Nanomaterials Laboratory
%
% Project:  ERASE
% Year:     2021
% MATLAB:   R2020a
% Authors:  Hassan Azzan (HA)
%
% Purpose: 
% Runs the ZLC setup. This function will provide set points to the 
% controllers, will read flow data.
%
% Last modified:
% - 2021-03-10, HA: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function runZLC
    % Maximum time of the experiment
    expInfo.maxTime = 10;
    % Sampling time for the device
    expInfo.samplingTime = 1;
    % Define gas for MFM
    expInfo.gasName_MFM = 'He';
    % Define gas for MFC1
    expInfo.gasName_MFC1 = 'He';
    % Define gas for MFC2
    expInfo.gasName_MFC2 = 'CO2';

    % Comm setup for the flow meter and controller
    serialObj.MFM = struct('portName','COM6','baudRate',19200,'terminator','CR');
    serialObj.MFC1 = struct('portName','COM7','baudRate',19200,'terminator','CR');
    serialObj.MFC2 = struct('portName','COM7','baudRate',19200,'terminator','CR');

    % Generate serial command for polling data
    serialObj.cmdPollData = generateSerialCommand('pollData',1);

    %% Initialize timer
    timerDevice = timer;
    timerDevice.ExecutionMode = 'fixedRate';
    timerDevice.BusyMode = 'drop';
    timerDevice.Period = expInfo.samplingTime; % [s]
    timerDevice.StartDelay = 0; % [s]
    timerDevice.TasksToExecute = floor((expInfo.maxTime)/expInfo.samplingTime);

    % Specify timer callbacks
    timerDevice.StartFcn = {@initializeTimerDevice,expInfo,serialObj};
    timerDevice.TimerFcn = {@executeTimerDevice,serialObj};
    timerDevice.StopFcn = {@stopTimerDevice};
    
    % Start the experiment
    % Get the date/time
    currentDateTime = datestr(now,'yyyymmdd_HHMMSS');
    disp([currentDateTime,'-> Starting the experiment!!'])
    % Start the timer
    start(timerDevice)
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
    [~] = controlAuxiliaryEquipments(serialObj.MFM, gasID_MFM,1);   % Set gas for MFM
    [~] = controlAuxiliaryEquipments(serialObj.MFC1, gasID_MFC1,1); % Set gas for MFC1
    [~] = controlAuxiliaryEquipments(serialObj.MFC2, gasID_MFC2,1); % Set gas for MFC2
    % Get the event date/time
    currentDateTime = datestr(now,'yyyymmdd_HHMMSS');
    disp([currentDateTime,'-> Initialization complete!!'])
end
%% executeTimerDevice: Execute function for the timer at each instant
function executeTimerDevice(timerObj, thisEvent, serialObj)
    % Get the event date/time
    currentDateTime = datestr(thisEvent.Data.time,'yyyymmdd_HHMMSS');
    disp([currentDateTime,'-> Performing task #', num2str(timerObj.tasksExecuted)])
    % Get the current state of the flow meter
    deviceOutput = controlAuxiliaryEquipments(serialObj.MFM, serialObj.cmdPollData,1);
end
%% stopTimerDevice: Stop timer device
function stopTimerDevice(~, thisEvent)
    % Get the event date/time
    currentDateTime = datestr(thisEvent.Data.time,'yyyymmdd_HHMMSS');
    disp([currentDateTime,'-> And its over babyyyyyy!!'])
end