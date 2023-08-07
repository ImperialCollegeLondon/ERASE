%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Imperial College London, United Kingdom
% Multifunctional Nanomaterials Laboratory
%
% Project:  ERASE
% Year:     2022
% MATLAB:   R2020a
% Authors:  Ashwin Kumar Rajagopalan (AK)
%           Hassan Azzan (HA)
%
% Purpose: 
% Function to define manual set points for the two flow controllers in the
% ZLC setup. For use with two helium controllers
%
% Last modified:
% - 2022-05-04, HA: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function defineSetPtManualDilute(MFC1_SP,MFC2_SP)
    % Define gas and set point for MFC1
    expInfo.gasName_MFC1 = 'He';
    gasName_MFC1 = expInfo.gasName_MFC1;
    expInfo.MFC1_SP = MFC1_SP;
    % Find the port corresponding MFC1
    portText = matchUSBport({'FT1EU0ACA'});
    if ~isempty(portText{1})
        portMFC1 = ['COM',portText{1}(regexp(portText{1},'COM[123456789] - FTDI')+3)];
    end
    % Generate Serial port object
    serialObj.MFC1 = struct('portName',portMFC1,'baudRate',19200,'terminator','CR');
    % Generate serial command for polling data
    serialObj.cmdPollData = generateSerialCommand('pollData',1);
    % Generate Gas ID for Alicat devices
    gasID_MFC1 = checkGasName(gasName_MFC1);
    % Set the gas for MFC1
    [~] = controlAuxiliaryEquipments(serialObj.MFC1, gasID_MFC1,1); % Set gas for MFC1
    % Generate serial command for volumteric flow rate set poin
    cmdSetPt = generateSerialCommand('setPoint',1,expInfo.MFC1_SP); % Same units as device
    [~] = controlAuxiliaryEquipments(serialObj.MFC1, cmdSetPt,1); % Set gas for MFC1
    % Check if the set point was sent to the controller
    outputMFC1 = controlAuxiliaryEquipments(serialObj.MFC1, serialObj.cmdPollData,1);

    % Define gas and set point for MFC2
    expInfo.gasName_MFC2 = 'He';
    gasName_MFC2 = expInfo.gasName_MFC2;
    expInfo.MFC2_SP = MFC2_SP;
    % Find the port corresponding MFC2
    portText = matchUSBport({'FT1EQDD6A'});
    if ~isempty(portText{1})
        portMFC2 = ['COM',portText{1}(regexp(portText{1},'COM[123456789] - FTDI')+3)];
    end
    % Generate Serial port object
    serialObj.MFC2 = struct('portName',portMFC2,'baudRate',19200,'terminator','CR');
    % Generate serial command for polling data
    serialObj.cmdPollData = generateSerialCommand('pollData',1);
    % Generate Gas ID for Alicat devices
    gasID_MFC2 = checkGasName(gasName_MFC2);
    % Set the gas for MFC2
    [~] = controlAuxiliaryEquipments(serialObj.MFC2, gasID_MFC2,1); % Set gas for MFC2
    % Generate serial command for volumteric flow rate set poin
    cmdSetPt = generateSerialCommand('setPoint',1,expInfo.MFC2_SP); % Same units as device
    [~] = controlAuxiliaryEquipments(serialObj.MFC2, cmdSetPt,1); % Set gas for MFC2
    % Check if the set point was sent to the controller
    outputMFC2 = controlAuxiliaryEquipments(serialObj.MFC2, serialObj.cmdPollData,1);
end
