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
%
%
% Last modified:
% - 2021-03-01, HA: Initial creation
%
% Input arguments:
% - portProperty    : Structure containing the properties of the comms 
%                     device
%
% - serialCommand   : Command that would be issued to the microcontoller.
%
% - flagAlicat      : Determines whether or not the equipment is from Alicat
% 
% - gasID           : ID input for the gas for alicat equipment
%
% Output arguments:
% - controllerOutput: variable output from the controller
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [controllerOutput] = controlAuxiliaryEquipments(portProperty, serialCommand, flagAlicat, gasID)

%% CREATE CONNECTION WITH uCONTROLLER & PARSE ARGUMENTS
% Create a serial object with the port and baudrate specified by the user
% Requires > MATLAB2020a
serialObj = serialport(portProperty.portName,portProperty.baudRate);
% Configure terminator as specified by the user
configureTerminator(serialObj,portProperty.terminator)

if flagAlicat
    % change name for initialisation
    writeline(serialObj,"a");
    % Send a command to check the gas
    writeline(serialObj, gasID);
    % Add other initialisation procedure (if required)
end
%% SEND THE COMMAND AND CLOSE THE CONNCETION
% Send command to controller
writeline(serialObj, serialCommand);

% Read response from the microcontroller and print it out
controllerOutput = readline(serialObj);

% Terminate the connection with the microcontroller
clear serialObj
end