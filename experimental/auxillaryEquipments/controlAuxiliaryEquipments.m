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
% - serialCommand   : Command that would be issued to the microcontoller.
% - varargin        : Variable arguments for the device type and gas
%
% Output arguments:
% - controllerOutput: variable output from the controller
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [controllerOutput] = controlAuxiliaryEquipments(portProperty, serialCommand, varargin)
% Create a serial object with the port and baudrate specified by the user
% Requires > MATLAB2020a
serialObj = serialport(portProperty.portName,portProperty.baudRate);
% Configure terminator as specified by the user
% Alicat: <CR>
configureTerminator(serialObj,portProperty.terminator)

% If using Alicat (flow meter or controller)
% varargin(1): Device type - Alicat: True
% varargin(2): Gas ID - If Alicat, then ID for the process gas
if varargin(1)
    % Perform a pseudo handshake for the alicat devices. Without this line
    % the communcation is usually not established (AK:10.03.21)
    writeline(serialObj,"a");
    % Send a command to check the gas
    writeline(serialObj, varargin(2));
end

%% SEND THE COMMAND AND CLOSE THE CONNCETION
% Send command to controller
writeline(serialObj, serialCommand);

% Read response from the microcontroller and print it out
controllerOutput = readline(serialObj);

% Terminate the connection with the microcontroller
clear serialObj
end