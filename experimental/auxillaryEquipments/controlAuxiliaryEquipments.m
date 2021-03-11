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
% Last modified:
% - 2021-03-11, HA: Add UMFM
% - 2021-03-01, HA: Remove gas selection (hard coded)
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

% If using Alicat (flow meter or controller)
% varargin(1): Device type - Alicat: True
if nargin>2 && varargin{1}
    % Configure terminator as specified by the user
    % Alicat: <CR>
    configureTerminator(serialObj,portProperty.terminator)
    % Perform a pseudo handshake for the alicat devices. Without this line
    % the communcation is usually not established (AK:10.03.21)
    writeline(serialObj,'a');
    pause(1); % Pause to ensure proper read
end

%% SEND THE COMMAND AND CLOSE THE CONNCETION
% Send command to controller
% Send a command if not the universal gas flow meter
if ~strcmp(serialCommand,"UMFM")
    writeline(serialObj, serialCommand);
end
% Read response from the microcontroller and print it out
% Read the output from the UMFM (this is always streaming)
if strcmp(serialCommand,"UMFM")
    controllerOutput = read(serialObj,10,"string");
% For everything else
else
    controllerOutput = readline(serialObj);
end
% Terminate the connection with the microcontroller
clear serialObj
end