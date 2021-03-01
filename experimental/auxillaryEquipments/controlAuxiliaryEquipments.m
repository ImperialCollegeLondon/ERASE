%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ETH Zurich, Switzerland
% Separation Processes Laboratory
%
% Project:  ERASE
% Year:     2021
% MATLAB:   R2020a
% Authors:  Hassan Azzan (HA)
%
% Purpose: 
% Controls the pump (Ismatec BVP series) which works with the IKA Magic Lab
% (Milling setup) and the IKA Magic Lab. Additionally, two relay switches 
% are available which can be used to control two overhead stirrers.
%
% Last modified:
% - 2021-03-01, HA: Initial creation
%
% Input arguments:
% - portProperty    : Enter the serial port ID for the connection to be
%                     made
% - serialCommand   : Command that would be issued to the microcontoller.
%
% Output arguments:
% - controllerOutput: Values ranging from -1 to 6, serves as an output from
%                     the microcontroller to indicate successful execution
%                     of the command. For the Mill, output from the
%                     controller would typically be an acknowledgment
%                     character or the speed in rpm.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [controllerOutput] = controlAuxiliaryEquipments(portProperty, serialCommand, flagAlicat,gasID)

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