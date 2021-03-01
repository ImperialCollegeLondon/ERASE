%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ETH Zurich, Switzerland
% Separation Processes Laboratory
%
% Project:  CrystOCAM 2.0
% Year:     2018
% MATLAB:   R2017b
% Authors:  Ashwin Kumar Rajagopalan (AK)
%
% Purpose: 
% Controls the pump (Ismatec BVP series) which works with the IKA Magic Lab
% (Milling setup) and the IKA Magic Lab. Additionally, two relay switches 
% are available which can be used to control two overhead stirrers.
%
% Last modified:
% - 2018-09-25, AK: Removed the command that would read from the mill. <A
%                   potential bug when the connection fails. Not sure 
%                   though!>
% - 2018-07-26, AK: Added new subpart to read the milling intensity from
%                   the mill 
% - 2018-07-25, AK: Additional functionality to control the IKA Magic Lab
% - 2018-07-20, AK: Initial creation
%
% Input arguments:
% - portName        : Enter the serial port ID for the connection to be
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
% Communication protocol:
% Commands to control the pump of the mill and stirrers: 
% PRC           : Turn on the pump and run it in clockwise direction
% PRA           : Turn on the pump and run it in anticlockwise direction
% PS            : Turn off the pump
% PSON          : Turn on the primary stirrer
% PSOFF         : Turn off the primary stirrer
% SSON          : Turn on the secondary stirrer
% SSOFF         : Turn off the secondary stirrer
%
% Commands to control the IKA Magic Lab
% The command is generated using generateMillingControllerCommand.m
%
% Output from the microcontroller for the mill and the stirrers:
% 6 : Succesfully turned off the secondary stirrer
% 5 : Succesfully turned on the secondary stirrer
% 4 : Succesfully turned off the primary stirrer
% 3 : Succesfully turned on the primary stirrer
% 2 : Successfully tuned on the pump and ran it in anticlockwise direction
% 1 : Successfully tuned on the pump and ran it in clockwise direction
% 0 : Successfully turned off the pump
% -1 : Unknown command
%
% Output from the IKA Magic Lab
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [controllerOutput] = controlAuxiliaryEquipments(portName, serialCommand)

%% CREATE CONNECTION WITH uCONTROLLER & PARSE ARGUMENTS
% Create a serial object with the port specified by the user
serialObj = serial(portName);

% Check if the command is directed towards the IKA Magic Lab. If the
% command is directed towards the Magic lab, then the serial port
% settings needs to be adapted accordingly.
if strcmpi(serialCommand(2),'@')
    set(serialObj,...
        'BaudRate',57600,...
        'DataBits',7,...
        'FlowControl','none',...
        'Parity','even',...
        'StopBits',1,...
        'Timeout',5);
    flagIKA = true;
else
    set(serialObj,...
        'BaudRate',9600,...
        'Timeout',5);
    flagIKA = false;
end

%% OPEN THE PORT, SEND THE COMMAND AND CLOSE THE CONNCETION
% Open connection with the microcontroller
fopen(serialObj);

% Send a command to the microcontroller to regulate the valve
fprintf(serialObj, '%s', serialCommand);

% Read response from the microcontroller and print it out
% If the mill is being commanded do not convert the ASCII text to num
if flagIKA
    controllerOutput = fscanf(serialObj,'%s');
% For all other cases convert the reply to a number (self defined)
else
    controllerOutput = str2num(fscanf(serialObj,'%s'));
end

% Terminate the connection with the microcontroller
fclose (serialObj);

% Delete the serial object
delete (serialObj);
end