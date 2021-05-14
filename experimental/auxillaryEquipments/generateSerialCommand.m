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
% - 2021-03-02, HA: Initial creation
%
% Input arguments:
% - commandToBeAnalyzed: Command that needs to be analysed to generate
%                        serial command
% - varargin:            Arguments to determine the device and the set pt    
%
% Output arguments:
% - serialCommand   : Command that would be issued to the device
%
% Communication protocol:
% Commands to be analysed to control Alicat MFC and MFM:
%       setPoint      : Set a new set-point for the MFC
%       pollData      : Poll the current data from the MFC/MFM
%       ...
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [serialCommand] = generateSerialCommand(commandToBeAnalyzed, varargin)
% Generate serial command for the device
% If Alicat is used: variable argument 1 is true!
if varargin{1}
    switch commandToBeAnalyzed
        case 'setPoint'
            % Set the set point value for the controller 
            setPointValue = round(varargin{2},2);
            serialCommand = ['as',num2str(setPointValue)];
        case 'pollData'
            serialCommand = 'a??d';
    end
end
end