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
% - commandToBeAnalyzed   : Command that needs to be analysed to generate
%                           serial command 
%
% - flagAlicat            : Determines whether or not the equipment is from Alicat
%
% Output arguments:
% - serialCommand   : Command that would be issued to the microcontoller.
%
% Communication protocol:
%   Commands to be analysed to control Alicat MFC and MFM: 
%       setPoint      : Set a new set-point for the MFC
%       pollData      : Poll the current data from the MFC/MFM
%       ...
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [serialCommand] = generateSerialCommand(commandToBeAnalyzed, flagAlicat, varargin)

%% GENERATE SERIAL COMMAND FOR THE CONTROLLER
if flagAlicat
    switch commandToBeAnalyzed
        case 'setPoint'
            serialCommand = ["as","vargin(1)"];
        case 'pollData'
            serialCommand = "a??d";
    end
end
end