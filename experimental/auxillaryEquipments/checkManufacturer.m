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
% Output arguments:
% - flagAlicat      : Determines whether or not the equipment is from Alicat
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function flagAlicat = checkManufacturer(portProperty)

%% CREATE CONNECTION WITH uCONTROLLER & PARSE ARGUMENTS
% Create a serial object with the port and baudrate specified by the user
% Requires > MATLAB2020a
flagAlicat = 0;

switch portProperty.portName
    case 'COM5'
        flagAlicat = 1;
    case 'COM6'
        flagAlicat = 0;      
end
end