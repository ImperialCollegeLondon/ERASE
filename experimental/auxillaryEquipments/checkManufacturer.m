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
% Function to check if the device connected to a port is Alicat or not
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
flagAlicat = 0;
switch portProperty.portName
    case 'COM6'
        flagAlicat = 1;
    case 'COM5'
        flagAlicat = 0;      
end
end