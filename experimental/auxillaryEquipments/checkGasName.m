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
% Function to generate the gas ID required for alicat devices
%
% Last modified:
% - 2021-03-01, HA: Initial creation
%
% Input arguments:
% - Gas name    : Name of the gas used in Alicat equipment (CO2, He, CH4, H2, N2)
%
% Output arguments:
% - gasID       : ID of the gas needed for 'controlAuxiliaryEquipments.m'
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gasID = checkGasName(gasName)
% Requires > MATLAB2020a
switch gasName
    case 'CO2'
        gasID = "ag4";
    case 'He'
        gasID = "ag7"; 
    case 'H2'
        gasID = "ag6";
    case 'CH4'
        gasID = "ag2";
    case 'N2'
        gasID = "ag8";            
end
end