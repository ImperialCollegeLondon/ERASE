%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Imperial College London, United Kingdom
% Multifunctional Nanomaterials Laboratory
%
% Project:  ERASE
% Year:     2021
% MATLAB:   R2020a
% Authors:  Ashwin Kumar Rajagopalan (AK)
%
% Purpose: 
% Runs multiple MS calibration with different total flow rates
%
% Last modified:
% - 2021-04-30, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the total flow rate of interest
totalFlowRate = [5, 7, 9 , 15, 30, 45, 60, 90];

% Loop through all total flow rates
for ii=1:length(totalFlowRate)
    % Define the set point for MFC2 (CO2)
    % This is done on a log scale to have a high reoslution at low CO2
    % concentrations
    MFC2 = round(logspace(log10(0.2),log10(totalFlowRate(ii)),20),1);
    % Call calibrateMS function
    calibrateMS(MFC2)
end