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
totalFlowRate = [5 10 15 30 45 60];

% Loop through all total flow rates
for ii=1:length(totalFlowRate)
    % Define the set point for MFC2 (CO2)
    % This is done on a log scale to have a high reoslution at low CO2
    % concentrations. Linear spaced flow rates for high compositions
    MFC2 = unique([0 round(logspace(log10(0.2),log10(1),10),1) ...
        round(linspace(1,max(totalFlowRate(ii)),10),1)]);
    pause(3600);
    % Call calibrateMS function
    calibrateMS(MFC2)
    % Change the flow rate of CO2 to 0 and of He to the next set point to
    % equilibrate
    if ii<length(totalFlowRate)
        defineSetPtManual(totalFlowRate(ii+1),0)
    end
end
% Turn off CO2 flow and set a low He flow so that there is constant supply
% of gas to the MS
defineSetPtManual(10,0)