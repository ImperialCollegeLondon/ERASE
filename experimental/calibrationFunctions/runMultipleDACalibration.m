%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Imperial College London, United Kingdom
% Multifunctional Nanomaterials Laboratory
%
% Project:  ERASE
% Year:     2022
% MATLAB:   R2020a
% Authors:  Ashwin Kumar Rajagopalan (AK)
%           Hassan Azzan (HA)
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
totalFlowRate = [60 100];

% Loop through all total flow rates
for ii=1:length(totalFlowRate)
    % Define the set point for MFC2 (CO2)
    % This is done on a log scale to have a high reoslution at low CO2
    % concentrations. Linear spaced flow rates for high compositions
    MFC2 = unique([round(linspace(0,max(totalFlowRate(ii)),20),1)]);
    pause(400);
    % Call calibrateMS function
    calibrateDA(MFC2)
    % Change the flow rate of CO2 to 0 and of He to the next set point to
    % equilibrate
    if ii<length(totalFlowRate)
        defineSetPtManualDilute(totalFlowRate(ii+1),0)
    end
end
% Turn off CO2 flow and set a low He flow so that there is constant supply
% of gas to the MS
defineSetPtManualDilute(0,0)