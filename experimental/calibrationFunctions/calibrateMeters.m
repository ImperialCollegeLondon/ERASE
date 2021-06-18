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
% Calibrates the flow meter and controller for different set point values
%
% Last modified:
% - 2021-04-19, AK: Change from individual flow to total flow rate
% - 2021-04-19, AK: Change functionality for mixtures
% - 2021-03-16, AK: Add calibrate meters flag
% - 2021-03-12, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function calibrateMeters
    % Experiment name
    expInfo.expName = ['ZLCCalibrateMeters','_',...
        datestr(datetime('now'),'yyyymmdd')];
    % Maximum time of the experiment
    expInfo.maxTime = 30;
    % Sampling time for the device
    expInfo.samplingTime = 5;
    % Define gas for MFM
    expInfo.gasName_MFM = 'He';
    % Define gas for MFC1
    expInfo.gasName_MFC1 = 'He';
    % Define gas for MFC2
    expInfo.gasName_MFC2 = 'CO2';
    % Set the total flow rate for the calibration
    totalFlowRate = [0.0, 2.0, 4.0, 15.0, 30.0, 45.0, 60.0, 80.0, 100.0];
    % Mole fraction of CO2 desired
    moleFracCO2 = 0:0.1:1;
    % Define set point for MFC1
    % Round the flow rate to the nearest first decimal (as this is the
    % resolution of the meter)    
    MFC1_SP = round(totalFlowRate'*(1-moleFracCO2),1);
    % Define set point for MFC2
    % Round the flow rate to the nearest first decimal (as this is the
    % resolution of the meter)    
    MFC2_SP = round(totalFlowRate'*moleFracCO2,1);
    % Start delay
    expInfo.equilibrationTime = 10; % [s]
    % Flag for meter calibration
    expInfo.calibrateMeters = true;
    % Mixtures Flag - When a T junction instead of 6 way valve used
    expInfo.runMixtures = false;    

    % Loop through all setpoints to calibrate the meters
    for ii=1:length(totalFlowRate)
        for jj=1:length(moleFracCO2)
            % Set point for MFC1
            expInfo.MFC1_SP = MFC1_SP(ii,jj);
            % Set point for MFC2           
            expInfo.MFC2_SP = MFC2_SP(ii,jj);            
            % Flag for meter calibration
            expInfo.calibrateMeters = true;
            % Run the setup for different calibrations
            runZLC(expInfo)
        end
    end
end
