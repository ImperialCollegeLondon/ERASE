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
% Runs multiple ZLC experiments in series
%
% Last modified:
% - 2021-04-20, AK: Add multiple equilibration time
% - 2021-04-15, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function runMultipleZLC
    % Series name for the experiments
    expSeries = 'ZLC_ActivatedCarbon_Exp25';
    % Maximum time of the experiment
    expInfo.maxTime = 400;
    % Sampling time for the device
    expInfo.samplingTime = 1;
    % Intervals for collecting MFC data
    expInfo.MFCInterval = 300;
    % Define gas for MFM
    expInfo.gasName_MFM = 'He';
    % Define gas for MFC1
    expInfo.gasName_MFC1 = 'He';
    % Define gas for MFC2
    expInfo.gasName_MFC2 = 'CO2';
    % Total flow rate
    expTotalFlowRate = [10, 10, 10, 10, 10, 10];
    % Fraction CO2
    fracCO2 = [1/8 1/3 1 2 4 10];
    % Define set point for MFC1
    % Round the flow rate to the nearest first decimal (as this is the
    % resolution of the meter)    
    MFC1_SP = round(expTotalFlowRate,1);
    % Define set point for MFC2
    % Round the flow rate to the nearest first decimal (as this is the
    % resolution of the meter)    
    MFC2_SP = round(fracCO2.*expTotalFlowRate,1);
    % Start delay (used for adsorbent equilibration)
    equilibrationTime = [14400 3600 3600 3600 3600 3600]; % [s] 
    % Flag for meter calibration
    expInfo.calibrateMeters = false;    
    % Mixtures Flag - When a T junction instead of 6 way valve used
    expInfo.runMixtures = true;
    % Loop through all setpoints to calibrate the meters
    for ii=1:length(MFC1_SP)
        % Experiment name
        expInfo.expName = [expSeries,char(64+ii)];
        expInfo.equilibrationTime = equilibrationTime(ii);
        expInfo.MFC1_SP = MFC1_SP(ii);
        expInfo.MFC2_SP = MFC2_SP(ii);
        % Run the setup for different calibrations
        runZLC(expInfo)
        % Wait for 1 min before starting the next experiment
        pause(30)
    end
    defineSetPtManual(10,0)
end
