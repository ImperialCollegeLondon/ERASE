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
    expInfo.maxTime = 60;
    % Sampling time for the device
    expInfo.samplingTime = 5;
    % Define gas for MFM
    expInfo.gasName_MFM = 'He';
    % Define gas for MFC1
    expInfo.gasName_MFC1 = 'He';
    % Define gas for MFC2
    expInfo.gasName_MFC2 = 'CO2';
    % Define set point for MFC1
    MFC1_SP = [0.0, 15.0, 30.0, 45.0, 60.0];
    % Start delay
    expInfo.equilibrationTime = 0; % [s]
    % Flag for meter calibration
    expInfo.calibrateMeters = true;
    % Mixtures Flag - When a T junction instead of 6 way valve used
    expInfo.runMixtures = false;    

    % Loop through all setpoints to calibrate the meters
    for ii=1:length(MFC1_SP)
        for jj=1:length(MFC1_SP)
            % Set point for MFC1
            expInfo.MFC1_SP = MFC1_SP(ii);
            % Set point for MFC2 (same as MFC1)            
            expInfo.MFC2_SP = MFC1_SP(jj);            
            % Flag for meter calibration
            expInfo.calibrateMeters = true;
            % Run the setup for different calibrations
            runZLC(expInfo)
        end
    end
end
