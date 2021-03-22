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
% Calibrates the mass specfor different set point values
%
% Last modified:
% - 2021-03-16, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function calibrateMS
    % Experiment name
    expInfo.expName = ['ZLCCalibrateMS','_',...
        datestr(datetime('now'),'yyyymmdd')];
    % Maximum time of the experiment
    expInfo.maxTime = 300;
    % Sampling time for the device
    expInfo.samplingTime = 2;
    % Define gas for MFM
    expInfo.gasName_MFM = 'He';
    % Define gas for MFC1
    expInfo.gasName_MFC1 = 'He';
    % Define gas for MFC2
    expInfo.gasName_MFC2 = 'CO2';
    % Define set point for MFC1
    MFC1_SP = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5 15.0];
    % Define set point for MFC2
    MFC2_SP = max(MFC1_SP)-MFC1_SP;
    % Loop through all setpoints to calibrate the meters
    for ii=1:length(MFC1_SP)
        expInfo.MFC1_SP = MFC1_SP(ii);
        expInfo.MFC2_SP = MFC2_SP(ii);
        % Flag for meter calibration
        expInfo.calibrateMeters = true;
        % Run the setup for different calibrations
        runZLC(expInfo)
    end
end
