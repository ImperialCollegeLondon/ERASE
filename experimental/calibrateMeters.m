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
    expInfo.samplingTime = 2;
    % Define gas for MFM
    expInfo.gasName_MFM = 'He';
    % Define gas for MFC1
    expInfo.gasName_MFC1 = 'He';
    % Define gas for MFC2
    expInfo.gasName_MFC2 = 'CO2';
    % Define set point for MFC1
    MFC1_SP = [0.0, 10.0, 20.0];
    
    % Loop through all setpoints to calibrate the meters
    for ii=1:length(MFC1_SP)
        expInfo.MFC1_SP = MFC1_SP(ii);
        % Run the setup for different calibrations
        runZLC(expInfo)
    end
end
