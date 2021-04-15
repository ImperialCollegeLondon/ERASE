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
% - 2021-04-15, AK: Modify function for mixture experiments
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
    % Sampling time for the device
    expInfo.samplingTime = 2;
    % Define gas for MFM
    expInfo.gasName_MFM = 'He';
    % Define gas for MFC1
    expInfo.gasName_MFC1 = 'He';
    % Define gas for MFC2
    expInfo.gasName_MFC2 = 'CO2';
    % Define set point for MFC1
    MFC1_SP = repmat([0.0 0.2 1.5 3.0 4.5, 6.0 7.5, 9.0 10.5 12.0 13.5 14.8 15.0],[1,2]);
    % Define set point for MFC2
    MFC2_SP = 15.0-MFC1_SP;
    % Start delay (used for adsorbent equilibration)
    expInfo.equilibrationTime = 1800; % [s]
    % Flag for meter calibration
    expInfo.calibrateMeters = false;
    % Mixtures Flag - When a T junction instead of 6 way valve used
    expInfo.runMixtures = true;
    % Loop through all setpoints to calibrate the meters
    for ii=1:length(MFC1_SP)
        expInfo.MFC1_SP = MFC1_SP(ii);
        expInfo.MFC2_SP = MFC2_SP(ii);
        % When the set point goes back to zero wait for 5 more min before
        % starting the measurement
        if ii == find(MFC1_SP == 0,1,'last')
            % Maximum time of the experiment
            % Change the max time to 10 min
            expInfo.maxTime = 600;
        else
            % Else use 5 min
            expInfo.maxTime = 300;
        end
        % Run the setup for different calibrations
        runZLC(expInfo)
    end
end
