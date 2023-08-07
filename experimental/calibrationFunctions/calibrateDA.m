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
% Calibrates the dilute analyser for different set point values
%
% Last modified:
% - 2022-05-04, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function calibrateDA(varargin)
    % Sampling time for the device
    expInfo.samplingTime = 1;
    % Define gas for MFM
    expInfo.gasName_MFM = 'He';
    % Define gas for MFC1
    expInfo.gasName_MFC1 = 'He';
    % Define gas for MFC2
    expInfo.gasName_MFC2 = 'He';
    % Define set point for MFC2
    % Round the flow rate to the nearest first decimal (as this is the
    % resolution of the meter)
    if ~varargin{1}
        MFC2_SP = round(repmat([0.0 0.2 0.4 3.0 6.0 9.0 10.5 12.0 15 29.6 29.8 30.0],[1,1]),1);
    else
        MFC2_SP = varargin{1};
    end
    % Define set point for MFC1
    % Round the flow rate to the nearest first decimal (as this is the
    % resolution of the meter)    
    MFC1_SP = round(max(MFC2_SP)-MFC2_SP,1);
    % Experiment name
    expInfo.expName = ['ZLCCalibrateDA','_',...
        datestr(datetime('now'),'yyyymmdd'),'_',num2str(max(MFC2_SP)),'ccm'];
    % Start delay
    expInfo.equilibrationTime = 5; % [s]
    % Flag for meter calibration
    expInfo.calibrateMeters = true;
    % Mixtures Flag - When a T junction instead of 6 way valve used
    expInfo.runMixtures = false;
    % Loop through all setpoints to calibrate the meters
    for ii=1:length(MFC1_SP)
        expInfo.MFC1_SP = MFC1_SP(ii);
        expInfo.MFC2_SP = MFC2_SP(ii);
        % When the set point goes back to zero wait for 5 more min before
        % starting the measurement
        if ii == find(MFC1_SP == 0,1,'last')
            % Maximum time of the experiment
            % Change the max time to 10 min
            expInfo.maxTime = 300;
        else
            % Else use 5 min
            expInfo.maxTime = 300;
        end
        % Run the setup for different calibrations
        runZLC(expInfo)
    end
end
