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
% - 2021-05-17, AK: Add check for CO2 set point
% - 2021-05-14, AK: Add flow rate sweep functionality
% - 2021-04-20, AK: Add multiple equilibration time
% - 2021-04-15, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function runMultipleZLCDA
    run('C:\Users\ha3215\Documents\ERASE\setPathERASE.m')
    % Series name for the experiments
    expSeries = {'ZLC_13X_DA_Exp11',...
                 'ZLC_13X_DA_Exp12'};
    % Maximum time of the experiment
    maxTime = [2000,2000;...
               600,600];
    % Sampling time for the device
    expInfo.samplingTime = 1;
    % Intervals for collecting MFC data
    expInfo.MFCInterval = 100;
    % Define gas for MFM
    expInfo.gasName_MFM = 'He';
    % Define gas for MFC1
    expInfo.gasName_MFC1 = 'He';
    % Define gas for MFC2
    expInfo.gasName_MFC2 = 'He';
    % Experimental Temperature [degC]
    expInfo.Temperature = 35;
    % Total flow rate
%     expTotalFlowRate = [10, 10;...
%                         60, 60];

    expTotalFlowRate = [15, 15;...
                        60, 60];
    % Fraction CO2
%     fracCO2 = [1/8, 16;...
%                1/8, 2.66];

    fracCO2 = [1/8, 10.66;...
               1/8, 2.66];

    % Define set point for MFC1
    % Round the flow rate to the nearest first decimal (as this is the
    % resolution of the meter)    
    MFC1_SP = round(expTotalFlowRate,1);
    % Define set point for MFC2
    % Round the flow rate to the nearest first decimal (as this is the
    % resolution of the meter)    
    MFC2_SP = round(fracCO2.*expTotalFlowRate,1);
    % Start delay (used for adsorbent equilibration)
%     equilibrationTime = repmat([900, 900],[length(expSeries),1]); % [s] 
    equilibrationTime = [300,600;...
                         500,500];
    % Flag for meter calibration
    expInfo.calibrateMeters = false;    
    % Mixtures Flag - When a T junction instead of 6 way valve used
    expInfo.runMixtures = true;
    % pause for temp equilibration
%     pause(2400)
    % Loop through all setpoints to calibrate the meters
    for jj=1:size(MFC1_SP,1) 
        for ii=1:size(MFC1_SP,2)
            % The MFC can support only 200 sccm (180 ccm is borderline)
            % Keep an eye out
            % If MFC2SP > 180 ccm break and move to the next operating
            % condition
            if MFC2_SP(jj,ii) >= 180
%                 break;
            end
            % Experiment name
            expInfo.expName = [expSeries{jj},char(64+ii)];
            expInfo.equilibrationTime = equilibrationTime(jj,ii);
            expInfo.maxTime = maxTime(jj,ii);
            expInfo.MFC1_SP = MFC1_SP(jj,ii);
            expInfo.MFC2_SP = MFC2_SP(jj,ii);
            % Run the setup for different calibrations
            runZLC(expInfo)
            % Wait for 1 min before starting the next experiment
            pause(60)
        end
    end
    defineSetPtManualDilute(0,0)
end