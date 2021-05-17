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
function runMultipleZLC
    % Series name for the experiments
    expSeries = {'ZLC_ActivatedCarbon_Exp26',...
                 'ZLC_ActivatedCarbon_Exp27',...
                 'ZLC_ActivatedCarbon_Exp28',...
                 'ZLC_ActivatedCarbon_Exp29',...
                 'ZLC_ActivatedCarbon_Exp30',...
                 'ZLC_ActivatedCarbon_Exp31'};
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
    expTotalFlowRate = [10, 10, 10, 10, 10, 10;...
                        15, 15, 15, 15, 15, 15;...
                        30, 30, 30, 30, 30, 30;...
                        45, 45, 45, 45, 45, 45;...
                        60, 60, 60, 60, 60, 60;...
                        80, 80, 80, 80, 80, 80];
    % Fraction CO2
    fracCO2 = repmat([1/8 1/3 1 2 4 10],[length(expSeries),1]);
    % Define set point for MFC1
    % Round the flow rate to the nearest first decimal (as this is the
    % resolution of the meter)    
    MFC1_SP = round(expTotalFlowRate,1);
    % Define set point for MFC2
    % Round the flow rate to the nearest first decimal (as this is the
    % resolution of the meter)    
    MFC2_SP = round(fracCO2.*expTotalFlowRate,1);
    % Start delay (used for adsorbent equilibration)
    equilibrationTime = repmat([7200 3600 3600 3600 3600 3600],[length(expSeries),1]); % [s] 
    % Flag for meter calibration
    expInfo.calibrateMeters = false;    
    % Mixtures Flag - When a T junction instead of 6 way valve used
    expInfo.runMixtures = true;
    % Loop through all setpoints to calibrate the meters
    for jj=1:size(MFC1_SP,1) 
        for ii=1:size(MFC1_SP,2)
            % The MFC can support only 200 sccm (180 ccm is borderline)
            % Keep an eye out
            % If MFC2SP > 180 ccm break and move to the next operating
            % condition
            if MFC2_SP(jj,ii) >= 180
                break;
            end
            % Experiment name
            expInfo.expName = [expSeries{jj},char(64+ii)];
            expInfo.equilibrationTime = equilibrationTime(jj,ii);
            expInfo.MFC1_SP = MFC1_SP(jj,ii);
            expInfo.MFC2_SP = MFC2_SP(jj,ii);
            % Run the setup for different calibrations
            runZLC(expInfo)
            % Wait for 1 min before starting the next experiment
            pause(30)
        end
    end
    defineSetPtManual(10,0)
end
