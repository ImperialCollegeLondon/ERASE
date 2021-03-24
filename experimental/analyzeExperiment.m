%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Imperial College London, United Kingdom
% Multifunctional Nanomaterials Laboratory
%
% Project:  ERASE
% Year:     2021
% MATLAB:   R2020a
% Authors:  Ashwin Kumar Rajagopalan (AK)
%           Hassan Azzan (HA)
%
% Purpose: 
% Script to define inputs to calibrate flowmeter and MS or to analyze a
% real experiment using calibrated flow meters and MS
%
% Last modified:
% - 2021-03-24, AK: Add flow rate computation and prepare structure for
%                   Python script
% - 2021-03-18, AK: Updates to structure
% - 2021-03-18, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

flagCalibration = false; % Flag to decide calibration or analysis

% Mode to switch between calibration and analyzing real experiment
if flagCalibration
    experimentStruct.calibrationFlow = 'ZLCCalibrateMeters_20210316__Model'; % Calibration file for meters (.mat)
    experimentStruct.flow = 'ZLCCalibrateMS_20210324'; % Experimental flow file (.mat)
    experimentStruct.MS = 'C:\Users\QCPML\Desktop\Ashwin\MS\ZLCCalibrateMS_20210324.asc'; % Experimental MS file (.asc)
    experimentStruct.interpMS = true; % Flag for interpolating MS data (true) or flow data (false)
    experimentStruct.numMean = 10; % Number of points for averaging
    experimentStruct.polyDeg = 3; % Degree of polynomial fit
    % Call reconcileData function for calibration of the MS
    analyzeCalibration([],experimentStruct) % Call the function to generate the calibration file
else
    setTotalFlowRate = 15;
    experimentStruct.calibrationFlow = 'ZLCCalibrateMeters_20210316__Model'; % Calibration file for meters (.mat)
    experimentStruct.flow = 'ZLC_DeadVolume_Exp05A'; % Experimental flow file (.mat)
    experimentStruct.calibrationMS = 'ZLCCalibrateMS_20210323_Model'; % Experimental flow file (.mat)
    experimentStruct.MS = 'C:\Users\QCPML\Desktop\Ashwin\MS\ZLC_DeadVolume_Exp05A.asc'; % Experimental MS file (.asc)
    experimentStruct.interpMS = true; % Flag for interpolating MS data (true) or flow data (false)
    % Call reconcileData function to get the output mole fraction for a
    % real experiment
    outputStruct = concatenateData(experimentStruct);
    
    % Clean mole fraction to remove negative values (due to calibration)
    % Replace all negative molefraction with eps
    outputStruct.moleFrac(outputStruct.moleFrac(:,1)<0,1)=eps; % He
    outputStruct.moleFrac(:,2)=1-outputStruct.moleFrac(:,1); % Compute CO2 with mass balance
    
    % Convert the MFM flow to real flow
    % Flow rate when there is pure He (He equivalent)
    flowRateAtPureHe = setTotalFlowRate*calibrationFlow.MFC_He/calibrationFlow.MFM_He;
    % Flow rate when there is pure CO2 (He equivalent)
    flowRateAtPureCO2 = setTotalFlowRate*calibrationFlow.MFC_CO2/calibrationFlow.MFM_CO2;
    % Compute the slope of the line (to relate mole fraction of CO2 to the
    % change in the flow rate of He equivalent in MFM)
    delFlowRate = flowRateAtPureHe - flowRateAtPureCO2;
    % Compute the flow of CO2 (He equivalent) at the different mole
    % fractions as the experiment progress (by doing a mass balance)
    flowRateCO2 = outputStruct.moleFrac(:,2).*(flowRateAtPureHe-delFlowRate.*outputStruct.moleFrac(:,2));
    % Compute the real flow rate of CO2 (by performing a transformation
    % from He equivalent CO2 to UMFM) [ccm]
    realFlowRateCO2 = flowRateCO2*calibrationFlow.MFM_CO2;
    % Compute the flow of He (He equivalent) at the different mole
    % fractions as the experiment progress (by doing a mass balance)
    flowRateHe = (1-outputStruct.moleFrac(:,2)).*(flowRateAtPureHe-delFlowRate.*outputStruct.moleFrac(:,2));
    % Compute the real flow rate of He (by performing a transformation
    % from He equivalent He to UMFM) [ccm]
    realFlowRateHe = flowRateHe*calibrationFlow.MFM_He;
    % Compute the total flow rate of the gas [ccm]
    totalFlowRate = realFlowRateHe+realFlowRateCO2;
    
    % Input for the ZLC script (Python)
    experimentOutput.timeExp = outputStruct.flow(:,1); % Time elapsed [s]
    experimentOutput.moleFrac = outputStruct.moleFrac(:,2); % Mole fraction CO2 [-]
    experimentOutput.totalFlowRate = totalFlowRate./60; % Total flow rate of the gas [ccs]
end