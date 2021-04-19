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
% - 2021-04-19, AK: Major revamp for flow rate computation
% - 2021-04-13, AK: Add threshold to cut data below a given mole fraction
% - 2021-04-08, AK: Add ratio of gas for calibration
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

% Get the git commit ID
gitCommitID = getGitCommit;

% Flag to decide calibration or analysis
flagCalibration = false;

% Flag to decide calibration of flow meters or MS
flagFlowMeter = true;

% Mode to switch between calibration and analyzing real experiment
% Analyze calibration data
if flagCalibration
    % Calibrate flow meter
    if flagFlowMeter
        % File with the calibration data to build a model for MFC/MFM
        experimentStruct = 'ZLCCalibrateMeters_20210419_E'; % Experimental flow file (.mat)     
        % Call analyzeCalibration function for calibration of the MS
        analyzeCalibration(experimentStruct,[]) % Call the function to generate the calibration file        
    % Calibrate MS
    else
        experimentStruct.calibrationFlow = 'ZLCCalibrateMeters_20210316__Model'; % Calibration file for meters (.mat)
        experimentStruct.flow = 'ZLCCalibrateMS_20210414'; % Experimental flow file (.mat)
        experimentStruct.MS = 'C:\Users\QCPML\Desktop\Ashwin\MS\ZLCCalibrateMS_20210414.asc'; % Experimental MS file (.asc)
        experimentStruct.interpMS = true; % Flag for interpolating MS data (true) or flow data (false)
        experimentStruct.numMean = 10; % Number of points for averaging
        experimentStruct.flagUseIndGas = false; % Flag to determine whether independent (true) or ratio of signals used for calibration
        experimentStruct.polyDeg = 3; % Degree of polynomial fit for independent gas calibraiton
        % Call analyzeCalibration function for calibration of the MS
        analyzeCalibration([],experimentStruct) % Call the function to generate the calibration file
    end
% Analyze real experiment    
else
    moleFracThreshold = 1e-3; % Threshold to cut data below a given mole fraction [-]
    experimentStruct.calibrationFlow = 'ZLCCalibrateMeters_20210419_E_Model'; % Calibration file for meters (.mat)
    experimentStruct.flow = 'ZLC_ActivatedCarbon_Exp10A'; % Experimental flow file (.mat)
    experimentStruct.calibrationMS = 'ZLCCalibrateMS_20210414_Model'; % Experimental calibration file (.mat)
    experimentStruct.MS = 'C:\Users\QCPML\Desktop\Ashwin\MS\ZLC_ActivatedCarbon_Exp10.asc'; % Experimental MS file (.asc)
    experimentStruct.interpMS = true; % Flag for interpolating MS data (true) or flow data (false)
    % Call reconcileData function to get the output mole fraction for a
    % real experiment
    [outputStruct,~] = concatenateData(experimentStruct);
    
    % Clean mole fraction to remove negative values (due to calibration)
    % Replace all negative molefraction with eps
    outputStruct.moleFrac(outputStruct.moleFrac(:,2)<0,1)=eps; % CO2
    outputStruct.moleFrac(:,1)=1-outputStruct.moleFrac(:,2); % Compute He with mass balance
   
    % Convert the MFM flow to real flow
    % Load the meter calibrations
    load(experimentStruct.calibrationFlow);
    % Get the MFM flow rate
    volFlow_MFM = outputStruct.flow(:,2);
    % Get the CO2 mole fraction for obtaining real flow rate
    moleFracCO2 = outputStruct.moleFrac(:,2);
    % Compute the total flow rate of the gas [ccm]
    totalFlowRate = calibrationFlow.MFM(moleFracCO2,volFlow_MFM);
    
    % Input for the ZLC script (Python)
    % Find the index for the mole fraction that corresponds to the
    % threshold mole fraction
    moleFracThresholdInd = find(outputStruct.moleFrac(:,2)<moleFracThreshold,1,'first');
    % Set the final index to be the length of the series, if threshold not
    % reached
    if isempty(moleFracThresholdInd)
        moleFracThresholdInd = length(outputStruct.moleFrac(:,2));
    end
    experimentOutput.timeExp = outputStruct.flow(1:moleFracThresholdInd,1); % Time elapsed [s]
    experimentOutput.moleFrac = outputStruct.moleFrac(1:moleFracThresholdInd,2); % Mole fraction CO2 [-]
    experimentOutput.totalFlowRate = totalFlowRate(1:moleFracThresholdInd)./60; % Total flow rate of the gas [ccs]

    % Save the experimental output into a .mat file
    % Check if runData data folder exists
    if exist(['experimentalData',filesep,...
            'runData'],'dir') == 7
        % Save the calibration data for further use
        save(['experimentalData',filesep,...
            'runData',filesep,experimentStruct.flow,'_Output'],'experimentOutput',...
            'gitCommitID');
    else
        % Create the calibration data folder if it does not exist
        mkdir(['experimentalData',filesep,'runData'])
        % Save the calibration data for further use
        save(['experimentalData',filesep,...
            'runData',filesep,experimentStruct.flow,'_Output'],'experimentOutput',...
            'gitCommitID');
    end
end