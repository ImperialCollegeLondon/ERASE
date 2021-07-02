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
% - 2021-07-02, AK: Bug fix for threshold
% - 2021-05-10, AK: Convert into a function
% - 2021-04-20, AK: Add experiment struct to output .mat file
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
function analyzeExperiment(experimentStruct,flagCalibration,flagFlowMeter)
    % Get the git commit ID
    gitCommitID = getGitCommit;

    % Mode to switch between calibration and analyzing real experiment
    % Analyze calibration data
    if flagCalibration
        % Calibrate flow meter
        if flagFlowMeter
            % File with the calibration data to build a model for MFC/MFM
            experimentStruct = 'ZLCCalibrateMeters_20210419'; % Experimental flow file (.mat)     
            % Call analyzeCalibration function for calibration of the MS
            analyzeCalibration(experimentStruct,[]) % Call the function to generate the calibration file        
        % Calibrate MS
        else
            % Call analyzeCalibration function for calibration of the MS
            analyzeCalibration([],experimentStruct) % Call the function to generate the calibration file
        end
    % Analyze real experiment
    else
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
        % Round the flow rate to the nearest first decimal (as this is the
        % resolution of the meter)
        totalFlowRate = round(calibrationFlow.MFM(moleFracCO2,volFlow_MFM),1);

        % Input for the ZLC script (Python)
        % Find the index for the mole fraction that corresponds to the
        % threshold mole fraction
        moleFracThresholdInd = min([find(outputStruct.moleFrac(:,2)<experimentStruct.moleFracThreshold,1,'first'),...
                                   find(~isnan(totalFlowRate),1,'last')]); % Additional check to weed out nan flow due to interpolation
        % Set the final index to be the length of the series, if threshold not
        % reached
        if isempty(moleFracThresholdInd)
            moleFracThresholdInd = length(outputStruct.moleFrac(:,2));
        end
        experimentOutput.timeExp = outputStruct.flow(1:moleFracThresholdInd,1); % Time elapsed [s]
        experimentOutput.moleFrac = outputStruct.moleFrac(1:moleFracThresholdInd,2); % Mole fraction CO2 [-]
        experimentOutput.totalFlowRate = totalFlowRate(1:moleFracThresholdInd)./60; % Total flow rate of the gas [ccs]
        % Save outputStruct to semiProcessedStruct
        semiProcessedStruct = outputStruct; % Check concatenateData for more (this is reconciledData there)
        % Save the experimental output into a .mat file
        % Check if runData data folder exists
        if exist(['..',filesep,'experimentalData',filesep,...
                'runData'],'dir') == 7
            % Save the calibration data for further use
            save(['..',filesep,'experimentalData',filesep,...
                'runData',filesep,experimentStruct.flow,'_Output'],'experimentOutput',...
                'experimentStruct','semiProcessedStruct','gitCommitID');
        else
            % Create the calibration data folder if it does not exist
            mkdir(['..',filesep,'experimentalData',filesep,'runData'])
            % Save the calibration data for further use
            save(['..',filesep,'experimentalData',filesep,...
                'runData',filesep,experimentStruct.flow,'_Output'],'experimentOutput',...
                'experimentStruct','semiProcessedStruct','gitCommitID');
        end
    end
end