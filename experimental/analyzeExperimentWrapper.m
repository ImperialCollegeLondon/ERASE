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
% A wrapper function that generates the calibration data to be used for
% experiments and subsequently analyze experimental data using calibrated
% MS model to generate the response curve from a "ZLC/Breakthrough"
% experiment
%
% Last modified:
% - 2021-05-10, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% NOTE: Move analysis scripts to an analyze folder (change path for file saving) - analyzeCalibration, analyzeExperiment, 

% List the flow meter calibration file (this is usually in calibraiton folder)
flowMeterCalibration = 'ZLCCalibrateMeters_20210419_Model';

%%%% Calibration files to be used %%%%
% List the MS calibration files (this is usually in experimental data folder)
msFileDir = 'C:\Users\QCPML\Desktop\Ashwin\MS'; % Directory with MS data
msRawFiles = {'ZLCCalibrateMS_20210505'}; % Raw MS data file names for all calibration
numExpForEachRawFile = [6]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files)

% Flow rate files for calibration 
msCalibrationFiles = {'ZLCCalibrateMS_20210505_5ccm',...
                    'ZLCCalibrateMS_20210506_10ccm',...
                    'ZLCCalibrateMS_20210506_15ccm',...
                    'ZLCCalibrateMS_20210506_30ccm',...
                    'ZLCCalibrateMS_20210506_45ccm',...
                    'ZLCCalibrateMS_20210507_60ccm'};

%%%% Experimet to be analyzed %%%%     
% List the experiments that have to be analyzed
msExpFile = 'ZLC_DeadVolume_Exp15'; % Raw MS data file name
% Flow rate files for experiments 
experimentFiles = {'ZLC_DeadVolume_Exp15A'};

% Initialize the name of the msRawFile to be used for all calibrations
startInd = 1;
msRawFileALL={};
% Loop over the number of experiments per calibration ms raw file
% Generate the name of the MS file that will be used for each flow rate
% .mat file
for ii = 1:length(numExpForEachRawFile)
    for jj = startInd:startInd+numExpForEachRawFile(ii)-1
        msRawFileALL{jj} = msRawFiles{ii};
    end
    startInd =  length(msRawFileALL) + 1;
end

% % Loop through all the MS calibration files
% for ii = 1:length(msCalibrationFiles)
%     experimentStruct.calibrationFlow = flowMeterCalibration; % Calibration file for meters (.mat)
%     experimentStruct.flow = msCalibrationFiles{ii}; % Experimental flow file (.mat)
%     experimentStruct.MS = [msFileDir,filesep,msRawFileALL{ii},'.asc']; % Experimental MS file (.asc)
%     experimentStruct.interpMS = true; % Flag for interpolating MS data (true) or flow data (false)
%     experimentStruct.numMean = 10; % Number of points for averaging
%     % Call the analyzeExperiment function to calibrate the MS at the conditions
%     % experiment was performed for calibration
%     % The output calibration model is usually in calibration folder
%     % Syntax: analyzeExperiment(experimentStruct,calibrationMode,calibrationFlowMeter)
%     analyzeExperiment(experimentStruct,true,false); % Calibrate MS 
% end

% Loop through all the experimental files
if ~isempty(experimentFiles)
    for ii = 1:length(experimentFiles)
        experimentStruct.calibrationFlow = flowMeterCalibration; % Calibration file for meters (.mat)
        experimentStruct.flow = experimentFiles{ii}; % Experimental flow file (.mat)
        experimentStruct.MS = [msFileDir,filesep,msExpFile,'.asc']; % Experimental MS file (.asc). Assumes name of file to be the date of the first flow rate
        experimentStruct.calibrationMS = msCalibrationFiles; % Experimental calibration file list
        experimentStruct.moleFracThreshold = 1e-3; % Threshold for cutting off data below a given mole fraction
        % Call the analyzeExperiment function to analyze the experimental data
        % using the calibration files given by msCalibrationFiles 
        % The output is usually in runData folder
        % Syntax: analyzeExperiment(experimentStruct,calibrationMode,calibrationFlowMeter)
        analyzeExperiment(experimentStruct,false,false); % Analyze experiment
    end
end