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
% - 2021-05-17, AK: Change MS interpolation flag
% - 2021-05-10, AK: Cosmetic changes to plots
% - 2021-05-10, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% List the flow meter calibration file (this is usually in calibraiton folder)
flowMeterCalibration = 'ZLCCalibrateMeters_20210419_Model';

%%%% Calibration files to be used %%%%
% List the MS calibration files (this is usually in experimental data folder)
msFileDir = 'C:\Users\QCPML\Desktop\Ashwin\MS'; % Directory with MS data
msRawFiles = {'ZLCCalibrateMS_20210505'}; % Raw MS data file names for all calibration
numExpForEachRawFile = [4]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files)

% Flow rate files for calibration 
msCalibrationFiles = {'ZLCCalibrateMS_20210506_15ccm',...
                    'ZLCCalibrateMS_20210506_30ccm',...
                    'ZLCCalibrateMS_20210506_45ccm',...
                    'ZLCCalibrateMS_20210507_60ccm'};

%%%% Experimet to be analyzed %%%%     
% List the experiments that have to be analyzed
msExpFile = 'ZLC_ActivatedCarbon_Exp26_28'; % Raw MS data file name
% Flow rate files for experiments 
experimentFiles = {'ZLC_ActivatedCarbon_Exp26A',...
                   'ZLC_ActivatedCarbon_Exp26B',...
                   'ZLC_ActivatedCarbon_Exp26C',...
                   'ZLC_ActivatedCarbon_Exp26D',...
                   'ZLC_ActivatedCarbon_Exp26E',...
                   'ZLC_ActivatedCarbon_Exp26F'};

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
for ii = 1:length(msCalibrationFiles)
    calibrationStruct.calibrationFlow = flowMeterCalibration; % Calibration file for meters (.mat)
    calibrationStruct.flow = msCalibrationFiles{ii}; % Experimental flow file (.mat)
    calibrationStruct.MS = [msFileDir,filesep,msRawFileALL{ii},'.asc']; % Experimental MS file (.asc)
    calibrationStruct.interpMS = true; % Flag for interpolating MS data (true) or flow data (false)
    calibrationStruct.numMean = 50; % Number of points for averaging
    % Call the analyzeExperiment function to calibrate the MS at the conditions
    % experiment was performed for calibration
    % The output calibration model is usually in calibration folder
    % Syntax: analyzeExperiment(experimentStruct,calibrationMode,calibrationFlowMeter)
    analyzeExperiment(calibrationStruct,true,false); % Calibrate MS 
end

% Loop through all the experimental files
if ~isempty(experimentFiles)
    for ii = 1:length(experimentFiles)
        experimentStruct.calibrationFlow = flowMeterCalibration; % Calibration file for meters (.mat)
        experimentStruct.flow = experimentFiles{ii}; % Experimental flow file (.mat)
        experimentStruct.MS = [msFileDir,filesep,msExpFile,'.asc']; % Experimental MS file (.asc). Assumes name of file to be the date of the first flow rate
        experimentStruct.calibrationMS = msCalibrationFiles; % Experimental calibration file list
        experimentStruct.interpMS = false; % Flag for interpolating flow data, to have a higher resolution for actual experiments
        experimentStruct.moleFracThreshold = 5e-3; % Threshold for cutting off data below a given mole fraction
        % Call the analyzeExperiment function to analyze the experimental data
        % using the calibration files given by msCalibrationFiles 
        % The output is usually in runData folder
        % Syntax: analyzeExperiment(experimentStruct,calibrationMode,calibrationFlowMeter)
        analyzeExperiment(experimentStruct,false,false); % Analyze experiment
    end
end

% Loop through all the experimental files and plot the output mole fraction
if ~isempty(experimentFiles)
    colorForPlot = {'5C73B9','7262C3','8852CD','9D41D7','B330E1'};
    f1 = figure('Units','inch','Position',[2 2 7 3.3]);
    for ii = 1:length(experimentFiles)
        load([experimentFiles{ii},'_Output']);
        % Plot the output from different experiments (in y and Ft plots)
        figure(f1);
        subplot(1,2,1)
        semilogy(experimentOutput.timeExp,experimentOutput.moleFrac,'color',['#',colorForPlot{ii}]);
        hold on
        box on;grid on;
        xlim([0,500]); ylim([0,1]);
        xlabel('{\it{t}} [s]'); ylabel('{\it{y}} [-]');
        set(gca,'FontSize',8)
        
        subplot(1,2,2)
        semilogy(experimentOutput.timeExp.*experimentOutput.totalFlowRate,experimentOutput.moleFrac,'color',['#',colorForPlot{ii}]);
        hold on
        xlim([0,100]); ylim([0,1]);      
        xlabel('{\it{Ft}} [cc]'); ylabel('{\it{y}} [-]');
        set(gca,'FontSize',8)
        box on;grid on;
        
        % Plot data from different calibrations
        figure('Units','inch','Position',[2 2 3.3 3.3])
        semilogy(semiProcessedStruct.flow(:,1),1-semiProcessedStruct.moleFracIndCalib);
        hold on
        semilogy(experimentOutput.timeExp,experimentOutput.moleFrac,'--k');
        xlim([0,500]); ylim([0,1]);
        xlabel('{\it{t}} [s]'); ylabel('{\it{y}} [-]');
        set(gca,'FontSize',8)
        box on;grid on;
    end
end