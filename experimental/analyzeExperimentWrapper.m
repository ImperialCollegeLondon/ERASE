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
% - 2021-07-23, AK: Change calibration files
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

%%%%%%%%%%% MS %%%%%%%%%%%%%%%%%%%
%% Calibration files to be used %%%%
% List the MS calibration files (this is usually in experimental data folder)
msFileDir = 'C:\Users\azxan\Documents\GitHub\ERASE\MS'; % Directory with MS data
%  % Raw MS data file names for all calibration
% msRawFiles = {'ZLCCalibrateMS_20220723.asc','ZLCCalibrateMS_20220725.asc'};
% 
% numExpForEachRawFile = [1,1]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files)
% 
% % Flow rate files for calibration 
% msCalibrationFiles = {'ZLCCalibrateMS_20220724_30ccm',...
%                       'ZLCCalibrateMS_20220725_60ccm'};

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Raw MS data file names for all calibration
% msRawFiles = {'ZLCCalibrateMS_20230418.asc'};
% 
% numExpForEachRawFile = [2]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files)
% 
% % Flow rate files for calibration 
% msCalibrationFiles = {'ZLCCalibrateMS_20230418_30ccm',...
%                       'ZLCCalibrateMS_20230419_60ccm'};

% Raw MS data file names for all calibration
% msRawFiles = {'ZLCCalibrateMS_20230224.asc'};
% 
% numExpForEachRawFile = [2]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files)
% 
% % Flow rate files for calibration 
% msCalibrationFiles = {'ZLCCalibrateMS_20230225_30ccm',...
%                       'ZLCCalibrateMS_20230227_60ccm'};

% Raw MS data file names for all calibration
% msRawFiles = {'ZLCCalibrateMS_20230224.asc'};
% 
% numExpForEachRawFile = [1]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files)
% 
% Flow rate files for calibration 
% msCalibrationFiles = {'ZLCCalibrateMS_20230227_60ccm'};
% % 
% 
% msRawFiles = {'ZLCCalibrateMS_20231019.asc'};
% 
% numExpForEachRawFile = [1]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files)
% 
% % Flow rate files for calibration 
% msCalibrationFiles = {'ZLCCalibrateMS_20231018_60ccm'};

msRawFiles = {'ZLCCalibrateTCD_20240628.txt','ZLCCalibrateTCD_010724_old.txt','ZLCCalibrateTCD_020724.txt'};

% numExpForEachRawFile = [2]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files). i.e. how many flowrates within same calibration file
numExpForEachRawFile = [1,1,1]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files). i.e. how many flowrates within same calibration file

% Flow rate files for calibration 
% msCalibrationFiles = {'ZLCCalibrateMS_20240518_60ccm', 'ZLCCalibrateMS_20240519_60ccm'};
msCalibrationFiles = {'ZLCCalibrateTCD_20240628_60ccm_old','ZLCCalibrateTCD_20240701_60ccm_old','ZLCCalibrateTCD_20240702_40ccm'};

% msRawFiles = {'ZLCCalibrateTCD_010724_old.txt','ZLCCalibrateTCD_010724.txt'};
% 
% % numExpForEachRawFile = [2]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files). i.e. how many flowrates within same calibration file
% numExpForEachRawFile = [1,1]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files). i.e. how many flowrates within same calibration file
% 
% % Flow rate files for calibration 
% % msCalibrationFiles = {'ZLCCalibrateMS_20240518_60ccm', 'ZLCCalibrateMS_20240519_60ccm'};
% msCalibrationFiles = {'ZLCCalibrateTCD_20240701_60ccm_old','ZLCCalibrateTCD_20240701_40ccm'};

% 

% msRawFiles = {'ZLCCalibrateMS_20240219.asc'};

% numExpForEachRawFile = [2]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files)

% Flow rate files for calibration 
% msCalibrationFiles = {'ZLCCalibrateMS_20240219_40ccm','ZLCCalibrateMS_20240220_60ccm'};

% 
% msRawFiles = {'ZLCCalibrateMS_20231017.asc'};
% 
% numExpForEachRawFile = [1]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files)
% 
% % Flow rate files for calibration 
% msCalibrationFiles = {'ZLCCalibrateMS_20231016_30ccm'};

%%%% Experiment to be analyzed %%%%     
% List the experiments that have to be analyzed
% MS Raw data should contain only two gases and the pressure. For now
% cannot handle more gases.

msExpFile = 'ZLC_Empty_Exp72_73.asc'; % Raw MS data file name
% Flow rate files for experiments
% experimentFiles =  {'ZLC_HCP-DETA-2_Exp01A',...
%                     'ZLC_HCP-DETA-2_Exp01B',...
%                     'ZLC_HCP-DETA-2_Exp02A',...
%                     'ZLC_HCP-DETA-2_Exp02B'};
experimentFiles =  {'ZLC_Empty_Exp72A',...
                    'ZLC_Empty_Exp72B',...
                    'ZLC_Empty_Exp73A',...
                    'ZLC_Empty_Exp73B'};


% msExpFile = 'ZLC_ZYNaCrush_Exp05_06.asc'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_ZYNaCrush_Exp05A',...
%                     'ZLC_ZYNaCrush_Exp05B',...
%                     'ZLC_ZYNaCrush_Exp06A',...
%                     'ZLC_ZYNaCrush_Exp06B'};
% 
% msExpFile = 'ZLC_ZYNaCrush_Exp09_10.asc'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_ZYNaCrush_Exp09A',...
%                     'ZLC_ZYNaCrush_Exp09B',...
%                     'ZLC_ZYNaCrush_Exp10A',...
%                     'ZLC_ZYNaCrush_Exp10B'};
% 
% msExpFile = 'ZLC_ZYNaCrush_Exp11_12.asc'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_ZYNaCrush_Exp11A',...
%                     'ZLC_ZYNaCrush_Exp11B',...
%                     'ZLC_ZYNaCrush_Exp12A',...
%                     'ZLC_ZYNaCrush_Exp12B'};
% 
% % 
% msExpFile = 'ZLC_ZYHCrush_Exp05_06.asc'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_ZYHCrush_Exp05A',...
%                     'ZLC_ZYHCrush_Exp05B',...
%                     'ZLC_ZYHCrush_Exp06A',...
%                     'ZLC_ZYHCrush_Exp06B'};
% 
% msExpFile = 'ZLC_ZYHCrush_Exp07_08.asc'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_ZYHCrush_Exp07A',...
%                     'ZLC_ZYHCrush_Exp07B',...
%                     'ZLC_ZYHCrush_Exp08A',...
%                     'ZLC_ZYHCrush_Exp08B'};
% 
% msExpFile = 'ZLC_ZYHCrush_Exp09_10.asc'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_ZYHCrush_Exp09A',...
%                     'ZLC_ZYHCrush_Exp09B',...
%                     'ZLC_ZYHCrush_Exp10A',...
%                     'ZLC_ZYHCrush_Exp10B'};
% 
% msExpFile = 'ZLC_ZYTMACrush_Exp03_04.asc'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_ZYTMACrush_Exp03A',...
%                     'ZLC_ZYTMACrush_Exp03B',...
%                     'ZLC_ZYTMACrush_Exp04A',...
%                     'ZLC_ZYTMACrush_Exp04B'};
% 
% msExpFile = 'ZLC_ZYTMACrush_Exp05_06.asc'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_ZYTMACrush_Exp05A',...
%                     'ZLC_ZYTMACrush_Exp05B',...
%                     'ZLC_ZYTMACrush_Exp06A',...
%                     'ZLC_ZYTMACrush_Exp06B'};
% 
% msExpFile = 'ZLC_ZYTMACrush_Exp09_10.asc'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_ZYTMACrush_Exp09A',...
%                     'ZLC_ZYTMACrush_Exp09B',...
%                     'ZLC_ZYTMACrush_Exp10A',...
%                     'ZLC_ZYTMACrush_Exp10B'};
% 
% msExpFile = 'ZLC_Empty_Exp60_61.asc'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_Empty_Exp60A',...
%                     'ZLC_Empty_Exp60B',...
%                     'ZLC_Empty_Exp61A',...
%                     'ZLC_Empty_Exp61B'};

% %%%%%%%%%%%%%%%% IR %%%%%%%%%%%%%%%%
% msRawFiles = {'ZLCCalibrateIR_20230822.txt'};
% 
% numExpForEachRawFile = [1]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files)
% 
% % Flow rate files for calibration 
% msCalibrationFiles = {'ZLCCalibrateIR_20230822_60ccm'};
% 
% %%%% Experiment to be analyzed %%%%     
% % List the experiments that have to be analyzed
% % MS Raw data should contain only two gases and the pressure. For now
% % cannot handle more gases.
% msExpFile = 'ZLC_13X_IR_231114.txt'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_13X_IR_Exp01A',...
%                     'ZLC_13X_IR_Exp01_repB',...
%                     'ZLC_13X_IR_Exp02A',...
%                     'ZLC_13X_IR_Exp02B',...
%                     'ZLC_13X_IR_Exp03A',...
%                     'ZLC_13X_IR_Exp03B',...
%                     'ZLC_13X_IR_Exp04A',...
%                     'ZLC_13X_IR_Exp04B'};

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%% OLD SETUP %%%%%%%%%%%%%%%%

% msRawFiles = {'ZLCCalibrateMS_20210726.asc'};
% 
% numExpForEachRawFile = [3]; % Number of experiments that use the same raw MS file (vector corresponding to number of MS files)
% 
% % Flow rate files for calibration 
% msCalibrationFiles = {'ZLCCalibrateMS_20210726_10ccm',...
%                     'ZLCCalibrateMS_20210726_30ccm',...
%                     'ZLCCalibrateMS_20210727_60ccm'};
% 
% msExpFile = 'ZLC_DeadVolume_Exp21.asc'; % Raw MS data file name
% % Flow rate files for experiments
% experimentFiles =  {'ZLC_DeadVolume_Exp21A',...
%                     'ZLC_DeadVolume_Exp21B',...
%                     'ZLC_DeadVolume_Exp21C',...
%                     'ZLC_DeadVolume_Exp21D'};
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
    calibrationStruct.MS = [msFileDir,filesep,msRawFileALL{ii}]; % Experimental MS file (.asc)
    calibrationStruct.interpMS = true; % Flag for interpolating MS data (true) or flow data (false)
    if ~contains(msExpFile,'IR')
        calibrationStruct.numMean = 50; % Number of points for averaging
    else
        calibrationStruct.numMean = 200; % Number of points for averaging
    end
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
        experimentStruct.initialCO2 = 1; % flag for Initial gas to be CO2
        experimentStruct.flow = experimentFiles{ii}; % Experimental flow file (.mat)
        experimentStruct.MS = [msFileDir,filesep,msExpFile]; % Experimental MS file (.asc). Assumes name of file to be the date of the first flow rate
        experimentStruct.calibrationMS = msCalibrationFiles; % Experimental calibration file list
        experimentStruct.interpMS = false; % Flag for interpolating flow data, to have a higher resolution for actual experiments
        if ~contains(msExpFile,'TCD')
            % experimentStruct.moleFracThreshold = 1e-2; % Threshold for cutting off data below a given mole fraction FOR MS
            experimentStruct.moleFracThreshold = 8e-3; % Threshold for cutting off data below a given mole fraction FOR MS
        else
            experimentStruct.moleFracThreshold = 1e-4; % Threshold for cutting off data below a given mole fraction FOR DA
        end

        % Call the analyzeExperiment function to analyze the experimental data
        % using the calibration files given by msCalibrationFiles 
        % The output is usually in runData folder 
        % Syntax: analyzeExperiment(experimentStruct,calibrationMode,calibrationFlowMeter)
        analyzeExperiment(experimentStruct,false,false); % Analyze experiment
    end
end

% Loop through all the experimental files and plot the output mole fraction
if ~isempty(experimentFiles)
    colorForPlot = {'5C73B9','7262C3','8852CD','9D41D7','B330E1',...
                    '5C73B9','7262C3','8852CD','9D41D7','B330E1',...
                    '5C73B9','7262C3','8852CD','9D41D7','B330E1',...
                    '5C73B9','7262C3','8852CD','9D41D7','B330E1',...
                    '5C73B9','7262C3','8852CD','9D41D7','B330E1',...
                    '5C73B9','7262C3','8852CD','9D41D7','B330E1'};
    f1 = figure('Units','inch','Position',[2 2 7 3.3]);
    for ii = 1:length(experimentFiles)
        load([experimentFiles{ii},'_Output']);
        % Plot the output from different experiments (in y and Ft plots)
        figure(f1);
        subplot(1,2,1)
        semilogy(experimentOutput.timeExp,experimentOutput.moleFrac,'color','b');
        hold on
        box on;grid on;
        xlim([0,300]); ylim([0,1.1*max(experimentOutput.moleFrac)]);
        xlabel('{\it{t}} [s]'); ylabel('{\it{y}} [-]');
        set(gca,'FontSize',8)
        
        subplot(1,2,2)
        semilogy(experimentOutput.timeExp.*experimentOutput.totalFlowRate,experimentOutput.moleFrac,'color',['#',colorForPlot{ii}]);
        hold on
        xlim([0,10]); ylim([0,1]);      
        xlabel('{\it{Ft}} [cc]'); ylabel('{\it{y}} [-]');
        set(gca,'FontSize',8)
        box on;grid on;
        
        % Plot data from different calibrations
        figure('Units','inch','Position',[2 2 3.3 3.3])
        semilogy(semiProcessedStruct.flow(:,1),1-semiProcessedStruct.moleFracIndCalib);
        hold on
        semilogy(experimentOutput.timeExp,experimentOutput.moleFrac,'--k');
        xlim([0,500]); ylim([0,1.1*max(experimentOutput.moleFrac)]);
        xlabel('{\it{t}} [s]'); ylabel('{\it{y}} [-]');
        set(gca,'FontSize',8)
        box on;grid on;
    end
end