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
% 
%
% Last modified:
% - 2021-03-24, AK: Remove k-means and replace with averaging of n points
% - 2021-03-19, HA: Added kmeans calculation to obtain mean ion current for
%                   polynomial fitting
% - 2021-03-18, AK: Fix variable names
% - 2021-03-17, AK: Change structure
% - 2021-03-17, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analyzeCalibration(parametersFlow,parametersMS)
% Find the directory of the file and move to the top folder
filePath = which('analyzeCalibration');
cd(filePath(1:end-21));

% Get the git commit ID
gitCommitID = getGitCommit;

% Load the file that contains the flow meter calibration
if ~isempty(parametersFlow)
    flowData = load(parametersFlow);
    % Analyse flow data
    MFM = [flowData.outputStruct.MFM]; % MFM 
    MFC1 = [flowData.outputStruct.MFC1]; % MFC1
    MFC2 = [flowData.outputStruct.MFC2]; % MFC2
    UMFM = [flowData.outputStruct.UMFM]; % UMFM
    % Get the volumetric flow rate
    volFlow_MFM = [MFM.volFlow];
    volFlow_MFC1 = [MFC1.volFlow];
    gas_MFC1 = [MFC1.gas];
    volFlow_UMFM = [UMFM.volFlow];
    % Find indices that corresponds to He/CO2 in the MFC
    indexHe = find(gas_MFC1 == 'He');
    indexCO2 = find(gas_MFC1 == 'CO2');
    % Parse the flow rate from the MFC, MFM, and UMFM for each gas
    % MFC
    volFlow_MFC1_He = volFlow_MFC1(indexHe);
    volFlow_MFC1_CO2 = volFlow_MFC1(indexCO2);
    % MFM
    volFlow_MFM_He = volFlow_MFM(indexHe);
    volFlow_MFM_CO2 = volFlow_MFM(indexCO2);
    % UMFM
    volFlow_UMFM_He = volFlow_UMFM(indexHe);
    volFlow_UMFM_CO2 = volFlow_UMFM(indexCO2);
    
    % Calibrate the meters
    % MFC
    calibrationFlow.MFC_He = volFlow_MFC1_He'\volFlow_UMFM_He';
    calibrationFlow.MFC_CO2 = volFlow_MFC1_CO2'\volFlow_UMFM_CO2';
    % MFM
    calibrationFlow.MFM_He = volFlow_MFM_He'\volFlow_UMFM_He';
    calibrationFlow.MFM_CO2 = volFlow_MFM_CO2'\volFlow_UMFM_CO2';
    
    % Save the calibration data into a .mat file
    % Check if calibration data folder exists
    if exist(['experimentalData',filesep,...
            'calibrationData'],'dir') == 7
        % Save the calibration data for further use
        save(['experimentalData',filesep,...
            'calibrationData',filesep,parametersFlow,'_Model'],'calibrationFlow',...
            'gitCommitID');
    else
        % Create the calibration data folder if it does not exist
        mkdir(['experimentalData',filesep,'calibrationData'])
        % Save the calibration data for further use
        save(['experimentalData',filesep,...
            'calibrationData',filesep,parametersFlow,'_Model'],'calibrationFlow',...
            'gitCommitID');
    end
   
    % Plot the raw and the calibrated data
    figure
    MFC1Set = 0:80;
    subplot(2,2,1)
    hold on
    scatter(volFlow_MFC1_He,volFlow_UMFM_He,'or')
    plot(MFC1Set,calibrationFlow.MFC_He*MFC1Set,'b')
    subplot(2,2,2)
    hold on
    scatter(volFlow_MFC1_CO2,volFlow_UMFM_CO2,'or')
    plot(MFC1Set,calibrationFlow.MFC_CO2*MFC1Set,'b')    
    subplot(2,2,3)
    hold on
    scatter(volFlow_MFM_He,volFlow_UMFM_He,'or')
    plot(MFC1Set,calibrationFlow.MFM_He*MFC1Set,'b')    
    subplot(2,2,4)
    hold on
    scatter(volFlow_MFM_CO2,volFlow_UMFM_CO2,'or')
    plot(MFC1Set,calibrationFlow.MFM_CO2*MFC1Set,'b')
end
% Load the file that contains the MS calibration
if ~isempty(parametersMS)
    % Call reconcileData function for calibration of the MS
    reconciledData = concatenateData(parametersMS);
    % Find the index that corresponds to the last time for a given set
    % point
    setPtMFC = unique(reconciledData.flow(:,4));
	% Find indices that corresponds to a given set point
    indList = ones(length(setPtMFC),2);
    % Loop over all the set points
    for ii=1:length(setPtMFC)
        % Indices for a given set point
        indList(ii,1) = find(reconciledData.flow(:,4)==setPtMFC(ii),1,'first');
        indList(ii,2) = find(reconciledData.flow(:,4)==setPtMFC(ii),1,'last');
        % Find the mean value of the signal for numMean number of points
        % for each set point
        indMean = find(reconciledData.flow(:,4)==setPtMFC(ii),...
            parametersMS.numMean,'last');
        % MS Signal mean
        meanHeSignal(ii) = mean(reconciledData.MS(indMean(1):indMean(end),2)); % He
        meanCO2Signal(ii) = mean(reconciledData.MS(indMean(1):indMean(end),3)); % CO2
        % Mole fraction mean
        meanMoleFrac(ii,1) = mean(reconciledData.moleFrac(indMean(1):indMean(end),1)); % He
        meanMoleFrac(ii,2) = mean(reconciledData.moleFrac(indMean(1):indMean(end),2)); % CO2
    end
            
    % Fit a polynomial function to get the model for MS
    % Fitting a 3rd order polynomial (check before accepting this)
    calibrationMS.He = polyfit(meanHeSignal,meanMoleFrac(:,1),parametersMS.polyDeg); % He
    calibrationMS.CO2 = polyfit(meanCO2Signal,meanMoleFrac(:,2),parametersMS.polyDeg); % COo2
    
    % Save the calibration data into a .mat file
    % Check if calibration data folder exists
    if exist(['experimentalData',filesep,...
            'calibrationData'],'dir') == 7
        % Save the calibration data for further use
        save(['experimentalData',filesep,...
            'calibrationData',filesep,parametersMS.flow,'_Model'],'calibrationMS',...
            'gitCommitID');
    else
        % Create the calibration data folder if it does not exist
        mkdir(['experimentalData',filesep,'calibrationData'])
        % Save the calibration data for further use
        save(['experimentalData',filesep,...
            'calibrationData',filesep,parametersMS.flow,'_Model'],'calibrationMS',...
            'gitCommitID');
    end
    
    % Plot the raw and the calibrated data
    figure(1)
    % He
    subplot(1,2,1)
    hold on
    plot(1e-13:1e-13:1e-8,polyval(calibrationMS.He,1e-13:1e-13:1e-8))
    scatter(meanHeSignal,meanMoleFrac(:,1))
    xlim([0 1.1*max(meanHeSignal)]);
    ylim([0 1]);
    
    % CO2
    subplot(1,2,2)
    hold on
    plot(1e-13:1e-13:1e-8,polyval(calibrationMS.CO2,1e-13:1e-13:1e-8))
    scatter(meanCO2Signal,meanMoleFrac(:,2))
    xlim([0 1.1*max(meanCO2Signal)]);
    ylim([0 1]);
end
end