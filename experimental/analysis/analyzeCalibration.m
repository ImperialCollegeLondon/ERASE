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
% - 2021-05-10, AK: Remove single gas calibration
% - 2021-04-27, AK: Change the calibration model to linear interpolation
% - 2021-04-23, AK: Change the calibration model to Fourier series based
% - 2021-04-21, AK: Change the calibration equation to mole fraction like
% - 2021-04-19, AK: Change MFC and MFM calibration (for mixtures)
% - 2021-04-08, AK: Add ratio of gas for calibration
% - 2021-04-07, AK: Modify for addition of MFM
% - 2021-03-26, AK: Fix for number of repetitions
% - 2021-03-19, HA: Add legends to the plots
% - 2021-03-24, AK: Remove k-means and replace with averaging of n points
% - 2021-03-19, HA: Add kmeans calculation to obtain mean ion current for
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
    volFlow_MFC1 = [MFC1.volFlow]; % Flow rate for MFC1 [ccm]
    setPt_MFC1 = [MFC1.setpoint]; % Set point for MFC1
    volFlow_MFC2 = [MFC2.volFlow]; % Flow rate for MFC2 [ccm]
    setPt_MFC2 = [MFC2.setpoint]; % Set point for MFC2
    volFlow_UMFM = [UMFM.volFlow]; % Flow rate for UMFM [ccm]
    % Find indices corresponding to pure gases
    indexPureHe = find(setPt_MFC2 == 0); % Find pure He index
    indexPureCO2 = find(setPt_MFC1 == 0); % Find pure CO2 index
    % Parse the flow rate from the MFC, MFM, and UMFM for pure gas
    % MFC
    volFlow_MFC1_PureHe = volFlow_MFC1(indexPureHe);
    volFlow_MFC2_PureCO2 = volFlow_MFC2(indexPureCO2);
    % UMFM for pure gases
    volFlow_UMFM_PureHe = volFlow_UMFM(indexPureHe);
    volFlow_UMFM_PureCO2 = volFlow_UMFM(indexPureCO2);
    % Calibrate the MFC
    calibrationFlow.MFC_He = volFlow_MFC1_PureHe'\volFlow_UMFM_PureHe'; % MFC 1
    calibrationFlow.MFC_CO2 = volFlow_MFC2_PureCO2'\volFlow_UMFM_PureCO2'; % MFC 2

    % Compute the mole fraction of CO2 using flow data
    moleFracCO2 = (calibrationFlow.MFC_CO2*volFlow_MFC2)./...
        (calibrationFlow.MFC_CO2*volFlow_MFC2 + calibrationFlow.MFC_He*volFlow_MFC1);
    indNoNan = ~isnan(moleFracCO2); % Find indices correponsing to no Nan
    % Calibrate the MFM
    % Fit a 23 (2nd order in mole frac and 3rd order in MFM flow) to UMFM
    % Note that the MFM flow rate corresponds to He gas configuration in
    % the MFM
    modelFlow = fit([moleFracCO2(indNoNan)',volFlow_MFM(indNoNan)'],volFlow_UMFM(indNoNan)','poly23');
    calibrationFlow.MFM = modelFlow;

    % Also save the raw data into the calibration file
    calibrationFlow.rawData = flowData;

    % Save the calibration data into a .mat file
    % Check if calibration data folder exists
    if exist(['..',filesep,'experimentalData',filesep,...
            'calibrationData'],'dir') == 7
        % Save the calibration data for further use
        save(['..',filesep,'experimentalData',filesep,...
            'calibrationData',filesep,parametersFlow,'_Model'],'calibrationFlow',...
            'gitCommitID');
    else
        % Create the calibration data folder if it does not exist
        mkdir(['..',filesep,'experimentalData',filesep,'calibrationData'])
        % Save the calibration data for further use
        save(['..',filesep,'experimentalData',filesep,...
            'calibrationData',filesep,parametersFlow,'_Model'],'calibrationFlow',...
            'gitCommitID');
    end

    % Plot the raw and the calibrated data (for pure gases at MFC)
    figure
    MFC1Set = 0:80;
    subplot(1,2,1)
    hold on
    scatter(volFlow_MFC1_PureHe,volFlow_UMFM_PureHe,'or')
    plot(MFC1Set,calibrationFlow.MFC_He*MFC1Set,'b')
    xlim([0 1.1*max(volFlow_MFC1_PureHe)]);
    ylim([0 1.1*max(volFlow_UMFM_PureHe)]);
    box on; grid on;
    xlabel('He MFC Flow Rate [ccm]')
    ylabel('He Actual Flow Rate [ccm]')
    subplot(1,2,2)
    hold on
    scatter(volFlow_MFC2_PureCO2,volFlow_UMFM_PureCO2,'or')
    plot(MFC1Set,calibrationFlow.MFC_CO2*MFC1Set,'b')
    xlim([0 1.1*max(volFlow_MFC2_PureCO2)]);
    ylim([0 1.1*max(volFlow_UMFM_PureCO2)]);
    box on; grid on;
    xlabel('CO2 MFC Flow Rate [ccm]')
    ylabel('CO2 Actual Flow Rate [ccm]')

    % Plot the raw and the calibrated data (for mixtures at MFM)
    figure
    x = 0:0.1:1; % Mole fraction
    y = 0:1:150; % Total flow rate
    [X,Y] = meshgrid(x,y); % Create a grid for the flow model
    Z = modelFlow(X,Y); % Actual flow rate from the model % [ccm]
    hold on
    surf(X,Y,Z,'FaceAlpha',0.25,'EdgeColor','none');
    scatter3(moleFracCO2,volFlow_MFM,volFlow_UMFM,'r');
    xlim([0 1.1*max(X(:))]);
    ylim([0 1.1*max(Y(:))]);
    zlim([0 1.1*max(Z(:))]);
    box on; grid on;
    xlabel('CO2 Mole Fraction [-]')
    ylabel('MFM Flow Rate [ccm]')
    zlabel('Actual Flow Rate [ccm]')
    view([30 30])
end

% Load the file that contains the MS calibration
if ~isempty(parametersMS)
    % Call reconcileData function for calibration of the MS
    [reconciledData, expInfo] = concatenateData(parametersMS);
    % Find the index that corresponds to the last time for a given set
    % point
    setPtMFC = unique(reconciledData.flow(:,5));
    % Find total number of data points
    numDataPoints = length(reconciledData.flow(:,1));
    % Total number of points per set point
    numPointsSetPt = expInfo.maxTime/expInfo.samplingTime;
    
    if ~isempty(strfind(parametersMS.MS,'DA'))
        numRepetitions = 1;
    elseif ~isempty(strfind(parametersMS.MS,'IR'))
        numRepetitions = 1;
    elseif ~isempty(strfind(parametersMS.MS,'TCD'))
        numRepetitions = 1;
    else 
        % Number of repetitions per set point (assuming repmat in calibrateMS)
        numRepetitions = floor((numDataPoints/numPointsSetPt)/length(setPtMFC));
        % numRepetitions = 1;
    end
    % Remove the 5 min idle time between repetitions
    % For two repetitions
    if numRepetitions == 2
        indRepFirst = numPointsSetPt*length(setPtMFC)+1;
        indRepLast = indRepFirst+numPointsSetPt-1;
        reconciledData.flow(indRepFirst:indRepLast,:) = [];
        reconciledData.MS(indRepFirst:indRepLast,:) = [];
        reconciledData.moleFrac(indRepFirst:indRepLast,:) = [];
        % For one repetition
    elseif numRepetitions == 1
        % Do nothing %
    else
        error('Currently more than two repetitions are not supported by analyzeCalibration.m');
    end
    % Find indices that corresponds to a given set point
    indList = ones(numRepetitions*length(setPtMFC),2);
    % Loop over all the set points
    if ~isempty(strfind(parametersMS.MS,'DA'))
        rangeVals = length(setPtMFC);
    elseif ~isempty(strfind(parametersMS.MS,'IR'))
        rangeVals = numDataPoints/numPointsSetPt;
    elseif ~isempty(strfind(parametersMS.MS,'TCD'))
        rangeVals = numDataPoints/numPointsSetPt;
    else
        rangeVals = numDataPoints/numPointsSetPt;
    end
    for kk = 1:round(numRepetitions)
        for ii=1:rangeVals
            % Indices for a given set point accounting for set point and
            % number of repetitions
            initInd = length(setPtMFC)*numPointsSetPt*(kk-1) + (ii-1)*numPointsSetPt + 1;
            finalInd = initInd + numPointsSetPt - 1;
            % Find the mean value of the signal for numMean number of points
            % for each set point
            indMean = (finalInd-parametersMS.numMean+1):finalInd;
            % MS Signal mean
            if ~isempty(strfind(parametersMS.MS,'CalibrateMS'))
                meanHeSignal((kk-1)*length(setPtMFC)+ii) = mean(reconciledData.MS(indMean,2)); % He
            end
            meanCO2Signal((kk-1)*length(setPtMFC)+ii) = mean(reconciledData.MS(indMean,3)); % CO2
            % Mole fraction mean
            if ~isempty(strfind(parametersMS.MS,'CalibrateMS'))
                meanMoleFrac(((kk-1)*length(setPtMFC)+ii),1) = mean(reconciledData.moleFrac(indMean,1)); % He
            end
            meanMoleFrac(((kk-1)*length(setPtMFC)+ii),2) = mean(reconciledData.moleFrac(indMean,2)); % CO2
        end
    end
    % Use a linear interpolation to fit the calibration data of the signal
    % ratio w.r.t He composition
    if ~isempty(strfind(parametersMS.MS,'DA'))
        calibrationMS.ratioHeCO2 = fit(meanCO2Signal',meanMoleFrac(:,2),'cubicspline');
    elseif ~isempty(strfind(parametersMS.MS,'IR'))
        calibrationMS.ratioHeCO2 = fit(meanCO2Signal',meanMoleFrac(:,2),'cubicspline');
    elseif ~isempty(strfind(parametersMS.MS,'TCD'))
        meanCO2Signal = meanCO2Signal-min(meanCO2Signal);
        calibrationMS.ratioHeCO2 = fit(meanCO2Signal'./max(meanCO2Signal),meanMoleFrac(:,2),'cubicspline');
        calibrationMS.maxVoltage = max(meanCO2Signal);
    else
        calibrationMS.ratioHeCO2 = fit((meanHeSignal./(meanCO2Signal+meanHeSignal))',meanMoleFrac(:,1),'cubicspline');
        % calibrationMS.ratioHeCO2 = fit((meanHeSignal./(meanCO2Signal+meanHeSignal))',meanMoleFrac(:,1),'linearinterp');
    end
    % Save the calibration data into a .mat file
    % Check if calibration data folder exists
    if exist(['..',filesep,'experimentalData',filesep,...
            'calibrationData'],'dir') == 7
        % Save the calibration data for further use
        save(['..',filesep,'experimentalData',filesep,...
            'calibrationData',filesep,parametersMS.flow,'_Model'],'calibrationMS',...
            'gitCommitID','parametersMS');
    else
        % Create the calibration data folder if it does not exist
        mkdir(['..',filesep,'experimentalData',filesep,'calibrationData'])
        % Save the calibration data for further use
        save(['..',filesep,'experimentalData',filesep,...
            'calibrationData',filesep,parametersMS.flow,'_Model'],'calibrationMS',...
            'gitCommitID','parametersMS');
    end

    % Plot the raw and the calibrated data
    figure(1)
    if ~isempty(strfind(parametersMS.MS,'DA'))
        plot(meanCO2Signal,meanMoleFrac(:,2),'or') % Experimental
        hold on
        plot(0:1e-5:2e-2,calibrationMS.ratioHeCO2(0:1e-5:2e-2),'b')
        xlim([0 2e-2]);
        ylim([0 2e-2]);
        box on; grid on;
        xlabel('CO_{2} Measured mole frac [-]')
        ylabel('CO_{2} Feed mole frac[-]')
        set(gca,'FontSize',8)
    elseif ~isempty(strfind(parametersMS.MS,'IR'))
        plot(meanCO2Signal,meanMoleFrac(:,2),'or') % Experimental
        hold on
        plot(0:0.001:1,calibrationMS.ratioHeCO2(0:0.001:1),'b')
        xlim([0 1]);
        ylim([0 1]);
        box on; grid on;
        xlabel('Helium Signal/(CO2 Signal+Helium Signal) [-]')
        ylabel('Helium mole frac [-]')
        set(gca,'FontSize',8)
    elseif ~isempty(strfind(parametersMS.MS,'TCD'))
        figure(1)
        plot(meanCO2Signal./max(meanCO2Signal),meanMoleFrac(:,2),'or') % Experimental
        hold on
        plot(0:0.001:1,calibrationMS.ratioHeCO2(0:0.001:1),'b')
        xlim([0 1]);
        ylim([0 1]);
        box on; grid on;
        xlabel('V/V_{max} [-]')
        ylabel('CO_{2} mole frac [-]')
        set(gca,'FontSize',8)
        figure(2)
        hold on
        box on; grid on;
        ylabel('V [uV]')
        xlabel('time [s]')
        ylim([0 1.05*max(reconciledData.MS(:,3))])
        set(gca,'FontSize',8)
        plot(reconciledData.MS(:,1),reconciledData.MS(:,3),'LineWidth',2,'DisplayName',[convertCharsToStrings(parametersMS.flow)]); 
        legend('Interpreter','latex','Location','southeast','FontSize',6);
%         yline(meanCO2Signal,'HandleVisibility','off','LineStyle','--','LineWidth',1)
    else
        plot(meanHeSignal./(meanHeSignal+meanCO2Signal),meanMoleFrac(:,1),'or') % Experimental
        hold on
        plot(0:0.001:1,calibrationMS.ratioHeCO2(0:0.001:1),'b')
        xlim([0 1]);
        ylim([0 1]);    
        box on; grid on;
        xlabel('Helium Signal/(CO2 Signal+Helium Signal) [-]')
        ylabel('Helium mole frac [-]')
        set(gca,'FontSize',8)
    end
end