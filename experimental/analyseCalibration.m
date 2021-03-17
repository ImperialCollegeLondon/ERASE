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
% - 2021-03-17, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analyseCalibration(fileToLoadMeter,fileToLoadMS)
% Find the directory of the file and move to the top folder
filePath = which('analyseCalibration');
cd(filePath(1:end-21));

% Get the git commit ID
gitCommitID = getGitCommit;

% Load the file that contains the flow meter calibration
if ~isempty(fileToLoadMeter)
    flowData = load(fileToLoadMeter);
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
    calibration.MFC_He = volFlow_MFC1_He'\volFlow_UMFM_He';
    calibration.MFC_CO2 = volFlow_MFC1_CO2'\volFlow_UMFM_CO2';
    % MFM
    calibration.MFM_He = volFlow_MFM_He'\volFlow_UMFM_He';
    calibration.MFM_CO2 = volFlow_MFM_CO2'\volFlow_UMFM_CO2';
    
    % Save the calibration data into a .mat file
    % Check if calibration data folder exists
    if exist(['experimentalData',filesep,...
            'calibrationData'],'dir') == 7
        % Save the calibration data for further use
        save(['experimentalData',filesep,...
            'calibrationData',filesep,fileToLoadMeter,'_Model'],'calibration',...
            'gitCommitID');
    else
        % Create the calibration data folder if it does not exist
        mkdir(['experimentalData',filesep,'calibrationData'])
        % Save the calibration data for further use
        save(['experimentalData',filesep,...
            'calibrationData',filesep,fileToLoadMeter,'_Model'],'calibration',...
            'gitCommitID');
    end
   
    % Plot the raw and the calibrated data
    figure
    MFC1Set = 0:80;
    subplot(2,2,1)
    hold on
    scatter(volFlow_MFC1_He,volFlow_UMFM_He,'or')
    plot(MFC1Set,calibration.MFC_He*MFC1Set,'b')
    subplot(2,2,2)
    hold on
    scatter(volFlow_MFC1_CO2,volFlow_UMFM_CO2,'or')
    plot(MFC1Set,calibration.MFC_CO2*MFC1Set,'b')    
    subplot(2,2,3)
    hold on
    scatter(volFlow_MFM_He,volFlow_UMFM_He,'or')
    plot(MFC1Set,calibration.MFM_He*MFC1Set,'b')    
    subplot(2,2,4)
    hold on
    scatter(volFlow_MFM_CO2,volFlow_UMFM_CO2,'or')
    plot(MFC1Set,calibration.MFM_CO2*MFC1Set,'b')
end
% Load the file that contains the MS calibration
if ~isempty(fileToLoadMS)
    % Load flow data
    flowMS = load(fileToLoadMS.flow);
    if ~isfield(fileToLoadMS,'calibration')
        error('You gotta calibrate your flow meters fisrst!! Or check the file name of the flow calibration!!')
    end
    calibrationMeters = load(fileToLoadMS.calibration);
    % Analyse flow data
    MFC1 = [flowMS.outputStruct.MFC1]; % MFC1 - He
    MFC2 = [flowMS.outputStruct.MFC2]; % MFC2 - CO2
    % Get the datetime and volumetric flow rate
    dateTimeFlow = datetime({flowMS.outputStruct.samplingDateTime},...
        'InputFormat','yyyyMMdd_HHmmss');
    volFlow_MFC1 = [MFC1.volFlow]; % He
    volFlow_MFC2 = [MFC2.volFlow]; % CO2
    % Apply the calibration for the flows
    volFlow_He = volFlow_MFC1*calibrationMeters.calibration.MFC_He;
    volFlow_CO2 = volFlow_MFC2*calibrationMeters.calibration.MFC_CO2;
    % Load MS Ascii data
    % Create file identifier
    fileId = fopen(fileToLoadMS.MS);
    % Load the MS Data into a cell array
    rawMSData = textscan(fileId,repmat('%s',1,9),'HeaderLines',8,'Delimiter','\t');
    % Get the date time for CO2
    dateTimeHe = datetime(cell2mat(rawMSData{1,4}),...
        'InputFormat','MM/dd/yyyy hh:mm:ss.SSS a');
    % Get the date time for He
    dateTimeCO2 = datetime(cell2mat(rawMSData{1,1}),...
        'InputFormat','MM/dd/yyyy hh:mm:ss.SSS a');
    % Reconcile all the data
    % Initial time
    initialTime = max([dateTimeFlow(1), dateTimeHe(1), dateTimeCO2(1)]);
    % Final time
    finalTime = min([dateTimeFlow(end), dateTimeHe(end), dateTimeCO2(end)]);
    
    % Find index corresponding to initial time for meters and MS
    indexInitial_Flow = find(dateTimeFlow>=initialTime,1,'first');
    indexInitial_He = find(dateTimeHe>=initialTime,1,'first');
    indexInitial_CO2 = find(dateTimeCO2>=initialTime,1,'first');    

    % Find index corresponding to final time for meters and MS
    indexFinal_Flow = find(dateTimeFlow<=finalTime,1,'last');
    indexFinal_He = find(dateTimeHe<=finalTime,1,'last');
    indexFinal_CO2 = find(dateTimeCO2<=finalTime,1,'last');    

    % Reconciled data (without interpolation)
    % The whole reconciliation assumes that the MS is running after the
    % flow meters to avoid any issues with interpolation!!!
    % Meters and the controllers
    reconciledData.raw.dateTimeFlow = dateTimeFlow(indexInitial_Flow:end);
    reconciledData.raw.volFlow_He = volFlow_He(indexInitial_Flow:end);
    reconciledData.raw.volFlow_CO2 = volFlow_CO2(indexInitial_Flow:end);
    % MS    
    reconciledData.raw.dateTimeMS_He = dateTimeHe(indexInitial_He:end);
    reconciledData.raw.dateTimeMS_CO2 = dateTimeCO2(indexInitial_CO2:end);
    reconciledData.raw.signalHe = str2num(cell2mat(rawMSData{1,6}(indexInitial_He:end)));
    reconciledData.raw.signalCO2 = str2num(cell2mat(rawMSData{1,3}(indexInitial_CO2:end)));  
    
    % Reconciled data (with interpolation)
    % Meters and the controllers
    reconciledData.flow(:,1) = seconds(reconciledData.raw.dateTimeFlow...
        -reconciledData.raw.dateTimeFlow(1)); % Time elapsed [s]
    reconciledData.flow(:,2) = reconciledData.raw.volFlow_He; % He Flow [ccm]
    reconciledData.flow(:,3) = reconciledData.raw.volFlow_CO2; % CO2 flow [ccm]

    % MS
    rawTimeElapsedHe = seconds(reconciledData.raw.dateTimeMS_He ...
        - reconciledData.raw.dateTimeMS_He(1)); % Time elapsed He [s]
    rawTimeElapsedCO2 = seconds(reconciledData.raw.dateTimeMS_CO2 ...
        - reconciledData.raw.dateTimeMS_CO2(1)); % Time elapsed CO2 [s]
    % Interpolate the MS signal at the times of flow meter/controller
    reconciledData.MS(:,1) = reconciledData.flow(:,1); % Use the time of the flow meter [s]
    reconciledData.MS(:,2) = interp1(rawTimeElapsedHe,reconciledData.raw.signalHe,...
                                    reconciledData.MS(:,1)); % Interpoloted MS signal He [-]
    reconciledData.MS(:,3) = interp1(rawTimeElapsedCO2,reconciledData.raw.signalCO2,...
                                    reconciledData.MS(:,1)); % Interpoloted MS signal CO2 [-]
    
    % Compute the mole fractions using the reconciled flow data
    reconciledData.moleFrac(:,1) = (reconciledData.flow(:,2))./(reconciledData.flow(:,2)+reconciledData.flow(:,3));
    reconciledData.moleFrac(:,2) = 1 - reconciledData.moleFrac(:,1);

    % Fit a polynomial function to get the model for MS
    % Fitting a 3rd order polynomial (check before accepting this)
    calibration.He = polyfit(reconciledData.moleFrac(:,1),reconciledData.MS(:,2),3); % He
    calibration.CO2 = polyfit(reconciledData.moleFrac(:,2),reconciledData.MS(:,3),3); % Co2
    
    % Save the calibration data into a .mat file
    % Check if calibration data folder exists
    if exist(['experimentalData',filesep,...
            'calibrationData'],'dir') == 7
        % Save the calibration data for further use
        save(['experimentalData',filesep,...
            'calibrationData',filesep,fileToLoadMS.flow,'_Model'],'calibration',...
            'gitCommitID');
    else
        % Create the calibration data folder if it does not exist
        mkdir(['experimentalData',filesep,'calibrationData'])
        % Save the calibration data for further use
        save(['experimentalData',filesep,...
            'calibrationData',filesep,fileToLoadMS.flow,'_Model'],'calibration',...
            'gitCommitID');
    end
    
    % Plot the raw and the calibrated data
    figure
    % He
    subplot(1,2,1)
    plot(0:0.01:1,polyval(calibration.He,0:0.01:1))
    hold on
    plot(reconciledData.moleFrac(:,1),reconciledData.MS(:,2),'or')
    
    % CO2
    subplot(1,2,2)
    plot(0:0.01:1,polyval(calibration.CO2,0:0.01:1))
    hold on
    plot(reconciledData.moleFrac(:,2),reconciledData.MS(:,3),'or')
end
end