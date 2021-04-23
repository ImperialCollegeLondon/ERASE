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
% - 2021-04-23, AK: Change the calibration model to Fourier series based 
% - 2021-04-21, AK: Change the calibration equation to mole fraction like
% - 2021-04-19, AK: Remove MFM calibration (check analyzeExperiment)
% - 2021-04-09, AK: Change output for calibration or non calibrate mode
% - 2021-04-08, AK: Add ratio of gas for calibration
% - 2021-04-07, AK: Modify for addition of MFM
% - 2021-03-26, AK: Add expInfo to output
% - 2021-03-22, AK: Bug fixes for finding indices
% - 2021-03-22, AK: Add checks for MS concatenation
% - 2021-03-18, AK: Add interpolation based on MS or flow meter
% - 2021-03-18, AK: Add experiment analysis mode
% - 2021-03-17, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [reconciledData, expInfo] = concatenateData(fileToLoad)
    % Load flow data
    flowMS = load(fileToLoad.flow);
    if ~isfield(fileToLoad,'calibrationFlow')
        error('You gotta calibrate your flow meters first!! Or check the file name of the flow calibration!!')
    end
    % Flow Calibration File
    load(fileToLoad.calibrationFlow);
    % Return expInfo
    expInfo = flowMS.expInfo;
    % Analyse flow data
    MFC1 = [flowMS.outputStruct.MFC1]; % MFC1 - He
    MFC2 = [flowMS.outputStruct.MFC2]; % MFC2 - CO2
    MFM = [flowMS.outputStruct.MFM]; % MFM - He
    % Get the datetime and volumetric flow rate
    dateTimeFlow = datetime({flowMS.outputStruct.samplingDateTime},...
        'InputFormat','yyyyMMdd_HHmmss');
    volFlow_MFC1 = [MFC1.volFlow]; % He
    setPt_MFC1 = [MFC1.setpoint]; % He setpoint for calibration
    volFlow_MFC2 = [MFC2.volFlow]; % CO2
    volFlow_MFM = [MFM.volFlow]; % CO2
    % Apply the calibration for the flows
    % Round the flow rate to the nearest first decimal (as this is the
    % resolution of the meter)    
    volFlow_He = round(volFlow_MFC1*calibrationFlow.MFC_He,1);
	volFlow_CO2 = round(volFlow_MFC2*calibrationFlow.MFC_CO2,1);
    
    % Load MS Ascii data
    % Create file identifier
    fileId = fopen(fileToLoad.MS);
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
  
    indexInitial_MS = max([find(dateTimeHe>=initialTime,1,'first'),...
                        find(dateTimeCO2>=initialTime,1,'first')]);
                    
    % Find index corresponding to final time for meters and MS
    indexFinal_Flow = find(dateTimeFlow<=finalTime,1,'last');
    indexFinal_MS = min([find(dateTimeHe<=finalTime,1,'last'),...
                        find(dateTimeCO2<=finalTime,1,'last')]);
  

    % Reconciled data (without interpolation)
    % NOTE: The whole reconciliation assumes that the MS is running after #
    % the flow meters to avoid any issues with interpolation!!!
    % Meters and the controllers
    reconciledData.raw.dateTimeFlow = dateTimeFlow(indexInitial_Flow:end);
    reconciledData.raw.volFlow_He = volFlow_He(indexInitial_Flow:end);
    reconciledData.raw.volFlow_CO2 = volFlow_CO2(indexInitial_Flow:end);
    reconciledData.raw.volFlow_MFM = volFlow_MFM(indexInitial_Flow:end);
    reconciledData.raw.setPt_He = setPt_MFC1(indexInitial_Flow:end);
    
    % MS    
    % Find the index of the last entry (from one of the two gases)
    concantenateLastInd = indexInitial_MS + min([size(dateTimeHe(indexInitial_MS:end),1), ...
        size(dateTimeCO2(indexInitial_MS:end),1)]) - 1;
    reconciledData.raw.dateTimeMS_He = dateTimeHe(indexInitial_MS:concantenateLastInd);
    reconciledData.raw.dateTimeMS_CO2 = dateTimeCO2(indexInitial_MS:concantenateLastInd);
    % Check if any element is negative for concatenation
    for ii=indexInitial_MS:concantenateLastInd
        % He
        % If negative element, initialize to eps
        if str2num(cell2mat(rawMSData{1,6}(ii))) < 0
            reconciledData.raw.signalHe(ii-indexInitial_MS+1) = eps;
        % If not, use the actual value
        else
            reconciledData.raw.signalHe(ii-indexInitial_MS+1) = str2num(cell2mat(rawMSData{1,6}(ii)));
        end
        % CO2        
        % If negative element, initialize to eps        
        if str2num(cell2mat(rawMSData{1,3}(ii))) < 0
            reconciledData.raw.signalCO2(ii-indexInitial_MS+1) = eps;
        % If not, use the actual value            
        else
            reconciledData.raw.signalCO2(ii-indexInitial_MS+1) = str2num(cell2mat(rawMSData{1,3}(ii)));
        end
    end
  
    % Reconciled data (with interpolation)
    % Interpolate based on flow
    if fileToLoad.interpMS
        % Meters and the controllers
        reconciledData.flow(:,1) = seconds(reconciledData.raw.dateTimeFlow...
            -reconciledData.raw.dateTimeFlow(1)); % Time elapsed [s]
        % Save the MFC and MFM values for the calibrate meters experiments
        if expInfo.calibrateMeters
            reconciledData.flow(:,2) = reconciledData.raw.volFlow_He; % He Flow [ccm]
            reconciledData.flow(:,3) = reconciledData.raw.volFlow_CO2; % CO2 flow [ccm]
            reconciledData.flow(:,4) = reconciledData.raw.volFlow_MFM; % MFM flow [ccm]
            reconciledData.flow(:,5) = reconciledData.raw.setPt_He; % He set point [ccm]
        % Save only the MFM values for the true experiments
        else
            reconciledData.flow(:,2) = reconciledData.raw.volFlow_MFM; % He set point [ccm]
        end
        
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
    % Interpolate based on MS
    else
        % MS
        rawTimeElapsedHe = seconds(reconciledData.raw.dateTimeMS_He ...
            - reconciledData.raw.dateTimeMS_He(1)); % Time elapsed He [s]
        rawTimeElapsedCO2 = seconds(reconciledData.raw.dateTimeMS_CO2 ...
            - reconciledData.raw.dateTimeMS_CO2(1)); % Time elapsed CO2 [s] 
        % Interpolate the MS signal at the times of flow meter/controller
        reconciledData.MS(:,1) = rawTimeElapsedHe; % Use the time of He [s]
        reconciledData.MS(:,2) = reconciledData.raw.signalHe; % Raw signal He [-]
        reconciledData.MS(:,3) = interp1(rawTimeElapsedCO2,reconciledData.raw.signalCO2,...
                                        reconciledData.MS(:,1)); % Interpoloted MS signal CO2 based on He time [-]
        
        % Meters and the controllers
        rawTimeElapsedFlow = seconds(reconciledData.raw.dateTimeFlow...
                                    -reconciledData.raw.dateTimeFlow(1)); 
        reconciledData.flow(:,1) = reconciledData.MS(:,1); % Time elapsed of MS [s]
        % Save the MFC and MFM values for the calibrate meters experiments
        if expInfo.calibrateMeters
            reconciledData.flow(:,2) = interp1(rawTimeElapsedFlow,reconciledData.raw.volFlow_He,...
                                            reconciledData.MS(:,1)); % Interpoloted He Flow [ccm]
            reconciledData.flow(:,3) = interp1(rawTimeElapsedFlow,reconciledData.raw.volFlow_CO2,...
                                            reconciledData.MS(:,1)); % Interpoloted CO2 flow [ccm]
            reconciledData.flow(:,4) = interp1(rawTimeElapsedFlow,reconciledData.raw.volFlow_MFM,...
                                            reconciledData.MS(:,1)); % Interpoloted MFM flow [ccm]
            reconciledData.flow(:,5) = interp1(rawTimeElapsedFlow,reconciledData.raw.setPt_He,...
                                            reconciledData.MS(:,1)); % Interpoloted He setpoint[ccm]                                    
        % Save only the MFM values for the true experiments
        else
            reconciledData.flow(:,2) = interp1(rawTimeElapsedFlow,reconciledData.raw.volFlow_MFM,...
                                            reconciledData.MS(:,1)); % Interpoloted MFM flow [ccm]
        end
    end
                                    
    % Get the mole fraction used for the calibration
    % This will be used in the analyzeCalibration script
    if expInfo.calibrateMeters
        % Compute the mole fractions using the reconciled flow data
        reconciledData.moleFrac(:,1) = (reconciledData.flow(:,2))./(reconciledData.flow(:,2)+reconciledData.flow(:,3));
        reconciledData.moleFrac(:,2) = 1 - reconciledData.moleFrac(:,1);
    % If actual experiment is analyzed, loads the calibration MS file
    else
        % MS Calibration File
        load(fileToLoad.calibrationMS);
        % Convert the raw signal to concentration
        % When independent gas signals are used
        if calibrationMS.flagUseIndGas
            reconciledData.moleFrac(:,1) = polyval(calibrationMS.He,reconciledData.MS(:,2)); % He [-]
            reconciledData.moleFrac(:,2) = polyval(calibrationMS.CO2,reconciledData.MS(:,3)); % CO2 [-]
        % When ratio of gas signals are used
        else
            % Parse out the fitting parameters
            paramFit = calibrationMS.ratioHeCO2;
            % Use a fourier series model to obtain the mole fraction
            reconciledData.moleFrac(:,1) = paramFit(reconciledData.MS(:,2)./...
                (reconciledData.MS(:,2)+reconciledData.MS(:,3))); % He [-]            
            reconciledData.moleFrac(:,2) = 1 - reconciledData.moleFrac(:,1); % CO2 [-]
        end
    end
end