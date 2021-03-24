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
    experimentStruct.calibrationFlow = 'ZLCCalibrateMeters_20210316__Model'; % Calibration file for meters (.mat)
    experimentStruct.flow = 'ZLCCalibrateMS_Short_20210324'; % Experimental flow file (.mat)
    experimentStruct.calibrationMS = 'ZLCCalibrateMS_20210324_Model'; % Experimental flow file (.mat)
    experimentStruct.MS = 'C:\Users\QCPML\Desktop\Ashwin\MS\ZLCCalibrateMS_Short_20210324.asc'; % Experimental MS file (.asc)
    experimentStruct.interpMS = true; % Flag for interpolating MS data (true) or flow data (false)
    % Call reconcileData function to get the output mole fraction for a
    % real experiment
    outputStruct = concatenateData(experimentStruct);
end