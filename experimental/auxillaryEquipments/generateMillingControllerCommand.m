%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ETH Zurich, Switzerland
% Separation Processes Laboratory
%
% Project:  CrystOCAM 2.0, Population Balance Solver
% Year:     2018
% MATLAB:   R2017b, Windows 64bit
% Authors:  Ashwin Kumar Rajagopalan (AK)
%
% Purpose:
% Generate commands that would be sent to the milling controller. This
% works for IKA Magic Lab with the MK/MKO module.
%
% Last modified:
% - 2018-07-26, AK: Added command to obtain the speed from the mill
% - 2018-07-25, AK: Initial creation
%
% Input arguments:
% - millOperation:      String that defines the operation of the mill.
%                       Includes starting, stopping, and changing the
%                       milling intensity
% - millingIntensity:   Milling intensity (1/min)
%
% Output arguments:
% millingControllerCommand: Command that would be eventually sent to the
%                           controller
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [millingControllerCommand] = generateMillingControllerCommand(millOperation,millingIntensity)
% Character that corresponds to the start of the text byte
startOfText = char(2);
% Character that corresponds to the end of the text byte
endOfText = char(3);
% Character that corresponds to the end of the transmission byte
endOfTransmission = char(4);
% Character that corresponds to the end of the transmission byte
enquiryChar = char(5);
% Character that corresponds to the address of the master/slave
% configuration: This is used for the controller of IKA Magic LAB
addressChar = '@';
% Flag to check if command is meant to be for commanding the mill or
% enquiring the mill about a particular setting
flagEnq = false;

% Generate codes based on the operation that needs to be performed on
% the mill
switch millOperation
    % Start the mill
    case 'START_MILL'
        commandForTheOperation = '20551=0001';
    % Stop the mill
    case 'STOP_MILL'
        commandForTheOperation = '20551=0000';
    % Set intensity of the rotor shaft
    case 'SET_INTENSITY'
        millingIntensityBin_IEEE754 = float2bin(single(millingIntensity));
        millingIntensityDec_IEEE754 = bin2dec(millingIntensityBin_IEEE754);
        millingIntensityHex_IEEE754 = dec2hex(millingIntensityDec_IEEE754);
        commandForTheOperation = ['20284=',millingIntensityHex_IEEE754];
	% Read the intensity of the rotor shaft
    case 'READ_INTENSITY'
        flagEnq = true;
        commandForTheOperation = '20284';
	% Reset error in the controller
    case 'RESET_ERROR'
        commandForTheOperation = '20074=0001';
    otherwise
        error('generateMillingControllerCommand:invalidmillOperation',...
            'Invalid millOperation input.');
end

% Generate the parity bit for the command to be sent to the controller. The
% approach applied in this specific controller is called the longitudinal
% parity check. The bytes of the command being sent are arranged and an
% exclusive or operation is performed on all the bytes. In this specific
% controller the parity byte is evaluated using the command for the
% specific operation (commandForTheOperation) and the end of the text byte
% (endOfText). Refer to the manual provided by IKA.
checksumChar = computeChecksumChar([commandForTheOperation,endOfText]);

% Generate the command that needs to be sent to the controller of the mill
% If a particular parameter needs to be read from the mill
if flagEnq
    millingControllerCommand = [endOfTransmission,addressChar,startOfText,...
        commandForTheOperation,enquiryChar];
% If a particular parameter needs to be sent to the mill
else
    millingControllerCommand = [endOfTransmission,addressChar,startOfText,...
        commandForTheOperation,endOfText,checksumChar];
end
end

%% Function to genereate the checksum/parity byte
function checksumChar = computeChecksumChar(commandToBeAnalyzed)
% Convert the command to hex form and stack it as a column vector
hexCode = strsplit(sprintf('%X\t',commandToBeAnalyzed))';
% Initialize to empty cells/vector
decCode = {}; binTemp = {}; binCode = [];
% Loop over all the characters in the commandToBeAnalyzed. Note: loops till
% size - 1, because of the \t used in the previous line. Generates an extra
% element.
for ii = 1:size(hexCode,1)-1
    % Convert the command from hexadecimal to decimal
    decCode{ii,1} = hex2dec(hexCode{ii,1});
    % Convert the command from decimal to binary
    binTemp{ii,1} = dec2bin(decCode{ii,1},8);
    % Obtain a vector of binary representation for the command. Each
    % character is represented as a row with n bits
    binCode(ii,:) = binTemp{ii,1} - '0';
end

% Obtain the parity byte by performing the XOR operation on the command
% genereated (this has to be done on the binary form of the command)
% Initialize the XOR output to the first character binary representation
xorValue = binCode (1,:);
% Loop over all the other characters and perform the XOR oepration
for jj = 2:size(binCode,1)
    xorValue = xor(xorValue,binCode(jj,:));
end

% Generate the binary representation from the logical values obtained from
% the XOR function
checksumBinary = sprintf('%d', xorValue);
% Convert the parity byte to decimal
checksumDecimal = bin2dec(checksumBinary);
% Convert the parity byte to hex (just to check. Not used!)
checksumHex = dec2hex(checksumDecimal);
% Convert the decimal representation to ASCII text to be used in the
% command to be sent to the controller
checksumChar = char(checksumDecimal);
end