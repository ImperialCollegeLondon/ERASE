%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Imperial College London, United Kingdom
% Multifunctional Nanomaterials Laboratory
%
% Project:  ERASE
% Year:     2021
% MATLAB:   R2020a
% Authors:  Ashwin Kumar Rajagopalan (AK)
%
% Purpose: 
% Used to find the COM ports for the different devices (inspired from AK's
% work at ETHZ)
%
% Last modified:
% - 2021-03-12, AK: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function portAddress = matchUSBport(varargin)
% Only one input argument is accepted (array of cells)
if nargin > 1
    error('Too many input arguments');
end

if exist('listComPorts.vbs','file')~=2
    warning('listComPorts.vbs not found. The script will be generated.');
    fileID = fopen('listComPorts.vbs','w');
    fprintf(fileID,['Set portList = GetComPorts()\n\nportnames = portList.Keys\nfor each pname in portnames\n\tSet portinfo = portList.item(pname)\n\t',...
        'wscript.echo pname & "~" & _\n\t\tportinfo.PNPDeviceID\nNext\n\n'...
        'Function GetComPorts()\n\tset portList = CreateObject("Scripting.Dictionary")\n\n\tstrComputer = "."\n\tset objWMIService = GetObject',...
        '("winmgmts:\\\\" & strComputer & "\\root\\cimv2")\n\tset colItems = objWMIService.ExecQuery ("Select * from Win32_PnPEntity")\n\t'...
        'for each objItem in colItems\n\t\tIf Not IsNull(objItem.Name) Then\n\t\t\tset objRgx = CreateObject("vbScript.RegExp")\n\t\t\t',...
        'objRgx.Pattern = "COM[0-9]+"\n\t\t\tSet objRegMatches = objRgx.Execute(objItem.Name)\n\t\t\tif objRegMatches.Count = 1 Then  portList.Add ',...
        'objRegMatches.Item(0).Value, objItem\n\t\tEnd if\n\tNext\n\tset GetComPorts = portList\nEnd Function']);
    fclose(fileID);
end
bashPath = which('listComPorts.vbs');
[~, bashOut] = system(['cscript.exe //nologo ',bashPath]);

stringBashOut = string(bashOut(1:end-1));
splitBashOut = split(split(stringBashOut,char(10)),'~'); %#ok<CHARTEN>

% If no argument was provided, just output all the ports available
if nargin == 0
    portAddress = splitBashOut;
    return;
else
    if ~iscell(varargin{1})
        if ischar(varargin{1}) && isrow(varargin{1})
            portIdentifier = varargin(1);
        else
            error('The list of ports must be given as a cell array of chars');
        end
    else
        % Get the input device names
        portIdentifier = varargin{1};
    end
end

% If anyway no device is available throw an error and exit
if strcmp(splitBashOut,"")
    error('matchUSBport:noUSBDeviceFound',...
        'No USB device was found. Aborting.');
end

% In case there is only one device (two entries), a column vector is output
% when performing the split, but we actually want a row vector
if isequal(size(splitBashOut),[2 1])
    splitBashOut = splitBashOut';
end

% Search for all the input device names (port identifiers)
for kk=1:length(portIdentifier)
    cellMatch = regexp(splitBashOut,portIdentifier{kk});
    cellMatchLogic = cellfun(@(x)~isempty(x),cellMatch);
    if sum(cellMatchLogic(:))>1
        warning('matchUSBport:nonUniqueIdentification',...
            ['The identifier ',portIdentifier{kk},' has multiple matches among the USB ports available. No port could be assigned.']);
        portAddress{kk} = ''; %#ok<*AGROW>
        continue;
    end
    portAddress{kk} = char(splitBashOut(fliplr(cellMatchLogic)));
end
end