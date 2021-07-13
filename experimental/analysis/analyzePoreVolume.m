%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Imperial College London, United Kingdom
% Multifunctional Nanomaterials Laboratory
%
% Project:  ERASE
% Year:     2021
% MATLAB:   R2020a
% Authors:  Hassan Azzan (HA)
%
% Purpose:
% Process pore volume data from Quantachrome (Ar/N2) and Autopore IV (Hg)
%
% Last modified:
% - 2021-07-13, HA: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% USER INPUT %%%
% File name to be used for analysis
rawDataFileName = 'dummy.mat';

% Name of files from QC and MIP (only for reference purposes)
poreVolume.rawFiles.quantachromeFileName = 'AC_S1_N2_77K_2021_07_09 (DFT method  Pore Size Distribution)';
poreVolume.rawFiles.mercuryIntrusionFileName = 'AC_P';
%%%%%%
%%%%%%

% Create poreVolumeData folder to have experiemntal data from Quantachrome
% and MIPporeVolume
if ~exist('poreVolumeData','dir')
    mkdir('poreVolumeData')
end

% Create a dummy file to process
if ~exist(['poreVolumeData',filesep,rawDataFileName],'file')
    % First column - pore diameter [nm]
    % Second column - cummulative volume, V [ml/g] (Ar/N2)
    poreVolume.QC = [];
    % First column - pore radius [nm]
    % Second column - incremental intrusion [ml/g] (Hg)
    poreVolume.MIP = [];
    % Save the file
    save(['poreVolumeData',filesep,rawDataFileName], 'poreVolume')
    % Print error message to let the user know about the create of the
    % dummy file that should be populated with the raw data
    load(['poreVolumeData',filesep,rawDataFileName]);
    clear quantachromeFileName mercuryIntrusionFileName rawDataFileName;
    error('Raw file in .mat format does not exist. Populate the data using raw files from QC or MIP and resave the file. A file with the file name is created in poreVolumeData!!')
end
% Load the saved raw data from the .mat file
load(['poreVolumeData',filesep,rawDataFileName])

% Quantachrome
% Compute incremental pore volume
poreVolume.QC(:,3) = [poreVolume.QC(1,2);  diff(poreVolume.QC(:,2))];

% MIP
% Sort MIP data based on ascending pore radius
poreVolume.MIP = sortrows(poreVolume.MIP,1);
% Convert pore radius to pore diameter [nm]
poreVolume.MIP(:,1) = 2.*poreVolume.MIP(:,1);

% Plot incremental pore volume data
figure('Units','inch','Position',[2 2 6.6 6.6])
semilogx(poreVolume.QC(:,1),poreVolume.QC(:,3),'or:');
hold on
semilogx(poreVolume.MIP(:,1),poreVolume.MIP(:,2),'xk:');
xlim([0,max(poreVolume.MIP(:,1))]); ylim([0,max([max(poreVolume.MIP(:,2)) max(poreVolume.QC(:,3))])]);
legend('QC', 'MIP','Location','northwest');
xlabel('{\it{D}} [nm]'); ylabel('{\it{dV}} [mL/g]');
set(gca,'FontSize',8)
box on;grid on;

% Prompt user to enter threshold for combining QC and MIP data based on
% plot
prompt = {'Enter pore width threshold [nm]:'};
dlgtitle = 'PoreVolume';
dims = [1 35];
definput = {'20','hsv'};
poreWidthThreshold = str2num(cell2mat(inputdlg(prompt,dlgtitle,dims,definput)));

% Close the plot window
close all

% Find the indices for QC and MIP to fit threshold
QCindexLast = find(poreVolume.QC(:,1)<poreWidthThreshold,1,'last');
MIPindexFirst = find(poreVolume.MIP(:,1)>=poreWidthThreshold,1,'first');

% Combine QC and MIP
% Note that low pore diameter values come from the QC and high pore
% diameter values come from MIP (due to the theory behind their working)
% First colume: Pore diamater [nm]
% Second column: Incremental volume [mL/g]
% Third column: Cummulative volume [mL/g]
poreVolume.combined = [poreVolume.QC(1:QCindexLast,[1 3]); poreVolume.MIP(MIPindexFirst:end,1:2)];
poreVolume.combined(:,3) = cumsum(poreVolume.combined(:,2));

% Prompt user to enter bulk density of sample from MIP
prompt = {'Enter bulk density [g/mL]:'};
dlgtitle = 'PoreVolume';
dims = [1 35];
definput = {'20','hsv'};
bulkDensity = str2num(cell2mat(inputdlg(prompt,dlgtitle,dims,definput)));

% Calculate material properties from data
poreVolume.properties.bulkDensity = bulkDensity;
poreVolume.properties.bulkVolume = 1./bulkDensity;
poreVolume.properties.totalPoreVolume = poreVolume.combined(end,3);
poreVolume.properties.skeletalDensity = 1/(poreVolume.properties.bulkVolume - poreVolume.properties.totalPoreVolume);
poreVolume.properties.totalVoidage = poreVolume.properties.totalPoreVolume./poreVolume.properties.bulkVolume;

% Get the git commit ID
poreVolume.gitCommitID = getGitCommit;

% Plot the combined cumulative pore volumne distribution
% Plot incremental pore volume data
figure('Units','inch','Position',[2 2 6.6 6.6])
semilogx(poreVolume.combined(1:QCindexLast,1),poreVolume.combined(1:QCindexLast,3),'or:');
hold on
semilogx(poreVolume.combined(QCindexLast+1:end,1),poreVolume.combined(QCindexLast+1:end,3),'ok:');
xlim([0,max(poreVolume.combined(:,1))]); ylim([0,1.1.*max(poreVolume.combined(:,3))]);
legend('QC', 'MIP','Location','northwest');
xlabel('{\it{D}} [nm]'); ylabel('{\it{V}} [mL/g]');
set(gca,'FontSize',8)
box on;grid on;

% Save the file with processed data in the poreVolumeData folder
save(['poreVolumeData',filesep,rawDataFileName], 'poreVolume')