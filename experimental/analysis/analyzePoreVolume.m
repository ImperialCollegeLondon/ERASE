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
% - 2021-07-19, HA: Add interpolation for cumulative pore size distribution
% - 2021-07-15, HA: Minor bug fixes
% - 2021-07-13, HA: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% USER INPUT %%%
clear
% File name to be used for analysis/plotting
rawDataFileName = 'ZYTMA_ZLC_HA';

% Name of files from QC and MIP (only for reference purposes)
poreVolume.rawFiles.quantachromeFileName = '';
poreVolume.rawFiles.mercuryIntrusionFileName = '';
%%%%%%
%%%%%%

% Create poreVolumeData folder to have experiemntal data from Quantachrome
% and MIPporeVolume
if ~exist('poreVolumeData','dir')
    mkdir('poreVolumeData')
end

% Create a dummy file to process
if ~exist(['poreVolumeData',filesep,rawDataFileName,'.mat'],'file')
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
    clear quantachromeFileName mercuryIntrusionFileName rawDataFileName processFlag;
    error('Raw file in .mat format does not exist. Populate the data using raw files from QC or MIP and resave the file. A file with the file name is created in poreVolumeData!!')
end
% Load the saved raw data from the .mat file
load(['poreVolumeData',filesep,rawDataFileName])

if ~isfield(poreVolume,'properties')
    % Save the raw data
    poreVolume.rawData.QC = poreVolume.QC;
    poreVolume.rawData.MIP = poreVolume.MIP;
    % Quantachrome
    % Compute interpolation query points
    interpQC = exp(1).^(linspace(log(min(poreVolume.QC(:,1))),log(max(poreVolume.QC(:,1))),200));
    poreVolume.interp.QC(:,1) = interpQC';
    % interpolate cumulative pore volume
    poreVolume.interp.QC(:,2) = interp1(poreVolume.QC(:,1),poreVolume.QC(:,2),interpQC','spline');
    % Compute incremental pore volume
    poreVolume.interp.QC(:,3) = [poreVolume.interp.QC(1,2);  diff(poreVolume.interp.QC(:,2))];
    % MIP
    % Sort MIP data based on ascending pore radius
    poreVolume.MIP = sortrows(poreVolume.MIP,1);
    % Convert pore radius to pore diameter [nm]
    poreVolume.MIP(:,1) = 2.*poreVolume.MIP(:,1);
    % Compute cumulative pore size distribution
    poreVolume.MIP(:,3) = cumsum(poreVolume.MIP(:,2));
    % Compute interpolation query points
    interpMIP = exp(1).^(linspace(log(min(poreVolume.MIP(:,1))),log(max(poreVolume.MIP(:,1))),200));
    poreVolume.interp.MIP(:,1) = interpMIP';
    % interpolate cumulative pore volume
    poreVolume.interp.MIP(:,2) = interp1(poreVolume.MIP(:,1),poreVolume.MIP(:,3),interpMIP');
    % Compute incremental pore volume
    poreVolume.interp.MIP(:,3) = [poreVolume.interp.MIP(1,2);  diff(poreVolume.interp.MIP(:,2))];
end

% Plot incremental pore volume data
figure('Units','inch','Position',[2 2 10 5])
subplot(1,2,1)
semilogx(poreVolume.interp.QC(:,1),poreVolume.interp.QC(:,3),'or:');
hold on
semilogx(poreVolume.interp.MIP(:,1),poreVolume.interp.MIP(:,3),'xk:');
xlim([0,max(poreVolume.interp.MIP(:,1))]); ylim([0,max([max(poreVolume.interp.MIP(:,3)) max(poreVolume.interp.QC(:,3))])]);
legend('QC', 'MIP','Location','northeast');
xlabel('{\it{D}} [nm]'); ylabel('{\it{dV}} [mL/g]');
set(gca,'FontSize',14)
box on;grid on;
% Plot cumulative pore volume data
subplot(1,2,2)
semilogx(poreVolume.interp.QC(:,1),poreVolume.interp.QC(:,2),'or:');
hold on
semilogx(poreVolume.interp.MIP(:,1),max(poreVolume.interp.QC(:,2))+poreVolume.interp.MIP(:,2),'xk:');
xlim([0,max(poreVolume.interp.MIP(:,1))]); ylim([0,max(poreVolume.interp.QC(:,2)+poreVolume.interp.MIP(:,2))]);
legend('QC', 'MIP','Location','northeast');
xlabel('{\it{D}} [nm]'); ylabel('{\it{dV}} [mL/g]');
set(gca,'FontSize',14)
box on;grid on;

if ~isfield(poreVolume,'properties')
    % Prompt user to enter threshold for combining QC and MIP data based on
    % plot
    prompt = {'Enter pore width threshold [nm]:'};
    dlgtitle = 'PoreVolume';
    dims = [1 35];
    definput = {'20','hsv'};
    poreVolume.options.poreWidthThreshold = str2num(cell2mat(inputdlg(prompt,dlgtitle,dims,definput)));
    
    % Close the plot window
    close all
    
    % Find the indices for QC and MIP to fit threshold
    poreVolume.options.QCindexLast = find(poreVolume.interp.QC(:,1)<poreVolume.options.poreWidthThreshold,1,'last');
    poreVolume.options.MIPindexFirst = find(poreVolume.interp.MIP(:,1)>=poreVolume.options.poreWidthThreshold,1,'first');
    
    % Combine QC and MIP
    % Note that low pore diameter values come from the QC and high pore
    % diameter values come from MIP (due to the theory behind their working)
    % First colume: Pore diamater [nm]
    % Second column: Incremental volume [mL/g]
    % Third column: Cummulative volume [mL/g]
    poreVolume.combined = [poreVolume.interp.QC(1:poreVolume.options.QCindexLast,[1 3]); poreVolume.interp.MIP(poreVolume.options.MIPindexFirst:end,[1 3])];
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
    poreVolume.properties.totalPoreVolume = poreVolume.combined(end-1,3);
    poreVolume.properties.skeletalDensity = 1/(poreVolume.properties.bulkVolume - poreVolume.properties.totalPoreVolume);
    poreVolume.properties.totalVoidage = poreVolume.properties.totalPoreVolume./poreVolume.properties.bulkVolume;
    
    % Get the git commit ID
    poreVolume.gitCommitID = getGitCommit;
end

% Plot the combined cumulative pore volumne distribution
figure('Units','inch','Position',[2 2 5 5])
semilogx(poreVolume.combined(1:poreVolume.options.QCindexLast,1),poreVolume.combined(1:poreVolume.options.QCindexLast,3),'or:');
hold on
semilogx(poreVolume.combined(poreVolume.options.QCindexLast+1:end,1),poreVolume.combined(poreVolume.options.QCindexLast+1:end,3),'ok:');
xlim([0,max(poreVolume.combined(:,1))]); ylim([0,1.1.*max(poreVolume.combined(:,3))]);
legend('QC', 'MIP','Location','southeast');
xlabel('{\it{D}} [nm]'); ylabel('{\it{V}} [mL/g]');
set(gca,'FontSize',14)
box on;grid on;

if isfield(poreVolume,'properties')
    % Save the file with processed data in the poreVolumeData folder
    save(['poreVolumeData',filesep,rawDataFileName], 'poreVolume')
end

fprintf('Skeletal Density = %5.4e g/mL \n',poreVolume.properties.skeletalDensity);
fprintf('Total pore volume = %5.4e mL/g \n',poreVolume.properties.totalPoreVolume);
fprintf('Total voidage = %5.4e \n',poreVolume.properties.totalVoidage);