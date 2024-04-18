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
% Analyze pore volume data from Quantachrome (Ar/N2) and Autopore IV (Hg)
% to obtain parallel pore diffusivity
%
% Last modified:
% - 2024-04-17, HA: Initial creation
%
% Input arguments:
%
% Output arguments:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear all;close all;

% Define Temperatures for evaluation
Tvals = [288.15, 298.15, 308.15];

poreData = load('ZYTMA_ZLC_HA.mat');
MIP = poreData.poreVolume.MIP;
macroporeIndex = find(poreData.poreVolume.MIP(1:end,1)>50,1,'first');
macroporeVolume = poreData.poreVolume.MIP(end,3)-poreData.poreVolume.MIP(macroporeIndex,3);
epVals = macroporeVolume./poreData.poreVolume.properties.bulkVolume;

% Chapman-Enskog equation
ChapmanEnskogVals = [4.50000000000000	0.861000000000000
4.60000000000000	0.856800000000000
4.70000000000000	0.853000000000000
4.80000000000000	0.849200000000000
4.90000000000000	0.845600000000000
5	0.842200000000000
6	0.812400000000000
7	0.789600000000000
8	0.771200000000000
9	0.755600000000000
10	0.742400000000000
20	0.664000000000000];

MwCO2 = 44; % molecular weight of CO2 [kg/mol]
MwCar = 4; % molecular weight of He [kg/mol]
sigmaCO2 = 3.941e-10; % collision diameter for CO2 [m]
sigmaCar = 2.551e-10; % collision diameter for Carrier (He) [m]
sigma12 = 1./2.*(sigmaCO2+sigmaCar);
epskCO2 = 195.2; %  force constant for CO2 from Lennard Jones potential divided by boltzmann constant [K]
epskCar = 10.22; %  force constant for Carrier (He) from Lennard Jones potential divided by boltzmann constant [K]
kb = 1.38e-23; % boltzmann constant [J/K]
eps12 = sqrt(epskCO2.*kb.*epskCar.*kb); %  force constant for CO2 and Carrier (He) from Lennard Jones potential divided by boltzmann constant [K]
kTbyeps12 = kb./eps12.*Tvals; %  kT divided by eps12 for interpolation
DmVal = zeros(1,length(Tvals));
for ii = 1:length(Tvals)
    omegaD = interp1(ChapmanEnskogVals(:,1),ChapmanEnskogVals(:,2),kTbyeps12(ii));
    DmVal(ii) = (0.001858.*Tvals(ii).^1.5 .*(1./44 + 1./4).^0.5) ./(1*(sigma12*1e10)^2.*omegaD)*1e-4; % Equimolar counter diffusivity [m2/s]
end


DpVal = zeros(1,3);

set(groot,'defaulttextInterpreter','latex') %latex axis labels
set(groot, 'DefaultLegendInterpreter', 'latex')
MarkersForPlot = ["o","o","o","o"];
MarkersForPlotH2 = ["o","v","square","diamond"];
ColorsForPlotAll = brewermap(9,'YlOrRd');
ColorsForPlot = ColorsForPlotAll([4 6 8],:);
sz = 50;
fsz = 15;
legfsz = 15;
capsz = 3;
LineStyles = [":","-.","-"];

distType = 'wbl';

tiledlayout(1,1, 'Padding', 'compact', 'TileSpacing', 'compact');
nexttile
hold on
x = poreData.poreVolume.MIP(macroporeIndex:77,1);
p = cumtrapz(x,poreData.poreVolume.MIP(macroporeIndex:77,4));
p = p - min(p);
pVals = linspace(min(p),max(p),12000)./max(p);
xVals = interp1(p./max(p),x,pVals)-poreData.poreVolume.MIP(macroporeIndex,1);
xVals = interp1(p./max(p),x,pVals);
dist = fitdist(xVals',distType)
dvals = linspace(MIP(macroporeIndex,1),MIP(end,1),100000);
distribPDF = pdf(dist,dvals);
for kk = 1:length(Tvals)
    DkVals = 97.*9./13./2.*dvals.*(1e-9).*sqrt(Tvals(kk)./44.01);
    Drvals = 1./(1./DkVals + 1./DmVal(kk));
    frvals = distribPDF;
    figure(1)

    yyaxis right
    hold on
    DrFr = Drvals.*frvals;
    semilogx(dvals,cumtrapz(dvals,DrFr)./epVals,'LineWidth',2, 'LineStyle',LineStyles(kk), 'HandleVisibility','off')
    DpVal(kk) = trapz(dvals,DrFr)./epVals;
    ylabel('$$\frac{1}{\epsilon_{\mathrm{p}}}$$$$\int_{50\mathrm{ nm}}^{W}\mathit{D(W)f(W)  \,dW}$$ [m$$^2$$s$$^{-1}$$]','FontSize',15)

    yyaxis left
    semilogx(dvals,frvals,'LineWidth',2)
    hold on
    DrFr = Drvals.*frvals;
    DpVal(kk) = trapz(dvals,DrFr)./epVals;
    ylabel('$$\mathit{f(W)}$$ [-]','FontSize',15)
    set(gca,'YScale','linear','XScale','log','FontSize',fsz,'LineWidth',0.8)
    grid on; axis square; box on
    set(gca,'fontname','arial')
    yyaxis left
    ylim([0 0.005])


    figure(100)
    set(gcf,'Position',  [0 0 350 350])
    semilogx(dvals,Drvals,'LineWidth',2,'Color','black', 'LineStyle',LineStyles(kk),'DisplayName',[num2str(Tvals(kk)),' K'])
    set(gca,'YScale','linear','XScale','log','FontSize',fsz,'LineWidth',0.8)
    grid on; axis square; box on
    set(gca,'fontname','arial')
    hold on
    ylabel('$$\mathit{D(W)}$$ [m$$^2$$s$$^{-1}$$]','FontSize',15)
    xlabel('Pore width [nm]','FontSize',15)
    ylim([0 7e-5])
    xlim([50 3e5])
    legend('Location','northwest')
end

tauVals = 1.42;
tauDelta = 0.016;
tauFac = epVals./tauVals;
tauFac2 = epVals'./(tauVals'+tauDelta);
DeVals = tauFac.*DpVal;
DeValsDelta = abs(tauFac2'.*DpVal-DeVals);
