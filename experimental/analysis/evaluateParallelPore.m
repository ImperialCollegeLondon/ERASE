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
clc;clear all;

% Define Temperatures for evaluation
Tvals = [288.15, 298.15, 308.15];

CarrierGas = 'He';
Ptotal = 1; % total pressure in atm

poreData = load('DummtKillian.mat');
MIP = poreData.poreVolume.MIP;
macroporeIndex = find(poreData.poreVolume.MIP(1:end,1)>50,1,'first');
macroporeVolume = poreData.poreVolume.MIP(end,4)-poreData.poreVolume.MIP(macroporeIndex,4);
epVals = macroporeVolume./poreData.poreVolume.properties.bulkVolume;

% Chapman-Enskog equation
ChapmanEnskogVals = UnpackCEVals;
MwCO2 = 44; % molecular weight of CO2 [kg/mol]
switch CarrierGas
    case 'He'
        sigmaCar = 2.551e-10; % collision diameter for Carrier (He) [m]
        epskCar = 10.22; %  force constant for Carrier (He) from Lennard Jones potential divided by boltzmann constant [K]
        MwCar = 4; % molecular weight of He [g/mol]
    case 'Ar'
        sigmaCar = 3.542e-10; % collision diameter for Carrier (Ar) [m]
        epskCar = 93.3; %  force constant for Carrier (Ar) from Lennard Jones potential divided by boltzmann constant [K]
        MwCar = 40; % molecular weight of Ar [g/mol]
    case 'N2'
        sigmaCar = 3.798e-10; % collision diameter for Carrier (N2) [m]
        epskCar = 71.4; %  force constant for Carrier (N2) from Lennard Jones potential divided by boltzmann constant [K]
        MwCar = 28; % molecular weight of N2 [g/mol]
end
sigmaCO2 = 3.941e-10; % collision diameter for CO2 [m]
sigma12 = 1./2.*(sigmaCO2+sigmaCar);
epskCO2 = 195.2; %  force constant for CO2 from Lennard Jones potential divided by boltzmann constant [K]
kb = 1.38e-23; % boltzmann constant [J/K]
eps12 = sqrt(epskCO2.*kb.*epskCar.*kb); %  force constant for CO2 and Carrier (He) from Lennard Jones potential divided by boltzmann constant [K]
kTbyeps12 = kb./eps12.*Tvals; %  kT divided by eps12 for interpolation
DmVal = zeros(1,length(Tvals));
for ii = 1:length(Tvals)
    omegaD = interp1(ChapmanEnskogVals(:,1),ChapmanEnskogVals(:,2),kTbyeps12(ii));
    DmVal(ii) = (0.001858.*Tvals(ii).^1.5 .*(1./MwCO2 + 1./MwCar).^0.5) ./(Ptotal*(sigma12*1e10)^2.*omegaD)*1e-4; % Equimolar counter diffusivity [m2/s]
end


DpVal = zeros(1,length(Tvals));

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
p = cumtrapz(x,poreData.poreVolume.MIP(macroporeIndex:77,3));
p = p - min(p);
pVals = linspace(min(p),max(p),12000)./max(p);
% xVals = interp1(p./max(p),x,pVals)-poreData.poreVolume.MIP(macroporeIndex,1);
xVals = interp1(p./max(p),x,pVals);
dist = fitdist(xVals',distType)
dvals = linspace(MIP(macroporeIndex,1),MIP(end,1),100000);
distribPDF = pdf(dist,dvals);
figure(1)
hold on
for kk = 1:length(Tvals)
    DkVals = 97.*9./13./2.*dvals.*(1e-9).*sqrt(Tvals(kk)./44.01);
    Drvals = 1./(1./DkVals + 1./DmVal(kk));
    frvals = distribPDF;
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

end

figure(2)
for kk = 1:length(Tvals)
    DkVals = 97.*9./13./2.*dvals.*(1e-9).*sqrt(Tvals(kk)./44.01);
    Drvals = 1./(1./DkVals + 1./DmVal(kk));
    frvals = distribPDF;
    hold on
    set(gcf,'Position',  [0 0 350 350])
    semilogx(dvals,Drvals,'LineWidth',2,'Color','black', 'LineStyle',LineStyles(kk),'DisplayName',[num2str(Tvals(kk)),' K'])
    set(gca,'YScale','linear','XScale','log','FontSize',fsz,'LineWidth',0.8)
    grid on; axis square; box on
    set(gca,'fontname','arial')
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

function ChapmanEnskogVals = UnpackCEVals
% tabulated values for kT/eps12 vs OmegaD from Mass Transfer in
% Heterogeneous Catalysis - Satterfield 1970 M.I.T. Press pg. 14-15
ChapmanEnskogVals = [0.300000000000000	2.66200000000000
0.350000000000000	2.47600000000000
0.400000000000000	2.31800000000000
0.450000000000000	2.18400000000000
0.500000000000000	2.06600000000000
0.550000000000000	1.96600000000000
0.600000000000000	1.87700000000000
0.650000000000000	1.79800000000000
0.700000000000000	1.72900000000000
0.750000000000000	1.66700000000000
0.800000000000000	1.61200000000000
0.850000000000000	1.56200000000000
0.900000000000000	1.51700000000000
0.950000000000000	1.47600000000000
1	1.43900000000000
1.05000000000000	1.40600000000000
1.10000000000000	1.37500000000000
1.15000000000000	1.34600000000000
1.20000000000000	1.32000000000000
1.25000000000000	1.29600000000000
1.30000000000000	1.27300000000000
1.35000000000000	1.25300000000000
1.40000000000000	1.23300000000000
1.45000000000000	1.21500000000000
1.50000000000000	1.19800000000000
1.55000000000000	1.18200000000000
1.60000000000000	1.16700000000000
1.65000000000000	1.15300000000000
1.70000000000000	1.14000000000000
1.75000000000000	1.12800000000000
1.80000000000000	1.11600000000000
1.85000000000000	1.10500000000000
1.90000000000000	1.09400000000000
1.95000000000000	1.08400000000000
2	1.07500000000000
2.10000000000000	1.05700000000000
2.20000000000000	1.04100000000000
2.30000000000000	1.02600000000000
2.40000000000000	1.01200000000000
2.50000000000000	0.999600000000000
2.60000000000000	0.987800000000000
2.70000000000000	0.977000000000000
2.80000000000000	0.967200000000000
2.90000000000000	0.957600000000000
3	0.949000000000000
3.10000000000000	0.940600000000000
3.20000000000000	0.932800000000000
3.30000000000000	0.925600000000000
3.40000000000000	0.918600000000000
3.50000000000000	0.912000000000000
3.60000000000000	0.905800000000000
3.70000000000000	0.899800000000000
3.80000000000000	0.894200000000000
3.90000000000000	0.888800000000000
4	0.883600000000000
4.10000000000000	0.878800000000000
4.20000000000000	0.874000000000000
4.30000000000000	0.869400000000000
4.40000000000000	0.865200000000000
4.50000000000000	0.861000000000000
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
20	0.664000000000000
30	0.623200000000000
40	0.596000000000000
50	0.575600000000000
60	0.559600000000000
70	0.546400000000000
80	0.535200000000000
90	0.525600000000000
100	0.513000000000000
200	0.464400000000000
400	0.417000000000000];
end