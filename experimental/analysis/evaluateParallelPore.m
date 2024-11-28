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
clc;clear all; close all;

% Define Temperatures for evaluation
Tvals = [288.15, 298.15, 308.15];
% Tvals = [288.15];
% Tvals = linspace(273.15,373.15,10);

CarrierGas = 'He';
Ptotal = 1; % total pressure in atm
Rg = 8.314;
poreData = load('Copy_of_ZYTMA_ZLC_HA.mat');
MIP = poreData.poreVolume.MIP;
macroporeIndex = find(poreData.poreVolume.MIP(1:end,1)>1,1,'first');
% macroporeIndex = 2;
% endIndex = find(poreData.poreVolume.MIP(1:end,1)>50,1,'first');
endIndex = length(poreData.poreVolume.MIP(1:end,1))-1;
for ii = 2:length(poreData.poreVolume.MIP(macroporeIndex:endIndex,4))
    if poreData.poreVolume.MIP(ii,4) == poreData.poreVolume.MIP(ii-1,4)
        poreData.poreVolume.MIP(ii,4) = 99;
    end
end

poreData.poreVolume.MIP(find(poreData.poreVolume.MIP(:,4)==99),:) = [];

MIP = poreData.poreVolume.MIP;
macroporeIndex = find(poreData.poreVolume.MIP(1:end,1)>50,1,'first');
% macroporeIndex = 3;
endIndex = length(poreData.poreVolume.MIP(1:end,1))-1;

macroporeVolume = poreData.poreVolume.MIP(end,4)-poreData.poreVolume.MIP(macroporeIndex,4);
epVals = macroporeVolume./poreData.poreVolume.properties.bulkVolume;
Rp = 0.5./(((sum(poreData.poreVolume.MIP(macroporeIndex:endIndex,2)./poreData.poreVolume.MIP(macroporeIndex:endIndex,1))))./(sum(poreData.poreVolume.MIP(macroporeIndex:endIndex,2)))).*(1e-9);
% Chapman-Enskog equation
ChapmanEnskogVals = UnpackCEVals;
MwCO2 = 44.01; % molecular weight of CO2 [kg/mol]
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
kTbyeps12 = kb.*Tvals./eps12; %  kT divided by eps12 for interpolation
DmVal = zeros(1,length(Tvals));
omegaDVals = zeros(1,length(Tvals));
for ii = 1:length(Tvals)
    omegaD = interp1(ChapmanEnskogVals(:,1),ChapmanEnskogVals(:,2),kTbyeps12(ii));
    DmVal(ii) = (0.001858.*Tvals(ii).^1.5 .*(1./MwCO2 + 1./MwCar).^0.5) ./(Ptotal*(sigma12*1e10)^2.*omegaD)*1e-4; % Equimolar counter diffusivity [m2/s]
    omegaDVals(ii) = omegaD;
end

Ddg = [];
DpVal = zeros(1,length(Tvals));

set(groot,'defaulttextInterpreter','latex') %latex axis labels
set(groot, 'DefaultLegendInterpreter', 'latex')
MarkersForPlot = ["o","o","o","o"];
MarkersForPlotH2 = ["o","v","square","diamond"];
sz = 50;
fsz = 15;
legfsz = 15;
capsz = 3;
LineStyles = [":","-.","-"];

distType = 'kernel';

tiledlayout(1,1, 'Padding', 'compact', 'TileSpacing', 'compact');
nexttile
hold on
x_old = poreData.poreVolume.MIP(macroporeIndex:endIndex,1);
p_old = cumtrapz(x_old,poreData.poreVolume.MIP(macroporeIndex:endIndex,3));
p_old = p_old - min(p_old);
p = p_old(find(diff(p_old)~=0)+1);
x = x_old(find(diff(p_old)~=0)+1);
pVals = linspace(min(p),max(p),20000)./max(p);
xVals = interp1(p./poreData.poreVolume.properties.bulkVolume,x,pVals);
dist = fitdist(xVals',distType);
dvals = logspace(log10(MIP(macroporeIndex,1)),log10(MIP(end,1)),200000);
% dvals = linspace((MIP(macroporeIndex,1)),(MIP(end,1)),200000);
distribPDF = pdf(dist,dvals);
distribPDF(distribPDF<1e-5) = 0;
% distribPDF = distribPDF./trapz(dvals,distribPDF);
figure(1)
hold on
for kk = 1:length(Tvals)
%     DkVals = 97.*9./13./2.*dvals.*(1e-9).*sqrt(Tvals(kk)./44.01);
    DkVals = @(x) 9./13.*2./3.*(x).*(1e-9)./2.*sqrt(8.*Rg.*Tvals(kk)./(pi.*0.04401));
    Drvals = @(x) 1./(1./DkVals(x) + 1./DmVal(kk));
    Ddg(kk) = 1./(1./(9./13.*2./3.*Rp.*sqrt(8.*Rg.*Tvals(kk)./(pi.*0.04401)))+1./DmVal(kk));
    % frvals = distribPDF;
    % frvals(frvals<1e-6) = 0;
    % frvals = frvals./sum(distribPDF);
    yyaxis right
    hold on
    DrFr = Drvals(dvals).*distribPDF;
    % semilogx(dvals,cumtrapz(dvals,DrFr)./max(p_old),'LineWidth',2, 'LineStyle','--', 'HandleVisibility','off')
    semilogx(dvals,cumtrapz(dvals,DrFr),'LineWidth',2, 'LineStyle','-', 'HandleVisibility','off')
    % semilogx(dvals,cumsum((dvals(2)-dvals(1)).*DrFr),'LineWidth',2, 'LineStyle','--', 'HandleVisibility','off')
    DpVal(kk) = trapz(dvals,DrFr)./epVals;
    ylabel('$$\frac{1}{\epsilon_{\mathrm{p}}}$$$$\int_{50\mathrm{ nm}}^{W}\mathit{D(W)f(W)  \,dW}$$ [m$$^2$$s$$^{-1}$$]','FontSize',15)
    ylabel('$$\int_{50\mathrm{ nm}}^{W}\mathit{D(W)f(W)  \,dW}$$ [m$$^2$$s$$^{-1}$$]','FontSize',15)

    yyaxis left
    hold on
    semilogx(dvals,distribPDF.*max(p_old),'LineWidth',2,'Marker','none')
    % semilogx(dvals,distribPDF,'LineWidth',2,'Marker','none','LineStyle','--')
    plot(poreData.poreVolume.MIP(:,1),poreData.poreVolume.MIP(:,3), 'HandleVisibility','off','LineWidth',1,'LineStyle','none','Marker','o','MarkerFaceColor',	"#0072BD")
    % plot(poreData.poreVolume.MIP(:,1),poreData.poreVolume.MIP(:,3)./max(p_old), 'HandleVisibility','off','LineWidth',1,'LineStyle','none','Marker','o')
    xlim([50 2e3])
    % DrFr = Drvals.*frvals;
    DpVal(kk) = trapz(dvals,DrFr)./epVals;
    ylabel('$$\mathit{f(W)}$$ [-]','FontSize',15)
    xlabel('$$W$$ [nm]','FontSize',15)
    set(gca,'YScale','linear','XScale','log','FontSize',fsz,'LineWidth',1)
    grid on; axis square; box on
    set(gca,'fontname','arial')
    yyaxis left
%     ylim([0 0.005])

end

figure(2)
for kk = 1:length(Tvals)
    DkVals = 9./13.*2./3.*dvals.*(1e-9)./2.*sqrt(8.*Rg.*Tvals(kk)./(pi.*0.04401));
    Drvals = 1./(1./DkVals + 1./DmVal(kk));
    hold on
    set(gcf,'Position',  [0 0 350 350])
    semilogx(dvals,Drvals,'LineWidth',2,'Color','black', 'LineStyle',':','DisplayName',[num2str(Tvals(kk)),' K'])
    set(gca,'YScale','linear','XScale','log','FontSize',fsz,'LineWidth',1)
    grid on; axis square; box on
    set(gca,'fontname','arial')
    ylabel('$$\mathit{D(W)}$$ [m$$^2$$s$$^{-1}$$]','FontSize',15)
    xlabel('Pore width [nm]','FontSize',15)
    ylim([0 7e-5])
    xlim([50 2e3])
    legend('Location','northwest')
    xline(DmVal(kk),"")
end

tauVals = 1.39;
tauDelta = 0.016;
tauFac = epVals./tauVals;
tauFac2 = epVals'./(tauVals'+tauDelta);
DeVals = tauFac.*DpVal;
DeValsDelta = abs(tauFac2'.*DpVal-DeVals);

DpVal.*epVals;

figure
scatter(Tvals,Ddg)
hold on
p = polyfit(Tvals, Ddg,1);
plot(linspace(0, 340), polyval(p, linspace(0, 340)),'Color','b','LineWidth',2,'LineStyle','-','HandleVisibility','off')
scatter(Tvals,DpVal)

figure
scatter(ChapmanEnskogVals(66:70,1).*eps12./kb,ChapmanEnskogVals(66:70,2),60,'filled','b')
hold on
Nt = length(ChapmanEnskogVals(66:70,1));
fun = @(x) log(sum(((x(1).* ((ChapmanEnskogVals(66:70,1).*eps12./kb)./288).^x(2)+0*x(3)) - ChapmanEnskogVals(66:70,2)).^2)); 
fun = @(x) Nt./2.*log(sum(((x(1).* ((ChapmanEnskogVals(66:70,1).*eps12./kb)./250).^x(2)) - ChapmanEnskogVals(66:70,2)).^2)); 
x0 = [0,0,0];
x0 = [1,-0.2];
% options = optimoptions('lsqnonlin','Algorithm','levenberg-marquardt')
% [x fval] = lsqnonlin(fun,x0,[-10,-10,-10],[10,10,10],options)
% [x fval] = ga(fun,3,[],[],[],[],[-100,-100,-100],[100,100,100]);
[x fval] = ga(fun,2,[],[],[],[],[0,-2],[5,5])
TvalsCE = linspace(0,500,1000);
% plot(TvalsCE,x(1).*(TvalsCE./288).^x(2)+x(3))
plot(TvalsCE,x(1).*(TvalsCE./250).^x(2),'LineWidth',2,'Color','b')
xlim([250 450])
set(gca,'YScale','linear','XScale','linear','FontSize',fsz,'LineWidth',1)
grid on; axis square; box on
set(gca,'fontname','arial')
ylabel('$$\Omega_{D,12}$$ [-]','FontSize',15)
xlabel('Temperature [K]','FontSize',15)

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