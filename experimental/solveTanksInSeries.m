function df = solveTanksInSeries(t, f, initMoleFrac, feedMoleFrac, flowRate, DVParams)
%
if length(DVParams) == 6
    deadVolume_1 = DVParams(1);
    deadVolume_2M = DVParams(2);
    deadVolume_2D = DVParams(3);
    numTanks_1 = DVParams(4);
    flowRate_2D = (1-feedMoleFrac).^DVParams(6).*DVParams(5);
else
    deadVolume_1 = DVParams(1);
    deadVolume_2M = DVParams(2);
    deadVolume_2D = DVParams(3);
    numTanks_1 = DVParams(4);
    flowRate_2D = DVParams(5);
end

numTanksTotal = numTanks_1 + 2;

df = zeros(numTanksTotal,1);

volTank_1 = deadVolume_1./numTanks_1;
residenceTime_1 = volTank_1./(flowRate);

df(1) = ((1./residenceTime_1).*(feedMoleFrac - f(1)));
df(2:numTanks_1) = (1./residenceTime_1).*(f(1:numTanks_1-1) - f(2:numTanks_1));

volTank_2M = deadVolume_2M;
volTank_2D = deadVolume_2D;

flowRate_M = flowRate - flowRate_2D;
residenceTime_2M = volTank_2M./(flowRate_M);
residenceTime_2D = volTank_2D./(flowRate_2D);
%
% # Solve the odes
%     # Volume 2: Mixing volume
df(numTanks_1+1) = (1./residenceTime_2M).*(f(numTanks_1) - f(numTanks_1+1));

%     # Volume 2: Diffusive volume
df(numTanks_1+2) = (1./residenceTime_2D).*(f(numTanks_1) - f(numTanks_1+2));
end