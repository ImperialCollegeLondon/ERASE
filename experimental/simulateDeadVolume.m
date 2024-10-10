function [timeElapsedDV, moleFracDV, FlowrateDV] = simulateDeadVolume(tspan, initMoleFrac, feedMoleFrac, flowRate, DVParams)

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

initialConditions = ones(1,numTanksTotal)*initMoleFrac;

options = odeset('RelTol',1e-6,'InitialStep',1e-4,'MaxStep',0.1);
[t,y] = ode15s(@(t,f) solveTanksInSeries(t, f, initMoleFrac, feedMoleFrac, flowRate, DVParams),tspan,initialConditions,options);

timeElapsedDV = t;

% # Inlet concentration
moleFracIn = ones(length(timeElapsedDV),1)*feedMoleFrac;
moleFracMix = y(:,numTanks_1+1);
moleFracDiff = y(:,numTanks_1+2);

flowRate_M = flowRate - flowRate_2D;
moleFracDV = (flowRate_M.*moleFracMix+flowRate_2D.*moleFracDiff)./flowRate;
FlowrateDV = ones(length(timeElapsedDV),1)*flowRate;
end

