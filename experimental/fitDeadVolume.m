function DVparamsFit = fitDeadVolume(fileNames,modelType)
rng default % For reproducibility
rng(1,'twister') % for reproducibility
if maxNumCompThreads < 5
    UseParallel = false;
else
    UseParallel = true;
end

% Lower bounds, and upper bounds for parameters
if modelType == 1
    lb = [1e-5,1e-5,1e-5,1,1e-5];
    ub = [10,2,3,30,0.1];
else
    lb = [1e-5,1e-5,1e-5,1,1e-5,0];
    ub = [10,2,3,30,0.1,10];
end

if UseParallel
    popSize = 500;
else
    popSize = 30;
end
initPop = lhsdesign(popSize,length(lb)).*(ub-lb)+lb;

optfunc = @(par) generateMLEfunDV(fileNames,par);
if UseParallel
    options =  optimoptions(@ga,'Display','iter','PopulationSize',popSize,'MaxGenerations',100,'UseParallel',UseParallel,'CrossoverFraction',0.25,'InitialPopulationMatrix',initPop);
else
    options =  optimoptions(@ga,'Display','iter','PopulationSize',popSize,'MaxGenerations',80,'PlotFcn','gaplotbestf','CrossoverFraction',0.25,'InitialPopulationMatrix',initPop);
end

% Solve the optimisation problem to obtain the DV parameters
% for the fit
[DVparamsFit, fval] = ga(optfunc,length(ub),[],[],[],[],lb,ub,[],4,options);

DVparamsFit(4) = round(DVparamsFit(4));

if ~UseParallel
    computedError = 0;
    Nt = 0;
    figure
    for ii = 1:length(fileNames)
        load([convertCharsToStrings(fileNames{ii})]);
        %         indFinal = length(experimentOutput.moleFrac)-200;
        indFinal = 300;
        tspan = experimentOutput.timeExp(1:indFinal);
        moleFracExp = experimentOutput.moleFrac(1:indFinal);
        moleFracExp(moleFracExp<0) = 0;
        if contains(fileNames{ii},'ads')
            feedMoleFrac = mean(experimentOutput.moleFrac(end));
            initMoleFrac = 0;
        else
            feedMoleFrac = 0;
            initMoleFrac = moleFracExp(1);
        end

        flowRate = mean(experimentOutput.volFlow_MFM(1:200))./60;

        try
            %             DVparamsFit = [3.663062325586897,1.785463697997650,2.876705759014518,27,0.053943629164603,7.824413092933892];
            %             DVparamsFit = [3.663062325586897,1.785463697997650,1.876705759014518,27,0.003943629164603];
            [timeElapsedDV, moleFracDV, FlowrateDV] = simulateDeadVolume(tspan, initMoleFrac, feedMoleFrac, flowRate, DVparamsFit);

            if contains(fileNames{ii},'ads')
                computedError = computedError + sum((moleFracDV.*flowRate./(feedMoleFrac.*flowRate(end))-moleFracExp.*flowRate./(feedMoleFrac.*flowRate(end))).^2);
            else
                computedError = computedError + sum((moleFracDV.*flowRate./(initMoleFrac.*flowRate(1))-moleFracExp.*flowRate./(initMoleFrac.*flowRate(1))).^2);
            end
            if contains(fileNames{ii},'ads')
                figure(99)
                hold on;
                plot(timeElapsedDV(1:end),(moleFracExp(1:end)-initMoleFrac).*flowRate./((feedMoleFrac-initMoleFrac).*flowRate(end)),'b',LineWidth=0.5,LineStyle='none',Marker='o',MarkerSize=7)
                plot(timeElapsedDV,(moleFracDV-initMoleFrac).*flowRate./((feedMoleFrac-initMoleFrac).*flowRate(end)),'k','LineWidth',2);
            else
                figure(99)
                hold on;
                plot(timeElapsedDV,(moleFracExp-feedMoleFrac).*flowRate./((initMoleFrac-feedMoleFrac).*flowRate(1)),'b',LineWidth=0.5,LineStyle='--',Marker='o',MarkerSize=7)
                plot(timeElapsedDV,(moleFracDV-feedMoleFrac).*flowRate./((initMoleFrac-feedMoleFrac).*flowRate(1)),'k','LineWidth',2);
            end

        catch
            computedError = 10000000;
        end
        Nt = Nt + length(tspan);
    end

    figure(99)
    hold on
    %     ylim([0 1.02])
%     xlim([0 150])
    xlabel('{\it{t}} [s]'); ylabel('{\it{Q(t)y(t)/(Q_{ref}y_{ref})}} [-]');
    set(gca,'FontSize',12)
    box on;grid off;set(gca,'YScale','linear','XScale','linear','LineWidth',2)
    hold on
    optfunc = Nt./2.*sqrt(computedError);
end

DVparamsFit
end