############################################################################
#
# Imperial College London, United Kingdom
# Multifunctional Nanomaterials Laboratory
#
# Project:  ERASE
# Year:     2020
# Python:   Python 3.7
# Authors:  Ashwin Kumar Rajagopalan (AK)
#
# Purpose:
# Plots for the experiment manuscript
#
# Last modified:
# - 2022-05-10, AK: Minor fix for time-resolved plots
# - 2022-04-11, AK: Minor fix for plots (RPv1)
# - 2022-02-22, AK: Minor fix for plots
# - 2021-11-16, AK: Add Ft plot and minor fixes
# - 2021-10-27, AK: Add plots for sensitivity analysis
# - 2021-10-15, AK: Add plots for SI
# - 2021-10-08, AK: Add plots for experimental fits
# - 2021-10-06, AK: Add plots for experimental and computational data (iso)
# - 2021-10-04, AK: Add N2/MIP plots
# - 2021-10-01, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def plotsForArticle_Experiment(**kwargs):
    import auxiliaryFunctions
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Flag for saving figure
    if 'saveFlag' in kwargs:
        if kwargs["saveFlag"]:
            saveFlag = kwargs["saveFlag"]
    else:
        saveFlag = False

    # Save file extension (png or pdf)
    if 'saveFileExtension' in kwargs:
        if kwargs["saveFileExtension"]:
            saveFileExtension = kwargs["saveFileExtension"]
    else:
        saveFileExtension = ".png"

    # If material characterization plot needs to be plotted
    if 'figureMat' in kwargs:
        if kwargs["figureMat"]:
            plotForArticle_figureMat(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)
            
    # If dead volume plot needs to be plotted
    if 'figureDV' in kwargs:
        if kwargs["figureDV"]:
            plotForArticle_figureDV(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)
            
    # If ZLC plot needs to be plotted
    if 'figureZLC' in kwargs:
        if kwargs["figureZLC"]:
            plotForArticle_figureZLC(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)
            
    # If ZLC and QC plot needs to be plotted
    if 'figureComp' in kwargs:
        if kwargs["figureComp"]:
            plotForArticle_figureComp(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)
            
    # If ZLC simulation plot needs to be plotted
    if 'figureZLCSim' in kwargs:
        if kwargs["figureZLCSim"]:              
            plotForArticle_figureZLCSim(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)

    # If ZLC fits needs to be plotted
    if 'figureZLCFit' in kwargs:
        if kwargs["figureZLCFit"]:
            plotForArticle_figureZLCFit(gitCommitID, currentDT,
                                       saveFlag, saveFileExtension)

    # If ZLC fits needs to be plotted
    if 'figureZLCFitALL' in kwargs:
        if kwargs["figureZLCFitALL"]:
            plotForArticle_figureZLCFitALL(gitCommitID, currentDT,
                                       saveFlag, saveFileExtension)

    # If ZLC simulation fits needs to be plotted
    if 'figureZLCSimFit' in kwargs:
        if kwargs["figureZLCSimFit"]:
            plotForArticle_figureZLCSimFit(gitCommitID, currentDT,
                                       saveFlag, saveFileExtension)

    # If ZLC fits needs to be plotted
    if 'figureZLCSimFitALL' in kwargs:
        if kwargs["figureZLCSimFitALL"]:
            plotForArticle_figureZLCSimFitALL(gitCommitID, currentDT,
                                       saveFlag, saveFileExtension)

    # If ZLC experimental repeats needs to be plotted
    if 'figureZLCRep' in kwargs:
        if kwargs["figureZLCRep"]:
            plotForArticle_figureZLCRep(gitCommitID, currentDT,
                                       saveFlag, saveFileExtension)

    # If ZLC objective functions
    if 'figureZLCObj' in kwargs:
        if kwargs["figureZLCObj"]:
            plotForArticle_figureZLCObj(gitCommitID, currentDT,
                                       saveFlag, saveFileExtension)
            
    # If raw textural characterization
    if 'figureRawTex' in kwargs:
        if kwargs["figureRawTex"]:
            plotForArticle_figureRawTex(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension)
            
    # If MS calibration comparison
    if 'figureMSCal' in kwargs:
        if kwargs["figureMSCal"]:
            plotForArticle_figureMSCal(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension)

    # If sensitivity plots 
    if 'figureSensitivity' in kwargs:
        if kwargs["figureSensitivity"]:
            plotForArticle_figureSensitivity(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension)

    # If DV sensitivity plots 
    if 'figureDVSensitivity' in kwargs:
        if kwargs["figureDVSensitivity"]:
            plotForArticle_figureDVSensitivity(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension)

    # If Ft plots 
    if 'figureFt' in kwargs:
        if kwargs["figureFt"]:
            plotForArticle_figureFt(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension)
            
    # If BN equilibrium and kinetics 
    if 'figureBNeqkin' in kwargs:
        if kwargs["figureBNeqkin"]:
            plotForArticle_figureBNeqkin(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension)
    
    # If ZLC fits needs to be plotted for BN materials
    if 'figureZLCFitALLBN' in kwargs:
        if kwargs["figureZLCFitALLBN"]:
            plotForArticle_figureZLCFitALLBN(gitCommitID, currentDT,
                                       saveFlag, saveFileExtension)
            
    # If ZLC fits needs to be plotted for Lewatit
    if 'figureZLCFitALLLewatit' in kwargs:
        if kwargs["figureZLCFitALLLewatit"]:
            plotForArticle_figureZLCFitALLLewatit(gitCommitID, currentDT,
                                       saveFlag, saveFileExtension)
            
    # If BN equilibrium with MB 
    if 'figureBNeqMB' in kwargs:
        if kwargs["figureBNeqMB"]:
            plotForArticle_figureBNeqMB(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension)     
            
    # If ZIF8 equilibrium and kinetics 
    if 'figureZIF8eqkin' in kwargs:
        if kwargs["figureZIF8eqkin"]:
            plotForArticle_figureZIF8eqkin(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension)
    
    # If ZLC fits needs to be plotted for BN materials
    if 'figureZLCFitALLZIF8' in kwargs:
        if kwargs["figureZLCFitALLZIF8"]:
            plotForArticle_figureZLCFitALLZIF8(gitCommitID, currentDT,
                                        saveFlag, saveFileExtension)
# fun: plotForArticle_figureMat
# Plots the Figure DV of the manuscript: Material characterization (N2/MIP and QC)
def plotForArticle_figureMat(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    import scipy.io as sio
    import os
    from matplotlib.ticker import FormatStrFormatter
    from computeEquilibriumLoading import computeEquilibriumLoading

    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()

    # Plot colors and markers (porosity)
    colorsForPlot_P = ["0fa3b1","f17300"]
    markersForPlot_P = ["^","v"]
    
    # Plot colors and markers (isotherm)    
    colorsForPlot_I = ["ffba08","d00000","03071e"]
    markersForPlot_I = ["^","d","v"]
    
    # Length of arrow 
    dx1 = [-13,-24,-30] # (for N2 porosity)
    dx2 = [27,60,60] # (for MIP porosity)
    
    # Head length of arrow
    hl1 = [2.5,5,6,] # (for N2 porosity)
    hl2 = [10,20,20] # (for N2 porosity)
    
    # Interval for plots
    numIntPorosity = 4

    # Main folder for material characterization
    mainDir = os.path.join('..','experimental','materialCharacterization')

    # Porosity folder
    porosityDir = os.path.join('porosityData','porosityResults')
    
    # File with pore characterization data
    porosityALL = ['AC_20nm_interp.mat',
                   'BNp_39nm_poreVolume_interp.mat',
                   '13X_H_50nm_poreVolume.mat',]

    # Isotherm folder
    isothermDir = os.path.join('isothermData','isothermResults')
    
    # File with pore characterization data
    isothermALL = ['AC_S1_DSL_100621.mat',
                   'BNp_SSL_100621.mat',
                   'Z13X_H_DSL_100621.mat',]

    # Loop over all the porosity files
    for kk in range(len(porosityALL)):
        # Path of the file name
        fileToLoad = os.path.join(mainDir,porosityDir,porosityALL[kk])

        # Get the porosity options
        porosityOptions = sio.loadmat(fileToLoad)["poreVolume"]["options"][0][0]

        # Indicies for N2 and MIP data
        QCindexLast = porosityOptions["QCindexLast"][0][0][0][0] # Last index for N2 sorption
        
        # Get the porosity options
        combinedPorosityData = sio.loadmat(fileToLoad)["poreVolume"]["combined"][0][0]
    
        # Create the instance for the plots
        ax = plt.subplot(2,3,kk+1)
        
        # Plot horizontal line for total pore volume
        ax.axhline(combinedPorosityData[-2,2],
                   linestyle = ':', linewidth = 0.75, color = '#7d8597')  

        # Plot vertical line to distinguish N2 and MIP
        ax.axvline(combinedPorosityData[QCindexLast-1,0],
                   linestyle = ':', linewidth = 0.75, color = '#7d8597')
        
        # Set background color for micropore region
        ax.axvspan(0,2, facecolor='#EEE0CB', alpha=0.3)
        ax.text(0.13, 1.5, "micro", fontsize=8, color = 'k')
        # Set background color for mesopore region
        ax.axvspan(2, 50, facecolor='#BAA898', alpha=0.3)
        ax.text(2.8, 1.5, "meso", fontsize=8, color = 'k')
        # Set background color for macropore region
        ax.axvspan(50, 1e6, facecolor='#848586', alpha=0.3)
        ax.text(2e3, 1.5, "macro", fontsize=8, color = 'k')
        
        # Plot N2 sorption
        ax.semilogx(combinedPorosityData[0:QCindexLast-1:numIntPorosity,0],
                     combinedPorosityData[0:QCindexLast-1:numIntPorosity,2],
                     linewidth = 0.25,linestyle = ':',
                     marker = markersForPlot_P[0],
                     color='#'+colorsForPlot_P[0],)
        # Plot MIP
        ax.semilogx(combinedPorosityData[QCindexLast:-1:numIntPorosity,0],
                     combinedPorosityData[QCindexLast:-1:numIntPorosity,2],
                     linewidth = 0.25,linestyle = ':',
                     marker = markersForPlot_P[1],
                     color='#'+colorsForPlot_P[1],)

        # N2 sorption measurements
        ax.arrow(combinedPorosityData[QCindexLast-1,0], 0.3, dx1[kk], 0,
                 length_includes_head = True, head_length = hl1[kk], head_width = 0.04,
                 color = '#'+colorsForPlot_P[0])
        ax.text(combinedPorosityData[QCindexLast-1,0]+1.2*dx1[kk], 0.15, "N$_2$", fontsize=8,
                 color = '#'+colorsForPlot_P[0])
        
        # MIP measurements        
        ax.arrow(combinedPorosityData[QCindexLast-1,0], 0.7, dx2[kk], 0,
                 length_includes_head = True, head_length = hl2[kk], head_width = 0.04,
                 color = '#'+colorsForPlot_P[1])
        ax.text(combinedPorosityData[QCindexLast-1,0]+0.25*dx2[kk], 0.77, "Hg", fontsize=8,
                 color = '#'+colorsForPlot_P[1])
        
        # Material specific text labels
        if kk == 0:
            ax.set(xlabel='$D$ [nm]', 
                    ylabel='$V_\mathregular{pore}$ [cm$^{3}$ g$^{-1}$]',
                    xlim = [0.1,1e6], ylim = [0, 2])
            ax.text(0.2, 1.82, "(a)", fontsize=8,)
            # ax.text(1.4e5, 0.1, "AC", fontsize=8, fontweight = 'bold',color = '#e71d36')
            ax.text(1.6e3, combinedPorosityData[-2,2]+0.07, 
                    str(round(combinedPorosityData[-2,2],2))+' cm$^{3}$ g$^{-1}$',
                    fontsize=8,color = '#7d8597')
            ax.text(2.5e2, 2.15, "AC", fontsize=8, fontweight = 'bold',color = 'k')
        elif kk == 1:
            ax.set(xlabel='$D$ [nm]', 
                    xlim = [0.1,1e6], ylim = [0, 2])
            ax.text(0.2, 1.82, "(b)", fontsize=8,)
            # ax.text(1.4e5, 0.1, "BN", fontsize=8, fontweight = 'bold',color = '#e71d36')
            ax.text(5, combinedPorosityData[-2,2]-0.15, 
                    str(round(combinedPorosityData[-2,2],2))+' cm$^{3}$ g$^{-1}$',
                    fontsize=8,color = '#7d8597')
            ax.text(2.5e2, 2.15, "BN", fontsize=8, fontweight = 'bold',color = 'k')
        elif kk == 2:
            ax.set(xlabel='$D$ [nm]', 
                    xlim = [0.1,1e6], ylim = [0, 2])
            ax.text(0.2, 1.82, "(c)", fontsize=8,)
            # ax.text(1e5, 0.1, "13X", fontsize=8, fontweight = 'bold',color = '#e71d36')
            ax.text(1.6e3, combinedPorosityData[-2,2]+0.07, 
                    str(round(combinedPorosityData[-2,2],2))+' cm$^{3}$ g$^{-1}$',
                    fontsize=8,color = '#7d8597')
            ax.text(2.5e2, 2.15, "13X", fontsize=8, fontweight = 'bold',color = 'k')            
        ax.locator_params(axis="y", nbins=5)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
    # Loop over all the isotherm files
    for kk in range(len(isothermALL)):
        # Create the instance for the plots
        ax = plt.subplot(2,3,kk+4)
        
        # Path of the file name
        fileToLoad = os.path.join(mainDir,isothermDir,isothermALL[kk])

        # Get the experimental points
        experimentALL = sio.loadmat(fileToLoad)["isothermData"]["experiment"][0][0]

        # Get the isotherm fits
        isothermFitALL = sio.loadmat(fileToLoad)["isothermData"]["isothermFit"][0][0]      

        # Find temperatures
        temperature = np.unique(experimentALL[:,2])

        # Find indices corresponding to each temperature
        for ll in range(len(temperature)):
            indexFirst = int(np.argwhere(experimentALL[:,2]==temperature[ll])[0])
            indexLast = int(np.argwhere(experimentALL[:,2]==temperature[ll])[-1])
        
            # Plot experimental isotherm
            ax.plot(experimentALL[indexFirst:indexLast,0],
                    experimentALL[indexFirst:indexLast,1],
                    linewidth = 0, marker = markersForPlot_I[ll],
                    color='#'+colorsForPlot_I[ll],
                    label = str(temperature[ll])) 
            # Removed isotherm fit from MATLAB code (bug)
            # ax.plot(isothermFitALL[1:-1,0],isothermFitALL[1:-1,ll+1],
            #         linewidth = 1,color='#'+colorsForPlot_I[ll],alpha=0.5)
            ax.legend(loc='best', handletextpad=0.0)
                  
            
            # Obtain the confidence bounds for the QC data
            # Load isotherm parameters from QC data
            isothermParameters = sio.loadmat(fileToLoad)["isothermData"]["isothermParameters"][0][0]
    
            # Create the grid for mole fractions
            y = np.linspace(0,1.,100)
    
            # Prepare x_VOL
            x_VOL = list(isothermParameters[0:-1:2,0]) + list(isothermParameters[1::2,0])
            x_VOL_CI = list(isothermParameters[0:-1:2,1]) + list(isothermParameters[1::2,1])
 
            # Initialize volumetric loading
            isoLoading_VOL = np.zeros([len(y),len(temperature)])   
 
            # Loop through all the temperature and mole fraction
            for jj in range(len(temperature)):
                for ii in range(len(y)):
                    isoLoading_VOL[ii,jj] = computeEquilibriumLoading(isothermModel=x_VOL,
                                                                      moleFrac = y[ii],
                                                                      temperature = temperature[jj])
 
            # Get the confidence bounds
            isoLoading_VOL_LowerBound, isoLoading_VOL_UpperBound = computeConfidenceBounds(x_VOL, x_VOL_CI, temperature, y)


            # Plot fitted isotherm and confidence bounds
            for jj in range(len(temperature)):
                ax.plot(y,isoLoading_VOL[:,jj],color='#'+colorsForPlot_I[jj],alpha=1.,linestyle=':') # QC
                ax.fill_between(y, isoLoading_VOL_LowerBound[:,jj], isoLoading_VOL_UpperBound[:,jj],
                      color='#'+colorsForPlot_I[jj],alpha = 0.1,linewidth=0.) # Lowest J
            
        # Material specific text labels
        if kk == 0:
            ax.set(xlabel='$P$ [bar]', 
                    ylabel='$q^*_\mathregular{CO_2}$ [mol kg$^{-1}$]',
                    xlim = [0,1], ylim = [0, 3])
            ax.text(0.89, 2.75, "(d)", fontsize=8,)
            # ax.text(0.87, 0.13, "AC", fontsize=8, fontweight = 'bold',color = '#4895EF')

        elif kk == 1:
            ax.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 2])
            ax.text(0.89, 1.82, "(e)", fontsize=8,)
            # ax.text(0.87, 0.09, "BN", fontsize=8, fontweight = 'bold',color = '#4895EF')

        elif kk == 2:
            ax.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 8])
            ax.text(0.89, 7.25, "(f)", fontsize=8,)
            # ax.text(0.85, 0.35, "13X", fontsize=8, fontweight = 'bold',color = '#4895EF')

        ax.locator_params(axis="x", nbins=4)            
        ax.locator_params(axis="y", nbins=4)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
 
    #  Save the figure
    if saveFlag:
        # FileName: figureMat_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureMat_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
 
    plt.show()

# fun: plotForArticle_figureDV
# Plots the Figure DV of the manuscript: Dead volume characterization
def plotForArticle_figureDV(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    from deadVolumeWrapper import deadVolumeWrapper
    from extractDeadVolume import filesToProcess # File processing script
    from numpy import load
    import os
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # File with parameter estimates
    fileParameterALL = ['deadVolumeCharacteristics_20210810_1323_eddec53.npz', # MS
                        'deadVolumeCharacteristics_20210810_1653_eddec53.npz', # With ball
                        'deadVolumeCharacteristics_20210817_2330_ea32ed7.npz',] # Without ball

    # Flag to plot simulations
    simulateModel = True

    # Plot colors and markers
    colorsForPlot = ["03045e","0077b6","00b4d8","90e0ef"]
    markersForPlot = ["^",">","v","<"]

    for kk in range(len(fileParameterALL)):
        fileParameter = fileParameterALL[kk] # Parse out the parameter estimate name
        # Dead volume parameter model path
        parameterPath = os.path.join('..','simulationResults',fileParameter)
        # Load file names and the model
        fileNameList = load(parameterPath, allow_pickle=True)["fileName"]

        # Generate .npz file for python processing of the .mat file 
        filesToProcess(True,os.path.join('..','experimental','runData'),fileNameList,'DV')
        # Get the processed file names
        fileName = filesToProcess(False,[],[],'DV')
        # Load the model
        modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
        x = modelOutputTemp[()]["variable"]
        print(x)

        # Get the MS fit flag, flow rates and msDeadVolumeFile (if needed)
        # Check needs to be done to see if MS file available or not
        # Checked using flagMSDeadVolume in the saved file
        dvFileLoadTemp = load(parameterPath)
        if 'flagMSDeadVolume' in dvFileLoadTemp.files:
            flagMSFit = dvFileLoadTemp["flagMSFit"]
            msFlowRate = dvFileLoadTemp["msFlowRate"]
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
        else:
            flagMSFit = False
            msFlowRate = -np.inf
            flagMSDeadVolume = False
            msDeadVolumeFile = []
        
        numPointsExp = np.zeros(len(fileName))
        for ii in range(len(fileName)): 
            fileToLoad = fileName[ii]
            # Load experimental molefraction
            timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
            numPointsExp[ii] = len(timeElapsedExp)
        
        # Downsample intervals
        downsampleInt = numPointsExp/np.min(numPointsExp)
    
        # Print the objective function and volume from model parameters
        print("Model Volume",round(sum(x[0:2]),2))
        moleFracExpALL = np.array([])
        moleFracSimALL = np.array([])
    
        # Create the instance for the plots
        ax1 = plt.subplot(1,3,1)
        ax2 = plt.subplot(1,3,2)
        ax3 = plt.subplot(1,3,3)
        
        # Initialize error for objective function
        # Loop over all available files    
        for ii in range(len(fileName)):
            # Initialize outputs
            moleFracSim = []
            # Path of the file name
            fileToLoad = fileName[ii]   
            # Load experimental time, molefraction and flowrate (accounting for downsampling)
            timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
            moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
            flowRateTemp = load(fileToLoad)["flowRate"].flatten()
            timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
            moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
            flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
            # Get the flow rates from the fit file
            # When MS used
            if flagMSFit:
                flowRateDV = msFlowRate
            else:
                flowRateDV = np.mean(flowRateExp[-1:-10:-1])
            
            # Integration and ode evaluation time
            timeInt = timeElapsedExp
            
            if simulateModel:    
                # Call the deadVolume Wrapper function to obtain the outlet mole fraction
                moleFracSim = deadVolumeWrapper(timeInt, flowRateDV, x, flagMSDeadVolume, msDeadVolumeFile)
                       
                # Stack mole fraction from experiments and simulation for error 
                # computation
                minExp = np.min(moleFracExp) # Compute the minimum from experiment
                normalizeFactor = np.max(moleFracExp - minExp) # Compute the max from normalized data
                moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
                moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))
                
            # Plot the expreimental and model output
            # Log scale
            if kk == 0:
                ax1.semilogy(timeElapsedExp,moleFracExp,
                              marker = markersForPlot[ii],linewidth = 0,
                              color='#'+colorsForPlot[ii],alpha=0.25,label=str(round(abs(np.mean(flowRateExp)),1))+" cm$^3$ s$^{-1}$") # Experimental response
                ax1.semilogy(timeElapsedExp,moleFracSim,
                                  color='#'+colorsForPlot[ii]) # Simulation response
                ax1.set(xlabel='$t$ [s]', 
                        ylabel='$y\mathregular{_{CO_2}}$ [-]',
                        xlim = [0,15], ylim = [1e-2, 1])
                ax1.locator_params(axis="x", nbins=5)
                ax1.legend(handletextpad=0.0,loc='center right')
                ax1.text(7, 1.3, "(a)", fontsize=8,)
                ax1.text(8.9, 0.64, "Segment II", fontsize=8, fontweight = 'bold',
                        backgroundcolor = 'w', color = '#e71d36')
                ax1.text(7.2, 0.385, "$V^\mathrm{S_{II}}$ = 0.02 cm$^3$", fontsize=8, 
                        backgroundcolor = 'w', color = '#7d8597')
                ax1.grid(which='minor', linestyle=':')
            elif kk == 1:
                ax2.semilogy(timeElapsedExp,moleFracExp,
                              marker = markersForPlot[ii],linewidth = 0,
                              color='#'+colorsForPlot[ii],alpha=0.25,label=str(round(abs(np.mean(flowRateExp)),1))+" cm$^3$ s$^{-1}$") # Experimental response
                ax2.semilogy(timeElapsedExp,moleFracSim,
                                  color='#'+colorsForPlot[ii]) # Simulation response
                ax2.set(xlabel='$t$ [s]', 
                        xlim = [0,150], ylim = [1e-2, 1])   
                ax2.locator_params(axis="x", nbins=5)
                ax2.legend(handletextpad=0.0,loc='center right')
                ax2.text(70, 1.3, "(b)", fontsize=8,)
                ax2.text(57, 0.64, "Segment I w/ Ball", fontsize=8,  fontweight = 'bold',
                        backgroundcolor = 'w', color = '#e71d36')
                ax2.text(75, 0.385, "$V^\mathrm{S_{I}}$ = 3.76 cm$^3$", fontsize=8, 
                        backgroundcolor = 'w', color = '#7d8597')
                ax2.grid(which='minor', linestyle=':')
            elif kk == 2:
                ax3.semilogy(timeElapsedExp,moleFracExp,
                              marker = markersForPlot[ii],linewidth = 0,
                              color='#'+colorsForPlot[ii],alpha=0.25,label=str(round(abs(np.mean(flowRateExp)),1))+" cm$^3$ s$^{-1}$") # Experimental response
                ax3.semilogy(timeElapsedExp,moleFracSim,
                                  color='#'+colorsForPlot[ii]) # Simulation response
                ax3.set(xlabel='$t$ [s]', 
                        xlim = [0,150], ylim = [1e-2, 1])   
                ax3.locator_params(axis="x", nbins=5)
                ax3.legend(handletextpad=0.0,loc='center right')
                ax3.text(70, 1.3, "(c)", fontsize=8,)
                ax3.text(51, 0.64, "Segment I w/o Ball", fontsize=8,  fontweight = 'bold',
                        backgroundcolor = 'w', color = '#e71d36')
                ax3.text(75, 0.385, "$V^\mathrm{S_{I}}$ = 3.93 cm$^3$", fontsize=8, 
                        backgroundcolor = 'w', color = '#7d8597')
                ax3.grid(which='minor', linestyle=':')

        # Remove all the .npz files genereated from the .mat
        # Loop over all available files    
        for ii in range(len(fileName)):
            os.remove(fileName[ii])

    #  Save the figure
    if saveFlag:
        # FileName: figureDV_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureDV_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
 
    plt.show()
        
# fun: plotForArticle_figureZLC
# Plots the Figure ZLC of the manuscript: ZLC parameter estimates
def plotForArticle_figureZLC(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    from numpy import load
    import os
    from computeEquilibriumLoading import computeEquilibriumLoading
    from matplotlib.ticker import FormatStrFormatter
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Plot colors and markers (isotherm)    
    colorsForPlot = ["ffba08","d00000","03071e"]
    
    # Universal gas constant
    Rg = 8.314
    
    # Total pressure
    pressureTotal = np.array([1.e5]);
    
    # Define temperature
    temperature = [308.15, 328.15, 348.15]
    
    # Parameter estimate files
                        # Activated Carbon Experiments
    zlcFileNameALL = [['zlcParameters_20210822_0926_c8173b1.npz',
                       'zlcParameters_20210822_1733_c8173b1.npz',
                       # 'zlcParameters_20210823_0133_c8173b1.npz', # DSL BAD (but lowest J)
                       # 'zlcParameters_20210823_1007_c8173b1.npz', # DSL BAD (but lowest J)
                       'zlcParameters_20210823_1810_c8173b1.npz'],
                       # Boron Nitride Experiments
                      ['zlcParameters_20210823_1731_c8173b1.npz',
                       'zlcParameters_20210824_0034_c8173b1.npz',
                       'zlcParameters_20210824_0805_c8173b1.npz',
                       'zlcParameters_20210824_1522_c8173b1.npz',
                       'zlcParameters_20210824_2238_c8173b1.npz',],
                       # Zeolite 13X Experiments
                      ['zlcParameters_20210824_1552_6b88505.npz',
                       'zlcParameters_20210825_0559_6b88505.npz',
                       'zlcParameters_20210825_1854_6b88505.npz',
                       'zlcParameters_20210826_0847_6b88505.npz',
                       'zlcParameters_20210827_0124_6b88505.npz',]]
    
    # Create the grid for mole fractions
    y = np.linspace(0,1.,100)
    
    for pp in range(len(zlcFileNameALL)):
        zlcFileName = zlcFileNameALL[pp]

        # Initialize isotherms 
        isoLoading_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
        kineticConstant_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
        objectiveFunction = np.zeros([len(zlcFileName)])

        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # ZLC Data 
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            parameterReference = load(parameterPath)["parameterReference"]
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
            modelNonDim = modelOutputTemp[()]["variable"] 
            # Multiply the paremeters by the reference values
            x_ZLC = np.multiply(modelNonDim,parameterReference)    
            adsorbentDensity = load(parameterPath, allow_pickle=True)["adsorbentDensity"]
    
            # Parse out the isotherm parameter
            isothermModel = x_ZLC[0:-2]
            rateConstant_1 = x_ZLC[-2]
            rateConstant_2 = x_ZLC[-1]
        
            for jj in range(len(temperature)):
                for ii in range(len(y)):
                    isoLoading_ZLC[kk,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                         moleFrac = y[ii], 
                                                                         temperature = temperature[jj]) # [mol/kg]
                    # Partial pressure of the gas
                    partialPressure = y[ii]*pressureTotal
                    # delta pressure to compute gradient
                    delP = 1e-3
                    # Mole fraction (up)
                    moleFractionUp = (partialPressure + delP)/pressureTotal
                    # Compute the loading [mol/m3] @ moleFractionUp
                    equilibriumLoadingUp  = computeEquilibriumLoading(temperature=temperature[jj],
                                                                    moleFrac=moleFractionUp,
                                                                    isothermModel=isothermModel) # [mol/kg]
                    
                    # Compute the gradient (delq*/dc)
                    dqbydc = (equilibriumLoadingUp-isoLoading_ZLC[kk,ii,jj])*adsorbentDensity/(delP/(Rg*temperature[jj])) # [-]
        
                    # Rate constant 1 (analogous to micropore resistance)
                    k1 = rateConstant_1
        
                    # Rate constant 2 (analogous to macropore resistance)
                    k2 = rateConstant_2/dqbydc
                                
                    # Overall rate constant
                    # The following conditions are done for purely numerical reasons
                    # If pure (analogous) macropore
                    if k1<1e-12:
                        rateConstant = k2
                    # If pure (analogous) micropore
                    elif k2<1e-12:
                        rateConstant = k1
                    # If both resistances are present
                    else:
                        rateConstant = 1/(1/k1 + 1/k2)
                    
                    # Rate constant (overall)
                    kineticConstant_ZLC[kk,ii,jj] = rateConstant
    
        # Plot the isotherms    
        ax1 = plt.subplot(2,3,pp+1)
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))

        for jj in range(len(temperature)):
            for qq in range(len(zlcFileName)):
                if qq == minJ[0]:
                    ax1.plot(y,isoLoading_ZLC[qq,:,jj],color='#'+colorsForPlot[jj],label=str(temperature[jj])+' K')  # Lowest J
                else:
                    ax1.plot(y,isoLoading_ZLC[qq,:,jj],color='#'+colorsForPlot[jj],alpha=0.2) # ALL

        # Plot the kinetic constants    
        ax2 = plt.subplot(2,3,pp+4)
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))

        for jj in range(len(temperature)):
            for qq in range(len(zlcFileName)):
                if qq == minJ[0]:
                    ax2.plot(y,kineticConstant_ZLC[qq,:,jj],color='#'+colorsForPlot[jj],label=str(temperature[jj])+' K') # Lowest J
                else:
                    ax2.plot(y,kineticConstant_ZLC[qq,:,jj],color='#'+colorsForPlot[jj],alpha=0.2) # ALL
        
        if pp == 0:
            # Isotherm
            ax1.set(ylabel='$q^*_\mathregular{CO_2}$ [mol kg$^{-1}$]',
                    xlim = [0,1], ylim = [0, 3])
            ax1.text(0.04, 2.75, "(a)", fontsize=8,)
            ax1.text(0.45, 3.2, "AC", fontsize=8, fontweight = 'bold',color = 'k')
            ax1.text(0.84, 0.30, "OPT", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax1.text(0.84, 0.12, "REP", fontsize=8, fontweight = 'bold',color = '#4895EF', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.axes.xaxis.set_ticklabels([])
            ax1.legend()
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    ylabel='$k\mathregular{_{CO_2}}$ [s$^{-1}$]',
                    xlim = [0,1], ylim = [0, 1])
            ax2.text(0.04, 0.9, "(d)", fontsize=8,)
            # ax2.text(0.87, 0.9, "AC", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 0.83, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=4)
            ax2.locator_params(axis="y", nbins=4)
        elif pp  == 1:
            # Isotherm
            ax1.set(xlim = [0,1], ylim = [0, 1.5])
            ax1.text(0.04, 1.35, "(b)", fontsize=8,)
            ax1.text(0.45, 1.6, "BN", fontsize=8, fontweight = 'bold',color = 'k')
            ax1.text(0.84, 0.15, "OPT", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax1.text(0.84, 0.06, "REP", fontsize=8, fontweight = 'bold',color = '#4895EF', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.axes.xaxis.set_ticklabels([])
            ax1.legend()
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 1])
            ax2.text(0.04, 0.9, "(e)", fontsize=8,)
            # ax2.text(0.87, 0.9, "BN", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 0.83, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=4)
            ax2.locator_params(axis="y", nbins=4)
        elif pp  == 2:
            # Isotherm
            ax1.set(xlim = [0,1], ylim = [0, 8])
            ax1.text(0.04, 7.3, "(c)", fontsize=8,)
            ax1.text(0.44, 8.5, "13X", fontsize=8, fontweight = 'bold',color = 'k')
            ax1.text(0.84, 0.86, "OPT", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax1.text(0.84, 0.32, "REP", fontsize=8, fontweight = 'bold',color = '#4895EF', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.axes.xaxis.set_ticklabels([])
            ax1.legend(loc='upper right')
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 2])
            ax2.text(0.04, 1.8, "(f)", fontsize=8,)
            # ax2.text(0.84, 1.8, "13X", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 1.66, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=4)
            ax2.locator_params(axis="y", nbins=4)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #  Save the figure
    if saveFlag:
        # FileName: figureZLC_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureZLC_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
 
    plt.show()
    
# fun: plotForArticle_figureComp
# Plots the Figure Comp of the manuscript: ZLC and QC comparison
def plotForArticle_figureComp(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    from matplotlib.pyplot import figure
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    from numpy import load
    import scipy.io as sio
    import os
    from computeEquilibriumLoading import computeEquilibriumLoading
    from matplotlib.ticker import FormatStrFormatter   
    from matplotlib.lines import Line2D
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Plot colors and markers (isotherm)    
    colorsForPlot = ["ffba08","d00000","03071e"]

    # Define temperature
    temperature = [308.15, 328.15, 348.15]
    
    # QC parameter estimates
    # Main folder for material characterization
    mainDir = os.path.join('..','experimental','materialCharacterization')

    # Isotherm folder
    isothermDir = os.path.join('isothermData','isothermResults')
    
    # File with pore characterization data
    isothermALL = ['AC_S1_DSL_100621.mat',
                   'BNp_SSL_100621.mat',
                   'Z13X_H_DSL_100621.mat',]
    
    # Parameter estimate files
                        # Activated Carbon Experiments
    zlcFileNameALL = [['zlcParameters_20210822_0926_c8173b1.npz',
                       'zlcParameters_20210822_1733_c8173b1.npz',
                       # 'zlcParameters_20210823_0133_c8173b1.npz', # DSL BAD (but lowest J)
                       # 'zlcParameters_20210823_1007_c8173b1.npz', # DSL BAD (but lowest J)
                       'zlcParameters_20210823_1810_c8173b1.npz'],
                       # Boron Nitride Experiments
                      ['zlcParameters_20210823_1731_c8173b1.npz',
                       'zlcParameters_20210824_0034_c8173b1.npz',
                       'zlcParameters_20210824_0805_c8173b1.npz',
                       'zlcParameters_20210824_1522_c8173b1.npz',
                       'zlcParameters_20210824_2238_c8173b1.npz',],
                       # Zeolite 13X Experiments
                      ['zlcParameters_20210824_1552_6b88505.npz',
                       'zlcParameters_20210825_0559_6b88505.npz',
                       'zlcParameters_20210825_1854_6b88505.npz',
                       'zlcParameters_20210826_0847_6b88505.npz',
                       'zlcParameters_20210827_0124_6b88505.npz',]]

    # Dead Volume
    methodLabel = ['VOL','OPT',]

    # Custom Legend Lines
    custom_lines = [Line2D([0], [0], linestyle=':', lw=1, color = '#4895EF'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#4895EF'),]

    
    # Create the grid for mole fractions
    y = np.linspace(0,1.,100)

    # Get the figure handle 
    fig = figure()
    
    # Compute the ZLC loadings    
    for pp in range(len(zlcFileNameALL)):
        # Initialize volumetric loading
        isoLoading_VOL = np.zeros([len(y),len(temperature)])
        # Path of the file name
        fileToLoad = os.path.join(mainDir,isothermDir,isothermALL[pp])
        # Load isotherm parameters from QC data
        isothermParameters = sio.loadmat(fileToLoad)["isothermData"]["isothermParameters"][0][0]

        # Prepare x_VOL
        x_VOL = list(isothermParameters[0:-1:2,0]) + list(isothermParameters[1::2,0])
        x_VOL_CI = list(isothermParameters[0:-1:2,1]) + list(isothermParameters[1::2,1])

        # Loop through all the temperature and mole fraction
        for jj in range(len(temperature)):
            for ii in range(len(y)):
                isoLoading_VOL[ii,jj] = computeEquilibriumLoading(isothermModel=x_VOL,
                                                                  moleFrac = y[ii],
                                                                  temperature = temperature[jj])

        # Get the confidence bounds
        isoLoading_VOL_LowerBound, isoLoading_VOL_UpperBound = computeConfidenceBounds(x_VOL, x_VOL_CI, temperature, y)
        
        # Get the ZLC Isotherms
        zlcFileName = zlcFileNameALL[pp]
        # Initialize isotherms 
        isoLoading_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
        objectiveFunction = np.zeros([len(zlcFileName)])
    
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # ZLC Data 
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            parameterReference = load(parameterPath)["parameterReference"]
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
            modelNonDim = modelOutputTemp[()]["variable"] 
            # Multiply the paremeters by the reference values
            x_ZLC = np.multiply(modelNonDim,parameterReference)    
    
            # Parse out the isotherm parameter
            isothermModel = x_ZLC[0:-2]
                        
            for jj in range(len(temperature)):
                for ii in range(len(y)):
                    isoLoading_ZLC[kk,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                         moleFrac = y[ii], 
                                                                         temperature = temperature[jj]) # [mol/kg]

        # Plot the isotherms    
        ax1 = plt.subplot(1,3,pp+1)
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))

        for jj in range(len(temperature)):
            ax1.plot(y,isoLoading_VOL[:,jj],color='#'+colorsForPlot[jj],linestyle=':',alpha=0.5) # QC
            # Get the confidence bounds
            ax1.fill_between(y, isoLoading_VOL_LowerBound[:,jj], isoLoading_VOL_UpperBound[:,jj],
                             color='#'+colorsForPlot[jj],alpha = 0.25,linewidth=0.) # Lowest J         
            for qq in range(len(zlcFileName)):
                if qq == minJ[0]:
                    ax1.plot(y,isoLoading_ZLC[qq,:,jj],color='#'+colorsForPlot[jj],alpha = 1,
                             label=str(temperature[jj])+' K') # Lowest J

        if pp == 0:
            # Isotherm
            ax1.set(xlabel='$P$ [bar]', 
                    ylabel='$q^*_\mathregular{CO_2}$ [mol kg$^{-1}$]',
                    xlim = [0,1], ylim = [0, 3])
            ax1.text(0.04, 2.75, "(a)", fontsize=8,)
            ax1.text(0.15, 2.74, "AC", fontsize=8, fontweight = 'bold',color = 'k')
            # ax1.text(0.84, 0.32, "VOL", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax1.text(0.84, 0.12, "OPT", fontsize=8, fontweight = 'bold',color = '#4895EF', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.legend()
        elif pp  == 1:
            # Isotherm
            ax1.set(xlabel='$P$ [bar]', xlim = [0,1], ylim = [0, 1.5])
            ax1.text(0.04, 1.35, "(b)", fontsize=8,)
            ax1.text(0.15, 1.345, "BN", fontsize=8, fontweight = 'bold',color = 'k')
            # ax1.text(0.84, 0.16, "VOL", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax1.text(0.84, 0.06, "OPT", fontsize=8, fontweight = 'bold',color = '#4895EF', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.legend()
        elif pp  == 2:
            # Isotherm
            ax1.set(xlabel='$P$ [bar]', xlim = [0,1], ylim = [0, 8])
            ax1.text(0.04, 7.3, "(c)", fontsize=8,)
            ax1.text(0.15, 7.25, "13X", fontsize=8, fontweight = 'bold',color = 'k')
            # ax1.text(0.84, 0.86, "VOL", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax1.text(0.84, 0.32, "OPT", fontsize=8, fontweight = 'bold',color = '#4895EF', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.legend(loc='upper right')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        fig.legend(custom_lines,methodLabel,bbox_to_anchor=(0.07,0.93,0.55,0.1), 
                   ncol=2, borderaxespad=0, labelcolor = '#4895EF')   

    #  Save the figure
    if saveFlag:
        # FileName: figureComp_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureComp_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath,bbox_inches='tight')
 
    plt.show()
    
# fun: plotForArticle_figureZLCSim
# Plots the Figure ZLC Sim of the manuscript: ZLC parameter estimates (simulated)
def plotForArticle_figureZLCSim(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    from numpy import load
    import scipy.io as sio
    import os
    from computeEquilibriumLoading import computeEquilibriumLoading
    from matplotlib.ticker import FormatStrFormatter
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Plot colors and markers (isotherm)    
    colorsForPlot = ["0091ad","5c4d7d","b7094c"]

    # Universal gas constant
    Rg = 8.314
    
    # Total pressure
    pressureTotal = np.array([1.e5]);
    
    # Define temperature
    temperature = [308.15, 328.15, 348.15]
    
    # .mat files with genereated simuation data
    # Main folder for material characterization
    mainDir = os.path.join('..','experimental','runData')
    
    # File with pore characterization data
    simData = ['ZLC_ActivatedCarbon_Sim01A_Output.mat',
               'ZLC_BoronNitride_Sim01A_Output.mat',
               'ZLC_Zeolite13X_Sim01A_Output.mat',]
    
    # Parameter estimate files
                        # Activated Carbon Simulations
    zlcFileNameALL = [['zlcParameters_20210823_1104_03c82f4.npz',
                       'zlcParameters_20210824_0000_03c82f4.npz',
                       'zlcParameters_20210824_1227_03c82f4.npz',
                       'zlcParameters_20210825_0017_03c82f4.npz',
                       'zlcParameters_20210825_1151_03c82f4.npz'],
                       # Boron Nitride Simulations
                      ['zlcParameters_20210823_1907_03c82f4.npz',
                       'zlcParameters_20210824_0555_03c82f4.npz',
                       'zlcParameters_20210824_2105_03c82f4.npz',
                       'zlcParameters_20210825_0833_03c82f4.npz',
                       'zlcParameters_20210825_2214_03c82f4.npz'],
                       # Zeolite 13X Simulations
                      ['zlcParameters_20210824_1102_c8173b1.npz',
                       'zlcParameters_20210825_0243_c8173b1.npz',
                       'zlcParameters_20210825_1758_c8173b1.npz',
                       'zlcParameters_20210826_1022_c8173b1.npz',
                       'zlcParameters_20210827_0104_c8173b1.npz',]]
    
    # Create the grid for mole fractions
    y = np.linspace(0,1.,100)
    
    for pp in range(len(zlcFileNameALL)):
        # Initialize simulated loading
        isoLoading_SIM = np.zeros([len(y),len(temperature)])
        # Path of the file name
        fileToLoad = os.path.join(mainDir,simData[pp])
        # Load isotherm parameters from simulated data
        isothermParameters = sio.loadmat(fileToLoad)["modelParameters"][0]

        # Prepare x_VOL
        x_SIM = isothermParameters[0:-2]
        
        # Loop through all the temperature and mole fraction
        for jj in range(len(temperature)):
            for ii in range(len(y)):
                isoLoading_SIM[ii,jj] = computeEquilibriumLoading(isothermModel=x_SIM,
                                                                  moleFrac = y[ii],
                                                                  temperature = temperature[jj])


        # Go through the ZLC files
        zlcFileName = zlcFileNameALL[pp]

        # Initialize isotherms 
        isoLoading_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
        kineticConstant_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
        kineticConstant_SIM = np.zeros([len(y),len(temperature)]) # For simulated data
        objectiveFunction = np.zeros([len(zlcFileName)])
    
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # ZLC Data 
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            parameterReference = load(parameterPath)["parameterReference"]
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
            modelNonDim = modelOutputTemp[()]["variable"] 
            # Multiply the paremeters by the reference values
            x_ZLC = np.multiply(modelNonDim,parameterReference)    
            adsorbentDensity = load(parameterPath, allow_pickle=True)["adsorbentDensity"]
    
            # Parse out the isotherm parameter
            isothermModel = x_ZLC[0:-2]
            rateConstant_1 = x_ZLC[-2]
            rateConstant_2 = x_ZLC[-1]
        
            for jj in range(len(temperature)):
                for ii in range(len(y)):
                    isoLoading_ZLC[kk,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                         moleFrac = y[ii], 
                                                                         temperature = temperature[jj]) # [mol/kg]
                    # Partial pressure of the gas
                    partialPressure = y[ii]*pressureTotal
                    # delta pressure to compute gradient
                    delP = 1e-3
                    # Mole fraction (up)
                    moleFractionUp = (partialPressure + delP)/pressureTotal
                    # Compute the loading [mol/m3] @ moleFractionUp
                    equilibriumLoadingUp  = computeEquilibriumLoading(temperature=temperature[jj],
                                                                    moleFrac=moleFractionUp,
                                                                    isothermModel=isothermModel) # [mol/kg]
                    
                    # Compute the gradient (delq*/dc)
                    dqbydc = (equilibriumLoadingUp-isoLoading_ZLC[kk,ii,jj])*adsorbentDensity/(delP/(Rg*temperature[jj])) # [-]
        
                    # Rate constant 1 (analogous to micropore resistance)
                    k1 = rateConstant_1
        
                    # Rate constant 2 (analogous to macropore resistance)
                    k2 = rateConstant_2/dqbydc
                                
                    # Overall rate constant
                    # The following conditions are done for purely numerical reasons
                    # If pure (analogous) macropore
                    if k1<1e-12:
                        rateConstant = k2
                    # If pure (analogous) micropore
                    elif k2<1e-12:
                        rateConstant = k1
                    # If both resistances are present
                    else:
                        rateConstant = 1/(1/k1 + 1/k2)
                    
                    # Rate constant (overall)
                    kineticConstant_ZLC[kk,ii,jj] = rateConstant
                    
                    # Compute the simulated "true" kinetic constant
                    if kk == 0:
                        # Rate constant 1 (analogous to micropore resistance)
                        k1 = isothermParameters[-2]
            
                        # Rate constant 2 (analogous to macropore resistance)
                        k2 = isothermParameters[-1]/dqbydc
                                    
                        # Overall rate constant
                        # The following conditions are done for purely numerical reasons
                        # If pure (analogous) macropore
                        if k1<1e-12:
                            rateConstant = k2
                        # If pure (analogous) micropore
                        elif k2<1e-12:
                            rateConstant = k1
                        # If both resistances are present
                        else:
                            rateConstant = 1/(1/k1 + 1/k2)
                        
                        # Rate constant (overall)
                        kineticConstant_SIM[ii,jj] = rateConstant
    
        # Plot the isotherms    
        ax1 = plt.subplot(2,3,pp+1)
        for jj in range(len(temperature)):
            ax1.plot(y,isoLoading_SIM[:,jj],color='#'+colorsForPlot[jj],label=str(temperature[jj])+' K') # Simulated "True" Data
            for qq in range(len(zlcFileName)):
                    ax1.plot(y,isoLoading_ZLC[qq,:,jj],color='#'+colorsForPlot[jj],alpha=0.1) # ALL

        # Plot the kinetic constants    
        ax2 = plt.subplot(2,3,pp+4)
        for jj in range(len(temperature)):
            ax2.plot(y,kineticConstant_SIM[:,jj],color='#'+colorsForPlot[jj],label=str(temperature[jj])+' K') # Simulated "True" Data
            for qq in range(len(zlcFileName)):
                ax2.plot(y,kineticConstant_ZLC[qq,:,jj],color='#'+colorsForPlot[jj],alpha=0.1) # ALL
        
        if pp == 0:
            # Isotherm
            ax1.set(ylabel='$q^*_\mathregular{CO_2}$ [mol kg$^{-1}$]',
                    xlim = [0,1], ylim = [0, 3])
            ax1.text(0.04, 2.75, "(a)", fontsize=8,)
            ax1.text(0.45, 3.2, "AC", fontsize=8, fontweight = 'bold',color = 'k')
            ax1.text(0.79, 0.30, "TRUE", fontsize=8, fontweight = 'bold',color = 'k')
            ax1.text(0.83, 0.12, "EST.", fontsize=8, fontweight = 'bold',color = 'k', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.axes.xaxis.set_ticklabels([])
            ax1.legend()
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    ylabel='$k\mathregular{_{CO_2}}$ [s$^{-1}$]',
                    xlim = [0,1], ylim = [0, 1])
            ax2.text(0.04, 0.9, "(d)", fontsize=8,)
            # ax2.text(0.87, 0.9, "AC", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 0.83, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=4)
            ax2.locator_params(axis="y", nbins=4)
        elif pp  == 1:
            # Isotherm
            ax1.set(xlim = [0,1], ylim = [0, 1.5])
            ax1.text(0.04, 1.35, "(b)", fontsize=8,)
            ax1.text(0.45, 1.6, "BN", fontsize=8, fontweight = 'bold',color = 'k')
            ax1.text(0.79, 0.15, "TRUE", fontsize=8, fontweight = 'bold',color = 'k')
            ax1.text(0.83, 0.06, "EST.", fontsize=8, fontweight = 'bold',color = 'k', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.axes.xaxis.set_ticklabels([])
            ax1.legend()
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 1])
            ax2.text(0.04, 0.9, "(e)", fontsize=8,)
            # ax2.text(0.87, 0.9, "BN", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 0.83, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=4)
            ax2.locator_params(axis="y", nbins=4)
        elif pp  == 2:
            # Isotherm
            ax1.set(xlim = [0,1], ylim = [0, 8])
            ax1.text(0.04, 7.3, "(c)", fontsize=8,)
            ax1.text(0.44, 8.5, "13X", fontsize=8, fontweight = 'bold',color = 'k')
            ax1.text(0.79, 0.86, "TRUE", fontsize=8, fontweight = 'bold',color = 'k')
            ax1.text(0.83, 0.32, "EST.", fontsize=8, fontweight = 'bold',color = 'k', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.axes.xaxis.set_ticklabels([])
            ax1.legend(loc='upper right')
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 2])
            ax2.text(0.04, 1.8, "(f)", fontsize=8,)
            # ax2.text(0.84, 1.8, "13X", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 1.66, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=4)
            ax2.locator_params(axis="y", nbins=4)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
    #  Save the figure
    if saveFlag:
        # FileName: figureZLCSim_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureZLCSim_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
 
    plt.show()
    
# fun: plotForArticle_figureZLCFit
# Plots the Figure ZLC Fit of the manuscript: ZLC goodness of fit for experimental results
def plotForArticle_figureZLCFit(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure    
    import auxiliaryFunctions
    from numpy import load
    import os
    from simulateCombinedModel import simulateCombinedModel
    from deadVolumeWrapper import deadVolumeWrapper
    from extractDeadVolume import filesToProcess # File processing script
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
        
    # Plot colors and markers
    colorsForPlot = ["ffba08","d00000","03071e"]
    markersForPlot = ["^","d","v"]    
    
    # X limits for the different materials
    XLIM_L = [[0, 200],[0, 150],[0, 600]]
    XLIM_H = [[0, 100],[0, 60],[0, 200]]
    
    # Label positions for the different materials
    panelLabel_L = [185, 150/200*185, 600/200*185]
    panelLabel_H = [185/2, 60/100*185/2, 200/100*185/2]
    materialLabel_L = [182, 150/200*182, 600/200*180]
    materialLabel_H = [182/2, 60/100*182/2, 200/100*180/2]
    flowLabel_L = [118, 150/200*118, 600/200*118]
    flowLabel_H = [118/2, 60/100*118/2, 200/100*118/2]
    materialText = ["AC", "BN", "13X"]

    # Parameter estimate files
                        # Activated Carbon Experiments
    zlcFileNameALL = [['zlcParameters_20210822_0926_c8173b1.npz',
                       'zlcParameters_20210822_1733_c8173b1.npz',
                       # 'zlcParameters_20210823_0133_c8173b1.npz', # DSL BAD (but lowest J)
                       # 'zlcParameters_20210823_1007_c8173b1.npz', # DSL BAD (but lowest J)
                       'zlcParameters_20210823_1810_c8173b1.npz'],
                        # Boron Nitride Experiments
                        ['zlcParameters_20210823_1731_c8173b1.npz',
                          'zlcParameters_20210824_0034_c8173b1.npz',
                          'zlcParameters_20210824_0805_c8173b1.npz',
                          'zlcParameters_20210824_1522_c8173b1.npz',
                          'zlcParameters_20210824_2238_c8173b1.npz',],
                          # Zeolite 13X Experiments
                        ['zlcParameters_20210824_1552_6b88505.npz',
                          'zlcParameters_20210825_0559_6b88505.npz',
                          'zlcParameters_20210825_1854_6b88505.npz',
                          'zlcParameters_20210826_0847_6b88505.npz',
                          'zlcParameters_20210827_0124_6b88505.npz',]]
    
    for pp in range(len(zlcFileNameALL)):
        fig = figure(figsize=(6.5,5))   
        zlcFileName = zlcFileNameALL[pp]
        objectiveFunction = np.zeros([len(zlcFileName)])
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # Obtain the onjective function values
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)

        # Find the experiment with the min objective function
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))
        fileParameter = zlcFileName[int(minJ[0])]
        
        # ZLC parameter model path
        parameterPath = os.path.join('..','simulationResults',fileParameter)
           
        # Parse out experiments names and temperature used for the fitting
        rawFileName = load(parameterPath)["fileName"]
        temperatureExp = load(parameterPath)["temperature"]

        # Generate .npz file for python processing of the .mat file 
        filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'ZLC')
        # Get the processed file names
        fileName = filesToProcess(False,[],[],'ZLC')
        
        numPointsExp = np.zeros(len(fileName))
        for ii in range(len(fileName)): 
            fileToLoad = fileName[ii]
            # Load experimental molefraction
            timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
            numPointsExp[ii] = len(timeElapsedExp)
        
        # Parse out all the necessary quantities to obtain model fit
        # Mass of sorbent and particle epsilon
        adsorbentDensity = load(parameterPath)["adsorbentDensity"]
        particleEpsilon = load(parameterPath)["particleEpsilon"]
        massSorbent = load(parameterPath)["massSorbent"]
        # Volume of sorbent material [m3]
        volSorbent = (massSorbent/1000)/adsorbentDensity
        # Volume of gas chamber (dead volume) [m3]
        volGas = volSorbent/(1-particleEpsilon)*particleEpsilon
        # Dead volume model
        deadVolumeFile = str(load(parameterPath)["deadVolumeFile"])
        # Isotherm parameter reference
        parameterReference = load(parameterPath)["parameterReference"]
        # Load the model
        modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
        modelNonDim = modelOutputTemp[()]["variable"] 
        # Multiply the paremeters by the reference values
        x = np.multiply(modelNonDim,parameterReference)
        print(x)
        # Downsample intervals
        downsampleInt = numPointsExp/np.min(numPointsExp)
        
        # Initialize loadings
        moleFracExpALL = np.array([])
        moleFracSimALL = np.array([])

        # Loop over all available files    
        for ii in range(len(fileName)):
            fileToLoad = fileName[ii]   
            
            # Initialize outputs
            moleFracSim = []  
            # Load experimental time, molefraction and flowrate (accounting for downsampling)
            timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
            moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
            flowRateTemp = load(fileToLoad)["flowRate"].flatten()
            timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
            moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
            flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
                    
            # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
            timeInt = timeElapsedExp

            # Parse out parameter values
            isothermModel = x[0:-2]
            rateConstant_1 = x[-2]
            rateConstant_2 = x[-1]
                    
            # Compute the dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                                initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                                flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                                expFlag = True,
                                                                isothermModel = isothermModel,
                                                                rateConstant_1 = rateConstant_1,
                                                                rateConstant_2 = rateConstant_2,
                                                                deadVolumeFile = deadVolumeFile,
                                                                volSorbent = volSorbent,
                                                                volGas = volGas,
                                                                temperature = temperatureExp[ii],
                                                                adsorbentDensity = adsorbentDensity)
            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
                                                                  moleFracSim),
                                                        timeElapsedExp),2))

            # Stack mole fraction from experiments and simulation for error 
            # computation
            minExp = np.min(moleFracExp) # Compute the minimum from experiment
            normalizeFactor = np.max(moleFracExp - np.min(moleFracExp)) # Compute the max from normalized data
            moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
            moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))

            # Call the deadVolume Wrapper function to obtain the outlet mole fraction
            deadVolumePath = os.path.join('..','simulationResults',deadVolumeFile)
            modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
            pDV = modelOutputTemp[()]["variable"]
            dvFileLoadTemp = load(deadVolumePath)
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
            moleFracDV = deadVolumeWrapper(timeInt, resultMat[3,:]*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = [moleFracExp[0]])
    
            if 300<temperatureExp[ii] and temperatureExp[ii]<310:
                colorTemp = colorsForPlot[0]
                markersTemp =markersForPlot[0]
            elif 320<temperatureExp[ii] and temperatureExp[ii]<330:
                colorTemp = colorsForPlot[1]
                markersTemp =markersForPlot[1]
            elif 340<temperatureExp[ii] and temperatureExp[ii]<350:
                colorTemp = colorsForPlot[2]
                markersTemp =markersForPlot[2]
    
            if ii in range(0,3):                    
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax1 = plt.subplot(2,2,1)
                ax1.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax1.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax1.semilogy(timeElapsedExp,moleFracDV,
                                 color='#76c893',alpha = 0.2, linestyle='-') # Dead volume response
                ax1.set(ylabel='$y\mathregular{_{CO_2}}$ [-]',
                        xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax1.locator_params(axis="x", nbins=4)
                ax1.axes.xaxis.set_ticklabels([])
                ax1.grid(which='minor', linestyle=':')

            if ii in range(3,6):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax2 = plt.subplot(2,2,3)
                ax2.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax2.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response
                if ii%3 == 0:
                    ax2.semilogy(timeElapsedExp,moleFracDV,
                                 color='#76c893',alpha = 0.2, linestyle='-') # Dead volume response
                ax2.set(xlabel='$t$ [s]',ylabel='$y\mathregular{_{CO_2}}$ [-]',
                        xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax2.locator_params(axis="x", nbins=4)
                ax2.grid(which='minor', linestyle=':')

            if ii in range(6,9):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax3 = plt.subplot(2,2,2)
                ax3.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax3.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response  
                if ii%3 == 0:
                    ax3.semilogy(timeElapsedExp,moleFracDV,
                                 color='#76c893',alpha = 0.2, linestyle='-') # Dead volume response
                ax3.set(xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                ax3.locator_params(axis="x", nbins=4)
                ax3.axes.xaxis.set_ticklabels([])
                ax3.locator_params(axis="x", nbins=4)
                ax3.axes.xaxis.set_ticklabels([])
                ax3.grid(which='minor', linestyle=':')
                
            if ii in range(9,12):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax4 = plt.subplot(2,2,4)
                ax4.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax4.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax4.semilogy(timeElapsedExp,moleFracDV,
                                 color='#76c893',alpha = 0.2, linestyle='-') # Dead volume response
                ax4.set(xlabel='$t$ [s]',
                        xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                ax4.locator_params(axis="x", nbins=4)
                ax4.grid(which='minor', linestyle=':')

        # Get the order of temperatures for each plot        
        temperatureOrder = np.argsort(temperatureExp[0:3])
        # Get the axis handles and labels and order it
        handles,labels = ax1.get_legend_handles_labels()
        ax1.legend([handles[idx] for idx in temperatureOrder],[labels[idx] for idx in temperatureOrder], loc = "upper center", ncol=3, columnspacing=1)
        handles,labels = ax2.get_legend_handles_labels()    
        ax2.legend([handles[idx] for idx in temperatureOrder],[labels[idx] for idx in temperatureOrder], loc = "upper center", ncol=3, columnspacing=1)
        handles,labels = ax3.get_legend_handles_labels()    
        ax3.legend([handles[idx] for idx in temperatureOrder],[labels[idx] for idx in temperatureOrder], loc = "upper center", ncol=3, columnspacing=1)
        handles,labels = ax4.get_legend_handles_labels()    
        ax4.legend([handles[idx] for idx in temperatureOrder],[labels[idx] for idx in temperatureOrder], loc = "upper center", ncol=3, columnspacing=1)
       
        # Put other text entries
        ax1.text(panelLabel_L[pp], 0.67, "(a)", fontsize=8,)
        ax1.text(materialLabel_L[pp], 0.45, materialText[pp], fontsize=8, fontweight = 'bold',color = '#4895EF')
        ax1.text(flowLabel_L[pp], 0.33, "$F^\mathregular{in}$ = 10 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')

        ax2.text(panelLabel_L[pp], 0.67, "(c)", fontsize=8,)
        ax2.text(materialLabel_L[pp], 0.45, materialText[pp], fontsize=8, fontweight = 'bold',color = '#4895EF')
        ax2.text(flowLabel_L[pp], 0.33, "$F^\mathregular{in}$ = 10 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')

        ax3.text(panelLabel_H[pp], 0.67, "(b)", fontsize=8,)
        ax3.text(materialLabel_H[pp], 0.45, materialText[pp], fontsize=8, fontweight = 'bold',color = '#4895EF')
        ax3.text(flowLabel_H[pp], 0.33, "$F^\mathregular{in}$ = 60 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')

        ax4.text(panelLabel_H[pp], 0.67, "(d)", fontsize=8,)
        ax4.text(materialLabel_H[pp], 0.45, materialText[pp], fontsize=8, fontweight = 'bold',color = '#4895EF')
        ax4.text(flowLabel_H[pp], 0.33, "$F^\mathregular{in}$ = 60 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')
                
        #  Save the figure
        if saveFlag:
            # FileName: figureZLC_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
            saveFileName = "figureZLCFit_" + materialText[pp] + "_" + currentDT + "_" + gitCommitID + saveFileExtension
            savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
            # Check if inputResources directory exists or not. If not, create the folder
            if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
                os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
            plt.savefig (savePath)
            
        plt.show()
        
        # Remove all the .npz files genereated from the .mat
        # Loop over all available files    
        for ii in range(len(fileName)):
            os.remove(fileName[ii])

# fun: plotForArticle_figureZLCSimFit
# Plots the Figure ZLC Fit of the manuscript: ZLC goodness for computational results
def plotForArticle_figureZLCSimFit(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import auxiliaryFunctions
    from numpy import load
    import os
    from simulateCombinedModel import simulateCombinedModel
    from deadVolumeWrapper import deadVolumeWrapper
    from extractDeadVolume import filesToProcess # File processing script
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
        
    # Plot colors and markers
    colorsForPlot = ["0091ad","5c4d7d","b7094c"]
    markersForPlot = ["^","d","v"]    
    
    # X limits for the different materials
    XLIM_L = [[0, 200],[0, 150],[0, 600]]
    XLIM_H = [[0, 100],[0, 60],[0, 200]]
    
    # Label positions for the different materials
    panelLabel_L = [185, 150/200*185, 600/200*185]
    panelLabel_H = [185/2, 60/100*185/2, 200/100*185/2]
    materialLabel_L = [182, 150/200*182, 600/200*180]
    materialLabel_H = [182/2, 60/100*182/2, 200/100*180/2]
    flowLabel_L = [118, 150/200*118, 600/200*118]
    flowLabel_H = [118/2, 60/100*118/2, 200/100*118/2]
    materialText = ["AC", "BN", "13X"]

    # Parameter estimate files
                        # Activated Carbon Simulations
    zlcFileNameALL = [['zlcParameters_20210823_1104_03c82f4.npz',
                       'zlcParameters_20210824_0000_03c82f4.npz',
                       'zlcParameters_20210824_1227_03c82f4.npz',
                       'zlcParameters_20210825_0017_03c82f4.npz',
                       'zlcParameters_20210825_1151_03c82f4.npz'],
                      # Boron Nitride Simulations
                      ['zlcParameters_20210823_1907_03c82f4.npz',
                       'zlcParameters_20210824_0555_03c82f4.npz',
                       'zlcParameters_20210824_2105_03c82f4.npz',
                       'zlcParameters_20210825_0833_03c82f4.npz',
                       'zlcParameters_20210825_2214_03c82f4.npz'],
                      # Zeolite 13X Simulations
                      ['zlcParameters_20210824_1102_c8173b1.npz',
                       'zlcParameters_20210825_0243_c8173b1.npz',
                       'zlcParameters_20210825_1758_c8173b1.npz',
                       'zlcParameters_20210826_1022_c8173b1.npz',
                       'zlcParameters_20210827_0104_c8173b1.npz']]
 
    for pp in range(len(zlcFileNameALL)):        
        fig = figure(figsize=(6.5,5))   
        zlcFileName = zlcFileNameALL[pp]
        objectiveFunction = np.zeros([len(zlcFileName)])
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # Obtain the onjective function values
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)

        # Find the experiment with the min objective function
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))
        fileParameter = zlcFileName[int(minJ[0])]
        
        # ZLC parameter model path
        parameterPath = os.path.join('..','simulationResults',fileParameter)
           
        # Parse out experiments names and temperature used for the fitting
        rawFileName = load(parameterPath)["fileName"]
        temperatureExp = load(parameterPath)["temperature"]

        # Generate .npz file for python processing of the .mat file 
        filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'ZLC')
        # Get the processed file names
        fileName = filesToProcess(False,[],[],'ZLC')
        
        numPointsExp = np.zeros(len(fileName))
        for ii in range(len(fileName)): 
            fileToLoad = fileName[ii]
            # Load experimental molefraction
            timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
            numPointsExp[ii] = len(timeElapsedExp)
        
        # Parse out all the necessary quantities to obtain model fit
        # Mass of sorbent and particle epsilon
        adsorbentDensity = load(parameterPath)["adsorbentDensity"]
        particleEpsilon = load(parameterPath)["particleEpsilon"]
        massSorbent = load(parameterPath)["massSorbent"]
        # Volume of sorbent material [m3]
        volSorbent = (massSorbent/1000)/adsorbentDensity
        # Volume of gas chamber (dead volume) [m3]
        volGas = volSorbent/(1-particleEpsilon)*particleEpsilon
        # Dead volume model
        deadVolumeFile = str(load(parameterPath)["deadVolumeFile"])
        # Isotherm parameter reference
        parameterReference = load(parameterPath)["parameterReference"]
        # Load the model
        modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
        modelNonDim = modelOutputTemp[()]["variable"] 
        # Multiply the paremeters by the reference values
        x = np.multiply(modelNonDim,parameterReference)
        # Downsample intervals
        downsampleInt = numPointsExp/np.min(numPointsExp)
        
        # Initialize loadings
        moleFracExpALL = np.array([])
        moleFracSimALL = np.array([])

        # Loop over all available files    
        for ii in range(len(fileName)):
            fileToLoad = fileName[ii]   
            
            # Initialize outputs
            moleFracSim = []  
            # Load experimental time, molefraction and flowrate (accounting for downsampling)
            timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
            moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
            flowRateTemp = load(fileToLoad)["flowRate"].flatten()
            timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
            moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
            flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
                    
            # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
            timeInt = timeElapsedExp

            # Parse out parameter values
            isothermModel = x[0:-2]
            rateConstant_1 = x[-2]
            rateConstant_2 = x[-1]
                    
            # Compute the dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                                initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                                flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                                expFlag = True,
                                                                isothermModel = isothermModel,
                                                                rateConstant_1 = rateConstant_1,
                                                                rateConstant_2 = rateConstant_2,
                                                                deadVolumeFile = deadVolumeFile,
                                                                volSorbent = volSorbent,
                                                                volGas = volGas,
                                                                temperature = temperatureExp[ii],
                                                                adsorbentDensity = adsorbentDensity)
            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
                                                                  moleFracSim),
                                                        timeElapsedExp),2))

            # Stack mole fraction from experiments and simulation for error 
            # computation
            minExp = np.min(moleFracExp) # Compute the minimum from experiment
            normalizeFactor = np.max(moleFracExp - np.min(moleFracExp)) # Compute the max from normalized data
            moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
            moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))

            # Call the deadVolume Wrapper function to obtain the outlet mole fraction
            deadVolumePath = os.path.join('..','simulationResults',deadVolumeFile)
            modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
            pDV = modelOutputTemp[()]["variable"]
            dvFileLoadTemp = load(deadVolumePath)
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
            moleFracDV = deadVolumeWrapper(timeInt, resultMat[3,:]*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = [moleFracExp[0]])
    
            if 300<temperatureExp[ii] and temperatureExp[ii]<310:
                colorTemp = colorsForPlot[0]
                markersTemp =markersForPlot[0]
            elif 320<temperatureExp[ii] and temperatureExp[ii]<330:
                colorTemp = colorsForPlot[1]
                markersTemp =markersForPlot[1]
            elif 340<temperatureExp[ii] and temperatureExp[ii]<350:
                colorTemp = colorsForPlot[2]
                markersTemp =markersForPlot[2]
     
            if ii in range(0,3):                    
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax1 = plt.subplot(2,2,1)
                ax1.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax1.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax1.semilogy(timeElapsedExp,moleFracDV,
                                 color='k',alpha = 0.2, linestyle='-') # Dead volume response
                ax1.set(ylabel='$y\mathregular{_{CO_2}}$ [-]',
                        xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax1.locator_params(axis="x", nbins=4)
                ax1.axes.xaxis.set_ticklabels([])
                ax1.grid(which='minor', linestyle=':')

            if ii in range(3,6):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax2 = plt.subplot(2,2,3)
                ax2.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax2.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response
                if ii%3 == 0:
                    ax2.semilogy(timeElapsedExp,moleFracDV,
                                 color='k',alpha = 0.2, linestyle='-') # Dead volume response
                ax2.set(xlabel='$t$ [s]',ylabel='$y\mathregular{_{CO_2}}$ [-]',
                        xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax2.locator_params(axis="x", nbins=4)
                ax2.grid(which='minor', linestyle=':')
                
            if ii in range(6,9):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax3 = plt.subplot(2,2,2)
                ax3.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax3.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response  
                if ii%3 == 0:
                    ax3.semilogy(timeElapsedExp,moleFracDV,
                                 color='k',alpha = 0.2, linestyle='-') # Dead volume response
                ax3.set(xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                ax3.locator_params(axis="x", nbins=4)
                ax3.axes.xaxis.set_ticklabels([])
                ax3.locator_params(axis="x", nbins=4)
                ax3.axes.xaxis.set_ticklabels([])
                ax3.grid(which='minor', linestyle=':')
                
            if ii in range(9,12):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax4 = plt.subplot(2,2,4)
                ax4.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax4.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax4.semilogy(timeElapsedExp,moleFracDV,
                                 color='k',alpha = 0.2, linestyle='-') # Dead volume response
                ax4.set(xlabel='$t$ [s]',
                        xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                ax4.locator_params(axis="x", nbins=4)
                ax4.grid(which='minor', linestyle=':')

        # Get the order of temperatures for each plot        
        temperatureOrder = np.argsort(temperatureExp[0:3])
        # Get the axis handles and labels and order it
        handles,labels = ax1.get_legend_handles_labels()
        ax1.legend([handles[idx] for idx in temperatureOrder],[labels[idx] for idx in temperatureOrder], loc = "upper center", ncol=3, columnspacing=1)
        handles,labels = ax2.get_legend_handles_labels()    
        ax2.legend([handles[idx] for idx in temperatureOrder],[labels[idx] for idx in temperatureOrder], loc = "upper center", ncol=3, columnspacing=1)
        handles,labels = ax3.get_legend_handles_labels()    
        ax3.legend([handles[idx] for idx in temperatureOrder],[labels[idx] for idx in temperatureOrder], loc = "upper center", ncol=3, columnspacing=1)
        handles,labels = ax4.get_legend_handles_labels()    
        ax4.legend([handles[idx] for idx in temperatureOrder],[labels[idx] for idx in temperatureOrder], loc = "upper center", ncol=3, columnspacing=1)
       
        # Put other text entries
        ax1.text(panelLabel_L[pp], 0.67, "(a)", fontsize=8,)
        ax1.text(materialLabel_L[pp], 0.45, materialText[pp], fontsize=8, fontweight = 'bold',color = '#7d8597')
        ax1.text(flowLabel_L[pp], 0.33, "$F^\mathregular{in}$ = 10 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#7d8597')

        ax2.text(panelLabel_L[pp], 0.67, "(c)", fontsize=8,)
        ax2.text(materialLabel_L[pp], 0.45, materialText[pp], fontsize=8, fontweight = 'bold',color = '#7d8597')
        ax2.text(flowLabel_L[pp], 0.33, "$F^\mathregular{in}$ = 10 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#7d8597')

        ax3.text(panelLabel_H[pp], 0.67, "(b)", fontsize=8,)
        ax3.text(materialLabel_H[pp], 0.45, materialText[pp], fontsize=8, fontweight = 'bold',color = '#7d8597')
        ax3.text(flowLabel_H[pp], 0.33, "$F^\mathregular{in}$ = 60 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#7d8597')

        ax4.text(panelLabel_H[pp], 0.67, "(d)", fontsize=8,)
        ax4.text(materialLabel_H[pp], 0.45, materialText[pp], fontsize=8, fontweight = 'bold',color = '#7d8597')
        ax4.text(flowLabel_H[pp], 0.33, "$F^\mathregular{in}$ = 60 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#7d8597')
                
        #  Save the figure
        if saveFlag:
            # FileName: figureZLCSimFit_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
            saveFileName = "figureZLCSimFit_" + materialText[pp] + "_" + currentDT + "_" + gitCommitID + saveFileExtension
            savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
            # Check if inputResources directory exists or not. If not, create the folder
            if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
                os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
            plt.savefig (savePath)
            
        plt.show()
        
        # Remove all the .npz files genereated from the .mat
        # Loop over all available files    
        for ii in range(len(fileName)):
            os.remove(fileName[ii])

# fun: plotForArticle_figureZLCRep
# Plots the Figure repetition of the manuscript: ZLC experimental repetitions
def plotForArticle_figureZLCRep(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    from numpy import load
    import os
    from extractDeadVolume import filesToProcess # File processing script
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
        
    # Plot colors and markers
    colorsForPlot = ["ffba08","d00000","03071e"]
    markersForPlot = ["^","d","v"]    
    
    # X limits for the different materials
    XLIM_L = [[0, 200],[0, 150],[0, 600]]
    XLIM_H = [[0, 100],[0, 60],[0, 200]]
    
    # Label positions for the different materials
    panelLabel_L = [175, 150/200*175, 600/200*175]
    panelLabel_H = [175/2, 60/100*175/2, 200/100*175/2]
    materialLabel_L = [172, 150/200*172, 600/200*165]
    materialLabel_H = [172/2, 60/100*172/2, 200/100*165/2]
    flowLabel_L = [75, 150/200*75, 600/200*72]
    flowLabel_H = [75/2, 60/100*75/2, 200/100*72/2]
    materialText = ["AC", "BN", "13X"]
    panelLabel = ["(a)","(b)","(c)","(d)","(e)","(f)"]
    repLabel_L = [10, 10*150/200, 10*600/200]
    repLabel_H = [80, 80*60/100, 80*200/100]

    # Names of experiment and their corresponding temperature
                      # Activated Carbon Experiments
    rawFileNameALL = [['ZLC_ActivatedCarbon_Exp76B_Output.mat',
                       'ZLC_ActivatedCarbon_Exp74B_Output.mat',
                       'ZLC_ActivatedCarbon_Exp72B_Output.mat',
                       'ZLC_ActivatedCarbon_Exp77B_Output.mat',
                       'ZLC_ActivatedCarbon_Exp75B_Output.mat',
                       'ZLC_ActivatedCarbon_Exp73B_Output.mat',
                       'ZLC_ActivatedCarbon_Exp78B_Output.mat',
                       'ZLC_ActivatedCarbon_Exp80B_Output.mat',
                       'ZLC_ActivatedCarbon_Exp82B_Output.mat',
                       'ZLC_ActivatedCarbon_Exp79B_Output.mat',
                       'ZLC_ActivatedCarbon_Exp81B_Output.mat',
                       'ZLC_ActivatedCarbon_Exp83B_Output.mat'],
                      # # Boron Nitride Experiments
                      ['ZLC_BoronNitride_Exp38B_Output.mat',
                       'ZLC_BoronNitride_Exp36B_Output.mat',
                       'ZLC_BoronNitride_Exp34B_Output.mat',
                       'ZLC_BoronNitride_Exp39B_Output.mat',
                       'ZLC_BoronNitride_Exp37B_Output.mat',
                       'ZLC_BoronNitride_Exp35B_Output.mat',
                       'ZLC_BoronNitride_Exp40B_Output.mat',
                       'ZLC_BoronNitride_Exp42B_Output.mat',
                       'ZLC_BoronNitride_Exp44B_Output.mat',
                       'ZLC_BoronNitride_Exp41B_Output.mat',
                       'ZLC_BoronNitride_Exp43B_Output.mat',
                       'ZLC_BoronNitride_Exp45B_Output.mat',],
                       # Zeolite 13X Experiments
                       ['ZLC_Zeolite13X_Exp62B_Output.mat',
                        'ZLC_Zeolite13X_Exp58B_Output.mat',
                        'ZLC_Zeolite13X_Exp54B_Output.mat',
                        'ZLC_Zeolite13X_Exp63B_Output.mat',
                        'ZLC_Zeolite13X_Exp59B_Output.mat',
                        'ZLC_Zeolite13X_Exp55B_Output.mat',
                        'ZLC_Zeolite13X_Exp66B_Output.mat',
                        'ZLC_Zeolite13X_Exp70B_Output.mat',
                        'ZLC_Zeolite13X_Exp68B_Output.mat',
                        'ZLC_Zeolite13X_Exp67B_Output.mat',
                        'ZLC_Zeolite13X_Exp71B_Output.mat',
                        'ZLC_Zeolite13X_Exp69B_Output.mat',]]
    
    temperatureALL = [[306,325,345]*4, # Activated carbon
                      [306,325,345]*4, # Boron Nitrode
                      [306,326,345]*4,] # Zeolite 13X
    
    for pp in range(len(rawFileNameALL)):
        rawFileName = rawFileNameALL[pp]
           
        # Parse out temperature used for the fitting
        temperatureExp = temperatureALL[pp]

        # Generate .npz file for python processing of the .mat file 
        filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'ZLC')
        # Get the processed file names
        fileName = filesToProcess(False,[],[],'ZLC')
        
        numPointsExp = np.zeros(len(fileName))
        for ii in range(len(fileName)): 
            fileToLoad = fileName[ii]
            # Load experimental molefraction
            timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
            numPointsExp[ii] = len(timeElapsedExp)
        
        # Downsample intervals
        downsampleInt = numPointsExp/np.min(numPointsExp)

        # Loop over all available files    
        for ii in range(len(fileName)):
            fileToLoad = fileName[ii]   
            # Load experimental time, molefraction and flowrate (accounting for downsampling)
            timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
            moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
            timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
            moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
                    
            if 300<temperatureExp[ii] and temperatureExp[ii]<310:
                colorTemp = colorsForPlot[0]
            elif 320<temperatureExp[ii] and temperatureExp[ii]<330:
                colorTemp = colorsForPlot[1]
            elif 340<temperatureExp[ii] and temperatureExp[ii]<350:
                colorTemp = colorsForPlot[2]
    
            if ii in range(0,3): 
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"                   
                # Plot the experimental data with model output
                ax1 = plt.subplot(2,3,pp+1)
                ax1.semilogy(timeElapsedExp,moleFracExp,
                             marker = markersForPlot[1],linewidth = 0,
                             markeredgecolor='#'+colorTemp, 
                             markerfacecolor='w',
                             markeredgewidth=0.5,markevery = 4,
                             label = legendStr) # Experimental response
                ax1.legend(loc = "center right", handletextpad=0.0)
                if pp == 0:
                    ax1.set(xlabel='$t$ [s]',ylabel='$y\mathregular{_{CO_2}}$ [-]',
                            xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                    ax1.locator_params(axis="x", nbins=4)
                else: 
                    ax1.set(xlabel='$t$ [s]',
                            xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                    ax1.locator_params(axis="x", nbins=4)
                ax1.grid(which='minor', linestyle=':')

            if ii in range(6,9):
                ax2 = plt.subplot(2,3,pp+1)
                ax2.semilogy(timeElapsedExp,moleFracExp,
                             marker = markersForPlot[0],linewidth = 0,
                             color='#'+colorTemp, alpha = 0.25,
                             markevery = 4) # Experimental response
                ax2.grid(which='minor', linestyle=':')
                
            if ii in range(3,6):
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"                   
                # Plot the experimental data with model output
                ax3 = plt.subplot(2,3,pp+4)
                ax3.semilogy(timeElapsedExp,moleFracExp,
                             marker = markersForPlot[1],linewidth = 0,
                             markeredgecolor='#'+colorTemp, 
                             markerfacecolor='w',
                             markeredgewidth=0.5,markevery = 4,
                             label = legendStr) # Experimental response
                ax3.legend(loc = "center right", handletextpad=0.0)
                if pp == 0:
                    ax3.set(xlabel='$t$ [s]',ylabel='$y\mathregular{_{CO_2}}$ [-]',
                            xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                    ax3.locator_params(axis="x", nbins=4)
                else: 
                    ax3.set(xlabel='$t$ [s]',
                            xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                    ax3.locator_params(axis="x", nbins=4)                
                ax3.grid(which='minor', linestyle=':')

            if ii in range(9,12):
                # Plot the experimental data with model output
                ax4 = plt.subplot(2,3,pp+4)
                ax4.semilogy(timeElapsedExp,moleFracExp,
                             marker = markersForPlot[0],linewidth = 0,
                             color='#'+colorTemp, alpha = 0.25,
                             markevery = 4) # Experimental response 
                ax4.grid(which='minor', linestyle=':')
     
        # # Put other text entries
        ax1.text(panelLabel_L[pp], 0.67, panelLabel[pp], fontsize=8,)
        ax1.text(materialLabel_L[pp], 0.45, materialText[pp], fontsize=8, fontweight = 'bold',color = '#4895EF')
        ax1.text(flowLabel_L[pp], 0.33, "$F^\mathregular{in}$ = 10 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')
        ax1.text(repLabel_L[pp], 0.015, "REP1", fontsize=8, fontweight = 'bold',color = 'k')
        ax1.text(repLabel_L[pp], 0.011, "REP2", fontsize=8, fontweight = 'bold',color = 'k', alpha = 0.25)

        ax3.text(panelLabel_H[pp], 0.67, panelLabel[pp+3], fontsize=8,)
        ax3.text(materialLabel_H[pp], 0.45, materialText[pp], fontsize=8, fontweight = 'bold',color = '#4895EF')
        ax3.text(flowLabel_H[pp], 0.33, "$F^\mathregular{in}$ = 60 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')
        ax3.text(repLabel_H[pp], 0.015, "REP1", fontsize=8, fontweight = 'bold',color = 'k')
        ax3.text(repLabel_H[pp], 0.011, "REP2", fontsize=8, fontweight = 'bold',color = 'k', alpha = 0.25)
                
    #  Save the figure
    if saveFlag:
        # FileName: figureZLCRep_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureZLCRep" + "_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
            
    plt.show()
    
    # Remove all the .npz files genereated from the .mat
    # Loop over all available files    
    for ii in range(len(fileName)):
        os.remove(fileName[ii])
        
# fun: plotForArticle_figureZLCObj
# Plots the Figure ZLC Fit of the manuscript: ZLC goodness of fit for experimental results
def plotForArticle_figureZLCObj(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    from numpy import load
    import os
    from matplotlib.ticker import MaxNLocator
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Minimum J for all the materials (computational) - HARD CODED
    minJ_True = [-1894, -1077, -10592]
    
    # X limits for the different materials
    YLIM_L = [[180, 200],[380, 400],[-80, -50]]
    YLIM_H = [[-2000, -1400],[-1200, -400],[-12000, -4000]]
    
    # Label positions for the different materials
    minJLabel = [-1875,-1045, -10300]
    materialText = ["AC", "BN", "13X"]
    materialLabel_X = [4.45, 4.45, 4.32]
    materialLabel_L = [196.75, 396.75, -55]
    materialLabel_H = [-1500, -535, -5400]
    panelLabel = ["(a)","(b)","(c)","(d)","(e)","(f)"]
    panelLabel_L = [198.5, 398.5, -52.5]
    panelLabel_H = [-1450, -470, -4750]

    # Parameter estimate files (Experimental)
                        # Activated Carbon Experiments
    zlcFileNameExpALL = [['zlcParameters_20210822_0926_c8173b1.npz',
                        'zlcParameters_20210822_1733_c8173b1.npz',
                        # 'zlcParameters_20210823_0133_c8173b1.npz', # DSL BAD (but lowest J)
                        # 'zlcParameters_20210823_1007_c8173b1.npz', # DSL BAD (but lowest J)
                        'zlcParameters_20210823_1810_c8173b1.npz'],
                        # Boron Nitride Experiments
                        ['zlcParameters_20210823_1731_c8173b1.npz',
                          'zlcParameters_20210824_0034_c8173b1.npz',
                          'zlcParameters_20210824_0805_c8173b1.npz',
                          'zlcParameters_20210824_1522_c8173b1.npz',
                          'zlcParameters_20210824_2238_c8173b1.npz',],
                          # Zeolite 13X Experiments
                        ['zlcParameters_20210824_1552_6b88505.npz',
                          'zlcParameters_20210825_0559_6b88505.npz',
                          'zlcParameters_20210825_1854_6b88505.npz',
                          'zlcParameters_20210826_0847_6b88505.npz',
                          'zlcParameters_20210827_0124_6b88505.npz',]]

    # Parameter estimate files (Computational)
                        # Activated Carbon Simulations
    zlcFileNameSimALL = [['zlcParameters_20210823_1104_03c82f4.npz',
                       'zlcParameters_20210824_0000_03c82f4.npz',
                       'zlcParameters_20210824_1227_03c82f4.npz',
                       'zlcParameters_20210825_0017_03c82f4.npz',
                       'zlcParameters_20210825_1151_03c82f4.npz'],
                      # Boron Nitride Simulations
                      ['zlcParameters_20210823_1907_03c82f4.npz',
                       'zlcParameters_20210824_0555_03c82f4.npz',
                       'zlcParameters_20210824_2105_03c82f4.npz',
                       'zlcParameters_20210825_0833_03c82f4.npz',
                       'zlcParameters_20210825_2214_03c82f4.npz'],
                      # Zeolite 13X Simulations
                      ['zlcParameters_20210824_1102_c8173b1.npz',
                       'zlcParameters_20210825_0243_c8173b1.npz',
                       'zlcParameters_20210825_1758_c8173b1.npz',
                       'zlcParameters_20210826_1022_c8173b1.npz',
                       'zlcParameters_20210827_0104_c8173b1.npz']]
    
    for pp in range(len(zlcFileNameExpALL)):
        # Experiments
        zlcFileName = zlcFileNameExpALL[pp]
        objectiveFunction = np.zeros([len(zlcFileName)])
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # Obtain the onjective function values
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
            
        ax1 = plt.subplot(2,3,pp+1)       
        xx = range(1,len(zlcFileName)+1)
        ax1.plot(xx,objectiveFunction,
                 linestyle = ':', linewidth = 0.5,
                 color = '#7d8597', marker = '^', markersize = 3,
                 markerfacecolor = '#d00000',
                 markeredgecolor = '#d00000') # ALL
    
        if pp == 0:
            ax1.set(ylabel='$J$ [-]',
                    xlim = [1,5],ylim = YLIM_L[pp]) 
        else:
            ax1.set(xlim = [1,5],ylim = YLIM_L[pp]) 
        ax1.locator_params(axis="y", nbins=4)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.axes.xaxis.set_ticklabels([])

        # Simulation
        zlcFileName = zlcFileNameSimALL[pp]
        objectiveFunction = np.zeros([len(zlcFileName)])
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # Obtain the onjective function values
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
            
        ax2 = plt.subplot(2,3,pp+4)       
        xx = range(1,len(zlcFileName)+1)
        ax2.plot(xx,objectiveFunction,
                 linestyle = ':', linewidth = 0.5,
                 color = '#7d8597', marker = 'v', markersize = 3,
                 markerfacecolor = '#5c4d7d',
                 markeredgecolor = '#5c4d7d') # ALL
        ax2.axhline(minJ_True[pp],
                   linestyle = ':', linewidth = 1, color = '#7d8597')
        ax2.text(1.25, minJLabel[pp], '$J_\mathregular{true}$ = ' + str(minJ_True[pp]), 
                 fontsize=8,color = '#7d8597')
        
        if pp == 0:
            ax2.set(xlabel='Repetition [-]', 
                    ylabel='$J$ [-]',
                    xlim = [1,5],ylim = YLIM_H[pp]) 
        else:
            ax2.set(xlabel='Repetition [-]', 
                    xlim = [1,5],ylim = YLIM_H[pp]) 
        ax2.locator_params(axis="y", nbins=4)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Put other text entries
        ax1.text(1.15, panelLabel_L[pp], panelLabel[pp], fontsize=8,)
        ax2.text(1.15, panelLabel_H[pp], panelLabel[pp+3], fontsize=8,)

        ax1.text(3.1,materialLabel_L[pp], "Experimental", fontsize=8, fontweight = 'bold',color = '#7d8597')
        ax1.text(materialLabel_X[pp],panelLabel_L[pp], materialText[pp], fontsize=8, fontweight = 'bold',color = '#4895EF')

        ax2.text(2.9,materialLabel_H[pp], "Computational", fontsize=8, fontweight = 'bold',color = '#7d8597')
        ax2.text(materialLabel_X[pp],panelLabel_H[pp], materialText[pp], fontsize=8, fontweight = 'bold',color = '#4895EF')
        
    #  Save the figure
    if saveFlag:
        # FileName: figureZLCObj_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureZLCObj" + "_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
            
    plt.show()
    
# fun: plotForArticle_figureRawTex
# Plots the Figure SX of the manuscript: Raw Textural Characterization (MIP, N2 and XRD)
def plotForArticle_figureRawTex(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    import scipy.io as sio
    import os
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Plot colors and markers (isotherm)    
    colorsForPlot_I = ["E4572E","76B041","FFC914"]
    markersForPlot_I = ["^","d","v"]

    # Folder for material characterization
    mainDir = os.path.join('..','experimental','materialCharacterization')

    
    # File with pore characterization data
    rawDataALL = ['rawTexData.mat']

    # Loop over all the raw files
    for kk in range(len(rawDataALL)):
        # Create the instance for the plots
        ax1 = plt.subplot(1,3,2)
        
        # Path of the file name
        fileToLoad = os.path.join(mainDir,rawDataALL[kk])

        # Get the MIP points
        MIPALL = sio.loadmat(fileToLoad)["rawTexturalData"]["MIP"][0][0]

        # Get the QC points
        QCALL = sio.loadmat(fileToLoad)["rawTexturalData"]["QC"][0][0]
        
        # Get the XRD points
        XRDALL = sio.loadmat(fileToLoad)["rawTexturalData"]["XRD"][0][0]
        
        adsorbentName = ['AC', 'BN', '13X']
        
        # Find indices corresponding to each material
        for ll in range(3):
        
            # Plot MIP data
            ax1.semilogx(MIPALL[:,0+2*ll],
                    MIPALL[:,1+2*ll],
                    linewidth = 0.5,
                    linestyle =':',
                    marker = markersForPlot_I[ll],
                    color='#'+colorsForPlot_I[ll],
                    label = str(adsorbentName[ll])) 
          
        # Text labels
        ax1.set(xlabel='$P$ [psia]', 
                ylabel='$V_{\mathrm{Hg}}$ [cm$^{3}$ g$^{-1}$]',
                xlim = [1e-1,1e5], ylim = [0, 2])
        # ax1.text(70, 1.3, "(a)", fontsize=8,)
        ax1.legend(loc='upper right', handletextpad=0.2)

        ax1.locator_params(axis="y", nbins=4)
        ax1.text(0.2, 1.82, "(b)", fontsize=8,)

        
        # Create the instance for the plots
        ax2 = plt.subplot(1,3,3)
        # Find indices corresponding to each material
        for ll in range(3):
        
            # Plot XRD data
            ax2.plot(XRDALL[:,0+2*ll],
                    XRDALL[:,1+2*ll]/np.max(XRDALL[:,1+2*ll])+ll,
                    linewidth = 0.5,
                    color='#'+colorsForPlot_I[ll],
                    label = str(adsorbentName[ll])) 
          
        # Text labels
        ax2.set(xlabel='2\u03B8 [deg]', 
        ylabel='$I$ [-]',
        xlim = [5,60], ylim = [0, 3.5])
        # ax2.text(70, 1.3, "(b)", fontsize=8,)
        ax2.legend(loc='best', handletextpad=0.2)

        ax2.locator_params(axis="x", nbins=6)            
        ax2.locator_params(axis="y", nbins=1)
        ax2.yaxis.set_ticklabels([])
        ax2.yaxis.set_ticks([])
        ax2.text(7.5, 3.2, "(c)", fontsize=8,)
        
        # Create the instance for the plots
        ax3 = plt.subplot(1,3,1)      
        # Find indices corresponding to each material
        for ll in range(3):
        
            # Plot N2 77 K isotherm data
            ax3.semilogx(QCALL[:,0+2*ll],
                    QCALL[:,1+2*ll],
                    linewidth = 0.5,
                    linestyle = ':',
                    marker = markersForPlot_I[ll],
                    color='#'+colorsForPlot_I[ll],
                    label = str(adsorbentName[ll])) 
          
        # Text labels
        ax3.set(xlabel='$P/P_0$ [-]', 
        ylabel='$q^*_{\mathrm{N}_2}$ [cm$^{3}$(STP) g$^{-1}$]',
        xlim = [1e-7,1], 
        ylim = [0, 600])
        # ax3.text(70, 1.3, "(c)", fontsize=8,)
        ax3.legend(loc='upper right', handletextpad=0.2)

        # ax3.locator_params(axis="x", nbins=4)            
        ax3.locator_params(axis="y", nbins=4)
        ax3.text(2e-7, 550, "(a)", fontsize=8,)
 
    #  Save the figure
    if saveFlag:
        # FileName: figureRawTex_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureRawTex_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
 
    plt.show()

# fun: plotForArticle_figureMSCal
# Plots the Figure SX of the manuscript: Repeats of MS calibration over the course of 83 days    
def plotForArticle_figureMSCal(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import auxiliaryFunctions
    import scipy.io as sio
    import os
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Plot colors and markers (isotherm)    
    colorsForPlot_I = ["797d62","d08c60"]
    markersForPlot_I = ["^","^","^"]

    # Folder for MS Calibration
    mainDir = os.path.join('..','experimental','materialCharacterization')

    
    # File with MS data
    msDataNEW = ['msData_072621.mat']
    msDataOLD = ['msData_050521.mat']
    
    # Create the instance for the plots
    fig = figure(figsize=(6,3))        
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
        
       
    # Loop over all the flowrate files
    for kk in range(len(msDataOLD)):
        # Path of the file name
        fileToLoad = os.path.join(mainDir,msDataOLD[kk])

        # Get the MIP points
        msDataOLDALL = sio.loadmat(fileToLoad)["msData_050521"]
        
        # Find indices corresponding to each flowrate
        for ll in range(3):
        
            # Plot MS calibration plot for old repeat
            if ll == 1:
                ax1.plot(msDataOLDALL[:,0+2*ll],
                    msDataOLDALL[:,1+2*ll],
                    linewidth = 0.5,
                    linestyle =':',
                    marker = markersForPlot_I[ll],
                    markersize = 2,
                    color='#'+colorsForPlot_I[1],
                    label = str('Day 1'))
                
                ax2.semilogy(msDataOLDALL[1:-1,0+2*ll],
                    1-msDataOLDALL[1:-1,1+2*ll],
                    linewidth = 0.5,
                    linestyle =':',
                    marker = markersForPlot_I[ll],
                    markersize = 2,
                    color='#'+colorsForPlot_I[1],
                    label = str('Day 1'))
            else:
                ax1.plot(msDataOLDALL[:,0+2*ll],
                    msDataOLDALL[:,1+2*ll],
                    linewidth = 0.5,
                    linestyle =':',
                    marker = markersForPlot_I[ll],
                    markersize = 2,
                    color='#'+colorsForPlot_I[1])
                ax2.semilogy(msDataOLDALL[1:-1,0+2*ll],
                    1-msDataOLDALL[1:-1,1+2*ll],
                    linewidth = 0.5,
                    linestyle =':',
                    marker = markersForPlot_I[ll],
                    markersize = 2,
                    color='#'+colorsForPlot_I[1])
          
        # Text labels
        ax1.set(xlabel='$I_{\mathrm{He}}$ [-]', 
                ylabel='$y_{\mathrm{He}}$ [-]',
                xlim = [0,1], ylim = [0,1])
        # ax1.text(70, 1.3, "(a)", fontsize=8,)
        ax1.legend(loc='best', handletextpad=0.25)
        ax1.text(0.48, 1.05, "(a)", fontsize=8,)
        
        ax2.set(xlabel='$I_{\mathrm{He}}$ [-]', 
        ylabel='$1-y_{\mathrm{He}}$ [-]',
        xlim = [0.8,1], ylim = [1e-3,0.1])
        # ax1.text(70, 1.3, "(a)", fontsize=8,)
        ax1.legend(loc='best', handletextpad=0.25)
        ax2.legend(loc='best', handletextpad=0.25)
        ax2.locator_params(axis="x", nbins=4)   
        ax2.text(0.895, 0.13, "(b)", fontsize=8,)
     
    markersForPlot_I = ["v","v","v"]    
    # Loop over all the flowrate files
    for kk in range(len(msDataNEW)):
        # Path of the file name
        fileToLoad = os.path.join(mainDir,msDataNEW[kk])

        # Get the MIP points
        msDataNEWALL = sio.loadmat(fileToLoad)["msData_072621"]
        
        
        # Find indices corresponding to each flowrate
        for ll in range(3):
        
            # Plot MS calibration plot for new repeat
            if ll == 1:
                ax1.plot(msDataNEWALL[:,0+2*ll],
                    msDataNEWALL[:,1+2*ll],
                    linewidth = 0.5,
                    linestyle =':',
                    marker = markersForPlot_I[ll],
                    markersize = 2,
                    color='#'+colorsForPlot_I[0],
                    label = str('Day 83')) 
                ax2.semilogy(msDataNEWALL[1:-1,0+2*ll],
                    1-msDataNEWALL[1:-1,1+2*ll],
                    linewidth = 0.5,
                    linestyle =':',
                    marker = markersForPlot_I[ll],
                    markersize = 2,
                    color='#'+colorsForPlot_I[0],
                    label = str('Day 83')) 
            else:
                ax1.plot(msDataNEWALL[:,0+2*ll],
                    msDataNEWALL[:,1+2*ll],
                    linewidth = 0.5,
                    linestyle =':',
                    marker = markersForPlot_I[ll],
                    markersize = 2,
                    color='#'+colorsForPlot_I[0])
                ax2.semilogy(msDataNEWALL[1:-1,0+2*ll],
                    1-msDataNEWALL[1:-1,1+2*ll],
                    linewidth = 0.5,
                    linestyle =':',
                    marker = markersForPlot_I[ll],
                    markersize = 2,
                    color='#'+colorsForPlot_I[0])
          
        # Text labels
        ax1.legend(loc='best', handletextpad=0.25)
        ax2.legend(loc='best', handletextpad=0.25)
 
        ax1.grid(which='minor', linestyle=':')
        ax2.grid(which='minor', linestyle=':')
        
    #  Save the figure
    if saveFlag:
        # FileName: figureMSCal_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureMSCal_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
 
    plt.show()
    
# fun: plotForArticle_figureSensitivity
# Plots the Figure Sensitivity of the manuscript: Variable sensitivity analysis
def plotForArticle_figureSensitivity(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    from matplotlib.lines import Line2D
    import auxiliaryFunctions
    from numpy import load
    import os
    from computeEquilibriumLoading import computeEquilibriumLoading
    from matplotlib.ticker import FormatStrFormatter
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Plot colors and markers (isotherm)    
    colorsForPlot = ["0091ad","5c4d7d","b7094c"]
    alphaForPlot = [0.5, 1., 0.5]
    lineForPlot = [':', '-', '--']

    # DV alpha and lines
    alphaForPlot_DV = [0.5, 0.5, 1.0]
    lineForPlot_DV = [':', '--', '-']
    
    # Porosity label
    porosityLabel = ['$\epsilon_\mathregular{T}$ = 0.35',
                     '$\epsilon_\mathregular{T}$ = 0.61',
                     '$\epsilon_\mathregular{T}$ = 0.90']

    # Mass label
    massLabel = ['$m_\mathregular{ads}$ = 59.38 mg',
                 '$m_\mathregular{ads}$ = 62.50 mg',
                 '$m_\mathregular{ads}$ = 65.63 mg']

    # Dead Volume
    deadLabel = ['TIS',
                 'TIS + D/M',
                 'TIS + D/M + MS']

    # Custom Legend Lines
    custom_lines = [Line2D([0], [0], linestyle=lineForPlot[0], lw=1, dash_capstyle = 'round', alpha = alphaForPlot[0], color = 'k'),
                    Line2D([0], [0], linestyle=lineForPlot[1], lw=1, dash_capstyle = 'round', alpha = alphaForPlot[1], color = 'k'),
                    Line2D([0], [0], linestyle=lineForPlot[2], lw=1, dash_capstyle = 'round', alpha = alphaForPlot[2], color = 'k')]

    # Custom Legend Lines (DV)
    custom_linesDV = [Line2D([0], [0], linestyle=lineForPlot_DV[0], lw=1, dash_capstyle = 'round', alpha = alphaForPlot_DV[0], color = 'k'),
                        Line2D([0], [0], linestyle=lineForPlot_DV[1], lw=1, dash_capstyle = 'round', alpha = alphaForPlot_DV[1], color = 'k'),
                        Line2D([0], [0], linestyle=lineForPlot_DV[2], lw=1, dash_capstyle = 'round', alpha = alphaForPlot_DV[2], color = 'k')]

    
    
    # Define temperature
    temperature = [308.15, 328.15, 348.15]
    
    # Parameter estimate files
    # Effect of porosity
                  
    porosityALL = [# Porosity - 0.35
                   ['zlcParameters_20210923_0816_c8173b1.npz',
                    'zlcParameters_20210923_2040_c8173b1.npz',
                    'zlcParameters_20210924_0952_c8173b1.npz',
                    'zlcParameters_20210924_2351_c8173b1.npz',
                    'zlcParameters_20210925_1243_c8173b1.npz'],
                    # Activated Carbon Simulation (Base)
                   ['zlcParameters_20210823_1104_03c82f4.npz',
                    'zlcParameters_20210824_0000_03c82f4.npz',
                    'zlcParameters_20210824_1227_03c82f4.npz',
                    'zlcParameters_20210825_0017_03c82f4.npz',
                    'zlcParameters_20210825_1151_03c82f4.npz'],
                   # Porosity - 0.90
                   ['zlcParameters_20210922_2242_c8173b1.npz',
                    'zlcParameters_20210923_0813_c8173b1.npz',
                    'zlcParameters_20210923_1807_c8173b1.npz',
                    'zlcParameters_20210924_0337_c8173b1.npz',
                    'zlcParameters_20210924_1314_c8173b1.npz']]

    # Effect of mass
    massALL = [    # Mass - 0.95
                   ['zlcParameters_20210926_2111_c8173b1.npz',
                    'zlcParameters_20210927_0817_c8173b1.npz',
                    'zlcParameters_20210927_1933_c8173b1.npz',
                    'zlcParameters_20210928_0647_c8173b1.npz',
                    'zlcParameters_20210928_1809_c8173b1.npz'],
                   # Activated Carbon Simulation (Base)
                   ['zlcParameters_20210823_1104_03c82f4.npz',
                    'zlcParameters_20210824_0000_03c82f4.npz',
                    'zlcParameters_20210824_1227_03c82f4.npz',
                    'zlcParameters_20210825_0017_03c82f4.npz',
                    'zlcParameters_20210825_1151_03c82f4.npz'],
                   # Mass - 1.05
                   ['zlcParameters_20210925_1104_c8173b1.npz',
                    'zlcParameters_20210925_2332_c8173b1.npz',
                    'zlcParameters_20210926_1132_c8173b1.npz',
                    'zlcParameters_20210926_2248_c8173b1.npz',
                    'zlcParameters_20210927_0938_c8173b1.npz']]

    # Effect of DV Model
    deadALL = [    # TIS
                   ['zlcParameters_20211018_1029_c8173b1.npz',
                   'zlcParameters_20211018_1648_c8173b1.npz',
                   'zlcParameters_20211018_2358_c8173b1.npz',
                   'zlcParameters_20211019_0625_c8173b1.npz',
                   'zlcParameters_20211019_1303_c8173b1.npz'],
                   # TIS + D/M
                   ['zlcParameters_20211026_1152_c8173b1.npz',
                    'zlcParameters_20211026_2220_c8173b1.npz',
                    'zlcParameters_20211027_0918_c8173b1.npz',
                    'zlcParameters_20211027_2016_c8173b1.npz',
                    'zlcParameters_20211028_0645_c8173b1.npz'],
                   # Activated Carbon Simulation (Base)
                   ['zlcParameters_20210823_1104_03c82f4.npz',
                    'zlcParameters_20210824_0000_03c82f4.npz',
                    'zlcParameters_20210824_1227_03c82f4.npz',
                    'zlcParameters_20210825_0017_03c82f4.npz',
                    'zlcParameters_20210825_1151_03c82f4.npz'],]

    # Create the grid for mole fractions
    y = np.linspace(0,1.,100)
    fig = figure(figsize=(7,2.65))
    # Effect of porosity
    for pp in range(len(porosityALL)):
        zlcFileName = porosityALL[pp]

        # Initialize isotherms 
        isoLoading_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
        objectiveFunction = np.zeros([len(zlcFileName)])

        # Loop over all available ZLC files for a given porosity
        for kk in range(len(zlcFileName)):
            # ZLC Data 
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            parameterReference = load(parameterPath)["parameterReference"]
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
            modelNonDim = modelOutputTemp[()]["variable"] 
            # Multiply the paremeters by the reference values
            x_ZLC = np.multiply(modelNonDim,parameterReference)    
    
            # Parse out the isotherm parameter
            isothermModel = x_ZLC[0:-2]
        
            for jj in range(len(temperature)):
                for ii in range(len(y)):
                    isoLoading_ZLC[kk,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                         moleFrac = y[ii], 
                                                                         temperature = temperature[jj]) # [mol/kg]
    
        # Plot the isotherms    
        ax1 = plt.subplot(1,3,1)
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))
        for jj in range(len(temperature)):
            ax1.plot(y,isoLoading_ZLC[int(minJ[0]),:,jj],color='#'+colorsForPlot[jj],
                     linestyle = lineForPlot[pp],alpha = alphaForPlot[pp],
                     dash_capstyle = 'round',)  # Lowest J
        if pp == 0:
            # Isotherm
            ax1.set(xlabel = '$P$ [bar]', 
                    ylabel='$q^*_\mathregular{CO_2}$ [mol kg$^{-1}$]',
                    xlim = [0,1], ylim = [0, 3])
            ax1.text(0.04, 2.75, "(a)", fontsize=8,)
            ax1.text(0.70, 0.15, "Porosity", fontsize=8, fontweight = 'bold',color = 'k')
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            # ax1.axes.xaxis.set_ticklabels([])
            ax1.legend(custom_lines, porosityLabel)

    # Effect of mass            
    for pp in range(len(massALL)):
        zlcFileName = massALL[pp]

        # Initialize isotherms 
        isoLoading_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
        objectiveFunction = np.zeros([len(zlcFileName)])

        # Loop over all available ZLC files for a given mass
        for kk in range(len(zlcFileName)):
            # ZLC Data 
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            parameterReference = load(parameterPath)["parameterReference"]
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
            modelNonDim = modelOutputTemp[()]["variable"] 
            # Multiply the paremeters by the reference values
            x_ZLC = np.multiply(modelNonDim,parameterReference)    
    
            # Parse out the isotherm parameter
            isothermModel = x_ZLC[0:-2]
        
            for jj in range(len(temperature)):
                for ii in range(len(y)):
                    isoLoading_ZLC[kk,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                         moleFrac = y[ii], 
                                                                         temperature = temperature[jj]) # [mol/kg]
    
        # Plot the isotherms    
        ax2 = plt.subplot(1,3,2)
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))
        for jj in range(len(temperature)):
            if pp == 1:
                ax2.plot(y,isoLoading_ZLC[int(minJ[0]),:,jj],color='#'+colorsForPlot[jj],
                         linestyle = lineForPlot[pp],alpha = alphaForPlot[pp],
                         dash_capstyle = 'round',
                         label = str(temperature[jj]) + ' K')  # Lowest J
            else:
                ax2.plot(y,isoLoading_ZLC[int(minJ[0]),:,jj],color='#'+colorsForPlot[jj],
                     linestyle = lineForPlot[pp],alpha = alphaForPlot[pp],
                     dash_capstyle = 'round',)  # Lowest J
        if pp == 0:
            # Isotherm
            ax2.set(xlabel = '$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 3])
            ax2.text(0.04, 2.75, "(b)", fontsize=8,)
            ax2.text(0.79, 0.15, "Mass", fontsize=8, fontweight = 'bold',color = 'k')
            ax2.locator_params(axis="x", nbins=4)
            ax2.locator_params(axis="y", nbins=4)
            ax2.legend(custom_lines, massLabel)

    # Effect of dead volume            
    for pp in range(len(deadALL)):
        zlcFileName = deadALL[pp]

        # Initialize isotherms 
        isoLoading_ZLC = np.zeros([len(zlcFileName),len(y),len(temperature)])
        objectiveFunction = np.zeros([len(zlcFileName)])

        # Loop over all available ZLC files for a given DV model
        for kk in range(len(zlcFileName)):
            # ZLC Data 
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            parameterReference = load(parameterPath)["parameterReference"]
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
            modelNonDim = modelOutputTemp[()]["variable"] 
            # Multiply the paremeters by the reference values
            x_ZLC = np.multiply(modelNonDim,parameterReference)    
    
            # Parse out the isotherm parameter
            isothermModel = x_ZLC[0:-2]
        
            for jj in range(len(temperature)):
                for ii in range(len(y)):
                    isoLoading_ZLC[kk,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                         moleFrac = y[ii], 
                                                                         temperature = temperature[jj]) # [mol/kg]
    
        # Plot the isotherms    
        ax3 = plt.subplot(1,3,3)
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))

        for jj in range(len(temperature)):
            ax3.plot(y,isoLoading_ZLC[int(minJ[0]),:,jj],color='#'+colorsForPlot[jj],
                     linestyle = lineForPlot_DV[pp],alpha = alphaForPlot_DV[pp],
                     dash_capstyle = 'round',)  # Lowest J
        if pp == 0:
            # Isotherm
            ax3.set(xlabel = '$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 3])
            ax3.text(0.04, 2.75, "(c)", fontsize=8,)
            ax3.text(0.53, 0.15, "Blank Volume", fontsize=8, fontweight = 'bold',color = 'k')
            ax3.locator_params(axis="x", nbins=4)
            ax3.locator_params(axis="y", nbins=4)
            ax3.legend(custom_linesDV, deadLabel, loc = 'upper right')
 
        # Temperature legend
        if pp == 1:
            fig.legend(bbox_to_anchor=(0.3,0.93,0.4,0.1), mode="expand", ncol=3, borderaxespad=0)   
 
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
 
    #  Save the figure
    if saveFlag:
        # FileName: figureSensitivity_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureSensitivity_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath, bbox_inches = "tight")
 
    plt.show()
    
    
# fun: plotForArticle_figureDVSensitivity
# Plots the Figure DV Sensitivity of the manuscript: Dead volume characterization different models
def plotForArticle_figureDVSensitivity(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    from deadVolumeWrapper import deadVolumeWrapper
    from extractDeadVolume import filesToProcess # File processing script
    from numpy import load
    import os
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    from matplotlib.lines import Line2D
    import auxiliaryFunctions
    plt.style.use('singleColumn.mplstyle') # Custom matplotlib style file
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # File with parameter estimates
    fileParameterALL = ['deadVolumeCharacteristics_20211002_1307_c8173b1.npz', # TIS
                        'deadVolumeCharacteristics_20211026_0025_c8173b1.npz', # TIS + M/D
                        'deadVolumeCharacteristics_20210810_1653_eddec53.npz',] # TIS + M/D + MS

    # Flag to plot simulations
    simulateModel = True

    # Plot colors and markers
    colorsForPlot = ["03045e","0077b6","00b4d8","90e0ef"]
    markersForPlot = ["^",">","v","<"]

    # Line style and alpha    
    alphaForPlot_DV = [0.5, 0.5, 1.0]
    lineForPlot_DV = [':', '--', '-']

    # Dead Volume
    deadLabel = ['TIS',
                 'TIS + D/M',
                 'TIS + D/M + MS']

    # Custom Legend Lines
    custom_lines = [Line2D([0], [0], linestyle=lineForPlot_DV[0], lw=1, dash_capstyle = 'round', alpha = alphaForPlot_DV[0], color = 'k'),
                    Line2D([0], [0], linestyle=lineForPlot_DV[1], lw=1, dash_capstyle = 'round', alpha = alphaForPlot_DV[1], color = 'k'),
                    Line2D([0], [0], linestyle=lineForPlot_DV[2], lw=1, dash_capstyle = 'round', alpha = alphaForPlot_DV[2], color = 'k')]

    fig = figure(figsize=(3.3,2.65))
    # Loop over all the files
    for kk in range(len(fileParameterALL)):
        fileParameter = fileParameterALL[kk] # Parse out the parameter estimate name
        # Dead volume parameter model path
        parameterPath = os.path.join('..','simulationResults',fileParameter)
        # Load file names and the model
        fileNameList = load(parameterPath, allow_pickle=True)["fileName"]

        # Generate .npz file for python processing of the .mat file 
        filesToProcess(True,os.path.join('..','experimental','runData'),fileNameList,'DV')
        # Get the processed file names
        fileName = filesToProcess(False,[],[],'DV')
        # Load the model
        modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
        x = modelOutputTemp[()]["variable"]
        print(x)

        # Get the MS fit flag, flow rates and msDeadVolumeFile (if needed)
        # Check needs to be done to see if MS file available or not
        # Checked using flagMSDeadVolume in the saved file
        dvFileLoadTemp = load(parameterPath)
        if 'flagMSDeadVolume' in dvFileLoadTemp.files:
            flagMSFit = dvFileLoadTemp["flagMSFit"]
            msFlowRate = dvFileLoadTemp["msFlowRate"]
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
        else:
            flagMSFit = False
            msFlowRate = -np.inf
            flagMSDeadVolume = False
            msDeadVolumeFile = []
        
        numPointsExp = np.zeros(len(fileName))
        for ii in range(len(fileName)): 
            fileToLoad = fileName[ii]
            # Load experimental molefraction
            timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
            numPointsExp[ii] = len(timeElapsedExp)
        
        # Downsample intervals
        downsampleInt = numPointsExp/np.min(numPointsExp)
    
        # Print the objective function and volume from model parameters
        print("Model Volume",round(sum(x[0:2]),2))
        moleFracExpALL = np.array([])
        moleFracSimALL = np.array([])
    
        # Create the instance for the plots
        ax1 = plt.subplot(1,1,1)
        
        # Initialize error for objective function
        # Loop over all available files    
        for ii in range(len(fileName)):
            # Initialize outputs
            moleFracSim = []
            # Path of the file name
            fileToLoad = fileName[ii]   
            # Load experimental time, molefraction and flowrate (accounting for downsampling)
            timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
            moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
            flowRateTemp = load(fileToLoad)["flowRate"].flatten()
            timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
            moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
            flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
            # Get the flow rates from the fit file
            # When MS used
            if flagMSFit:
                flowRateDV = msFlowRate
            else:
                flowRateDV = np.mean(flowRateExp[-1:-10:-1])
            
            # Integration and ode evaluation time
            timeInt = timeElapsedExp
            
            if simulateModel:    
                # Call the deadVolume Wrapper function to obtain the outlet mole fraction
                moleFracSim = deadVolumeWrapper(timeInt, flowRateDV, x, flagMSDeadVolume, msDeadVolumeFile)
                       
                # Stack mole fraction from experiments and simulation for error 
                # computation
                minExp = np.min(moleFracExp) # Compute the minimum from experiment
                normalizeFactor = np.max(moleFracExp - minExp) # Compute the max from normalized data
                moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
                moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))
                
            # Plot the expreimental and model output
            # Log scale
            if kk == 0:
                ax1.semilogy(timeElapsedExp,moleFracExp,
                              marker = markersForPlot[ii],linewidth = 0,
                              color='#'+colorsForPlot[ii],alpha=0.25,label=str(round(abs(np.mean(flowRateExp)),1))+" cm$^3$ s$^{-1}$") # Experimental response
            ax1.semilogy(timeElapsedExp,moleFracSim,
                              color='#'+colorsForPlot[ii],
                              linestyle = lineForPlot_DV[kk],
                              alpha = alphaForPlot_DV[kk],
                              dash_capstyle = 'round',) # Simulation response
            ax1.set(xlabel='$t$ [s]', 
                    ylabel='$y\mathregular{_{CO_2}}$ [-]',
                    xlim = [0,150], ylim = [1e-2, 1])
            ax1.locator_params(axis="x", nbins=5)
            ax1.legend(handletextpad=0.0)
            ax1.grid(which='minor', linestyle=':')

        # Model legend
        fig.legend(custom_lines,deadLabel,bbox_to_anchor=(0.07,0.93,0.9,0.1), mode="expand", ncol=3, borderaxespad=0)   
            
        # Remove all the .npz files genereated from the .mat
        # Loop over all available files    
        for ii in range(len(fileName)):
            os.remove(fileName[ii])

    #  Save the figure
    if saveFlag:
        # FileName: figureDVSensitivity<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureDVSensitivity_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath, bbox_inches = "tight")
 
    plt.show()
    
    
# fun: plotForArticle_figureFt
# Plots the Figure Ftof the manuscript: Ft plots for parameter estimates
def plotForArticle_figureFt(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):    
    from simulateCombinedModel import simulateCombinedModel
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure    
    import scipy.io as sio    
    from numpy import load
    from matplotlib.lines import Line2D  
    os.chdir(".."+os.path.sep+"plotFunctions")
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    os.chdir(".."+os.path.sep+"experimental")
    
    # Move to top level folder (to avoid path issues)    
    os.chdir("..")
    import auxiliaryFunctions    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    os.chdir("experimental")
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Temperature of the simulate experiment [K]
    temperature = 308.15
    
    # Inlet flow rate [ccm]
    flowRate = [10, 60]
    
    # Saturation mole fraction (works for a binary system)
    initMoleFrac = np.array(([0.11, 0.94], [0.11, 0.94]))
    
    # Parameter estimate files
                        # Activated Carbon Experiments
    zlcFileNameALL = [['zlcParameters_20210822_0926_c8173b1.npz',
                       'zlcParameters_20210822_1733_c8173b1.npz',
                       # 'zlcParameters_20210823_0133_c8173b1.npz', # DSL BAD (but lowest J)
                       # 'zlcParameters_20210823_1007_c8173b1.npz', # DSL BAD (but lowest J)
                       'zlcParameters_20210823_1810_c8173b1.npz'],
                        # Boron Nitride Experiments
                        ['zlcParameters_20210823_1731_c8173b1.npz',
                          'zlcParameters_20210824_0034_c8173b1.npz',
                          'zlcParameters_20210824_0805_c8173b1.npz',
                          'zlcParameters_20210824_1522_c8173b1.npz',
                          'zlcParameters_20210824_2238_c8173b1.npz',],
                          # Zeolite 13X Experiments
                        ['zlcParameters_20210824_1552_6b88505.npz',
                          'zlcParameters_20210825_0559_6b88505.npz',
                          'zlcParameters_20210825_1854_6b88505.npz',
                          'zlcParameters_20210826_0847_6b88505.npz',
                          'zlcParameters_20210827_0124_6b88505.npz',]]
    
    # Create the instance for the plots
    fig = plt.figure
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)
    
    # Plot colors
    colorsForPlot = ["#ef233c","#8d99ae"]*2
    styleForPlot = [":","-"]*2
    alphaForPlot = [0.5,1.0]*2
    
    # Flow labels
    flowStr = [str(int(flowRate[0]))+ " cm$^3$ min$^{-1}$",
               str(int(flowRate[1]))+ " cm$^3$ min$^{-1}$"]
    
    # Legend labels
    legendStr = ["$y^\mathregular{in}$ = " + str(initMoleFrac[0,0]),
                 "$y^\mathregular{in}$ = " + str(initMoleFrac[0,1])]
    
    # Custom Legend Lines
    custom_lines = [Line2D([0], [0], linestyle=':', lw=1, dash_capstyle = 'round', alpha = alphaForPlot[0], color = 'k'),
                    Line2D([0], [0], linestyle='-', lw=1, dash_capstyle = 'round', alpha = alphaForPlot[1], color = 'k')]
    
    
    # Loop over all materials
    for pp in range(len(zlcFileNameALL)): 
        zlcFileName = zlcFileNameALL[pp]
        objectiveFunction = np.zeros([len(zlcFileName)])
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # Obtain the onjective function values
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)
    
        # Find the experiment with the min objective function
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))
        fileParameter = zlcFileName[int(minJ[0])]
        
        # ZLC parameter model path
        parameterPath = os.path.join('..','simulationResults',fileParameter)
           
        # Parse out experiments names and temperature used for the fitting
        rawFileName = load(parameterPath)["fileName"]
        temperatureExp = load(parameterPath)["temperature"]
    
        # Parse out all the necessary quantities to obtain model fit
        # Mass of sorbent and particle epsilon
        adsorbentDensity = load(parameterPath)["adsorbentDensity"]
        particleEpsilon = load(parameterPath)["particleEpsilon"]
        massSorbent = load(parameterPath)["massSorbent"]
        # Volume of sorbent material [m3]
        volSorbent = (massSorbent/1000)/adsorbentDensity
        # Volume of gas chamber (dead volume) [m3]
        volGas = volSorbent/(1-particleEpsilon)*particleEpsilon
        # Dead volume model
        deadVolumeFile = str(load(parameterPath)["deadVolumeFile"])
        # Isotherm parameter reference
        parameterReference = load(parameterPath)["parameterReference"]
        # Load the model
        modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
        modelNonDim = modelOutputTemp[()]["variable"] 
        # Multiply the paremeters by the reference values
        x = np.multiply(modelNonDim,parameterReference)
        # Integration time (set to 1000 s, default)
        timeInt = (0.0,1000.0)
        
        # Loop over all the conditions
        for ii in range(len(flowRate)):
            for jj in range(np.size(initMoleFrac,1)):
                # Initialize the output dictionary
                experimentOutput = {}
                # Compute the composite response using the optimizer parameters
                timeElapsedSim , _ , resultMat = simulateCombinedModel(isothermModel = x[0:-2],
                                                                                 rateConstant_1 = x[-2], # Last but one element is rate constant (analogous to micropore)
                                                                                 rateConstant_2 = x[-1], # Last element is activation energy (analogous to macropore)
                                                                                 temperature = temperature, # Temperature [K]
                                                                                 timeInt = timeInt,
                                                                                 initMoleFrac = [initMoleFrac[ii,jj]], # Initial mole fraction assumed to be the first experimental point
                                                                                 flowIn = flowRate[ii]*1e-6/60, # Flow rate [m3/s] for ZLC considered to be the mean of last 10 points (equilibrium)
                                                                                 expFlag = False,
                                                                                 deadVolumeFile = str(deadVolumeFile),
                                                                                 volSorbent = volSorbent,
                                                                                 volGas = volGas,
                                                                                 adsorbentDensity = adsorbentDensity)
                
                # Find the index that corresponds to 1e-2 (to be consistent with the 
                # experiments)
                lastIndThreshold = int(np.argwhere(resultMat[0,:]<=1e-2)[0])
                # Cut the time, mole fraction and the flow rate to the last index
                # threshold
                timeExp = timeElapsedSim[0:lastIndThreshold] # Time elapsed [s]
                moleFrac = resultMat[0,0:lastIndThreshold] # Mole fraction [-]
                totalFlowRate = resultMat[3,0:lastIndThreshold]*1e6 # Total flow rate[ccs]
                
                # Acticated Carbon
                if pp == 0:
                    # Ft - Log scale        
                    ax1.semilogy(np.multiply(totalFlowRate,timeExp),moleFrac,
                                  color=colorsForPlot[ii],linestyle=styleForPlot[jj],
                                  dash_capstyle = 'round',alpha=alphaForPlot[jj])
                    ax1.set(xlabel='$Ft$ [cm$^3$]', ylabel='$y\mathregular{_{CO_2}}$ [-]',
                            xlim = [0,60], ylim = [1e-2, 1])         
                    ax1.locator_params(axis="x", nbins=4)
                    ax1.legend(custom_lines, legendStr, loc = 'upper right')
                    ax1.text(2, 0.67, "(a)", fontsize=8,)
                    if ii == 0:
                        ax1.text(5, 0.2, flowStr[ii], fontsize=8, color=colorsForPlot[ii])
                    if ii == 1:
                        ax1.text(30, 0.03, flowStr[ii], fontsize=8, color=colorsForPlot[ii])
                    ax1.text(28, 1.3, "AC", fontsize=8, fontweight = 'bold',color = 'k')
                    ax1.grid(which='minor', linestyle=':')
                # Boron Nitride
                if pp == 1:
                    # Ft - Log scale        
                    ax2.semilogy(np.multiply(totalFlowRate,timeExp),moleFrac,
                                  color=colorsForPlot[ii],linestyle=styleForPlot[jj],
                                  dash_capstyle = 'round',alpha=alphaForPlot[jj])
                    ax2.set(xlabel='$Ft$ [cm$^3$]', 
                            xlim = [0,60], ylim = [1e-2, 1])         
                    ax2.locator_params(axis="x", nbins=4)
                    ax2.legend(custom_lines, legendStr, loc = 'upper right')
                    ax2.text(2, 0.67, "(b)", fontsize=8,)
                    if ii == 0:
                        ax2.text(5, 0.2, flowStr[ii], fontsize=8, color=colorsForPlot[ii])
                    if ii == 1:
                        ax2.text(22, 0.03, flowStr[ii], fontsize=8, color=colorsForPlot[ii])
                    ax2.text(28, 1.3, "BN", fontsize=8, fontweight = 'bold',color = 'k')
                    ax2.grid(which='minor', linestyle=':')
                # Zeolite 13X
                if pp == 2:
                    # Ft - Log scale        
                    ax3.semilogy(np.multiply(totalFlowRate,timeExp),moleFrac,
                                  color=colorsForPlot[ii],linestyle=styleForPlot[jj],
                                  dash_capstyle = 'round',alpha=alphaForPlot[jj])
                    ax3.set(xlabel='$Ft$ [cm$^3$]', 
                            xlim = [0,150], ylim = [1e-2, 1])         
                    ax3.locator_params(axis="x", nbins=4)
                    ax3.legend(custom_lines, legendStr, loc = 'upper right')
                    ax3.text(2*150/60, 0.67, "(c)", fontsize=8,)
                    if ii == 0:
                        ax3.text(10, 0.2, flowStr[ii], fontsize=8, color=colorsForPlot[ii])
                    if ii == 1:
                        ax3.text(65, 0.03, flowStr[ii], fontsize=8, color=colorsForPlot[ii])
                    ax3.text(67, 1.3, "13X", fontsize=8, fontweight = 'bold',color = 'k')
                    ax3.grid(which='minor', linestyle=':')
                    
    #  Save the figure
    if saveFlag:
        # FileName: figureFt_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureFt_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath, bbox_inches = "tight")
 
    plt.show()

# fun: plotForArticle_figureZLCFitALL
# Plots the Figure ZLC Fit of the manuscript: ZLC goodness of fit for experimental results
def plotForArticle_figureZLCFitALL(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure    
    import auxiliaryFunctions
    from numpy import load
    import os
    from simulateCombinedModel import simulateCombinedModel
    from deadVolumeWrapper import deadVolumeWrapper
    from extractDeadVolume import filesToProcess # File processing script
    from matplotlib.lines import Line2D
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
        
    # Plot colors and markers
    colorsForPlot = ["ffba08","d00000","03071e"]
    markersForPlot = ["^","d","v"]    
    
    # X limits for the different materials
    XLIM_L = [[0, 200],[0, 150],[0, 600]]
    XLIM_H = [[0, 60],[0, 40],[0, 150]]
    
    # Label positions for the different materials
    panelLabel_L = [170, 150/200*170, 600/200*170]
    panelLabel_H = [60/200*170, 60/200*40/60*170, 60/200*150/60*170]

    # Parameter estimate files
                        # Activated Carbon Experiments
    zlcFileNameALL = [['zlcParameters_20210822_0926_c8173b1.npz',
                       'zlcParameters_20210822_1733_c8173b1.npz',
                       # 'zlcParameters_20210823_0133_c8173b1.npz', # DSL BAD (but lowest J)
                       # 'zlcParameters_20210823_1007_c8173b1.npz', # DSL BAD (but lowest J)
                       'zlcParameters_20210823_1810_c8173b1.npz'],
                        # Boron Nitride Experiments
                        ['zlcParameters_20210823_1731_c8173b1.npz',
                          'zlcParameters_20210824_0034_c8173b1.npz',
                          'zlcParameters_20210824_0805_c8173b1.npz',
                          'zlcParameters_20210824_1522_c8173b1.npz',
                          'zlcParameters_20210824_2238_c8173b1.npz',],
                          # Zeolite 13X Experiments
                        ['zlcParameters_20210824_1552_6b88505.npz',
                          'zlcParameters_20210825_0559_6b88505.npz',
                          'zlcParameters_20210825_1854_6b88505.npz',
                          'zlcParameters_20210826_0847_6b88505.npz',
                          'zlcParameters_20210827_0124_6b88505.npz',]]
    
    fig = figure(figsize=(6.5,5))     
    for pp in range(len(zlcFileNameALL)):
 
        zlcFileName = zlcFileNameALL[pp]
        objectiveFunction = np.zeros([len(zlcFileName)])
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # Obtain the onjective function values
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)

        # Find the experiment with the min objective function
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))
        fileParameter = zlcFileName[int(minJ[0])]
        
        # ZLC parameter model path
        parameterPath = os.path.join('..','simulationResults',fileParameter)
           
        # Parse out experiments names and temperature used for the fitting
        rawFileName = load(parameterPath)["fileName"]
        temperatureExp = load(parameterPath)["temperature"]

        # Generate .npz file for python processing of the .mat file 
        filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'ZLC')
        # Get the processed file names
        fileName = filesToProcess(False,[],[],'ZLC')
        
        numPointsExp = np.zeros(len(fileName))
        for ii in range(len(fileName)): 
            fileToLoad = fileName[ii]
            # Load experimental molefraction
            timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
            numPointsExp[ii] = len(timeElapsedExp)
        
        # Parse out all the necessary quantities to obtain model fit
        # Mass of sorbent and particle epsilon
        adsorbentDensity = load(parameterPath)["adsorbentDensity"]
        particleEpsilon = load(parameterPath)["particleEpsilon"]
        massSorbent = load(parameterPath)["massSorbent"]
        # Volume of sorbent material [m3]
        volSorbent = (massSorbent/1000)/adsorbentDensity
        # Volume of gas chamber (dead volume) [m3]
        volGas = volSorbent/(1-particleEpsilon)*particleEpsilon
        # Dead volume model
        deadVolumeFile = str(load(parameterPath)["deadVolumeFile"])
        # Isotherm parameter reference
        parameterReference = load(parameterPath)["parameterReference"]
        # Load the model
        modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
        modelNonDim = modelOutputTemp[()]["variable"] 
        # Multiply the paremeters by the reference values
        x = np.multiply(modelNonDim,parameterReference)
        print(x)
        # Downsample intervals
        downsampleInt = numPointsExp/np.min(numPointsExp)
        
        # Initialize loadings
        moleFracExpALL = np.array([])
        moleFracSimALL = np.array([])

        # Loop over all available files    
        for ii in range(len(fileName)):
            fileToLoad = fileName[ii]   
            
            # Initialize outputs
            moleFracSim = []  
            # Load experimental time, molefraction and flowrate (accounting for downsampling)
            timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
            moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
            flowRateTemp = load(fileToLoad)["flowRate"].flatten()
            timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
            moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
            flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
                    
            # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
            timeInt = timeElapsedExp

            # Parse out parameter values
            isothermModel = x[0:-2]
            rateConstant_1 = x[-2]
            rateConstant_2 = x[-1]
                    
            # Compute the dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                                initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                                flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                                expFlag = True,
                                                                isothermModel = isothermModel,
                                                                rateConstant_1 = rateConstant_1,
                                                                rateConstant_2 = rateConstant_2,
                                                                deadVolumeFile = deadVolumeFile,
                                                                volSorbent = volSorbent,
                                                                volGas = volGas,
                                                                temperature = temperatureExp[ii],
                                                                adsorbentDensity = adsorbentDensity)
            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
                                                                  moleFracSim),
                                                        timeElapsedExp),2))

            # Stack mole fraction from experiments and simulation for error 
            # computation
            minExp = np.min(moleFracExp) # Compute the minimum from experiment
            normalizeFactor = np.max(moleFracExp - np.min(moleFracExp)) # Compute the max from normalized data
            moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
            moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))

            # Call the deadVolume Wrapper function to obtain the outlet mole fraction
            deadVolumePath = os.path.join('..','simulationResults',deadVolumeFile)
            modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
            pDV = modelOutputTemp[()]["variable"]
            dvFileLoadTemp = load(deadVolumePath)
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
            moleFracDV = deadVolumeWrapper(timeInt, resultMat[3,:]*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = [moleFracExp[0]])
    
            if 300<temperatureExp[ii] and temperatureExp[ii]<310:
                colorTemp = colorsForPlot[0]
                markersTemp =markersForPlot[0]
            elif 320<temperatureExp[ii] and temperatureExp[ii]<330:
                colorTemp = colorsForPlot[1]
                markersTemp =markersForPlot[1]
            elif 340<temperatureExp[ii] and temperatureExp[ii]<350:
                colorTemp = colorsForPlot[2]
                markersTemp =markersForPlot[2]
    
            if ii in range(0,3):                    
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax1 = plt.subplot(3,4,4*pp+1)
                ax1.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax1.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax1.semilogy(timeElapsedExp,moleFracDV,
                                 color='#76c893',alpha = 0.4, linestyle='-') # Dead volume response
                ax1.set(xlabel='$t$ [s]',ylabel='$y\mathregular{_{CO_2}}$ [-]',
                        xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax1.locator_params(axis="x", nbins=4)
                ax1.grid(which='minor', linestyle=':')

            if ii in range(3,6):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax2 = plt.subplot(3,4,4*pp+2)
                ax2.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax2.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response
                if ii%3 == 0:
                    ax2.semilogy(timeElapsedExp,moleFracDV,
                                 color='#76c893',alpha = 0.4, linestyle='-') # Dead volume response
                ax2.set(xlabel='$t$ [s]',
                        xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax2.locator_params(axis="x", nbins=4)
                ax2.grid(which='minor', linestyle=':')
                ax2.axes.yaxis.set_ticklabels([])

            if ii in range(6,9):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax3 = plt.subplot(3,4,4*pp+3)
                ax3.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax3.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response  
                if ii%3 == 0:
                    ax3.semilogy(timeElapsedExp,moleFracDV,
                                 color='#76c893',alpha = 0.4, linestyle='-') # Dead volume response
                ax3.set(xlabel='$t$ [s]',xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                ax3.locator_params(axis="x", nbins=4)
                ax3.locator_params(axis="x", nbins=4)
                ax3.axes.yaxis.set_ticklabels([])
                ax3.grid(which='minor', linestyle=':')
                
            if ii in range(9,12):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax4 = plt.subplot(3,4,4*pp+4)
                ax4.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax4.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax4.semilogy(timeElapsedExp,moleFracDV,
                                 color='#76c893',alpha = 0.4, linestyle='-') # Dead volume response
                ax4.set(xlabel='$t$ [s]',xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                ax4.locator_params(axis="x", nbins=4)
                ax4.grid(which='minor', linestyle=':')
                ax4.axes.yaxis.set_ticklabels([])

        # Put panel labels
        ax1.text(panelLabel_L[pp], 0.6, '('+chr(96+4*pp+1)+')', fontsize=8,)
        ax2.text(panelLabel_L[pp], 0.6, '('+chr(96+4*pp+2)+')', fontsize=8,)
        ax3.text(panelLabel_H[pp], 0.6, '('+chr(96+4*pp+3)+')', fontsize=8,)
        ax4.text(panelLabel_H[pp], 0.6, '('+chr(96+4*pp+4)+')', fontsize=8,)

        # Remove all the .npz files genereated from the .mat
        # Loop over all available files    
        for ii in range(len(fileName)):
            os.remove(fileName[ii])

    # Put other text entries
    plt.figtext(0.23, 0.98, "$F^\mathregular{in}$ = 10 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')
    plt.figtext(0.685, 0.98, "$F^\mathregular{in}$ = 60 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')
    plt.figtext(-0.02, 0.83, "AC", fontsize=8, fontweight = 'bold',color = '#4895EF')
    plt.figtext(-0.02, 0.51, "BN", fontsize=8, fontweight = 'bold',color = '#4895EF')
    plt.figtext(-0.02, 0.19, "13X", fontsize=8, fontweight = 'bold',color = '#4895EF')
    
    # Dead Volume
    tempLabel = ['306 K','325 K', '345 K']

    # Custom Legend Lines
    custom_lines = [Line2D([0], [0], linestyle='-', lw=1, color = '#ffba08'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#d00000'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#03071e'),]
    
    fig.legend(custom_lines,tempLabel,bbox_to_anchor=(0.04,0.95,0.675,0.1), 
                   ncol=3, borderaxespad=0)   

    #  Save the figure
    if saveFlag:
        # FileName: figureZLCALL_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureZLCFitALL_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath,bbox_inches='tight')
    
    plt.show()
    
# fun: plotForArticle_figureZLCSimFit
# Plots the Figure ZLC Fit of the manuscript: ZLC goodness for computational results
def plotForArticle_figureZLCSimFitALL(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import auxiliaryFunctions
    from numpy import load
    import os
    from simulateCombinedModel import simulateCombinedModel
    from deadVolumeWrapper import deadVolumeWrapper
    from extractDeadVolume import filesToProcess # File processing script
    from matplotlib.lines import Line2D
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
        
    # Plot colors and markers
    colorsForPlot = ["0091ad","5c4d7d","b7094c"]
    markersForPlot = ["^","d","v"]    
    
    # X limits for the different materials
    XLIM_L = [[0, 200],[0, 150],[0, 600]]
    XLIM_H = [[0, 60],[0, 40],[0, 150]]
    
    # Label positions for the different materials
    panelLabel_L = [170, 150/200*170, 600/200*170]
    panelLabel_H = [60/200*170, 60/200*40/60*170, 60/200*150/60*170]
    
    # Parameter estimate files
                        # Activated Carbon Simulations
    zlcFileNameALL = [['zlcParameters_20210823_1104_03c82f4.npz',
                       'zlcParameters_20210824_0000_03c82f4.npz',
                       'zlcParameters_20210824_1227_03c82f4.npz',
                       'zlcParameters_20210825_0017_03c82f4.npz',
                       'zlcParameters_20210825_1151_03c82f4.npz'],
                      # Boron Nitride Simulations
                      ['zlcParameters_20210823_1907_03c82f4.npz',
                       'zlcParameters_20210824_0555_03c82f4.npz',
                       'zlcParameters_20210824_2105_03c82f4.npz',
                       'zlcParameters_20210825_0833_03c82f4.npz',
                       'zlcParameters_20210825_2214_03c82f4.npz'],
                      # Zeolite 13X Simulations
                      ['zlcParameters_20210824_1102_c8173b1.npz',
                       'zlcParameters_20210825_0243_c8173b1.npz',
                       'zlcParameters_20210825_1758_c8173b1.npz',
                       'zlcParameters_20210826_1022_c8173b1.npz',
                       'zlcParameters_20210827_0104_c8173b1.npz']]

    fig = figure(figsize=(6.5,5))  
    for pp in range(len(zlcFileNameALL)):         
        zlcFileName = zlcFileNameALL[pp]
        objectiveFunction = np.zeros([len(zlcFileName)])
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # Obtain the onjective function values
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)

        # Find the experiment with the min objective function
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))
        fileParameter = zlcFileName[int(minJ[0])]
        
        # ZLC parameter model path
        parameterPath = os.path.join('..','simulationResults',fileParameter)
           
        # Parse out experiments names and temperature used for the fitting
        rawFileName = load(parameterPath)["fileName"]
        temperatureExp = load(parameterPath)["temperature"]

        # Generate .npz file for python processing of the .mat file 
        filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'ZLC')
        # Get the processed file names
        fileName = filesToProcess(False,[],[],'ZLC')
        
        numPointsExp = np.zeros(len(fileName))
        for ii in range(len(fileName)): 
            fileToLoad = fileName[ii]
            # Load experimental molefraction
            timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
            numPointsExp[ii] = len(timeElapsedExp)
        
        # Parse out all the necessary quantities to obtain model fit
        # Mass of sorbent and particle epsilon
        adsorbentDensity = load(parameterPath)["adsorbentDensity"]
        particleEpsilon = load(parameterPath)["particleEpsilon"]
        massSorbent = load(parameterPath)["massSorbent"]
        # Volume of sorbent material [m3]
        volSorbent = (massSorbent/1000)/adsorbentDensity
        # Volume of gas chamber (dead volume) [m3]
        volGas = volSorbent/(1-particleEpsilon)*particleEpsilon
        # Dead volume model
        deadVolumeFile = str(load(parameterPath)["deadVolumeFile"])
        # Isotherm parameter reference
        parameterReference = load(parameterPath)["parameterReference"]
        # Load the model
        modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
        modelNonDim = modelOutputTemp[()]["variable"] 
        # Multiply the paremeters by the reference values
        x = np.multiply(modelNonDim,parameterReference)
        # Downsample intervals
        downsampleInt = numPointsExp/np.min(numPointsExp)
        
        # Initialize loadings
        moleFracExpALL = np.array([])
        moleFracSimALL = np.array([])

        # Loop over all available files    
        for ii in range(len(fileName)):
            fileToLoad = fileName[ii]   
            
            # Initialize outputs
            moleFracSim = []  
            # Load experimental time, molefraction and flowrate (accounting for downsampling)
            timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
            moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
            flowRateTemp = load(fileToLoad)["flowRate"].flatten()
            timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
            moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
            flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
                    
            # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
            timeInt = timeElapsedExp

            # Parse out parameter values
            isothermModel = x[0:-2]
            rateConstant_1 = x[-2]
            rateConstant_2 = x[-1]
                    
            # Compute the dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                                initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                                flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                                expFlag = True,
                                                                isothermModel = isothermModel,
                                                                rateConstant_1 = rateConstant_1,
                                                                rateConstant_2 = rateConstant_2,
                                                                deadVolumeFile = deadVolumeFile,
                                                                volSorbent = volSorbent,
                                                                volGas = volGas,
                                                                temperature = temperatureExp[ii],
                                                                adsorbentDensity = adsorbentDensity)
            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
                                                                  moleFracSim),
                                                        timeElapsedExp),2))

            # Stack mole fraction from experiments and simulation for error 
            # computation
            minExp = np.min(moleFracExp) # Compute the minimum from experiment
            normalizeFactor = np.max(moleFracExp - np.min(moleFracExp)) # Compute the max from normalized data
            moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
            moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))

            # Call the deadVolume Wrapper function to obtain the outlet mole fraction
            deadVolumePath = os.path.join('..','simulationResults',deadVolumeFile)
            modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
            pDV = modelOutputTemp[()]["variable"]
            dvFileLoadTemp = load(deadVolumePath)
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
            moleFracDV = deadVolumeWrapper(timeInt, resultMat[3,:]*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = [moleFracExp[0]])
    
            if 300<temperatureExp[ii] and temperatureExp[ii]<310:
                colorTemp = colorsForPlot[0]
                markersTemp =markersForPlot[0]
            elif 320<temperatureExp[ii] and temperatureExp[ii]<330:
                colorTemp = colorsForPlot[1]
                markersTemp =markersForPlot[1]
            elif 340<temperatureExp[ii] and temperatureExp[ii]<350:
                colorTemp = colorsForPlot[2]
                markersTemp =markersForPlot[2]
     
            if ii in range(0,3):                    
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax1 = plt.subplot(3,4,4*pp+1)
                ax1.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax1.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax1.semilogy(timeElapsedExp,moleFracDV,
                                 color='k',alpha = 0.2, linestyle='-') # Dead volume response
                ax1.set(xlabel='$t$ [s]',ylabel='$y\mathregular{_{CO_2}}$ [-]',
                        xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax1.locator_params(axis="x", nbins=4)
                ax1.grid(which='minor', linestyle=':')

            if ii in range(3,6):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax2 = plt.subplot(3,4,4*pp+2)
                ax2.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax2.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response
                if ii%3 == 0:
                    ax2.semilogy(timeElapsedExp,moleFracDV,
                                 color='k',alpha = 0.2, linestyle='-') # Dead volume response
                ax2.set(xlabel='$t$ [s]',
                        xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax2.locator_params(axis="x", nbins=4)
                ax2.grid(which='minor', linestyle=':')
                ax2.axes.yaxis.set_ticklabels([])
                
            if ii in range(6,9):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax3 = plt.subplot(3,4,4*pp+3)
                ax3.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax3.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response  
                if ii%3 == 0:
                    ax3.semilogy(timeElapsedExp,moleFracDV,
                                 color='k',alpha = 0.2, linestyle='-') # Dead volume response
                ax3.set(xlabel='$t$ [s]',xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                ax3.locator_params(axis="x", nbins=4)
                ax3.locator_params(axis="x", nbins=4)
                ax3.axes.yaxis.set_ticklabels([])
                ax3.grid(which='minor', linestyle=':')
                
            if ii in range(9,12):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax4 = plt.subplot(3,4,4*pp+4)
                ax4.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.25) # Experimental response
                ax4.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax4.semilogy(timeElapsedExp,moleFracDV,
                                 color='k',alpha = 0.2, linestyle='-') # Dead volume response
                ax4.set(xlabel='$t$ [s]',xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                ax4.locator_params(axis="x", nbins=4)
                ax4.grid(which='minor', linestyle=':')
                ax4.axes.yaxis.set_ticklabels([])

        # Put panel labels
        ax1.text(panelLabel_L[pp], 0.6, '('+chr(96+4*pp+1)+')', fontsize=8,)
        ax2.text(panelLabel_L[pp], 0.6, '('+chr(96+4*pp+2)+')', fontsize=8,)
        ax3.text(panelLabel_H[pp], 0.6, '('+chr(96+4*pp+3)+')', fontsize=8,)
        ax4.text(panelLabel_H[pp], 0.6, '('+chr(96+4*pp+4)+')', fontsize=8,)
        
        # Remove all the .npz files genereated from the .mat
        # Loop over all available files    
        for ii in range(len(fileName)):
            os.remove(fileName[ii])

    # Put other text entries
    plt.figtext(0.23, 0.98, "$F^\mathregular{in}$ = 10 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#7d8597')
    plt.figtext(0.685, 0.98, "$F^\mathregular{in}$ = 60 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#7d8597')
    plt.figtext(-0.02, 0.83, "AC", fontsize=8, fontweight = 'bold',color = '#7d8597')
    plt.figtext(-0.02, 0.51, "BN", fontsize=8, fontweight = 'bold',color = '#7d8597')
    plt.figtext(-0.02, 0.19, "13X", fontsize=8, fontweight = 'bold',color = '#7d8597')
    
    # Dead Volume
    tempLabel = ['308 K','328 K', '348 K']

    # Custom Legend Lines
    custom_lines = [Line2D([0], [0], linestyle='-', lw=1, color = '#0091ad'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#5c4d7d'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#b7094c'),]
    
    fig.legend(custom_lines,tempLabel,bbox_to_anchor=(0.04,0.95,0.675,0.1), 
                   ncol=3, borderaxespad=0)
            
    #  Save the figure
    if saveFlag:
        # FileName: figureZLCSimFit_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureZLCSimFitALL_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath,bbox_inches='tight')
        
    plt.show()
    
    
# fun: plotForArticle_figureBNeqkin
# Plots the Equilibrium isotherms and kinetics for BNp, BNp-FAS, and BN-FASp
def plotForArticle_figureBNeqkin(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    import scipy.io as sio
    import os
    from matplotlib.ticker import FormatStrFormatter
    from computeEquilibriumLoading import computeEquilibriumLoading

    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()

    # Plot colors and markers (porosity)
    colorsForPlot_P = ["0fa3b1","f17300"]
    markersForPlot_P = ["^","v"]
    
    # Plot colors and markers (isotherm)    
    colorsForPlot_I = ["f27a18","c3351e","0f0b05"]
    markersForPlot_I = ["^","d","v"]


    # Main folder for material characterization
    mainDir = os.path.join('..','experimental','materialCharacterization')

    # Isotherm folder
    isothermDir = os.path.join('isothermData','isothermResults')
    
    # File with pore characterization data
    isothermALL = ['BNp_SSL_100621.mat',
                   'BNFASp_DSL_101922_101922.mat',
                   'BNpFAS_SSL_101922_101922.mat',]
    # Universal gas constant
    Rg = 8.314
    
    # Total pressure
    pressureTotal = np.array([1.e5]);
    
    # Define temperature
    temperature_k = [283.15, 293.15, 303.15]
    # Create the grid for mole fractions
    y = np.linspace(0,1.,100)
    # Rate constants from parameter estimation
    rateConstant_1ALL = [0.07, 922, 867]
    rateConstant_2ALL = [831.77, 603, 41]
    isothermModelALL = [[3.12, 11.62e-7,22.99e3,0 ,0, 0],
                        [9.44e-02, 2.94e-07, 3.43e4, 1.74, 5.61e-10, 4.10e4],
                        [1.13, 3.94e-08, 3.24e+04,0,0,0]]
    adsorbentDensityALL = [2040, 1250, 2320]
    isoLoading_ZLC = np.zeros([len(isothermALL),len(y),len(temperature_k)])
    kineticConstant_ZLC = np.zeros([len(isothermALL),len(y),len(temperature_k)])


        # Loop over all the isotherm files
    for pp in range(len(isothermALL)):
        adsorbentDensity = adsorbentDensityALL[pp]
        # Parse out the isotherm parameter
        isothermModel = isothermModelALL[pp]
        rateConstant_1 = rateConstant_1ALL[pp]
        rateConstant_2 = rateConstant_2ALL[pp]
        for jj in range(len(temperature_k)):
            for ii in range(len(y)):
                isoLoading_ZLC[pp,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                             moleFrac = y[ii], 
                                                                             temperature = temperature_k[jj]) # [mol/kg]
                # Partial pressure of the gas
                partialPressure = y[ii]*pressureTotal
                # delta pressure to compute gradient
                delP = 1e-3
                # Mole fraction (up)
                moleFractionUp = (partialPressure + delP)/pressureTotal
                # Compute the loading [mol/m3] @ moleFractionUp
                equilibriumLoadingUp  = computeEquilibriumLoading(temperature=temperature_k[jj],
                                                                moleFrac=moleFractionUp,
                                                                isothermModel=isothermModel) # [mol/kg]
                
                # Compute the gradient (delq*/dc)
                dqbydc = (equilibriumLoadingUp-isoLoading_ZLC[pp,ii,jj])*adsorbentDensity/(delP/(Rg*temperature_k[jj])) # [-]
    
                # Rate constant 1 (analogous to micropore resistance)
                k1 = rateConstant_1
    
                # Rate constant 2 (analogous to macropore resistance)
                k2 = rateConstant_2/dqbydc
                            
                # Overall rate constant
                # The following conditions are done for purely numerical reasons
                # If pure (analogous) macropore
                if k1<1e-12:
                    rateConstant = k2
                # If pure (analogous) micropore
                elif k2<1e-12:
                    rateConstant = k1
                # If both resistances are present
                else:
                    rateConstant = 1/(1/k1 + 1/k2)
                
                # Rate constant (overall)
                kineticConstant_ZLC[pp,ii,jj] = rateConstant
    # Loop over all the isotherm files
    for kk in range(len(isothermALL)):
       # Create the instance for the plots
        ax = plt.subplot(2,3,kk+1)
        
        # Path of the file name
        fileToLoad = os.path.join(mainDir,isothermDir,isothermALL[kk])

        # Get the experimental points
        experimentALL = sio.loadmat(fileToLoad)["isothermData"]["experiment"][0][0]

        # Get the isotherm fits
        # isothermFitALL = sio.loadmat(fileToLoad)["isothermData"]["isothermFit"][0][0]      

        # Find temperatures
        temperature = np.unique(experimentALL[:,2])

        # Find indices corresponding to each temperature
        for ll in range(len(temperature)):
            indexFirst = int(np.argwhere(experimentALL[:,2]==temperature[ll])[0])
            indexLast = int(np.argwhere(experimentALL[:,2]==temperature[ll])[-1])
        
            # Plot experimental isotherm
            ax.plot(experimentALL[indexFirst:indexLast,0],
                    experimentALL[indexFirst:indexLast,1],
                    linewidth = 0, marker = markersForPlot_I[ll],
                    color='#'+colorsForPlot_I[ll],
                    label = str(temperature[ll]-273.15)+"$^\circ$C") 
            # Removed isotherm fit from MATLAB code (bug)
            # ax.plot(isothermFitALL[1:-1,0],isothermFitALL[1:-1,ll+1],
            #         linewidth = 1,color='#'+colorsForPlot_I[ll],alpha=0.5)
            ax.legend(loc='upper left', handletextpad=0.0)
                  
            
            # Obtain the confidence bounds for the QC data
            # Load isotherm parameters from QC data
            isothermParameters = sio.loadmat(fileToLoad)["isothermData"]["isothermParameters"][0][0]
    
            # Create the grid for mole fractions
            y = np.linspace(0,1.,100)
    
            # Prepare x_VOL
            x_VOL = list(isothermParameters[0:-1:2,0]) + list(isothermParameters[1::2,0])
            x_VOL_CI = list(isothermParameters[0:-1:2,1]) + list(isothermParameters[1::2,1])
 
            # Initialize volumetric loading
            isoLoading_VOL = np.zeros([len(y),len(temperature)])   
 
            # Loop through all the temperature and mole fraction
            for jj in range(len(temperature)):
                for ii in range(len(y)):
                    isoLoading_VOL[ii,jj] = computeEquilibriumLoading(isothermModel=x_VOL,
                                                                      moleFrac = y[ii],
                                                                      temperature = temperature[jj])
 
            # Get the confidence bounds
            isoLoading_VOL_LowerBound, isoLoading_VOL_UpperBound = computeConfidenceBounds(x_VOL, x_VOL_CI, temperature, y)


            # Plot fitted isotherm and confidence bounds
            for jj in range(len(temperature)):
                ax.plot(y,isoLoading_VOL[:,jj],color='#'+colorsForPlot_I[jj],alpha=1.,linestyle=':') # QC
                ax.fill_between(y, isoLoading_VOL_LowerBound[:,jj], isoLoading_VOL_UpperBound[:,jj],
                      color='#'+colorsForPlot_I[jj],alpha = 0.1,linewidth=0.) # Lowest J
            
        # Material specific text labels
        if kk == 0:
            ax.set(xlabel='$P$ [bar]', 
                    ylabel='$q^*_\mathregular{CO_2}$ [mol kg$^{-1}$]',
                    xlim = [0,1], ylim = [0, 2])
            ax.text(0.89, 1.82, "(a)", fontsize=8,)
            # ax.text(0.87, 0.13, "AC", fontsize=8, fontweight = 'bold',color = '#4895EF')

        elif kk == 1:
            ax.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 2])
            ax.text(0.89, 1.82, "(b)", fontsize=8,)
            # ax.text(0.87, 0.09, "BN", fontsize=8, fontweight = 'bold',color = '#4895EF')

        elif kk == 2:
            ax.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 2])
            ax.text(0.89, 1.82, "(c)", fontsize=8,)
            # ax.text(0.85, 0.35, "13X", fontsize=8, fontweight = 'bold',color = '#4895EF')

        ax.locator_params(axis="x", nbins=4)            
        ax.locator_params(axis="y", nbins=4)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
 
        # Material specific text labels
        if kk == 0:
            ax.text(0.43, 2.15, "BNp", fontsize=8, fontweight = 'bold',color = 'k')
        elif kk == 1:
            ax.text(0.33, 2.15, "BN-FASp", fontsize=8, fontweight = 'bold',color = 'k')
        elif kk == 2:
            ax.text(0.33, 2.15, "BNp-FAS", fontsize=8, fontweight = 'bold',color = 'k')            
        ax.locator_params(axis="y", nbins=5)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    colorsForPlot_I = ["ffd900","f27a18","c3351e","0f0b05"]
    # Loop over all the isotherm files
    for kk in range(len(isothermALL)):
  
        # Plot the kinetic constants    
        ax2 = plt.subplot(2,3,kk+4)

        for jj in range(len(temperature)):
            ax2.plot(y,kineticConstant_ZLC[kk,:,jj],color='#'+colorsForPlot_I[jj],label=" "+str(temperature_k[jj]-273.15)+"$^\circ$C") # Lowest J

        
        if kk == 0:
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    ylabel='$k\mathregular{_{CO_2}}$ [s$^{-1}$]',
                    xlim = [0,1], ylim = [0, 1])
            ax2.text(0.89, 0.9, "(d)", fontsize=8,)
            # ax2.text(0.87, 0.9, "AC", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 0.83, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=4)
            ax2.locator_params(axis="y", nbins=4)
        elif kk  == 1:
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 70])
            ax2.text(0.89, 63, "(e)", fontsize=8,)
            # ax2.text(0.87, 0.9, "BN", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 0.83, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=4)
            ax2.locator_params(axis="y", nbins=8)
        elif kk  == 2:
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 3])
            ax2.text(0.89, 2.7, "(f)", fontsize=8,)
            # ax2.text(0.84, 1.8, "13X", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 1.66, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=4)
            ax2.locator_params(axis="y", nbins=4)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.legend(loc='upper left', handletextpad=0.0)
    #  Save the figure
    if saveFlag:
        # FileName: figureMat_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureMat_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
 
    plt.show()
    
# fun: plotForArticle_figureZLCFitALLBN
# Plots the Figure ZLC Fit of the manuscript: ZLC goodness of fit for experimental results
def plotForArticle_figureZLCFitALLBN(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure    
    import auxiliaryFunctions
    from numpy import load
    import os
    from simulateCombinedModel import simulateCombinedModel
    from deadVolumeWrapper import deadVolumeWrapper
    from extractDeadVolume import filesToProcess # File processing script
    from matplotlib.lines import Line2D
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
        
    
    markersForPlot = ["^","d","v"]    
    
    # X limits for the different materials
    XLIM_L = [[0, 200],[0, 200],[0, 200]]
    XLIM_H = [[0, 40],[0, 40],[0, 40]]
    
    # Label positions for the different materials
    panelLabel_L = [170, 170, 170]
    panelLabel_H = [60/200*40/60*170, 60/200*40/60*170, 60/200*40/60*170]
    # panelLabel_H = [60/200*170, 60/200*40/60*170, 60/200*150/60*170]


    # Parameter estimate files
                        
    zlcFileNameALL = [# Boron Nitride Experiments
                        ['zlcParameters_20210824_1522_c8173b1.npz',],
                        # BN-FASpExperiments
                        ['zlcParameters_20221001_0006_1016ac9.npz',],
                        # BNp-FAS Experiments
                        ['zlcParameters_20220815_0727_e81a19e.npz',]]
    
    rateConstant_1ALL = [0.07, 922, 867]
    rateConstant_2ALL = [831.77, 603, 41]
    isothermModelALL = [[3.12, 11.62e-7,22.99e3],
                        [9.44e-02, 2.94e-07, 3.43e4, 1.74, 5.61e-10, 4.10e4],
                        [1.13, 3.94e-08, 3.24e+04]]
    
    rawFileNameALL = [[  'ZLC_BoronNitride_Exp34A_Output.mat',
                         'ZLC_BoronNitride_Exp36A_Output.mat',
                         'ZLC_BoronNitride_Exp38A_Output.mat',
                         'ZLC_BoronNitride_Exp34B_Output.mat',
                         'ZLC_BoronNitride_Exp36B_Output.mat',
                         'ZLC_BoronNitride_Exp38B_Output.mat',
                         'ZLC_BoronNitride_Exp35A_Output.mat',
                         'ZLC_BoronNitride_Exp37A_Output.mat',
                         'ZLC_BoronNitride_Exp39A_Output.mat', 
                         'ZLC_BoronNitride_Exp35B_Output.mat',
                         'ZLC_BoronNitride_Exp37B_Output.mat',
                         'ZLC_BoronNitride_Exp39B_Output.mat'],
                      [ 'ZLC_BNFASp_Exp35A_Output.mat',
    		            'ZLC_BNFASp_Exp37A_Output.mat',
    		            'ZLC_BNFASp_Exp33A_Output.mat',
                        'ZLC_BNFASp_Exp35B_Output.mat',
    		            'ZLC_BNFASp_Exp37B_Output.mat',
    		            'ZLC_BNFASp_Exp33B_Output.mat',
    		            'ZLC_BNFASp_Exp36A_Output.mat',
    		            'ZLC_BNFASp_Exp38A_Output.mat',
    		            'ZLC_BNFASp_Exp34A_Output.mat',
    		            'ZLC_BNFASp_Exp36B_Output.mat',
    		            'ZLC_BNFASp_Exp38B_Output.mat',
    		            'ZLC_BNFASp_Exp34B_Output.mat',],
                      [ 'ZLC_BNpFAS_Exp01A_Output.mat',
                        'ZLC_BNpFAS_Exp03A_Output.mat',
                        'ZLC_BNpFAS_Exp05A_Output.mat',                        
                        'ZLC_BNpFAS_Exp01B_Output.mat',
                        'ZLC_BNpFAS_Exp03B_Output.mat',
                        'ZLC_BNpFAS_Exp05B_Output.mat',
                        'ZLC_BNpFAS_Exp02A_Output.mat',
                        'ZLC_BNpFAS_Exp04A_Output.mat',
                        'ZLC_BNpFAS_Exp06A_Output.mat',
                        'ZLC_BNpFAS_Exp02B_Output.mat',
                        'ZLC_BNpFAS_Exp04B_Output.mat',
                        'ZLC_BNpFAS_Exp06B_Output.mat',]]
    
    fig = figure(figsize=(6.5,5))     
    for pp in range(len(zlcFileNameALL)):
        # Plot colors and markers
        if pp == 0:
            colorsForPlot = ["ffba08","d00000","03071e"]
        else:
            colorsForPlot = ["05c70c","32a287","19612e"]
        # Parse out the isotherm parameter
        isothermModel = isothermModelALL[pp]
        rateConstant_1 = rateConstant_1ALL[pp]
        rateConstant_2 = rateConstant_2ALL[pp]
 
        zlcFileName = zlcFileNameALL[pp]
        objectiveFunction = np.zeros([len(zlcFileName)])
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # Obtain the onjective function values
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)

        # Find the experiment with the min objective function
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))
        fileParameter = zlcFileName[int(minJ[0])]
        
        # ZLC parameter model path
        parameterPath = os.path.join('..','simulationResults',fileParameter)
           
        # Parse out experiments names and temperature used for the fitting
        rawFileName = rawFileNameALL[pp]
        temperatureExp = load(parameterPath)["temperature"]

        # Generate .npz file for python processing of the .mat file 
        filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'ZLC')
        # Get the processed file names
        fileName = filesToProcess(False,[],[],'ZLC')
        
        numPointsExp = np.zeros(len(fileName))
        for ii in range(len(fileName)): 
            fileToLoad = fileName[ii]
            # Load experimental molefraction
            timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
            numPointsExp[ii] = len(timeElapsedExp)
        
        # Parse out all the necessary quantities to obtain model fit
        # Mass of sorbent and particle epsilon
        adsorbentDensity = load(parameterPath)["adsorbentDensity"]
        particleEpsilon = load(parameterPath)["particleEpsilon"]
        massSorbent = load(parameterPath)["massSorbent"]
        # Volume of sorbent material [m3]
        volSorbent = (massSorbent/1000)/adsorbentDensity
        # Volume of gas chamber (dead volume) [m3]
        volGas = volSorbent/(1-particleEpsilon)*particleEpsilon
        # Dead volume model
        deadVolumeFile = str(load(parameterPath)["deadVolumeFile"])
        # Isotherm parameter reference
        parameterReference = load(parameterPath)["parameterReference"]
        # Load the model
        modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
        modelNonDim = modelOutputTemp[()]["variable"] 
        # Multiply the paremeters by the reference values
        x = np.multiply(modelNonDim,parameterReference)
        x[0:-2] = isothermModel
        x[-2] = rateConstant_1
        x[-1] = rateConstant_2
        print(x)
        # Downsample intervals
        downsampleInt = numPointsExp/np.min(numPointsExp)
        
        # Initialize loadings
        moleFracExpALL = np.array([])
        moleFracSimALL = np.array([])

        # Loop over all available files    
        for ii in range(len(fileName)):
            fileToLoad = fileName[ii]   
            
            # Initialize outputs
            moleFracSim = []  
            # Load experimental time, molefraction and flowrate (accounting for downsampling)
            timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
            moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
            flowRateTemp = load(fileToLoad)["flowRate"].flatten()
            timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
            moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
            flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
                    
            # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
            timeInt = timeElapsedExp

            # Parse out parameter values
            isothermModel = x[0:-2]
            rateConstant_1 = x[-2]
            rateConstant_2 = x[-1]
                    
            # Compute the dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                                initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                                flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                                expFlag = True,
                                                                isothermModel = isothermModel,
                                                                rateConstant_1 = rateConstant_1,
                                                                rateConstant_2 = rateConstant_2,
                                                                deadVolumeFile = deadVolumeFile,
                                                                volSorbent = volSorbent,
                                                                volGas = volGas,
                                                                temperature = temperatureExp[ii],
                                                                adsorbentDensity = adsorbentDensity)
            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
                                                                  moleFracSim),
                                                        timeElapsedExp),2))

            # Stack mole fraction from experiments and simulation for error 
            # computation
            minExp = np.min(moleFracExp) # Compute the minimum from experiment
            normalizeFactor = np.max(moleFracExp - np.min(moleFracExp)) # Compute the max from normalized data
            moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
            moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))

            # Call the deadVolume Wrapper function to obtain the outlet mole fraction
            deadVolumePath = os.path.join('..','simulationResults',deadVolumeFile)
            modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
            pDV = modelOutputTemp[()]["variable"]
            dvFileLoadTemp = load(deadVolumePath)
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
            moleFracDV = deadVolumeWrapper(timeInt, resultMat[3,:]*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = [moleFracExp[0]])
            if pp == 0:
                if 300<temperatureExp[ii] and temperatureExp[ii]<310:
                    colorTemp = colorsForPlot[0]
                    markersTemp =markersForPlot[0]
                elif 320<temperatureExp[ii] and temperatureExp[ii]<330:
                    colorTemp = colorsForPlot[1]
                    markersTemp =markersForPlot[1]
                elif 340<temperatureExp[ii] and temperatureExp[ii]<350:
                    colorTemp = colorsForPlot[2]
                    markersTemp =markersForPlot[2]
            else:
                if 280<temperatureExp[ii] and temperatureExp[ii]<290:
                    colorTemp = colorsForPlot[0]
                    markersTemp =markersForPlot[0]
                elif 290<temperatureExp[ii] and temperatureExp[ii]<300:
                    colorTemp = colorsForPlot[1]
                    markersTemp =markersForPlot[1]
                elif 300<temperatureExp[ii] and temperatureExp[ii]<310:
                    colorTemp = colorsForPlot[2]
                    markersTemp =markersForPlot[2]       
    
            if ii in range(0,3):                    
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii]-273.15,0)))+"$^\circ$C"
                ax1 = plt.subplot(3,4,4*pp+1)
                ax1.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax1.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax1.semilogy(timeElapsedExp,moleFracDV,
                                 color='#9467bd',alpha = 0.4, linestyle='-') # Dead volume response
                ax1.set(xlabel='$t$ [s]',ylabel='$y\mathregular{_{CO_2}}$ [-]',
                        xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax1.locator_params(axis="x", nbins=4)
                ax1.grid(which='minor', linestyle=':')

            if ii in range(3,6):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax2 = plt.subplot(3,4,4*pp+2)
                ax2.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax2.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response
                if ii%3 == 0:
                    ax2.semilogy(timeElapsedExp,moleFracDV,
                                 color='#9467bd',alpha = 0.4, linestyle='-') # Dead volume response
                ax2.set(xlabel='$t$ [s]',
                        xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax2.locator_params(axis="x", nbins=4)
                ax2.grid(which='minor', linestyle=':')
                ax2.axes.yaxis.set_ticklabels([])

            if ii in range(6,9):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax3 = plt.subplot(3,4,4*pp+3)
                ax3.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax3.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response  
                if ii%3 == 0:
                    ax3.semilogy(timeElapsedExp,moleFracDV,
                                 color='#9467bd',alpha = 0.4, linestyle='-') # Dead volume response
                ax3.set(xlabel='$t$ [s]',xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                ax3.locator_params(axis="x", nbins=4)
                ax3.locator_params(axis="x", nbins=4)
                ax3.axes.yaxis.set_ticklabels([])
                ax3.grid(which='minor', linestyle=':')
                
            if ii in range(9,12):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax4 = plt.subplot(3,4,4*pp+4)
                ax4.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax4.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax4.semilogy(timeElapsedExp,moleFracDV,
                                 color='#9467bd',alpha = 0.4, linestyle='-') # Dead volume response
                ax4.set(xlabel='$t$ [s]',xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                ax4.locator_params(axis="x", nbins=4)
                ax4.grid(which='minor', linestyle=':')
                ax4.axes.yaxis.set_ticklabels([])

        # Put panel labels
        ax1.text(panelLabel_L[pp], 0.6, '('+chr(96+4*pp+1)+')', fontsize=8,)
        ax2.text(panelLabel_L[pp], 0.6, '('+chr(96+4*pp+2)+')', fontsize=8,)
        ax3.text(panelLabel_H[pp], 0.6, '('+chr(96+4*pp+3)+')', fontsize=8,)
        ax4.text(panelLabel_H[pp], 0.6, '('+chr(96+4*pp+4)+')', fontsize=8,)

        # Remove all the .npz files genereated from the .mat
        # Loop over all available files    
        for ii in range(len(fileName)):
            os.remove(fileName[ii])

    # Put other text entries
    plt.figtext(0.23, 0.98, "$F^\mathregular{in}$ = 10 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')
    plt.figtext(0.685, 0.98, "$F^\mathregular{in}$ = 60 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')
    plt.figtext(-0.02, 0.83, "BNp", fontsize=8, fontweight = 'bold',color = '#4895EF',rotation='vertical')
    plt.figtext(-0.02, 0.51, "BN-FASp", fontsize=8, fontweight = 'bold',color = '#4895EF',rotation='vertical')
    plt.figtext(-0.02, 0.19, "BNp-FAS", fontsize=8, fontweight = 'bold',color = '#4895EF',rotation='vertical')
    
    # Dead Volume
    tempLabel = ['10$^\circ$C','20$^\circ$C', '30$^\circ$C', '33$^\circ$C','52$^\circ$C', '72$^\circ$C']

    # Custom Legend Lines
    custom_lines = [Line2D([0], [0], linestyle='-', lw=1, color = '#05c70c'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#32a287'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#19612e'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#ffba08'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#d00000'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#03071e'),]
    
    fig.legend(custom_lines,tempLabel,bbox_to_anchor=(0.04,0.95,0.85,0.1), 
                   ncol=6, borderaxespad=0)   


    #  Save the figure
    if saveFlag:
        # FileName: figureZLCALL_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureZLCFitALLBN_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath,bbox_inches='tight')
    
    plt.show()
    
# fun: plotForArticle_figureZLCFitALLLewatit
# Plots the Figure ZLC Fit of the manuscript: ZLC goodness of fit for experimental results
def plotForArticle_figureZLCFitALLLewatit(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure    
    import auxiliaryFunctions
    from numpy import load
    import os
    from simulateCombinedModel import simulateCombinedModel
    from deadVolumeWrapper import deadVolumeWrapper
    from extractDeadVolume import filesToProcess # File processing script
    from matplotlib.lines import Line2D
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
        
    
    markersForPlot = ["^","d","v","o","<"]    
    
    # X limits for the different materials
    XLIM_L = [[0, 8000]]
    XLIM_H = [[0, 2000]]
    

    # Parameter estimate files
                        
    zlcFileNameALL = [['zlcParameters_20221013_1056_1016ac9.npz']]
    ######################### HIGH COMP ONLY
    rateConstant_1ALL = [0.722]
    rateConstant_2ALL = [15.8]
    ######################### HIGH AND LOW COMP
    rateConstant_1ALL = [1000]
    rateConstant_2ALL = [13.6]
    rateConstant_1ALL = [1000]
    rateConstant_2ALL = [13]
    # rateConstant_1ALL = [426]
    # rateConstant_2ALL = [112]
    
    
    ##########################
    isothermModelALL = [[1.05, 1.6e-12,  7.26e4, 0.99, 2.02e-14,9.22e4]]
    isothermModelALL = [[1.02, 2.18e-10,  5e4, 1.46, 1.92e-13, 8.44e4]]
#     isothermModelALL = [1.512499999999999956e+00,
# 5.307847700000000901e-06,
# 4.731298200000000361e+04,
# 1.494574499999999917e+00,
# 4.541271000000000722e-07,
# 4.310757809999999881e+04
# ]
    
    rawFileNameALL = [[ 'ZLC_Lewatit_DA_Exp05A_Output.mat',
    		            'ZLC_Lewatit_DA_Exp07A_Output.mat',
    		            'ZLC_Lewatit_DA_Exp09A_Output.mat',
    		            'ZLC_Lewatit_DA_Exp11A_Output.mat',
    		            'ZLC_Lewatit_DA_Exp13A_Output.mat',
                        'ZLC_Lewatit_DA_Exp05B_Output.mat',
    		            'ZLC_Lewatit_DA_Exp07B_Output.mat',
    		            'ZLC_Lewatit_DA_Exp09B_Output.mat',
    		            'ZLC_Lewatit_DA_Exp11B_Output.mat',
    		            'ZLC_Lewatit_DA_Exp13B_Output.mat',
                        'ZLC_Lewatit_DA_Exp06A_Output.mat',
    		            'ZLC_Lewatit_DA_Exp08A_Output.mat',
    		            'ZLC_Lewatit_DA_Exp10A_Output.mat',
    		            'ZLC_Lewatit_DA_Exp12A_Output.mat',
    		            'ZLC_Lewatit_DA_Exp14A_Output.mat',
                        'ZLC_Lewatit_DA_Exp06B_Output.mat',
    		            'ZLC_Lewatit_DA_Exp08B_Output.mat',
    		            'ZLC_Lewatit_DA_Exp10B_Output.mat',
    		            'ZLC_Lewatit_DA_Exp12B_Output.mat',
    		            'ZLC_Lewatit_DA_Exp14B_Output.mat',]]
    
    fig = figure(figsize=(6.5,2.2))     
    for pp in range(len(zlcFileNameALL)):
        # Plot colors and markers
        if pp == 0:
            colorsForPlot = ["4d9de0","e15554","e1bc29","3bb273","7768ae"]
        else:
            colorsForPlot = ["4d9de0","e15554","e1bc29","3bb273","7768ae"]
        # Parse out the isotherm parameter
        isothermModel = isothermModelALL[pp]
        rateConstant_1 = rateConstant_1ALL[pp]
        rateConstant_2 = rateConstant_2ALL[pp]
 
        zlcFileName = zlcFileNameALL[pp]
        objectiveFunction = np.zeros([len(zlcFileName)])
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # Obtain the onjective function values
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)

        # Find the experiment with the min objective function
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))
        fileParameter = zlcFileName[int(minJ[0])]
        
        # ZLC parameter model path
        parameterPath = os.path.join('..','simulationResults',fileParameter)
           
        # Parse out experiments names and temperature used for the fitting
        rawFileName = rawFileNameALL[pp]
        temperatureExp = load(parameterPath)["temperature"]
        print(temperatureExp)
        # Generate .npz file for python processing of the .mat file 
        filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'ZLC')
        # Get the processed file names
        fileName = filesToProcess(False,[],[],'ZLC')
        
        numPointsExp = np.zeros(len(fileName))
        for ii in range(len(fileName)): 
            fileToLoad = fileName[ii]
            # Load experimental molefraction
            timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
            numPointsExp[ii] = len(timeElapsedExp)
        
        # Parse out all the necessary quantities to obtain model fit
        # Mass of sorbent and particle epsilon
        adsorbentDensity = 988
        particleEpsilon = 0.44
        massSorbent = load(parameterPath)["massSorbent"]
        # Volume of sorbent material [m3]
        volSorbent = (massSorbent/1000)/adsorbentDensity
        # Volume of gas chamber (dead volume) [m3]
        volGas = volSorbent/(1-particleEpsilon)*particleEpsilon
        # Dead volume model
        deadVolumeFile = str(load(parameterPath)["deadVolumeFile"])
        # Isotherm parameter reference
        parameterReference = load(parameterPath)["parameterReference"]
        # Load the model
        modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
        modelNonDim = modelOutputTemp[()]["variable"] 
        # Multiply the paremeters by the reference values
        x = np.multiply(modelNonDim,parameterReference)
        x[0:-2] = isothermModel
        x[-2] = rateConstant_1
        x[-1] = rateConstant_2
        print(x)
        # Downsample intervals
        downsampleInt = numPointsExp/np.min(numPointsExp)
        
        # Initialize loadings
        moleFracExpALL = np.array([])
        moleFracSimALL = np.array([])

        # Loop over all available files    
        for ii in range(len(fileName)):
            fileToLoad = fileName[ii]   
            
            # Initialize outputs
            moleFracSim = []  
            # Load experimental time, molefraction and flowrate (accounting for downsampling)
            timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
            moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
            flowRateTemp = load(fileToLoad)["flowRate"].flatten()
            timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
            moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
            flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
                    
            # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
            timeInt = timeElapsedExp

            # Parse out parameter values
            isothermModel = x[0:-2]
            rateConstant_1 = x[-2]
            rateConstant_2 = x[-1]
                    
            # Compute the dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                                initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                                flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                                expFlag = True,
                                                                isothermModel = isothermModel,
                                                                rateConstant_1 = rateConstant_1,
                                                                rateConstant_2 = rateConstant_2,
                                                                deadVolumeFile = deadVolumeFile,
                                                                volSorbent = volSorbent,
                                                                volGas = volGas,
                                                                temperature = temperatureExp[ii],
                                                                adsorbentDensity = adsorbentDensity)
            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
                                                                  moleFracSim),
                                                        timeElapsedExp),2))

            # Stack mole fraction from experiments and simulation for error 
            # computation
            minExp = np.min(moleFracExp) # Compute the minimum from experiment
            normalizeFactor = np.max(moleFracExp - np.min(moleFracExp)) # Compute the max from normalized data
            moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
            moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))

            # Call the deadVolume Wrapper function to obtain the outlet mole fraction
            deadVolumePath = os.path.join('..','simulationResults',deadVolumeFile)
            modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
            pDV = modelOutputTemp[()]["variable"]
            dvFileLoadTemp = load(deadVolumePath)
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
            moleFracDV = deadVolumeWrapper(timeInt, resultMat[3,:]*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = [moleFracExp[0]])
            if pp == 0:
                if 300<temperatureExp[ii] and temperatureExp[ii]<305:
                    colorTemp = colorsForPlot[0]
                    markersTemp =markersForPlot[0]
                elif 305<temperatureExp[ii] and temperatureExp[ii]<320:
                    colorTemp = colorsForPlot[1]
                    markersTemp =markersForPlot[1]
                elif 320<temperatureExp[ii] and temperatureExp[ii]<335:
                    colorTemp = colorsForPlot[2]
                    markersTemp =markersForPlot[2]
                elif 335<temperatureExp[ii] and temperatureExp[ii]<350:
                    colorTemp = colorsForPlot[3]
                    markersTemp =markersForPlot[3]
                elif 350<temperatureExp[ii] and temperatureExp[ii]<365:
                    colorTemp = colorsForPlot[4]
                    markersTemp =markersForPlot[4]
            else:
                if 300<temperatureExp[ii] and temperatureExp[ii]<305:
                    colorTemp = colorsForPlot[0]
                    markersTemp =markersForPlot[0]
                elif 305<temperatureExp[ii] and temperatureExp[ii]<320:
                    colorTemp = colorsForPlot[1]
                    markersTemp =markersForPlot[1]
                elif 320<temperatureExp[ii] and temperatureExp[ii]<335:
                    colorTemp = colorsForPlot[2]
                    markersTemp =markersForPlot[2]
                elif 335<temperatureExp[ii] and temperatureExp[ii]<350:
                    colorTemp = colorsForPlot[3]
                    markersTemp =markersForPlot[3]
                elif 350<temperatureExp[ii] and temperatureExp[ii]<365:
                    colorTemp = colorsForPlot[4]
                    markersTemp =markersForPlot[4]      
    
            if ii in range(0,5):                    
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii]-273.15,0)))+"$^\circ$C"
                ax1 = plt.subplot(1,4,4*pp+1)
                ax1.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.2) # Experimental response
                ax1.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%5 == 0:
                    ax1.semilogy(timeElapsedExp,moleFracDV,
                                 color='#498AAF',alpha = 0.4, linestyle='-') # Dead volume response
                ax1.set(xlabel='$t$ [s]',ylabel='$y\mathregular{_{CO_2}}$ [-]',
                        xlim = [0, 7000], ylim = [1e-4, 3e-2])    
                ax1.locator_params(axis="x", nbins=4)
                ax1.grid(which='minor', linestyle=':')
            if ii in range(5,10):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax2 = plt.subplot(1,4,4*pp+2)
                ax2.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.2) # Experimental response
                ax2.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response
                if ii%3 == 0:
                    ax2.semilogy(timeElapsedExp,moleFracDV,
                                 color='#498AAF',alpha = 0.4, linestyle='-') # Dead volume response
                ax2.set(xlabel='$t$ [s]',
                        xlim = [0, 8000], ylim = [1e-4, 3e-2])     
                ax2.locator_params(axis="x", nbins=4)
                ax2.grid(which='minor', linestyle=':')
                ax2.axes.yaxis.set_ticklabels([])

            if ii in range(10,15):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax3 = plt.subplot(1,4,4*pp+3)
                ax3.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.2) # Experimental response
                ax3.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response  
                if ii%3 == 0:
                    ax3.semilogy(timeElapsedExp,moleFracDV,
                                 color='#498AAF',alpha = 0.4, linestyle='-') # Dead volume response
                ax3.set(xlabel='$t$ [s]',xlim = [0, 1000], ylim = [1e-4, 3e-2])      
                ax3.locator_params(axis="x", nbins=4)
                ax3.axes.yaxis.set_ticklabels([])
                ax3.grid(which='minor', linestyle=':')
                
            if ii in range(15,20):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax4 = plt.subplot(1,4,4*pp+4)
                ax4.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.2) # Experimental response
                ax4.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax4.semilogy(timeElapsedExp,moleFracDV,
                                 color='#498AAF',alpha = 0.4, linestyle='-') # Dead volume response
                ax4.set(xlabel='$t$ [s]',xlim = [0,2000], ylim = [1e-4, 3e-2])    
                ax4.locator_params(axis="x", nbins=4)
                ax4.grid(which='minor', linestyle=':')
                ax4.axes.yaxis.set_ticklabels([])
                
        

        # Remove all the .npz files genereated from the .mat
        # Loop over all available files    
        for ii in range(len(fileName)):
            os.remove(fileName[ii])

    # Put other text entries
    plt.figtext(0.23, 0.98, "$F^\mathregular{in}$ = 15 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')
    plt.figtext(0.685, 0.98, "$F^\mathregular{in}$ = 60 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')
    plt.figtext(-0.02, 0.5, "Lewatit", fontsize=8, fontweight = 'bold',color = '#4895EF',rotation='vertical')
    
    # Dead Volume
    tempLabel = ['30$^\circ$C','45$^\circ$C', '60$^\circ$C', '75$^\circ$C','90$^\circ$C']
    # Custom Legend Lines
    custom_lines = [Line2D([0], [0], linestyle='-', lw=1, color = '#4d9de0'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#e15554'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#e1bc29'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#3bb273'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#7768ae'),]
    
    fig.legend(custom_lines,tempLabel,bbox_to_anchor=(0.01,1.05,0.8,0.1), 
                   ncol=6, borderaxespad=0)   


    #  Save the figure
    if saveFlag:
        # FileName: figureZLCALL_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureZLCFitALLLewatit_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath,bbox_inches='tight')
    
    plt.show()
    
# fun: plotForArticle_figureBNeqMB
# Plots the Equilibrium isotherms with MB points for BNp, BNp-FAS, and BN-FASp
def plotForArticle_figureBNeqMB(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    import scipy.io as sio
    import os
    from matplotlib.ticker import FormatStrFormatter
    from computeEquilibriumLoading import computeEquilibriumLoading

    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()

    # Plot colors and markers (porosity)
    colorsForPlot_P = ["0fa3b1","f17300"]
    markersForPlot_P = ["^","v"]
    
    # Plot colors and markers (isotherm)    
    colorsForPlot_I = ["f27a18","c3351e","0f0b05"]
    markersForPlot_I = ["^","d","v"]


    # Main folder for material characterization
    mainDir = os.path.join('..','experimental','materialCharacterization')

    # Isotherm folder
    isothermDir = os.path.join('isothermData','isothermResults')
    
    # File with pore characterization data
    isothermALL = ['BNp_SSL_100621.mat',
                   'BNFASp_DSL_101922_101922.mat',
                   'BNpFAS_SSL_101922_101922.mat',]
    # Universal gas constant
    Rg = 8.314
    
    # Total pressure
    pressureTotal = np.array([1.e5]);
    
    # Define temperature
    temperature_k = [283.15, 293.15, 303.15]
    # Create the grid for mole fractions
    y = np.linspace(0,1.,100)
    # Rate constants from parameter estimation
    rateConstant_1ALL = [0.07, 922, 867]
    rateConstant_2ALL = [831.77, 603, 41]
    isothermModelALL = [[3.12, 11.62e-7,22.99e3,0 ,0, 0],
                        [9.44e-02, 2.94e-07, 3.43e4, 1.74, 5.61e-10, 4.10e4],
                        [1.13, 3.94e-08, 3.24e+04,0,0,0]]
    adsorbentDensityALL = [2040, 1250, 2320]
    isoLoading_ZLC = np.zeros([len(isothermALL),len(y),len(temperature_k)])
    kineticConstant_ZLC = np.zeros([len(isothermALL),len(y),len(temperature_k)])
    
    
    MassBalancepALL = [[0.109214,
0.117185,
0.114149,
0.942466,
0.942509,
0.941403],
[0.109610000000000,
0.113034000000000,
0.105163000000000,
0.946172000000000,
0.945105000000000,
0.946374000000000],
[0.105043000000000,
0.108162000000000,
0.107561000000000,
0.949932000000000,
0.949591000000000,
0.949454000000000]]
    MassBalanceQALL = [[0.174712,
0.108303,
0.0775624,
0.855556,
0.458426,
0.334479],
[0.147819,
0.137383,
0.0845643,
0.765847,
0.522119,
0.338531],
[0.169678,
0.127234,
0.0997316,
0.627848,
0.454012,
0.324763,
]]


        # Loop over all the isotherm files
    for pp in range(len(isothermALL)):
        adsorbentDensity = adsorbentDensityALL[pp]
        # Parse out the isotherm parameter
        isothermModel = isothermModelALL[pp]
        rateConstant_1 = rateConstant_1ALL[pp]
        rateConstant_2 = rateConstant_2ALL[pp]
        for jj in range(len(temperature_k)):
            for ii in range(len(y)):
                isoLoading_ZLC[pp,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                             moleFrac = y[ii], 
                                                                             temperature = temperature_k[jj]) # [mol/kg]
                # Partial pressure of the gas
                partialPressure = y[ii]*pressureTotal
                # delta pressure to compute gradient
                delP = 1e-3
                # Mole fraction (up)
                moleFractionUp = (partialPressure + delP)/pressureTotal
                # Compute the loading [mol/m3] @ moleFractionUp
                equilibriumLoadingUp  = computeEquilibriumLoading(temperature=temperature_k[jj],
                                                                moleFrac=moleFractionUp,
                                                                isothermModel=isothermModel) # [mol/kg]
                
                # Compute the gradient (delq*/dc)
                dqbydc = (equilibriumLoadingUp-isoLoading_ZLC[pp,ii,jj])*adsorbentDensity/(delP/(Rg*temperature_k[jj])) # [-]
    
                # Rate constant 1 (analogous to micropore resistance)
                k1 = rateConstant_1
    
                # Rate constant 2 (analogous to macropore resistance)
                k2 = rateConstant_2/dqbydc
                            
                # Overall rate constant
                # The following conditions are done for purely numerical reasons
                # If pure (analogous) macropore
                if k1<1e-12:
                    rateConstant = k2
                # If pure (analogous) micropore
                elif k2<1e-12:
                    rateConstant = k1
                # If both resistances are present
                else:
                    rateConstant = 1/(1/k1 + 1/k2)
                
                # Rate constant (overall)
                kineticConstant_ZLC[pp,ii,jj] = rateConstant
    # Loop over all the isotherm files
    for kk in range(len(isothermALL)):
       # Create the instance for the plots
        ax = plt.subplot(1,3,kk+1)
        MassBalancep = MassBalancepALL[kk]
        MassBalanceQ = MassBalanceQALL[kk]

        # Path of the file name
        fileToLoad = os.path.join(mainDir,isothermDir,isothermALL[kk])

        # Get the experimental points
        experimentALL = sio.loadmat(fileToLoad)["isothermData"]["experiment"][0][0]

        # Get the isotherm fits
        # isothermFitALL = sio.loadmat(fileToLoad)["isothermData"]["isothermFit"][0][0]      

        # Find temperatures
        temperature = np.unique(experimentALL[:,2])
        if kk == 0:
            temperature = [306.17,325.49,344.6]
        else:
            temperature = temperature_k
        if kk == 0:
            colorsForPlot_I = ["ffba08","d00000","03071e"]
        else:
            colorsForPlot_I = ["05c70c","32a287","19612e"]
            # Find indices corresponding to each temperature
        for ll in range(len(temperature)):
            # indexFirst = int(np.argwhere(experimentALL[:,2]==temperature[ll])[0])
            # indexLast = int(np.argwhere(experimentALL[:,2]==temperature[ll])[-1])
        
            # Plot experimental isotherm
            # ax.plot(experimentALL[indexFirst:indexLast,0],
            #         experimentALL[indexFirst:indexLast,1],
            #         linewidth = 0, marker = markersForPlot_I[ll],
            #         color='#'+colorsForPlot_I[ll],
            #         label = str(temperature[ll]-273.15)+"$^\circ$C") 
            # Removed isotherm fit from MATLAB code (bug)
            # ax.plot(isothermFitALL[1:-1,0],isothermFitALL[1:-1,ll+1],
            #         linewidth = 1,color='#'+colorsForPlot_I[ll],alpha=0.5)
            # ax.legend(loc='upper left', handletextpad=0.0)
                  
            
            # Obtain the confidence bounds for the QC data
            # Load isotherm parameters from QC data
            isothermParameters = sio.loadmat(fileToLoad)["isothermData"]["isothermParameters"][0][0]
    
            # Create the grid for mole fractions
            y = np.linspace(0,1.,100)
    
            # Prepare x_VOL
            x_VOL = list(isothermParameters[0:-1:2,0]) + list(isothermParameters[1::2,0])
            x_VOL_CI = list(isothermParameters[0:-1:2,1]) + list(isothermParameters[1::2,1])
 
            # Initialize volumetric loading
            isoLoading_VOL = np.zeros([len(y),len(temperature)])   
 
            # Loop through all the temperature and mole fraction
            for jj in range(len(temperature)):
                for ii in range(len(y)):
                    isoLoading_VOL[ii,jj] = computeEquilibriumLoading(isothermModel=x_VOL,
                                                                      moleFrac = y[ii],
                                                                      temperature = temperature[jj])
 
            # Get the confidence bounds
            isoLoading_VOL_LowerBound, isoLoading_VOL_UpperBound = computeConfidenceBounds(x_VOL, x_VOL_CI, temperature, y)


            # Plot fitted isotherm and confidence bounds
            for jj in range(len(temperature)):
                ax.plot(y,isoLoading_VOL[:,jj],color='#'+colorsForPlot_I[jj],alpha=1.,linestyle=':') # QC
                ax.fill_between(y, isoLoading_VOL_LowerBound[:,jj], isoLoading_VOL_UpperBound[:,jj],
                      color='#'+colorsForPlot_I[jj],alpha = 0.1,linewidth=0.,label='_nolegend_') # Lowest J
            MPL = MassBalancep[ll]
            MPH = MassBalancep[ll+3]
            MQL = MassBalanceQ[ll]
            MQH = MassBalanceQ[ll+3]
            ax.plot([MPL, MPH], [MQL, MQH],color='#'+colorsForPlot_I[ll],alpha = 1,linewidth=0.,marker = '*', markersize=4,
                  label = str(np.around(temperature[ll]-273.15,decimals=1))+"$^\circ$C") 
            ax.legend(loc='upper left', handletextpad=0.0)
        # Material specific text labels
        if kk == 0:
            ax.set(xlabel='$P$ [bar]', 
                    ylabel='$q^*_\mathregular{CO_2}$ [mol kg$^{-1}$]',
                    xlim = [0,1], ylim = [0, 1.5])
            ax.text(0.89, 1.35, "(a)", fontsize=8,)
            # ax.text(0.87, 0.13, "AC", fontsize=8, fontweight = 'bold',color = '#4895EF')

        elif kk == 1:
            ax.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 1.5])
            ax.text(0.89, 1.35, "(b)", fontsize=8,)
            # ax.text(0.87, 0.09, "BN", fontsize=8, fontweight = 'bold',color = '#4895EF')

        elif kk == 2:
            ax.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 1.5])
            ax.text(0.89, 1.35, "(c)", fontsize=8,)
            # ax.text(0.85, 0.35, "13X", fontsize=8, fontweight = 'bold',color = '#4895EF')

        ax.locator_params(axis="x", nbins=4)            
        ax.locator_params(axis="y", nbins=4)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
 
        # Material specific text labels
        if kk == 0:
            ax.text(0.43, 1.6, "BNp", fontsize=8, fontweight = 'bold',color = 'k')
        elif kk == 1:
            ax.text(0.33, 1.6, "BN-FASp", fontsize=8, fontweight = 'bold',color = 'k')
        elif kk == 2:
            ax.text(0.33, 1.6, "BNp-FAS", fontsize=8, fontweight = 'bold',color = 'k')            
        ax.locator_params(axis="y", nbins=5)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    #  Save the figure
    if saveFlag:
        # FileName: figureMat_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureMat_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
 
    plt.show()
    
# fun: plotForArticle_figureZIF8eqkin
# Plots the Equilibrium isotherms and kinetics for ZIF-8 MT, MCB20, and MCB30
def plotForArticle_figureZIF8eqkin(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    import scipy.io as sio
    import os
    from matplotlib.ticker import FormatStrFormatter
    from computeEquilibriumLoading import computeEquilibriumLoading
    import pdb

    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()

    # Plot colors and markers (porosity)
    colorsForPlot_P = ["0fa3b1","f17300"]
    markersForPlot_P = ["^","v"]
    
    # Plot colors and markers (isotherm)    
    colorsForPlot_I = ["ffba08","d00000","03071e"]

    markersForPlot_I = ["^","d","v"]


    # Main folder for material characterization
    mainDir = os.path.join('..','experimental','materialCharacterization')

    # Isotherm folder
    isothermDir = os.path.join('isothermData','isothermResults')
    
    # File with isotherm characterization data
    isothermALL = ['ZIF8_MT_SSS_QC_021323.mat',
                   'ZIF8_MCB20_SSS_QC_021323.mat',
                   'ZIF8_MCB30_SSS_QC_021323.mat',]
    # Universal gas constant
    Rg = 8.314
    
    # Total pressure
    pressureTotal = np.array([1.e5]);
    
    # Define temperature
    temperature_k = [283, 293, 303]
    # Create the grid for mole fractions
    y = np.linspace(0,1.,1000)
    # Rate constants from parameter estimation
    rateConstant_1ALL = [0.29, 0.23, 1.09]
    rateConstant_2ALL = [983, 1.30, 2.96]
    isothermModelALL = [[20, 5.51151743793683e-07, 19272.6327577962,1.12005137776376],
                        [9.6369, 2.74e-7, 2.14e+04,0.9893],
                        [6.51504325937662, 1.11248296208017e-07, 24942.1235836143,1.06047499753496]]
    adsorbentDensityALL = [1555, 2400, 2100]
    isoLoading_ZLC = np.zeros([len(isothermALL),len(y),len(temperature_k)])
    kineticConstant_ZLC = np.zeros([len(isothermALL),len(y),len(temperature_k)])


        # Loop over all the isotherm files
    for pp in range(len(isothermALL)):
        adsorbentDensity = adsorbentDensityALL[pp]
        # Parse out the isotherm parameter
        isothermModel = isothermModelALL[pp]
        rateConstant_1 = rateConstant_1ALL[pp]
        rateConstant_2 = rateConstant_2ALL[pp]
        for jj in range(len(temperature_k)):
            for ii in range(len(y)):
                isoLoading_ZLC[pp,ii,jj] = computeEquilibriumLoading(isothermModel=isothermModel,
                                                                             moleFrac = y[ii], 
                                                                             temperature = temperature_k[jj]) # [mol/kg]
                # Partial pressure of the gas
                partialPressure = y[ii]*pressureTotal
                # delta pressure to compute gradient
                delP = 1e-3
                # Mole fraction (up)
                moleFractionUp = (partialPressure + delP)/pressureTotal
                # Compute the loading [mol/m3] @ moleFractionUp
                equilibriumLoadingUp  = computeEquilibriumLoading(temperature=temperature_k[jj],
                                                                moleFrac=moleFractionUp,
                                                                isothermModel=isothermModel) # [mol/kg]
                
                # Compute the gradient (delq*/dc)
                dqbydc = (equilibriumLoadingUp-isoLoading_ZLC[pp,ii,jj])*adsorbentDensity/(delP/(Rg*temperature_k[jj])) # [-]
    
                # Rate constant 1 (analogous to micropore resistance)
                k1 = rateConstant_1
    
                # Rate constant 2 (analogous to macropore resistance)
                k2 = rateConstant_2/dqbydc
                            
                # Overall rate constant
                # The following conditions are done for purely numerical reasons
                # If pure (analogous) macropore
                if k1<1e-12:
                    rateConstant = k2
                # If pure (analogous) micropore
                elif k2<1e-12:
                    rateConstant = k1
                # If both resistances are present
                else:
                    rateConstant = 1/(1/k1 + 1/k2)
                
                # Rate constant (overall)
                kineticConstant_ZLC[pp,ii,jj] = rateConstant
    # Loop over all the isotherm files
    for kk in range(len(isothermALL)):
       # Create the instance for the plots
        ax2 = plt.subplot(1,3,kk+1)
        
        # Path of the file name
        fileToLoad = os.path.join(mainDir,isothermDir,isothermALL[kk])

        # Get the experimental points
        experimentALL = sio.loadmat(fileToLoad)["isothermData"]["experiment"][0][0]

        # Get the isotherm fits
        # isothermFitALL = sio.loadmat(fileToLoad)["isothermData"]["isothermFit"][0][0]      

        # Find temperatures
        temperature = np.unique(experimentALL[:,2])

        
        for jj in range(len(temperature)):
            ax2.plot(y,kineticConstant_ZLC[kk,:,jj],color='#'+colorsForPlot_I[jj],label=" "+str(temperature_k[jj])+" K") # Lowest J

        
        if kk == 0:
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    ylabel='$k\mathregular{_{CO_2}}$ [s$^{-1}$]',
                    xlim = [0,1], ylim = [0, 0.5])
            ax2.text(0.89, 0.45, "(a)", fontsize=8,)
            ax2.text(0.3, 1.1/2, "ZIF8-ext100", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 0.83, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=5)
            ax2.locator_params(axis="y", nbins=5)
        elif kk  == 1:
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 0.5])
            ax2.text(0.89, 0.45, "(b)", fontsize=8,)
            ax2.text(0.34, 1.1/2, "ZIF8-pel75", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 0.83, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=5)
            ax2.locator_params(axis="y", nbins=5)
        elif kk  == 2:
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 0.5])
            ax2.text(0.89, 0.45, "(c)", fontsize=8,)
            ax2.text(0.34, 1.1/2, "ZIF8-pel65", fontsize=8, fontweight = 'bold',color = '#4895EF')
            # ax2.text(0.53, 1.66, "Experimental", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax2.locator_params(axis="x", nbins=5)
            ax2.locator_params(axis="y", nbins=5)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.legend(loc='upper left', handletextpad=0.0)
    #  Save the figure
    if saveFlag:
        # FileName: figureMat_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureKinZIF8_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
 
    plt.show()   
    
# fun: plotForArticle_figureZLCFitALLZIF8
# Plots the Figure ZLC Fit of the manuscript: ZLC goodness of fit for experimental results
def plotForArticle_figureZLCFitALLZIF8(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure    
    import auxiliaryFunctions
    from numpy import load
    import os
    from simulateCombinedModel import simulateCombinedModel
    from deadVolumeWrapper import deadVolumeWrapper
    from extractDeadVolume import filesToProcess # File processing script
    from matplotlib.lines import Line2D
    import pdb
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
        
    
    markersForPlot = ["o","o","o"]    
    
    # X limits for the different materials
    XLIM_L = [[0, 250],[0, 250],[0, 250]]
    XLIM_H = [[0, 50],[0, 50],[0, 50]]
    
    # Label positions for the different materials
    panelLabel_L = [213, 213, 213]
    panelLabel_H = [60/250*50/60*213, 60/250*50/60*213, 60/250*50/60*213]
    # panelLabel_H = [60/200*170, 60/200*40/60*170, 60/200*150/60*170]


    # Parameter estimate files
                        
    zlcFileNameALL = [# ZIF8-ext100
                        ['zlcParameters_20230223_1239_7e5a5aa.npz',],
                        # ZIF8-pel75
                        ['zlcParameters_20230227_0924_7e5a5aa.npz',],
                        # ZIF8-pel65
                        ['zlcParameters_20230225_1844_7e5a5aa.npz',]]
    
    rateConstant_1ALL = [0.29, 0.23, 1.09]
    rateConstant_2ALL = [984, 1.30, 2.96]
    isothermModelALL = [[20, 5.51151743793683e-07, 19272.6327577962,1.12005137776376],
                        [9.6369, 2.74e-7, 2.14e+04,0.9893],
                        [6.51504325937662, 1.11248296208017e-07, 24942.1235836143,1.06047499753496]]
    
    rawFileNameALL = [[  'ZLC_ZIF8_MT_Exp01A_Output.mat',
            'ZLC_ZIF8_MT_Exp03A_Output.mat',
            'ZLC_ZIF8_MT_Exp07A_Output.mat',
            'ZLC_ZIF8_MT_Exp01B_Output.mat',
            'ZLC_ZIF8_MT_Exp03B_Output.mat',
            'ZLC_ZIF8_MT_Exp07B_Output.mat',
            'ZLC_ZIF8_MT_Exp02A_Output.mat',
            'ZLC_ZIF8_MT_Exp04A_Output.mat',
            'ZLC_ZIF8_MT_Exp08A_Output.mat',
            'ZLC_ZIF8_MT_Exp02B_Output.mat',
            'ZLC_ZIF8_MT_Exp04B_Output.mat',
            'ZLC_ZIF8_MT_Exp08B_Output.mat',],
                       ['ZLC_ZIF8_MCB20_Exp01A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp05A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp07A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp01B_Output.mat',
            'ZLC_ZIF8_MCB20_Exp05B_Output.mat',
            'ZLC_ZIF8_MCB20_Exp07B_Output.mat',
            'ZLC_ZIF8_MCB20_Exp02A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp06A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp08A_Output.mat',
            'ZLC_ZIF8_MCB20_Exp02B_Output.mat',
            'ZLC_ZIF8_MCB20_Exp06B_Output.mat',
            'ZLC_ZIF8_MCB20_Exp08B_Output.mat',],
                      [ 'ZLC_ZIF8_MCB30_Exp01A_Output.mat',
            'ZLC_ZIF8_MCB30_Exp03A_Output.mat',
            'ZLC_ZIF8_MCB30_Exp07A_Output.mat',
            'ZLC_ZIF8_MCB30_Exp01B_Output.mat',
            'ZLC_ZIF8_MCB30_Exp03B_Output.mat',
            'ZLC_ZIF8_MCB30_Exp07B_Output.mat',
            'ZLC_ZIF8_MCB30_Exp02A_Output.mat',
            'ZLC_ZIF8_MCB30_Exp04A_Output.mat',
            'ZLC_ZIF8_MCB30_Exp08A_Output.mat',
            'ZLC_ZIF8_MCB30_Exp02B_Output.mat',
            'ZLC_ZIF8_MCB30_Exp04B_Output.mat',
            'ZLC_ZIF8_MCB30_Exp08B_Output.mat',]]
    
    fig = figure(figsize=(6.5,5))     
    for pp in range(len(zlcFileNameALL)):
        # Plot colors and markers
        colorsForPlot = ["ffba08","d00000","03071e"]
        # Parse out the isotherm parameter
        isothermModel = isothermModelALL[pp]
        rateConstant_1 = rateConstant_1ALL[pp]
        rateConstant_2 = rateConstant_2ALL[pp]
 
        zlcFileName = zlcFileNameALL[pp]
        objectiveFunction = np.zeros([len(zlcFileName)])
        # Loop over all available ZLC files for a given material
        for kk in range(len(zlcFileName)):
            # Obtain the onjective function values
            parameterPath = os.path.join('..','simulationResults',zlcFileName[kk])
            modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
            objectiveFunction[kk] = round(modelOutputTemp[()]["function"],0)

        # Find the experiment with the min objective function
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))
        fileParameter = zlcFileName[int(minJ[0])]
        
        # ZLC parameter model path
        parameterPath = os.path.join('..','simulationResults',fileParameter)
           
        # Parse out experiments names and temperature used for the fitting
        rawFileName = rawFileNameALL[pp]
        temperatureExp = load(parameterPath)["temperature"]

        # Generate .npz file for python processing of the .mat file 
        filesToProcess(True,os.path.join('..','experimental','runData'),rawFileName,'ZLC')
        # Get the processed file names
        fileName = filesToProcess(False,[],[],'ZLC')
        
        numPointsExp = np.zeros(len(fileName))
        for ii in range(len(fileName)): 
            fileToLoad = fileName[ii]
            # Load experimental molefraction
            timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
            numPointsExp[ii] = len(timeElapsedExp)
        
        # Parse out all the necessary quantities to obtain model fit
        # Mass of sorbent and particle epsilon
        adsorbentDensity = load(parameterPath)["adsorbentDensity"]
        particleEpsilon = load(parameterPath)["particleEpsilon"]
        massSorbent = load(parameterPath)["massSorbent"]
        # Volume of sorbent material [m3]
        volSorbent = (massSorbent/1000)/adsorbentDensity
        # Volume of gas chamber (dead volume) [m3]
        volGas = volSorbent/(1-particleEpsilon)*particleEpsilon
        # Dead volume model
        deadVolumeFile = load(parameterPath)["deadVolumeFile"]
        # Isotherm parameter reference
        parameterReference = load(parameterPath)["parameterReference"]
        # Load the model
        modelOutputTemp = load(parameterPath, allow_pickle=True)["modelOutput"]
        modelNonDim = modelOutputTemp[()]["variable"] 
        # Multiply the paremeters by the reference values
        x = np.zeros(6)
        x[0:-2] = isothermModel
        x[-2] = rateConstant_1
        x[-1] = rateConstant_2
        print(x)
        # Downsample intervals
        downsampleInt = numPointsExp/np.min(numPointsExp)
        
        # Initialize loadings
        moleFracExpALL = np.array([])
        moleFracSimALL = np.array([])

        # Loop over all available files    
        for ii in range(len(fileName)):
            fileToLoad = fileName[ii]   
            
            # Initialize outputs
            moleFracSim = []  
            # Load experimental time, molefraction and flowrate (accounting for downsampling)
            timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
            moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
            flowRateTemp = load(fileToLoad)["flowRate"].flatten()
            timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
            moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
            flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
            if len(deadVolumeFile) == 1: # 1 DV fo
                deadVolumeFileTemp = str(deadVolumeFile[0])
            else:
                if np.absolute(flowRateExp[-1] - 1) > 0.2: # for lowflowrate experiments!
                    deadVolumeFileTemp = str(deadVolumeFile[0])
                else:
                    deadVolumeFileTemp = str(deadVolumeFile[1]) 

            # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
            timeInt = timeElapsedExp

            # Parse out parameter values
            isothermModel = x[0:-2]
            rateConstant_1 = x[-2]
            rateConstant_2 = x[-1]
                    
            # Compute the dead volume response using the optimizer parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                                initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                                flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                                expFlag = True,
                                                                isothermModel = isothermModel,
                                                                rateConstant_1 = rateConstant_1,
                                                                rateConstant_2 = rateConstant_2,
                                                                deadVolumeFile = deadVolumeFileTemp,
                                                                volSorbent = volSorbent,
                                                                volGas = volGas,
                                                                temperature = temperatureExp[ii],
                                                                adsorbentDensity = adsorbentDensity)
            # Print simulation volume    
            print("Simulation",str(ii+1),round(np.trapz(np.multiply(resultMat[3,:]*1e6,
                                                                  moleFracSim),
                                                        timeElapsedExp),2))

            # Stack mole fraction from experiments and simulation for error 
            # computation
            minExp = np.min(moleFracExp) # Compute the minimum from experiment
            normalizeFactor = np.max(moleFracExp - np.min(moleFracExp)) # Compute the max from normalized data
            moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
            moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))

            # Call the deadVolume Wrapper function to obtain the outlet mole fraction
            deadVolumePath = os.path.join('..','simulationResults',deadVolumeFileTemp)
            modelOutputTemp = load(deadVolumePath, allow_pickle=True)["modelOutput"]
            pDV = modelOutputTemp[()]["variable"]
            dvFileLoadTemp = load(deadVolumePath)
            flagMSDeadVolume = dvFileLoadTemp["flagMSDeadVolume"]
            msDeadVolumeFile = dvFileLoadTemp["msDeadVolumeFile"]
            moleFracDV = deadVolumeWrapper(timeInt, resultMat[3,:]*1e6, pDV, flagMSDeadVolume, msDeadVolumeFile, initMoleFrac = [moleFracExp[0]])

            if 280<temperatureExp[ii] and temperatureExp[ii]<290:
                colorTemp = colorsForPlot[0]
                markersTemp =markersForPlot[0]
            elif 290<temperatureExp[ii] and temperatureExp[ii]<300:
                colorTemp = colorsForPlot[1]
                markersTemp =markersForPlot[1]
            elif 300<temperatureExp[ii] and temperatureExp[ii]<310:
                colorTemp = colorsForPlot[2]
                markersTemp =markersForPlot[2]       
    
            if ii in range(0,3):                    
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii]-0.15,0)))+" K"
                ax1 = plt.subplot(3,4,4*pp+1)
                ax1.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax1.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax1.semilogy(timeElapsedExp,moleFracDV,
                                 color='#9467bd',alpha = 0.4, linestyle='-') # Dead volume response

                if ii == 0: 
                    ax1.set(xlabel='$t$ [s]',ylabel='$y\mathregular{_{CO_2}}$ [-]',
                            xlim = np.multiply(XLIM_L[pp],0.4), ylim = [1e-2, 1])    
                elif ii == 2:
                    ax1.set(xlabel='$t$ [s]',ylabel='$y\mathregular{_{CO_2}}$ [-]',
                            xlim = np.multiply(XLIM_L[pp],0.4), ylim = [1e-2, 1])    
                else:
                    ax1.set(xlabel='$t$ [s]',ylabel='$y\mathregular{_{CO_2}}$ [-]',
                            xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax1.locator_params(axis="x", nbins=5)
                ax1.grid(which='minor', linestyle=':')
            if ii in range(3,6):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax2 = plt.subplot(3,4,4*pp+2)
                ax2.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax2.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response
                if ii%3 == 0:
                    ax2.semilogy(timeElapsedExp,moleFracDV,
                                 color='#9467bd',alpha = 0.4, linestyle='-') # Dead volume response
                if ii == 0: 
                    ax2.set(xlabel='$t$ [s]',
                            xlim = np.multiply(XLIM_L[pp],0.4), ylim = [1e-2, 1])    
                elif ii == 2:
                    ax2.set(xlabel='$t$ [s]',
                            xlim = np.multiply(XLIM_L[pp],0.4), ylim = [1e-2, 1])    
                else:
                    ax2.set(xlabel='$t$ [s]',
                            xlim = XLIM_L[pp], ylim = [1e-2, 1])    
                ax2.locator_params(axis="x", nbins=5)
                ax2.grid(which='minor', linestyle=':')
                ax2.axes.yaxis.set_ticklabels([])

            if ii in range(6,9):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax3 = plt.subplot(3,4,4*pp+3)
                ax3.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax3.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response  
                if ii%3 == 0:
                    ax3.semilogy(timeElapsedExp,moleFracDV,
                                 color='#9467bd',alpha = 0.4, linestyle='-') # Dead volume response
                if ii == 0: 
                    ax3.set(xlabel='$t$ [s]',
                            xlim = np.multiply(XLIM_H[pp],0.4), ylim = [1e-2, 1])    
                elif ii == 2:
                    ax3.set(xlabel='$t$ [s]',
                            xlim = np.multiply(XLIM_H[pp],0.4), ylim = [1e-2, 1])    
                else:
                    ax3.set(xlabel='$t$ [s]',
                            xlim = np.multiply(XLIM_H[pp],0.4), ylim = [1e-2, 1])    
                ax3.locator_params(axis="x", nbins=5)
                ax3.grid(which='minor', linestyle=':')
                ax3.axes.yaxis.set_ticklabels([])
            if ii in range(9,12):
                # Plot the experimental data with model output
                legendStr = str(int(round(temperatureExp[ii],0)))+" K"
                ax4 = plt.subplot(3,4,4*pp+4)
                ax4.semilogy(timeElapsedExp,moleFracExp,
                        marker = markersTemp,linewidth = 0,
                        color='#'+colorTemp,alpha=0.1) # Experimental response
                ax4.semilogy(timeElapsedExp,moleFracSim,
                             color='#'+colorTemp,label=legendStr) # Simulation response    
                if ii%3 == 0:
                    ax4.semilogy(timeElapsedExp,moleFracDV,
                                 color='#9467bd',alpha = 0.4, linestyle='-') # Dead volume response
                if ii == 0: 
                    ax4.set(xlabel='$t$ [s]',
                            xlim = np.multiply(XLIM_H[pp],0.4), ylim = [1e-2, 1])    
                elif ii == 2:
                    ax4.set(xlabel='$t$ [s]',
                            xlim = np.multiply(XLIM_H[pp],0.4), ylim = [1e-2, 1])    
                else:
                    ax4.set(xlabel='$t$ [s]',
                            xlim = XLIM_H[pp], ylim = [1e-2, 1])    
                ax4.locator_params(axis="x", nbins=5)
                ax4.grid(which='minor', linestyle=':')
                ax4.axes.yaxis.set_ticklabels([])

        # Put panel labels
        # if ii > 5:
        ax1.text(np.multiply(panelLabel_L[pp],0.4), 0.6, '('+chr(96+4*pp+1)+')', fontsize=8,)
        ax2.text(panelLabel_L[pp], 0.6, '('+chr(96+4*pp+2)+')', fontsize=8,)
        ax3.text(np.multiply(panelLabel_H[pp],0.4), 0.6, '('+chr(96+4*pp+3)+')', fontsize=8,)
        ax4.text(panelLabel_H[pp], 0.6, '('+chr(96+4*pp+4)+')', fontsize=8,)
        # else:
        #     ax1.text(panelLabel_L[pp], 0.6, '('+chr(96+4*pp+1)+')', fontsize=8,)
        #     ax2.text(panelLabel_L[pp], 0.6, '('+chr(96+4*pp+2)+')', fontsize=8,)
        #     ax3.text(panelLabel_H[pp], 0.6, '('+chr(96+4*pp+3)+')', fontsize=8,)
        #     ax4.text(panelLabel_H[pp], 0.6, '('+chr(96+4*pp+4)+')', fontsize=8,)

        # Remove all the .npz files genereated from the .mat
        # Loop over all available files    
        for ii in range(len(fileName)):
            os.remove(fileName[ii])

    # Put other text entries
    plt.figtext(0.23, 0.98, "$F^\mathregular{in}$ = 10 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')
    plt.figtext(0.685, 0.98, "$F^\mathregular{in}$ = 60 cm$^3$ min$^{-1}$", fontsize=8, fontweight = 'bold',color = '#4895EF')
    plt.figtext(-0.015, 0.795, "ZIF8-ext100", fontsize=8, fontweight = 'bold',color = '#4895EF',rotation='vertical')
    plt.figtext(-0.015, 0.48, "ZIF8-pel75", fontsize=8, fontweight = 'bold',color = '#4895EF',rotation='vertical')
    plt.figtext(-0.015, 0.16, "ZIF8-pel65", fontsize=8, fontweight = 'bold',color = '#4895EF',rotation='vertical')
    
    # Dead Volume
    tempLabel = ['283 K','293 K', '303 K']

    # Custom Legend Lines
    custom_lines = [Line2D([0], [0], linestyle='-', lw=1, color = '#ffba08'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#d00000'),
                    Line2D([0], [0], linestyle='-', lw=1, color = '#03071e'),]

    
    fig.legend(custom_lines,tempLabel,bbox_to_anchor=(0.04,0.95,0.67,0.1), 
                   ncol=3, borderaxespad=0)   


    #  Save the figure
    if saveFlag:
        # FileName: figureZLCALL_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureZLCFitALLZIF8_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath,bbox_inches='tight')
    
    plt.show()
    
# fun: computeConfidenceBounds
# Generate LHS sampled isotherms using the confidence bounds
def computeConfidenceBounds(isoParam, ciParam, temperature, y):
    from smt.sampling_methods import LHS
    import numpy as np
    from computeEquilibriumLoading import computeEquilibriumLoading
    
    # Generate numIso isotherm parameters in the confidence region
    numIso = 1000
    # Initialize the bounds for the parameter values
    if len(np.array(isoParam[0])) == 3:
        isoBound = np.zeros([6,2])
        # Lower Bound
        isoBound[0:2,0] = np.array(isoParam[0:2]) - np.array(ciParam[0:2])
        isoBound[2,0] = np.array(isoParam[2]) + np.array(ciParam[2])
        
        # Upper Bound
        isoBound[0:2,1] = np.array(isoParam[0:2]) + np.array(ciParam[0:2])
        isoBound[2,1] = np.array(isoParam[2]) - np.array(ciParam[2])
        
    elif len(np.array(isoParam[0])) == 6:
        isoBound = np.zeros([6,2])
        # Lower Bound
        isoBound[0:2,0] = np.array(isoParam[0:2]) - np.array(ciParam[0:2])
        isoBound[3:5,0] = np.array(isoParam[3:5]) - np.array(ciParam[3:5])
        isoBound[2:6:3,0] = np.array(isoParam[2:6:3]) + np.array(ciParam[2:6:3])
    
        # Upper Bound
        isoBound[0:2,1] = np.array(isoParam[0:2]) + np.array(ciParam[0:2])
        isoBound[3:5,1] = np.array(isoParam[3:5]) + np.array(ciParam[3:5])
        isoBound[2:6:3,1] = np.array(isoParam[2:6:3]) - np.array(ciParam[2:6:3])
    else:
        isoBound = np.zeros([4,2])
        # Lower Bound
        isoBound[0:3,0] = np.array(isoParam[0][0:3]) - np.array(ciParam[0][0:3])
    
        # Upper Bound
        isoBound[0:3,1] = np.array(isoParam[0][0:3]) + np.array(ciParam[0][0:3])


    # Generate a LHS method with the isotherm parameter bounds
    lhsPopulation = LHS(xlimits=isoBound)

    # Generate numIso isotherm parameters
    ciIsothermParameters = lhsPopulation(numIso)

    # Initialize isoLoading
    isoLoading_VOL = np.zeros([len(ciIsothermParameters),len(y),len(temperature)])
    # Loop through all the isotherm parameters, temperature and mole fraction
    for kk in range(len(ciIsothermParameters)):
        ciIsothermParametersTemp = [np.float64(qq) for qq in ciIsothermParameters[kk,:]]
        for jj in range(len(temperature)):
            for ii in range(len(y)):
                isoLoading_VOL[kk,ii,jj] = computeEquilibriumLoading(isothermModel=ciIsothermParametersTemp,
                                                                      moleFrac = y[ii],
                                                                      temperature = temperature[jj])

    # Find the maximum and minimum loading at each partial pressure & temperature
    isoLoadingLowerBound = isoLoading_VOL.min(axis = 0)
    isoLoadingUpperBound = isoLoading_VOL.max(axis = 0)
    
    return isoLoadingLowerBound, isoLoadingUpperBound

