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

# fun: plotForArticle_figureMat
# Plots the Figure DV of the manuscript: Material characterization (N2/MIP and QC)
def plotForArticle_figureMat(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    import scipy.io as sio
    import os
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
                    ylabel='$V$ [mL g$^{-1}$]',
                    xlim = [0.1,1e6], ylim = [0, 2])
            ax.text(0.2, 1.82, "(a)", fontsize=8,)
            ax.text(1.4e5, 0.1, "AC", fontsize=8, fontweight = 'bold',color = '#e71d36')
            ax.text(2e3, combinedPorosityData[-2,2]+0.07, 
                    str(round(combinedPorosityData[-2,2],2))+' mL g$^{-1}$',
                    fontsize=8,color = '#7d8597')
        elif kk == 1:
            ax.set(xlabel='$D$ [nm]', 
                    xlim = [0.1,1e6], ylim = [0, 2])
            ax.text(0.2, 1.82, "(b)", fontsize=8,)
            ax.text(1.4e5, 0.1, "BN", fontsize=8, fontweight = 'bold',color = '#e71d36')
            ax.text(5, combinedPorosityData[-2,2]-0.15, 
                    str(round(combinedPorosityData[-2,2],2))+' mL g$^{-1}$',
                    fontsize=8,color = '#7d8597')
        elif kk == 2:
            ax.set(xlabel='$D$ [nm]', 
                    xlim = [0.1,1e6], ylim = [0, 2])
            ax.text(0.2, 1.82, "(c)", fontsize=8,)
            ax.text(1e5, 0.1, "13X", fontsize=8, fontweight = 'bold',color = '#e71d36')
            ax.text(2e3, combinedPorosityData[-2,2]+0.07, 
                    str(round(combinedPorosityData[-2,2],2))+' mL g$^{-1}$',
                    fontsize=8,color = '#7d8597')
            
        ax.locator_params(axis="y", nbins=5)
        
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
          
            # Plot fitted isotherm
            ax.plot(isothermFitALL[1:-1,0],isothermFitALL[1:-1,ll+1],
                    linewidth = 1,color='#'+colorsForPlot_I[ll],alpha=0.5)
            ax.legend(loc='best')
        # Material specific text labels
        if kk == 0:
            ax.set(xlabel='$P$ [bar]', 
                    ylabel='$q^*$ [mol kg$^{-1}$]',
                    xlim = [0,1], ylim = [0, 3])
            ax.text(0.89, 2.75, "(d)", fontsize=8,)
            ax.text(0.87, 0.13, "AC", fontsize=8, fontweight = 'bold',color = '#4895EF')

        elif kk == 1:
            ax.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 2])
            ax.text(0.89, 1.82, "(e)", fontsize=8,)
            ax.text(0.87, 0.09, "BN", fontsize=8, fontweight = 'bold',color = '#4895EF')

        elif kk == 2:
            ax.set(xlabel='$P$ [bar]', 
                    xlim = [0,1], ylim = [0, 8])
            ax.text(0.89, 7.25, "(f)", fontsize=8,)
            ax.text(0.85, 0.35, "13X", fontsize=8, fontweight = 'bold',color = '#4895EF')

        ax.locator_params(axis="x", nbins=4)            
        ax.locator_params(axis="y", nbins=4)
 
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
                              color='#'+colorsForPlot[ii],alpha=0.25,label=str(round(abs(np.mean(flowRateExp)),1))+" ccs") # Experimental response
                ax1.semilogy(timeElapsedExp,moleFracSim,
                                  color='#'+colorsForPlot[ii]) # Simulation response
                ax1.set(xlabel='$t$ [s]', 
                        ylabel='$y$ [-]',
                        xlim = [0,15], ylim = [1e-2, 1])
                ax1.locator_params(axis="x", nbins=5)
                ax1.legend(handletextpad=0.0,loc='center right')
                ax1.text(7, 1.3, "(a)", fontsize=8,)
                ax1.text(12.2, 0.64, "MS", fontsize=8, fontweight = 'bold',
                        backgroundcolor = 'w', color = '#e71d36')
                ax1.text(8.2, 0.39, "$V_\mathrm{d}$ = 0.02 cc", fontsize=8, 
                        backgroundcolor = 'w', color = '#7d8597')

            elif kk == 1:
                ax2.semilogy(timeElapsedExp,moleFracExp,
                              marker = markersForPlot[ii],linewidth = 0,
                              color='#'+colorsForPlot[ii],alpha=0.25,label=str(round(abs(np.mean(flowRateExp)),1))+" ccs") # Experimental response
                ax2.semilogy(timeElapsedExp,moleFracSim,
                                  color='#'+colorsForPlot[ii]) # Simulation response
                ax2.set(xlabel='$t$ [s]', 
                        xlim = [0,150], ylim = [1e-2, 1])   
                ax2.locator_params(axis="x", nbins=5)
                ax2.legend(handletextpad=0.0,loc='center right')
                ax2.text(70, 1.3, "(b)", fontsize=8,)
                ax2.text(67, 0.64, "Column w/ Ball", fontsize=8,  fontweight = 'bold',
                        backgroundcolor = 'w', color = '#e71d36')
                ax2.text(83, 0.39, "$V_\mathrm{d}$ = 3.76 cc", fontsize=8, 
                        backgroundcolor = 'w', color = '#7d8597')
                
            elif kk == 2:
                ax3.semilogy(timeElapsedExp,moleFracExp,
                              marker = markersForPlot[ii],linewidth = 0,
                              color='#'+colorsForPlot[ii],alpha=0.25,label=str(round(abs(np.mean(flowRateExp)),1))+" ccs") # Experimental response
                ax3.semilogy(timeElapsedExp,moleFracSim,
                                  color='#'+colorsForPlot[ii]) # Simulation response
                ax3.set(xlabel='$t$ [s]', 
                        xlim = [0,150], ylim = [1e-2, 1])   
                ax3.locator_params(axis="x", nbins=5)
                ax3.legend(handletextpad=0.0,loc='center right')
                ax3.text(70, 1.3, "(c)", fontsize=8,)
                ax3.text(61, 0.64, "Column w/o Ball", fontsize=8,  fontweight = 'bold',
                        backgroundcolor = 'w', color = '#e71d36')
                ax3.text(83, 0.39, "$V_\mathrm{d}$ = 3.93 cc", fontsize=8, 
                        backgroundcolor = 'w', color = '#7d8597')

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
    
    # Remove all the .npy files genereated from the .mat
    # Loop over all available files    
    for ii in range(len(fileName)):
        os.remove(fileName[ii])
        
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
            for kk in range(len(zlcFileName)):
                if kk == minJ[0]:
                    ax1.plot(y,isoLoading_ZLC[kk,:,jj],color='#'+colorsForPlot[jj],label=str(temperature[jj])+' K')  # Lowest J
                else:
                    ax1.plot(y,isoLoading_ZLC[kk,:,jj],color='#'+colorsForPlot[jj],alpha=0.2) # ALL

        # Plot the kinetic constants    
        ax2 = plt.subplot(2,3,pp+4)
        minJ = np.argwhere(objectiveFunction == min(objectiveFunction))

        for jj in range(len(temperature)):
            for kk in range(len(zlcFileName)):
                if kk == minJ[0]:
                    ax2.plot(y,kineticConstant_ZLC[kk,:,jj],color='#'+colorsForPlot[jj],label=str(temperature[jj])+' K') # Lowest J
                else:
                    ax2.plot(y,kineticConstant_ZLC[kk,:,jj],color='#'+colorsForPlot[jj],alpha=0.2) # ALL
        
        if pp == 0:
            # Isotherm
            ax1.set(ylabel='$q^*$ [mol kg$^{-1}$]',
                    xlim = [0,1], ylim = [0, 3])
            ax1.text(0.04, 2.75, "(a)", fontsize=8,)
            ax1.text(0.45, 3.2, "AC", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax1.text(0.84, 0.30, "OPT", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax1.text(0.84, 0.12, "REP", fontsize=8, fontweight = 'bold',color = '#4895EF', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.axes.xaxis.set_ticklabels([])
            ax1.legend()
            # Kinetics
            ax2.set(xlabel='$P$ [bar]', 
                    ylabel='$k$ [s$^{-1}$]',
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
            ax1.text(0.45, 1.6, "BN", fontsize=8, fontweight = 'bold',color = '#4895EF')
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
            ax1.text(0.44, 8.5, "13X", fontsize=8, fontweight = 'bold',color = '#4895EF')
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
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    from numpy import load
    import scipy.io as sio
    import os
    from computeEquilibriumLoading import computeEquilibriumLoading
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
    
    # Create the grid for mole fractions
    y = np.linspace(0,1.,100)

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
        
        # Loop through all the temperature and mole fraction
        for jj in range(len(temperature)):
            for ii in range(len(y)):
                isoLoading_VOL[ii,jj] = computeEquilibriumLoading(isothermModel=x_VOL,
                                                                  moleFrac = y[ii],
                                                                  temperature = temperature[jj])

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
            ax1.plot(y,isoLoading_VOL[:,jj],color='#'+colorsForPlot[jj],label=str(temperature[jj])+' K') # QC
            for kk in range(len(zlcFileName)):
                if kk == minJ[0]:
                    ax1.plot(y,isoLoading_ZLC[kk,:,jj],color='#'+colorsForPlot[jj],alpha = 0.2) # Lowest J

        if pp == 0:
            # Isotherm
            ax1.set(xlabel='$P$ [bar]', 
                    ylabel='$q^*$ [mol kg$^{-1}$]',
                    xlim = [0,1], ylim = [0, 3])
            ax1.text(0.04, 2.75, "(a)", fontsize=8,)
            ax1.text(0.45, 3.2, "AC", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax1.text(0.84, 0.32, "VOL", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax1.text(0.84, 0.12, "OPT", fontsize=8, fontweight = 'bold',color = '#4895EF', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.legend()
        elif pp  == 1:
            # Isotherm
            ax1.set(xlabel='$P$ [bar]', xlim = [0,1], ylim = [0, 1.5])
            ax1.text(0.04, 1.35, "(b)", fontsize=8,)
            ax1.text(0.45, 1.6, "BN", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax1.text(0.84, 0.16, "VOL", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax1.text(0.84, 0.06, "OPT", fontsize=8, fontweight = 'bold',color = '#4895EF', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.legend()
        elif pp  == 2:
            # Isotherm
            ax1.set(xlabel='$P$ [bar]', xlim = [0,1], ylim = [0, 8])
            ax1.text(0.04, 7.3, "(c)", fontsize=8,)
            ax1.text(0.44, 8.5, "13X", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax1.text(0.84, 0.86, "VOL", fontsize=8, fontweight = 'bold',color = '#4895EF')
            ax1.text(0.84, 0.32, "OPT", fontsize=8, fontweight = 'bold',color = '#4895EF', alpha = 0.3)
            ax1.locator_params(axis="x", nbins=4)
            ax1.locator_params(axis="y", nbins=4)
            ax1.legend(loc='upper right')
 
    #  Save the figure
    if saveFlag:
        # FileName: figureComp_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "figureComp_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','experimentManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','experimentManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','experimentManuscript'))
        plt.savefig (savePath)
 
    plt.show()
    
# fun: plotForArticle_figureZLCSim
# Plots the Figure ZLC of the manuscript: ZLC parameter estimates (simulated)
def plotForArticle_figureZLCSim(gitCommitID, currentDT, 
                           saveFlag, saveFileExtension):
    import numpy as np
    import matplotlib.pyplot as plt
    import auxiliaryFunctions
    from numpy import load
    import scipy.io as sio
    import os
    from computeEquilibriumLoading import computeEquilibriumLoading
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
        if len(isothermParameters) == 8:
            x_SIM = isothermParameters[0:6]
        else:
            x_SIM = isothermParameters[0:3]
        
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
            for kk in range(len(zlcFileName)):
                    ax1.plot(y,isoLoading_ZLC[kk,:,jj],color='#'+colorsForPlot[jj],alpha=0.1) # ALL

        # Plot the kinetic constants    
        ax2 = plt.subplot(2,3,pp+4)
        for jj in range(len(temperature)):
            ax2.plot(y,kineticConstant_SIM[:,jj],color='#'+colorsForPlot[jj],label=str(temperature[jj])+' K') # Simulated "True" Data
            for kk in range(len(zlcFileName)):
                ax2.plot(y,kineticConstant_ZLC[kk,:,jj],color='#'+colorsForPlot[jj],alpha=0.1) # ALL
        
        if pp == 0:
            # Isotherm
            ax1.set(ylabel='$q^*$ [mol kg$^{-1}$]',
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
                    ylabel='$k$ [s$^{-1}$]',
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