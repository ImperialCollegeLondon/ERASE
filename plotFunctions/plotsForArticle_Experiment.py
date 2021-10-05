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

    # If dead volume plot needs to be plotted
    if 'figureMat' in kwargs:
        if kwargs["figureMat"]:
            plotForArticle_figureMat(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)
            
    # If dead volume plot needs to be plotted
    if 'figureDV' in kwargs:
        if kwargs["figureDV"]:
            plotForArticle_figureDV(gitCommitID, currentDT, 
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

    # Plot colors and markers
    colorsForPlot = ["0fa3b1","f17300"]
    markersForPlot = ["^","v"]
    
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

    # Loop over all the porosity files
    for kk in range(len(porosityALL)):
        # Path of the file name
        fileToLoad = os.path.join(mainDir,porosityDir,porosityALL[kk])
        
        # Load .mat file
        rawData = sio.loadmat(fileToLoad)["poreVolume"]
        # Convert the nDarray to list
        nDArrayToList = np.ndarray.tolist(rawData)
        
        # Get the porosity options
        porosityOptionsTemp = nDArrayToList[0][0][5]
    
        # Unpack another time (due to the structure of loadmat)
        porosityOptions = np.ndarray.tolist(porosityOptionsTemp[0])
        
        # Indicies for N2 and MIP data
        QCindexLast = porosityOptions[0][1][0][0] # Last index for N2 sorption
        
        # Get the porosity options
        combinedPorosityData = nDArrayToList[0][0][6]
    
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
                     marker = markersForPlot[0],
                     color='#'+colorsForPlot[0],)
        # Plot MIP
        ax.semilogx(combinedPorosityData[QCindexLast:-1:numIntPorosity,0],
                     combinedPorosityData[QCindexLast:-1:numIntPorosity,2],
                     linewidth = 0.25,linestyle = ':',
                     marker = markersForPlot[1],
                     color='#'+colorsForPlot[1],)

        # N2 sorption measurements
        ax.arrow(combinedPorosityData[QCindexLast-1,0], 0.3, dx1[kk], 0,
                 length_includes_head = True, head_length = hl1[kk], head_width = 0.04,
                 color = '#'+colorsForPlot[0])
        ax.text(combinedPorosityData[QCindexLast-1,0]+1.2*dx1[kk], 0.15, "N$_2$", fontsize=8,
                 color = '#'+colorsForPlot[0])
        
        # MIP measurements        
        ax.arrow(combinedPorosityData[QCindexLast-1,0], 0.7, dx2[kk], 0,
                 length_includes_head = True, head_length = hl2[kk], head_width = 0.04,
                 color = '#'+colorsForPlot[1])
        ax.text(combinedPorosityData[QCindexLast-1,0]+0.25*dx2[kk], 0.77, "Hg", fontsize=8,
                 color = '#'+colorsForPlot[1])
        
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