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
# Plots for the simulation manuscript
#
# Last modified:
# - 2021-02-11, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def plotsForArticle_Simulation(**kwargs):
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

    # If sensor array plot needs to be plotted
    if 'sensorArray' in kwargs:
        if kwargs["sensorArray"]:
            plotForArticle_SensorArray(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)
# fun: plotForArticle_SensorArray
# Plots the histogram of gas compositions for a one and two material 
# sensor array
def plotForArticle_SensorArray(gitCommitID, currentDT, 
                               saveFlag, saveFileExtension):
    import numpy as np
    from numpy import load
    import os
    import matplotlib.pyplot as plt
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file
    # For now load a given adsorbent isotherm material file
    loadFileName = ("arrayConcentration_20210211_1818_b02f8c3.npz", # 1 material w/o constraint
                    "arrayConcentration_20210212_1055_b02f8c3.npz", # 2 material w/o constraint
                    "arrayConcentration_20210212_1050_b02f8c3.npz", # 1 material with constraint
                    "arrayConcentration_20210211_1822_b02f8c3.npz") # 2 material with constraint
    
    # Git commit id of the loaded isotherm file
    simID_loadedFile = loadFileName[0][-11:-4]
    
    # Loop through the two files to get the histogram
    for ii in range(len(loadFileName)):
        # Create the file name with the path to be loaded
        simResultsFile = os.path.join('..','simulationResults',loadFileName[ii]);
    
        # Check if the file with the adsorbent properties exist 
        if os.path.exists(simResultsFile):
                resultOutput = load(simResultsFile)["arrayConcentration"][:,:]
                if resultOutput.shape[1] == 3:
                    resultOutput = np.delete(resultOutput,[0],1)
                elif resultOutput.shape[1] == 4:
                    resultOutput = np.delete(resultOutput,[0,1],1)
        else:
            errorString = "Simulation result file " + simResultsFile + " does not exist."
            raise Exception(errorString)
    
        # Gas concentration
        molFracG1 = 0.05
        molFracG2 = 0.95
        
        # Xlimits and Ylimits
        xLimits = [0,1]
        yLimits = [0,60] 
        
        # Histogram properties
        nBins = 50
        rangeX = (xLimits)
        histTypeX = 'stepfilled'
        alphaX=0.75
        densityX = True
    
        # Plot the histogram of the gas compositions
        ax = plt.subplot(2,2,ii+1)
        # Histogram for 1 material array
        ax.axvline(x=molFracG1, linewidth=1, linestyle='dotted', color = '#e5383b', alpha = 0.6)
        ax.hist(resultOutput[:,0], bins = nBins, range = rangeX, density = densityX,
                 linewidth=1.5, histtype = histTypeX, color='#e5383b', alpha = alphaX, label = '$g_1$')
    
        # Histogram for 2 material array    
        ax.axvline(x=molFracG2, linewidth=1, linestyle='dotted', color = '#343a40', alpha = 0.6)
        ax.hist(resultOutput[:,1], bins = nBins, range = rangeX, density = densityX, 
                 linewidth=1.5, histtype = histTypeX, color='#343a40', alpha = alphaX, label = '$g_2$')
        
        ax.set(xlim = xLimits, ylim = yLimits)
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=4)
        if ii == 0 or ii == 2:
            ax.set(ylabel='$f$ [-]')
            ax.text(0.85, 55, "$n$ = 1", fontsize=10, 
                    backgroundcolor = 'w', color = '#0077b6')
            if ii == 0:
                ax.text(0.025, 55, "(a)", fontsize=10, 
                        backgroundcolor = 'w')
                ax.text(0.625, 51, "Without Constraint", fontsize=10, 
                        backgroundcolor = 'w', color = '#0077b6')
            else:
                ax.set(xlabel='$y$ [-]')
                ax.text(0.025, 55, "(c)", fontsize=10, 
                    backgroundcolor = 'w')
                ax.text(0.625, 51, "With Constraint", fontsize=10, 
                        backgroundcolor = 'w', color = '#0077b6')
            ax.text(0.085, 25, "$y_1$ = 0.05", fontsize=10, 
                    backgroundcolor = 'w', color = '#e5383b')
            ax.text(0.705, 25, "$y_2$ = 0.95", fontsize=10, 
                    backgroundcolor = 'w', color = '#343a40')
        elif ii == 1 or ii == 3:
            ax.text(0.85, 55, "$n$ = 2", fontsize=10, 
                    backgroundcolor = 'w', color = '#0077b6')
            if ii == 1:
                ax.text(0.025, 55, "(b)", fontsize=10, 
                        backgroundcolor = 'w')
                ax.text(0.625, 51, "Without Constraint", fontsize=10, 
                        backgroundcolor = 'w', color = '#0077b6')
            else:
                ax.set(xlabel='$y$ [-]')
                ax.text(0.025, 55, "(d)", fontsize=10, 
                        backgroundcolor = 'w')
                ax.text(0.625, 51, "With Constraint", fontsize=10, 
                        backgroundcolor = 'w', color = '#0077b6')
            ax.text(0.085, 25, "$y_1$ = 0.05", fontsize=10, 
                    backgroundcolor = 'w', color = '#e5383b')
            ax.text(0.705, 25, "$y_2$ = 0.95", fontsize=10, 
                    backgroundcolor = 'w', color = '#343a40')
            
    #  Save the figure
    if saveFlag:
        # FileName: sensorArray_<currentDateTime>_<GitCommitID_Current>_<GitCommitID_Data>
        saveFileName = "sensorArray_" + currentDT + "_" + gitCommitID + "_" + simID_loadedFile + saveFileExtension
        savePath = os.path.join('..','simulationFigures','simulationManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','simulationManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','simulationManuscript'))
        plt.savefig (savePath)
       
    # For the figure to be saved show should appear after the save
    plt.show()