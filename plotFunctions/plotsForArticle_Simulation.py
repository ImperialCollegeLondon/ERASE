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
# - 2021-08-13, AK: Add plot for absolute sensor response
# - 2021-05-03, AK: Cosmetic changes to all plots
# - 2021-04-07, AK: Add plot for design variables
# - 2021-03-05, AK: Add plot for full model
# - 2021-03-05, AK: Add plot for three materials
# - 2021-03-05, AK: Add plot for comparing sensor array with graphical tool
# - 2021-02-24, AK: Add function to generate sensitive region for each material
# - 2021-02-23, AK: Add mean error to sensor shape plot
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

    # If sensor response curve needs to be plotted
    if 'responseShape' in kwargs:
        if kwargs["responseShape"]:
            plotForArticle_ResponseShape(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)

    # If graphical tool needs to be plotted
    if 'graphicalTool' in kwargs:
        if kwargs["graphicalTool"]:
            plotForArticle_GraphicalTool(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)

    # If absolute sensor response curve needs to be plotted
    if 'absoluteResponse' in kwargs:
        if kwargs["absoluteResponse"]:
            plotForArticle_AbsoluteResponse(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)

    # If three materials needs to be plotted
    if 'threeMaterials' in kwargs:
        if kwargs["threeMaterials"]:
            plotForArticle_ThreeMaterials(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)

    # If kinetic importance needs to be plotted
    if 'kineticsImportance' in kwargs:
        if kwargs["kineticsImportance"]:
            plotForArticle_KineticsImportance(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)

    # If kinetic importance needs to be plotted
    if 'designVariables' in kwargs:
        if kwargs["designVariables"]:
            plotForArticle_DesignVariables(gitCommitID, currentDT, 
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
    # For now load the results file
    loadFileName = ("arrayConcentration_20210211_1818_b02f8c3.npz", # 1 material w/o constraint
                    "arrayConcentration_20210212_1055_b02f8c3.npz", # 2 material w/o constraint
                    "arrayConcentration_20210212_1050_b02f8c3.npz", # 1 material with constraint
                    "arrayConcentration_20210211_1822_b02f8c3.npz") # 2 material with constraint
    
    # Git commit id of the loaded file
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
            ax.text(0.82, 55, "$n$ = 1", fontsize=10, 
                    backgroundcolor = 'w', color = '#0077b6')
            if ii == 0:
                ax.text(0.075, 55, "(a)", fontsize=10, 
                        backgroundcolor = 'w')
                ax.text(0.53, 51, "Without Constraint", fontsize=10, 
                        backgroundcolor = 'w', color = '#0077b6')
            else:
                ax.set(xlabel='$y$ [-]')
                ax.text(0.075, 55, "(c)", fontsize=10, 
                    backgroundcolor = 'w')
                ax.text(0.595, 51, "With Constraint", fontsize=10, 
                        backgroundcolor = 'w', color = '#0077b6')
            ax.text(0.085, 25, "$y_1$ = 0.05", fontsize=10, 
                    backgroundcolor = 'w', color = '#e5383b')
            ax.text(0.705, 25, "$y_2$ = 0.95", fontsize=10, 
                    backgroundcolor = 'w', color = '#343a40')
        elif ii == 1 or ii == 3:
            ax.text(0.81, 55, "$n$ = 2", fontsize=10, 
                    backgroundcolor = 'w', color = '#0077b6')
            if ii == 1:
                ax.text(0.075, 55, "(b)", fontsize=10, 
                        backgroundcolor = 'w')
                ax.text(0.53, 51, "Without Constraint", fontsize=10, 
                        backgroundcolor = 'w', color = '#0077b6')
            else:
                ax.set(xlabel='$y$ [-]')
                ax.text(0.075, 55, "(d)", fontsize=10, 
                        backgroundcolor = 'w')
                ax.text(0.595, 51, "With Constraint", fontsize=10, 
                        backgroundcolor = 'w', color = '#0077b6')
            ax.text(0.085, 25, "$y_1$ = 0.05", fontsize=10, 
                    backgroundcolor = 'w', color = '#e5383b')
            ax.text(0.705, 25, "$y_2$ = 0.95", fontsize=10, 
                    backgroundcolor = 'w', color = '#343a40')
        # Remove tick labels
        if ii == 0 or ii == 1:
            ax.axes.xaxis.set_ticklabels([])
        if ii == 1 or ii == 3:
            ax.axes.yaxis.set_ticklabels([])
            
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

# fun: plotForArticle_SensorArray
# Plots the sensor response for a given sensor array
def plotForArticle_ResponseShape(gitCommitID, currentDT, 
                               saveFlag, saveFileExtension):
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns 
    import matplotlib.pyplot as plt
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    
    # Materials to be plotted
    sensorID = np.array([17,16,6])
    sensorText = ["A", "B", "C"]
    
    # File to be loaded for the simulation results
    # No Noise
    # loadFileName = ["sensitivityAnalysis_17_20210212_1259_b02f8c3.npz", # No Noise
    #                 "sensitivityAnalysis_16_20210212_1300_b02f8c3.npz", # No Noise
    #                 "sensitivityAnalysis_6_20210212_1259_b02f8c3.npz"] # No Noise
    # Noise (0.1 g/kg)
    loadFileName = ["sensitivityAnalysis_17_20210706_2258_ecbbb3e.npz", # Noise
                    "sensitivityAnalysis_16_20210707_0842_ecbbb3e.npz", # Noise
                    "sensitivityAnalysis_6_20210707_1125_ecbbb3e.npz"] # Noise

    # Colors for plot
    colorsForPlot = ("#5fad56","#f78154","#b4436c")
    
    # Get the sensor response and the sensor sensitive region
    os.chdir("..")
    moleFractionRange, arraySimResponse, _ = getSensorSensitiveRegion(sensorID)
    os.chdir("plotFunctions")
    
    # Plot the figure
    fig = plt.figure
    ax1 = plt.subplot(1,3,1)        
    # Loop through all sensors
    for kk in range(arraySimResponse.shape[1]):
        ax1.plot(moleFractionRange[:,0],arraySimResponse[:,kk],
                 color=colorsForPlot[kk]) # Simulated Response

    ax1.set(xlabel='$y_1$ [-]', 
           ylabel='$m$ [g kg$^{-1}$]',
           xlim = [0,1], ylim = [0, 300])     
    ax1.locator_params(axis="x", nbins=4)
    ax1.locator_params(axis="y", nbins=4)
    ax1.text(0.05, 270, "(a)", fontsize=10, 
            backgroundcolor = 'w')
    
    # Label for the materials     
    ax1.text(0.85, 225, sensorText[0], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[0])
    ax1.text(0.1, 150, sensorText[1], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[1])
    ax1.text(0.8, 75, sensorText[2], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[2])

    # Call the concatenateConcEstimate function
    meanErr, cvData = concatenateConcEstimate(loadFileName[0:3],sensorText)

    # Mean Error - No noise 
    ax2 = plt.subplot(1,3,2)
    meanErr["x"] = pd.to_numeric(meanErr["x"], downcast="float")
    sns.lineplot(data=meanErr, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1),(1,1),(1,1)], markers = ['o','s','D'],
                 palette = colorsForPlot[0:len(loadFileName)], linewidth = 0.5,
                 markersize = 5)
        
    ax2.set(xlabel='$y_1$ [-]', 
            ylabel='$\psi$ [-]',
            xlim = [0.,1.], ylim = [1e-8,100])
    ax2.locator_params(axis="x", nbins=4)
    ax2.set_yscale('log')
    plt.legend([],[], frameon=False)
    ax2.text(0.05, 8, "(b)", fontsize=10, 
            backgroundcolor = 'w')

    # Label for the materials         
    ax2.text(0.85, 1e-2, sensorText[0], fontsize=10, 
        backgroundcolor = 'w', color = colorsForPlot[0])
    ax2.text(0.3, 4e-4, sensorText[1], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[1])
    ax2.text(0.6, 3e-6, sensorText[2], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[2])

    # Label for the formula
    ax2.text(0.38, 7, "$\psi = |\mu - \hat{\mu}|/\mu$", fontsize=10, 
            backgroundcolor = 'w', color = '#0077b6')
    
    # CV - No noise 
    ax3 = plt.subplot(1,3,3)
    cvData["x"] = pd.to_numeric(cvData["x"], downcast="float")
    sns.lineplot(data=cvData, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1),(1,1),(1,1)], markers = ['o','s','D'],
                 palette = colorsForPlot[0:len(loadFileName)], linewidth = 0.5,
                 markersize = 5)
        
    ax3.set(xlabel='$y_1$ [-]', 
            ylabel='$\chi$ [-]',
            xlim = [0.,1.], ylim = [1e-8,100])
    ax3.locator_params(axis="x", nbins=4)
    ax3.set_yscale('log')
    plt.legend([],[], frameon=False)
    ax3.text(0.05, 8, "(c)", fontsize=10, 
            backgroundcolor = 'w')

    # Label for the materials         
    ax3.text(0.85, 3e-1, sensorText[0], fontsize=10, 
        backgroundcolor = 'w', color = colorsForPlot[0])
    ax3.text(0.81, 4e-5, sensorText[1], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[1])
    ax3.text(0.6, 3e-4, sensorText[2], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[2])

    # Label for the formula
    ax3.text(0.62, 7, "$\chi = \hat{\sigma}/\hat{\mu}$", fontsize=10, 
            backgroundcolor = 'w', color = '#0077b6')
    
    #  Save the figure
    if saveFlag:
        # FileName: responseShape_<sensorID>_<currentDateTime>_<GitCommitID_Current>
        sensorText = str(sensorID).replace('[','').replace(']','').replace('  ','-').replace(' ','-')
        saveFileName = "responseShape_" + sensorText + "_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','simulationManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','simulationManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','simulationManuscript'))
        plt.savefig (savePath)
    plt.show()
    
# fun: plotForArticle_AbsoluteResponse
# Plots the sensor response for a given sensor array
def plotForArticle_AbsoluteResponse(gitCommitID, currentDT, 
                               saveFlag, saveFileExtension):
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns 
    import matplotlib.pyplot as plt
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    
    # Materials to be plotted
    sensorID = np.array([6,2,2,2])
    multiplierError = np.array([1,1,3,10])
    sensorText = ["$\\alpha$", "$\\beta$", "$\\beta_3$", "$\\beta_{10}$"]
    arrayText = ["D", "D$_3$", "D$_{10}$"]
    
    # File to be loaded for the simulation results
    # Noise (0.1 g/kg)
    loadFileName =  ["sensitivityAnalysis_6-2_20210719_1117_ecbbb3e.npz", # 1
                     "sensitivityAnalysis_6-2_20210719_2145_ecbbb3e.npz", # 3
                     "sensitivityAnalysis_6-2_20210719_1458_ecbbb3e.npz"] # 10

    # Colors for plot
    colorsForPlot = ("#5fad56","#FF9E00","#D66612","#AD2E24")
    
    # Get the sensor response and the sensor sensitive region
    os.chdir("..")
    moleFractionRange, arraySimResponse, _ = getSensorSensitiveRegion(sensorID)
    os.chdir("plotFunctions")
    
    # Plot the figure
    fig = plt.figure
    ax1 = plt.subplot(1,3,1)        
    # Loop through all sensors
    for kk in range(arraySimResponse.shape[1]):
        ax1.plot(moleFractionRange[:,0],arraySimResponse[:,kk]*multiplierError[kk],
                 color=colorsForPlot[kk]) # Simulated Response

    ax1.set(xlabel='$y_1$ [-]', 
           ylabel='$m$ [g kg$^{-1}$]',
           xlim = [0,1], ylim = [0, 150])     
    ax1.locator_params(axis="x", nbins=4)
    ax1.locator_params(axis="y", nbins=4)
    ax1.text(0.05, 135, "(a)", fontsize=10, 
            backgroundcolor = 'w')
    
    # Label for the materials     
    ax1.text(0.30, 75, sensorText[0], fontsize=10, 
            color = colorsForPlot[0])
    ax1.text(0.8, 17, sensorText[1], fontsize=10, 
            color = colorsForPlot[1])
    ax1.text(0.8, 40, sensorText[2], fontsize=10, 
            color = colorsForPlot[2])
    ax1.text(0.8, 130, sensorText[3], fontsize=10, 
            color = colorsForPlot[3])

    # Call the concatenateConcEstimate function
    meanErr, cvData = concatenateConcEstimate(loadFileName[0:3],sensorText)

    # Mean Error - No noise 
    ax2 = plt.subplot(1,3,2)
    meanErr["x"] = pd.to_numeric(meanErr["x"], downcast="float")
    sns.lineplot(data=meanErr, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1),(1,1),(1,1)], markers = ['o','s','D'],
                 palette = colorsForPlot[1:len(loadFileName)+1], linewidth = 0.5,
                 markersize = 5)
        
    ax2.set(xlabel='$y_1$ [-]', 
            ylabel='$\psi$ [-]',
            xlim = [0.,1.], ylim = [1e-6,1])
    ax2.locator_params(axis="x", nbins=4)
    ax2.set_yscale('log')
    plt.legend([],[], frameon=False)
    ax2.text(0.05, 0.25, "(b)", fontsize=10,)
    plt.minorticks_off()

    # Label for the materials         
    ax2.text(0.85, 8e-4, arrayText[0], fontsize=10, 
        backgroundcolor = 'w', color = colorsForPlot[1])
    ax2.text(0.3, 1.2e-4, arrayText[1], fontsize=10, 
            color = colorsForPlot[2])
    ax2.text(0.63, 3e-6, arrayText[2], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[3])

    # Label for the formula
    ax2.text(0.38, 0.25, "$\psi = |\mu - \hat{\mu}|/\mu$", fontsize=10, 
            backgroundcolor = 'w', color = '#0077b6')
    
    # CV - No noise 
    ax3 = plt.subplot(1,3,3)
    cvData["x"] = pd.to_numeric(cvData["x"], downcast="float")
    sns.lineplot(data=cvData, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1),(1,1),(1,1)], markers = ['o','s','D'],
                 palette = colorsForPlot[1:len(loadFileName)+1], linewidth = 0.5,
                 markersize = 5)
        
    ax3.set(xlabel='$y_1$ [-]', 
            ylabel='$\chi$ [-]',
            xlim = [0.,1.], ylim = [1e-6,1])
    ax3.locator_params(axis="x", nbins=4)
    ax3.set_yscale('log')
    plt.legend([],[], frameon=False)
    ax3.text(0.05, 0.25, "(c)", fontsize=10,)
    plt.minorticks_off()

    # Label for the materials         
    ax3.text(0.8, 1.3e-2, arrayText[0], fontsize=10, 
        color = colorsForPlot[1])
    ax3.text(0.8, 3e-4, arrayText[2], fontsize=10, 
            color = colorsForPlot[3])

    # Label for the formula
    ax3.text(0.62, 0.25, "$\chi = \hat{\sigma}/\hat{\mu}$", fontsize=10, 
            backgroundcolor = 'w', color = '#0077b6')
    
    #  Save the figure
    if saveFlag:
        # FileName: responseShape_<sensorID>_<currentDateTime>_<GitCommitID_Current>
        sensorText = str(sensorID).replace('[','').replace(']','').replace('  ','-').replace(' ','-')
        saveFileName = "absoluteResponse_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','simulationManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','simulationManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','simulationManuscript'))
        plt.savefig (savePath)
    plt.show()

# fun: plotForArticle_GraphicalTool
# Plots the graphical tool to screen for materials
def plotForArticle_GraphicalTool(gitCommitID, currentDT,
                                 saveFlag, saveFileExtension):
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns 
    import matplotlib.pyplot as plt
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # File to be loaded for the simulation results
    # No noise
    # loadFileName = ["sensitivityAnalysis_6-2_20210305_1109_b02f8c3.npz", # 6,2
    #                 "sensitivityAnalysis_17-16_20210305_1050_b02f8c3.npz"] # 17,16
    # Noise
    loadFileName = ["sensitivityAnalysis_6-2_20210719_1117_ecbbb3e.npz", # 6,2
                    "sensitivityAnalysis_17-16_20210706_2120_ecbbb3e.npz"] # 17,16

    # Materials to be plotted
    sensorID = np.array([[6,2],[17,16]])
    arrayText = ["D", "E"]
    materialText = ["$\\alpha$", "$\\beta$", "$\gamma$", "$\delta$"]
    
    # Colors for plot
    colorsForPlot = ("#5fad56","#ff9e00")
    colorLeft = ("#e5383b","#6c757d")
    colorRight = ("#6c757d","#e5383b")
         
    # Plot the figure
    fig = plt.figure
    ax1 = plt.subplot(2,2,1)        
    # Get the sensor response and the sensor sensitive region
    os.chdir("..")
    moleFractionRange, arraySimResponse, sensitiveRegion = getSensorSensitiveRegion(sensorID[0,:])
    os.chdir("plotFunctions")
    # Loop through all sensors
    for kk in range(arraySimResponse.shape[1]):
        ax1.plot(moleFractionRange[:,0],arraySimResponse[:,kk],
                 color=colorsForPlot[kk]) # Simulated Response
        ax1.fill_between(sensitiveRegion[kk,:],1.5*np.max(arraySimResponse), 
                        facecolor=colorsForPlot[kk], alpha=0.25)

    ax1.set(ylabel='$m$ [g kg$^{-1}$]',
           xlim = [0,1], ylim = [0, 150])
    ax1.axes.xaxis.set_ticklabels([])     
    ax1.locator_params(axis="x", nbins=4)
    ax1.locator_params(axis="y", nbins=4)
    ax1.text(0.025, 138, "(a)", fontsize=10)
    ax1.text(0.78, 138, "Array D", fontsize=10, 
             color = '#0077b6')
    
    # Label for the materials     
    ax1.text(0.9, 120, materialText[0], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[0])
    ax1.text(0.9, 23, materialText[1], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[1])    

    ax2 = plt.subplot(2,2,2)
    # Call the concatenateConcEstimate function
    meanErr, cvData = concatenateConcEstimate([loadFileName[0]],arrayText[0])

    # Mean Error - No noise 
    meanErr["x"] = pd.to_numeric(meanErr["x"], downcast="float")
    sns.lineplot(data=meanErr, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1)], markers = ['o'],
                 palette = colorLeft, linewidth = 0.5,markersize = 6)
    ax2.set(ylabel='$\psi$ [-]',
            xlim = [0.,1.], ylim = [1e-8,1])
    ax2.locator_params(axis="x", nbins=4)
    ax2.set(xlabel=None)
    ax2.axes.xaxis.set_ticklabels([])     
    ax2.set_yscale('log')
    ax2.yaxis.label.set_color(colorLeft[0])
    ax2.tick_params(axis='y', colors=colorLeft[0])
    plt.legend([],[], frameon=False)
    # CV - No noise 
    ax2r = plt.twinx()
    cvData["x"] = pd.to_numeric(cvData["x"], downcast="float")
    sns.lineplot(data=cvData, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1)], markers = ['D'],
                 palette = colorRight, linewidth = 0.5,markersize = 6,
                 ax = ax2r)
    # Plot sensitive region
    for kk in range(arraySimResponse.shape[1]):
        ax2r.fill_between(sensitiveRegion[kk,:],1.5*np.max(arraySimResponse), 
                        facecolor=colorsForPlot[kk], alpha=0.25)

    ax2r.set(ylabel='$\chi$ [-]',ylim = [1e-8,1])
    ax2r.locator_params(axis="x", nbins=4)
    ax2r.axes.xaxis.set_ticklabels([]) 
    ax2r.set_yscale('log')
    ax2r.yaxis.label.set_color(colorLeft[1])
    ax2r.tick_params(axis='y', colors=colorLeft[1])
    plt.legend([],[], frameon=False)
    ax2r.annotate("", xy=(0.55, 1e-4), xytext=(0.65, 1e-4), 
                  arrowprops=dict(arrowstyle="-|>", color = colorLeft[0]))
    ax2r.annotate("", xy=(0.95, 3e-2), xytext=(0.85, 3e-2), 
                  arrowprops=dict(arrowstyle="-|>", color = colorLeft[1]))
    ax2r.text(0.025, 0.2, "(b)", fontsize=10)
    ax2r.spines["left"].set_color(colorLeft[0])
    ax2r.spines["right"].set_color(colorLeft[1])
    
    ax2r.text(0.78, 0.2, "Array D", fontsize=10, 
             color = '#0077b6')

    ax3 = plt.subplot(2,2,3)        
    # Get the sensor response and the sensor sensitive region
    os.chdir("..")
    moleFractionRange, arraySimResponse, sensitiveRegion = getSensorSensitiveRegion(sensorID[1,:])
    os.chdir("plotFunctions")
    # Loop through all sensors
    for kk in range(arraySimResponse.shape[1]):
        ax3.plot(moleFractionRange[:,0],arraySimResponse[:,kk],
                 color=colorsForPlot[kk]) # Simulated Response
        ax3.fill_between(sensitiveRegion[kk,:],1.5*np.max(arraySimResponse), 
                        facecolor=colorsForPlot[kk], alpha=0.25)

    ax3.set(xlabel='$y_1$ [-]', 
           ylabel='$m$ [g kg$^{-1}$]',
           xlim = [0,1], ylim = [0, 300])     
    ax3.locator_params(axis="x", nbins=4)
    ax3.locator_params(axis="y", nbins=4)
    ax3.text(0.025, 275, "(c)", fontsize=10)
    ax3.text(0.78, 275, "Array E", fontsize=10,
             color = '#0077b6')
 
    # Label for the materials     
    ax3.text(0.78, 225, materialText[2], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[0])
    ax3.text(0.1, 150, materialText[3], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[1])

    ax4 = plt.subplot(2,2,4)
    # Call the concatenateConcEstimate function
    meanErr, cvData = concatenateConcEstimate([loadFileName[1]],arrayText[1])

    # Mean Error - No noise 
    meanErr["x"] = pd.to_numeric(meanErr["x"], downcast="float")
    sns.lineplot(data=meanErr, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1)], markers = ['o'],
                 palette = colorLeft, linewidth = 0.5,markersize = 6)
    ax4.set(xlabel='$y_1$ [-]',
            ylabel='$\psi$ [-]',
            xlim = [0.,1.], ylim = [1e-8,1])
    ax4.locator_params(axis="x", nbins=4)
    ax4.set_yscale('log')
    ax4.yaxis.label.set_color(colorLeft[0])
    ax4.tick_params(axis='y', colors=colorLeft[0])
    plt.legend([],[], frameon=False)
    # CV - No noise 
    ax4r = plt.twinx()
    cvData["x"] = pd.to_numeric(cvData["x"], downcast="float")
    sns.lineplot(data=cvData, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1)], markers = ['D'],
                 palette = colorRight, linewidth = 0.5,markersize = 6,
                 ax = ax4r)
    # Plot sensitive region
    for kk in range(arraySimResponse.shape[1]):
        ax4r.fill_between(sensitiveRegion[kk,:],1.5*np.max(arraySimResponse), 
                        facecolor=colorsForPlot[kk], alpha=0.25)

    ax4r.set(ylabel='$\chi$ [-]',ylim = [1e-8,1])
    ax4r.locator_params(axis="x", nbins=4)
    ax4r.set_yscale('log')
    ax4r.yaxis.label.set_color(colorLeft[1])
    ax4r.tick_params(axis='y', colors=colorLeft[1])
    plt.legend([],[], frameon=False)
    ax4r.annotate("", xy=(0.2, 5e-4), xytext=(0.3, 5e-4), 
                  arrowprops=dict(arrowstyle="-|>", color = colorLeft[0]))
    ax4r.annotate("", xy=(0.7, 3e-2), xytext=(0.6, 3e-2), 
                  arrowprops=dict(arrowstyle="-|>", color = colorLeft[1]))
    ax4r.text(0.025, 0.2, "(d)", fontsize=10)
    ax4r.spines["left"].set_color(colorLeft[0])
    ax4r.spines["right"].set_color(colorLeft[1])

    ax4r.text(0.78, 0.2, "Array E", fontsize=10, 
             color = '#0077b6')

    #  Save the figure
    if saveFlag:
        # FileName: graphicalTool_<currentDateTime>_<GitCommitID_Current>
        saveFileName = "graphicalTool_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','simulationManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','simulationManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','simulationManuscript'))
        plt.savefig (savePath)
    plt.show()

# fun: plotForArticle_GraphicalTool
# Plots the analaysis for three material arrays
def plotForArticle_ThreeMaterials(gitCommitID, currentDT,
                                  saveFlag, saveFileExtension):
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns 
    import matplotlib.pyplot as plt
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # File to be loaded for the simulation results
    # No Noise
    # loadFileName = ["sensitivityAnalysis_17-15-6_20210306_1515_b02f8c3.npz", # 17,15,6
    #                 "sensitivityAnalysis_17-15-16_20210306_1515_b02f8c3.npz", # 17,15,16
    #                 "sensitivityAnalysis_17-15_20210308_1002_b02f8c3.npz"] # 17,15
    # Noise
    loadFileName = ["sensitivityAnalysis_17-15-6_20210707_2036_ecbbb3e.npz", # 17,15,6
                    "sensitivityAnalysis_17-15-16_20210708_0934_ecbbb3e.npz", # 17,15,16
                    "sensitivityAnalysis_17-15_20210709_1042_ecbbb3e.npz"] # 17,15

    # Materials to be plotted
    sensorID = np.array([[17,15,6],[17,15,16]])
    arrayText = ["F", "G", "Ref"]
    materialText = ["$\\alpha$", "$\\beta$", "$\gamma$", "$\delta$", "$\zeta$"]
    
    # Colors for plot
    colorsForPlot = ("#5fad56","#98c1d9","#ff9e00")
    colorLeft = ("#e5383b","#6c757d")
    colorRight = ("#6c757d","#e5383b")
         
    # Plot the figure
    fig = plt.figure
    ax1 = plt.subplot(2,2,1)        
    # Get the sensor response and the sensor sensitive region
    os.chdir("..")
    moleFractionRange, arraySimResponse, sensitiveRegion = getSensorSensitiveRegion(sensorID[0,:])
    os.chdir("plotFunctions")
    # Loop through all sensors
    for kk in range(arraySimResponse.shape[1]):
        ax1.plot(moleFractionRange[:,0],arraySimResponse[:,kk],
                 color=colorsForPlot[kk]) # Simulated Response
        ax1.fill_between(sensitiveRegion[kk,:],1.5*np.max(arraySimResponse), 
                        facecolor=colorsForPlot[kk], alpha=0.25)

    ax1.set(ylabel='$m$ [g kg$^{-1}$]',
           xlim = [0,1], ylim = [0, 300])
    ax1.axes.xaxis.set_ticklabels([])     
    ax1.locator_params(axis="x", nbins=4)
    ax1.locator_params(axis="y", nbins=4)
    ax1.text(0.025, 275, "(a)", fontsize=10)
    ax1.text(0.78, 275, "Array F", fontsize=10,
             color = '#0077b6')
        
    # Label for the materials     
    ax1.text(0.78, 225, materialText[2], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[0])
    ax1.text(0.78, 25, materialText[4], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[1]) 
    ax1.text(0.78, 80, materialText[0], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[2]) 

    ax2 = plt.subplot(2,2,2)
    # Call the concatenateConcEstimate function
    meanErrRef, cvDataRef = concatenateConcEstimate([loadFileName[2]],arrayText[2])
    meanErr, cvData = concatenateConcEstimate([loadFileName[0]],arrayText[0])

    # Mean Error - No noise 
    meanErr["x"] = pd.to_numeric(meanErr["x"], downcast="float")
    sns.lineplot(data=meanErr, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1)], markers = ['o'],
                 palette = colorLeft, linewidth = 0.5,markersize = 6)
    meanErrRef["x"] = pd.to_numeric(meanErrRef["x"], downcast="float")
    sns.lineplot(data=meanErrRef, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(5,5)], markers = ['o'],
                 palette = colorLeft, linewidth = 0.5, alpha = 0.35,markersize = 6)
    ax2.set(ylabel='$\psi$ [-]',
            xlim = [0.,1.], ylim = [1e-8,1])
    ax2.locator_params(axis="x", nbins=4)
    ax2.set(xlabel=None)
    ax2.axes.xaxis.set_ticklabels([])     
    ax2.set_yscale('log')
    ax2.yaxis.label.set_color(colorLeft[0])
    ax2.tick_params(axis='y', colors=colorLeft[0])
    plt.legend([],[], frameon=False)
    # CV - No noise 
    ax2r = plt.twinx()
    cvData["x"] = pd.to_numeric(cvData["x"], downcast="float")
    sns.lineplot(data=cvData, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1)], markers = ['D'],
                 palette = colorRight, linewidth = 0.5,markersize = 6,
                 ax = ax2r)
    cvDataRef["x"] = pd.to_numeric(cvDataRef["x"], downcast="float")
    sns.lineplot(data=cvDataRef, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(5,5)], markers = ['D'],
                 palette = colorRight, linewidth = 0.5, alpha = 0.35,markersize = 6,
                 ax = ax2r)
    # Plot sensitive region
    for kk in range(arraySimResponse.shape[1]):
        ax2r.fill_between(sensitiveRegion[kk,:],1.5*np.max(arraySimResponse), 
                        facecolor=colorsForPlot[kk], alpha=0.25)

    ax2r.set(ylabel='$\chi$ [-]',ylim = [1e-8,1])
    ax2r.locator_params(axis="x", nbins=4)
    ax2r.axes.xaxis.set_ticklabels([]) 
    ax2r.set_yscale('log')
    ax2r.yaxis.label.set_color(colorLeft[1])
    ax2r.tick_params(axis='y', colors=colorLeft[1])
    plt.legend([],[], frameon=False)
    ax2r.annotate("", xy=(0.4, 3e-5), xytext=(0.5, 3e-5), 
                  arrowprops=dict(arrowstyle="-|>", color = colorLeft[0]))
    ax2r.annotate("", xy=(0.95, 5e-3), xytext=(0.85, 5e-3), 
                  arrowprops=dict(arrowstyle="-|>", color = colorLeft[1]))
    ax2r.text(0.025, 0.2, "(b)", fontsize=10)
    ax2r.spines["left"].set_color(colorLeft[0])
    ax2r.spines["right"].set_color(colorLeft[1])

    ax2r.text(0.78, 1e-6, "Array F", fontsize=10, 
             color = '#0077b6')    
    ax2r.text(0.4, 0.3, "Reference ($\gamma \zeta$)", fontsize=10, 
             color = '#0077b6', alpha = 0.35)

    ax3 = plt.subplot(2,2,3)        
    # Get the sensor response and the sensor sensitive region
    os.chdir("..")
    moleFractionRange, arraySimResponse, sensitiveRegion = getSensorSensitiveRegion(sensorID[1,:])
    os.chdir("plotFunctions")
    # Loop through all sensors
    for kk in range(arraySimResponse.shape[1]):
        ax3.plot(moleFractionRange[:,0],arraySimResponse[:,kk],
                 color=colorsForPlot[kk]) # Simulated Response
        ax3.fill_between(sensitiveRegion[kk,:],1.5*np.max(arraySimResponse), 
                        facecolor=colorsForPlot[kk], alpha=0.25)

    ax3.set(xlabel='$y_1$ [-]', 
           ylabel='$m$ [g kg$^{-1}$]',
           xlim = [0,1], ylim = [0, 300])     
    ax3.locator_params(axis="x", nbins=4)
    ax3.locator_params(axis="y", nbins=4)
    ax3.text(0.025, 275, "(c)", fontsize=10)
    ax3.text(0.78, 275, "Array G", fontsize=10,
             color = '#0077b6')
 
    # Label for the materials     
    ax3.text(0.78, 225, materialText[2], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[0])
    ax3.text(0.78, 25, materialText[4], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[1]) 
    ax3.text(0.1, 150, materialText[3], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[2])

    ax4 = plt.subplot(2,2,4)
    # Call the concatenateConcEstimate function
    meanErr, cvData = concatenateConcEstimate([loadFileName[1]],arrayText[1])

    # Mean Error - No noise 
    meanErr["x"] = pd.to_numeric(meanErr["x"], downcast="float")
    sns.lineplot(data=meanErr, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1)], markers = ['o'],
                 palette = colorLeft, linewidth = 0.5,markersize = 6)
    sns.lineplot(data=meanErrRef, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(5,5)], markers = ['o'],
                 palette = colorLeft, linewidth = 0.5, alpha = 0.35,markersize = 6)
    ax4.set(xlabel='$y_1$ [-]',
            ylabel='$\psi$ [-]',
            xlim = [0.,1.], ylim = [1e-8,1])
    ax4.locator_params(axis="x", nbins=4)
    ax4.set_yscale('log')
    ax4.yaxis.label.set_color(colorLeft[0])
    ax4.tick_params(axis='y', colors=colorLeft[0])
    plt.legend([],[], frameon=False)
    # CV - No noise 
    ax4r = plt.twinx()
    cvData["x"] = pd.to_numeric(cvData["x"], downcast="float")
    sns.lineplot(data=cvData, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1)], markers = ['D'],
                 palette = colorRight, linewidth = 0.5,markersize = 6,
                 ax = ax4r)
    sns.lineplot(data=cvDataRef, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(5,5)], markers = ['o'],
                 palette = colorRight, linewidth = 0.5, alpha = 0.35,markersize = 6,
                 ax = ax4r)
    # Plot sensitive region
    for kk in range(arraySimResponse.shape[1]):
        ax4r.fill_between(sensitiveRegion[kk,:],1.5*np.max(arraySimResponse), 
                        facecolor=colorsForPlot[kk], alpha=0.25)

    ax4r.set(ylabel='$\chi$ [-]',ylim = [1e-8,1])
    ax4r.locator_params(axis="x", nbins=4)
    ax4r.set_yscale('log')
    ax4r.yaxis.label.set_color(colorLeft[1])
    ax4r.tick_params(axis='y', colors=colorLeft[1])
    plt.legend([],[], frameon=False)
    ax4r.annotate("", xy=(0.08, 5e-4), xytext=(0.18, 5e-4), 
                  arrowprops=dict(arrowstyle="-|>", color = colorLeft[0]))
    ax4r.annotate("", xy=(0.72, 1e-2), xytext=(0.62, 1e-2), 
                  arrowprops=dict(arrowstyle="-|>", color = colorLeft[1]))
    ax4r.text(0.025, 0.2, "(d)", fontsize=10)
    ax4r.spines["left"].set_color(colorLeft[0])
    ax4r.spines["right"].set_color(colorLeft[1])

    ax4r.text(0.6, 1e-5, "Array G", fontsize=10, 
             color = '#0077b6')    
    ax4r.text(0.3, 0.3, "Reference ($\gamma \zeta$)", fontsize=10, 
             color = '#0077b6', alpha = 0.35)
    #  Save the figure
    if saveFlag:
        # FileName: threeMaterials_<currentDateTime>_<GitCommitID_Current>
        saveFileName = "threeMaterials_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','simulationManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','simulationManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','simulationManuscript'))
        plt.savefig (savePath)
    plt.show()

# fun: plotForArticle_KineticsImportance
# Plots to highlight the importance of incorporating kinetics
def plotForArticle_KineticsImportance(gitCommitID, currentDT,
                                  saveFlag, saveFileExtension):
    import numpy as np
    from numpy import load
    import os
    import matplotlib.pyplot as plt
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file    
    
    # Colors for plot
    colorsForPlot = ("#5fad56","#ff9e00","#e5383b","#6c757d")
    
    # Labels for materials
    materialText = ["$\\alpha$", "$\\beta$"]
    
    # File name for equilibrium model and full model estimates
    loadFileName_E = "fullModelConcentrationEstimate_6-2_20210320_1336_4b80775.npz" # Eqbm
    loadFileName_F = "fullModelConcentrationEstimate_6-2_20210320_1338_4b80775.npz" # Full model
    
    # Parse out equilbirum results file
    simResultsFile = os.path.join('..','simulationResults',loadFileName_E);
    loadedFile_E = load(simResultsFile, allow_pickle=True)
    concentrationEstimate_E = loadedFile_E["arrayConcentration"]
    
    # Parse out full model results file
    simResultsFile = os.path.join('..','simulationResults',loadFileName_F);
    loadedFile_F = load(simResultsFile, allow_pickle=True)
    concentrationEstimate_F = loadedFile_F["arrayConcentration"]

    # Parse out true responses (this should be the same for both eqbm and full
    # model (here loaded from full model)
    trueResponseStruct = loadedFile_F["outputStruct"].item()
    # Parse out time
    timeSim = []
    timeSim = trueResponseStruct[0]["timeSim"]
    # Parse out feed mole fraction
    feedMoleFrac = trueResponseStruct[0]["inputParameters"][5]
    # Parse out true sensor finger print
    sensorFingerPrint = np.zeros([len(timeSim),len(trueResponseStruct)])
    for ii in range(len(trueResponseStruct)):
        sensorFingerPrint[:,ii] = trueResponseStruct[ii]["sensorFingerPrint"]
        
    # Points that will be taken for sampling (for plots)
    lenSampling = 6
    fig = plt.figure
    # Plot the true sensor response (using the full model)
    ax1 = plt.subplot(1,2,1)
    ax1.plot(timeSim[0:len(timeSim):lenSampling],
            sensorFingerPrint[0:len(timeSim):lenSampling,0],
            marker = 'o', markersize = 2, linestyle = 'dotted', linewidth = 0.5,
            color=colorsForPlot[0])
    ax1.plot(timeSim[0:len(timeSim):lenSampling],
            sensorFingerPrint[0:len(timeSim):lenSampling,1],
            marker = 'D', markersize = 2, linestyle = 'dotted', linewidth = 0.5,
            color=colorsForPlot[1])
    ax1.locator_params(axis="x", nbins=4)
    ax1.locator_params(axis="y", nbins=4)
    ax1.set(xlabel='$t$ [s]', 
            ylabel='$m$ [g kg$^{\mathregular{-1}}$]',
            xlim = [0, 1000.], ylim = [0, 40])
    
    ax1.text(20, 37, "(a)", fontsize=10)
    ax1.text(800, 37, "Array D", fontsize=10, 
             color = '#0077b6')
    ax1.text(720, 33.5, "$y^{\mathregular{in}}_{\mathregular{1}} (t)$ = 0.1", fontsize=10, 
             color = '#0077b6')
    
    # Label for the materials     
    ax1.text(900, 26, materialText[0], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[0])
    ax1.text(900, 4, materialText[1], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[1])    
    
    # Plot the evolution of the gas composition with respect to time
    ax2 = plt.subplot(1,2,2)
    ax2.plot(timeSim[0:len(timeSim):lenSampling],
            concentrationEstimate_E[0:len(timeSim):lenSampling,2],
            marker = 'v', markersize = 2, linestyle = 'dotted', linewidth = 0.5,
            color=colorsForPlot[2])
    ax2.plot(timeSim[0:len(timeSim):lenSampling],
            concentrationEstimate_F[0:len(timeSim):lenSampling,2],
            marker = '^', markersize = 2, linestyle = 'dotted', linewidth = 0.5,
            color=colorsForPlot[3])  
    ax2.locator_params(axis="x", nbins=4)
    ax2.locator_params(axis="y", nbins=4)
    ax2.set(xlabel='$t$ [s]', 
            ylabel='$y_1$ [-]',
            xlim = [0, 1000.], ylim = [0, 0.2])

    ax2.text(20, 0.185, "(b)", fontsize=10)
    ax2.text(800, 0.185, "Array D", fontsize=10, 
             color = '#0077b6')
    ax2.text(720, 0.11, "$y^{\mathregular{in}}_{\mathregular{1}} (t)$ = 0.1", fontsize=10, 
             color = '#0077b6')
    
    # Label for the materials     
    ax2.text(280, 0.06, "Equilibrium Model", fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[2])
    ax2.text(50, 0.11, "Full Model", fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[3])   
    
    #  Save the figure
    if saveFlag:
        # FileName: kineticsImportance_<currentDateTime>_<GitCommitID_Current>
        saveFileName = "kineticsImportance_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','simulationManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','simulationManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','simulationManuscript'))
        plt.savefig (savePath)
    plt.show()

# fun: plotForArticle_DesignVariables
# Plots to highlight the effect of different design variables
def plotForArticle_DesignVariables(gitCommitID, currentDT,
                                   saveFlag, saveFileExtension):
    import numpy as np
    from numpy import load
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.transforms import Bbox

    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file   

    # Curved arrow properties    
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="#0077b6")
    
    colorsForPlot = ["#E5383B","#CD4448","#B55055",
                     "#9C5D63","#846970","#6C757D"]
    plotIndex = ["(a)", "(b)", "(c)", "(d)"]
    
    # For now load the results file
    loadFileName = ("fullModelSensitivity_rateConstant_20210324_0957_c9e8179.npz", # Varying rate constant
                    "fullModelSensitivity_flowIn_20210407_1656_c9e8179.npz", # Varying flow in 
                    "fullModelSensitivity_volTotal_20210324_1013_c9e8179.npz", # Varying total volume 
                    "fullModelSensitivity_voidFrac_20210324_1006_c9e8179.npz") # Varying void fraction
        
    # Loop through the two files to get the histogram
    for ii in range(len(loadFileName)):
        ax = plt.subplot(2,2,ii+1)
        # Create the file name with the path to be loaded
        simResultsFile = os.path.join('..','simulationResults',loadFileName[ii]);
    
        # Check if the file with the simultaion results exist 
        if os.path.exists(simResultsFile):
            # Load the file
            loadedFileTemp = load(simResultsFile,allow_pickle=True)["outputStruct"]
             # Unpack to get the dictionary
            loadedFile = loadedFileTemp[()]
            
        # Prepare the data for plotting
        timeSim = loadedFile[0]["timeSim"]; # Time elapsed [s]
        adsorbentDensity = loadedFile[0]["inputParameters"][1]; # [kg/m3]        
        sensorFingerPrint = np.zeros((len(timeSim),len(loadedFile)));
        adsorbentVolume = np.zeros((len(loadedFile),1));
        # Loop over to get the sensor finger print and adsorbent volume
        for jj in range(len(loadedFile)):
            fingerPrintTemp = (loadedFile[jj]["sensorFingerPrint"]
                               *loadedFile[jj]["inputParameters"][9]
                               *adsorbentDensity) # Compute true response [g]
            ax.plot(timeSim, fingerPrintTemp,
                     linewidth=1.5,color=colorsForPlot[jj])
        ax.set(xlim = [timeSim[0], 2000], ylim = [0, 0.12])
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=5)
        if ii == 2 or ii  == 3:
            ax.set(xlabel = '$t$ [s]')
        if ii == 0 or ii  == 2:
            ax.set(ylabel = '$m$ [g]')
        # Remove ticks
        if ii == 0 or ii == 1:
            ax.axes.xaxis.set_ticklabels([])            
        if ii == 1 or ii == 3:
            ax.axes.yaxis.set_ticklabels([])
        # Create labels for the plots
        ax.text(50, 0.11, plotIndex[ii], fontsize=10)
        ax.text(1600, 0.11, "Array C", fontsize=10, 
             color = '#0077b6')
        if ii == 0:
            curvArr = patches.FancyArrowPatch((800, 0.02), (300, 0.06),
                                 connectionstyle="arc3,rad=0.35", **kw)
            ax.add_patch(curvArr)            
            ax.text(300, 0.065, "$k$", fontsize=10, 
                    color = '#0077b6')
            ax.text(1240, 0.101, "Varying Kinetics", fontsize=10, 
                    color = '#0077b6')
            ax.text(1700, 0.0025, "0.0001", fontsize=8, 
                        color = colorsForPlot[0])
            ax.text(1700, 0.023, "0.0005", fontsize=8, 
                        color = colorsForPlot[1], rotation=10)
            ax.text(1700, 0.037, "0.001", fontsize=8, 
                        color = colorsForPlot[2], rotation=8)
            ax.text(200, 0.027, "0.005", fontsize=8, 
                        color = colorsForPlot[3], rotation=45)
            ax.text(60, 0.042, "0.01", fontsize=8, 
                        color = colorsForPlot[4], rotation=45)
            ax.text(1700, 0.056, "10000", fontsize=8, 
                        color = colorsForPlot[5], rotation=0)
                
        if ii == 1:
            curvArr = patches.FancyArrowPatch((800, 0.02), (300, 0.06),
                                 connectionstyle="arc3,rad=0.35", **kw)
            ax.add_patch(curvArr)   
            ax.text(300, 0.065, "$F^\mathregular{in}$", fontsize=10, 
                    color = '#0077b6')
            ax.text(1140, 0.101, "Varying Flow Rate", fontsize=10, 
                    color = '#0077b6')
            ax.text(1700, 0.005, "0.001", fontsize=8, 
                    color = colorsForPlot[0])
            ax.text(1700, 0.017, "0.005", fontsize=8, 
                        color = colorsForPlot[1], rotation=10)
            ax.text(1700, 0.033, "0.01", fontsize=8, 
                        color = colorsForPlot[2], rotation=12)
            ax.text(330, 0.023, "0.05", fontsize=8, 
                        color = colorsForPlot[3], rotation=45)
            ax.text(230, 0.032, "0.1", fontsize=8, 
                        color = colorsForPlot[4], rotation=60)
            ax.text(1800, 0.056, "1", fontsize=8, 
                        color = colorsForPlot[5], rotation=0)
        if ii == 2:
            curvArr = patches.FancyArrowPatch((800, 0.01), (300, 0.06),
                                 connectionstyle="arc3,rad=0.35", **kw)
            ax.add_patch(curvArr)
            ax.text(300, 0.065, "$V_\mathregular{T}$", fontsize=10, 
                    color = '#0077b6')
            ax.text(1020, 0.101, "Varying Total Volume", fontsize=10, 
                    color = '#0077b6')
            ax.text(1820, 0.008, "0.1", fontsize=8, 
                        color = colorsForPlot[2], rotation=0)
            ax.text(1820, 0.018, "0.3", fontsize=8, 
                        color = colorsForPlot[3], rotation=0)
            ax.text(1820, 0.034, "0.6", fontsize=8, 
                        color = colorsForPlot[4], rotation=0)
            ax.text(1820, 0.056, "1.0", fontsize=8, 
                        color = colorsForPlot[5], rotation=0)
        if ii == 3:
            curvArr = patches.FancyArrowPatch((30, 0.08), (800, 0.015),
                                 connectionstyle="arc3,rad=-0.35", **kw)
            ax.add_patch(curvArr)
            ax.text(800, 0.035, "$\epsilon$", fontsize=10, 
                    color = '#0077b6')
            ax.text(960, 0.101, "Varying Dead Voidage", fontsize=10, 
                    color = '#0077b6')
            ax.text(1800, 0.090, "0.10", fontsize=8, 
                    color = colorsForPlot[0])
            ax.text(1800, 0.074, "0.25", fontsize=8, 
                        color = colorsForPlot[1])
            ax.text(1800, 0.056, "0.50", fontsize=8, 
                        color = colorsForPlot[2])
            ax.text(1820, 0.029, "0.75", fontsize=8, 
                        color = colorsForPlot[3])
            ax.text(1820, 0.013, "0.90", fontsize=8, 
                        color = colorsForPlot[4])
            ax.text(1820, 0.003, "0.99", fontsize=8, 
                        color = colorsForPlot[5])
        
    #  Save the figure
    if saveFlag:
        # FileName: designVariables_<currentDateTime>_<GitCommitID_Current>
        saveFileName = "designVariables_" + currentDT + "_" + gitCommitID + saveFileExtension
        savePath = os.path.join('..','simulationFigures','simulationManuscript',saveFileName)
        # Check if inputResources directory exists or not. If not, create the folder
        if not os.path.exists(os.path.join('..','simulationFigures','simulationManuscript')):
            os.mkdir(os.path.join('..','simulationFigures','simulationManuscript'))
        plt.savefig (savePath)
    plt.show()


# fun: getSensorSensitiveRegion
# Simulate the sensor array and obtain the region of sensitivity
def getSensorSensitiveRegion(sensorID):
    import numpy as np
    from simulateSensorArray import simulateSensorArray
    from kneed import KneeLocator # To compute the knee/elbow of a curve

    # Total pressure of the gas [Pa]
    pressureTotal = np.array([1.e5]);
    
    # Temperature of the gas [K]
    # Can be a vector of temperatures
    temperature = np.array([298.15]);
    
    # Number of molefractions
    numMolFrac= 101
    moleFractionRange = np.array([np.linspace(0,1,numMolFrac), 1 - np.linspace(0,1,numMolFrac)]).T

    # Simulate the sensor response for all possible concentrations
    arraySimResponse = np.zeros([moleFractionRange.shape[0],sensorID.shape[0]])
    for ii in range(moleFractionRange.shape[0]):
        arraySimResponse[ii,:] = simulateSensorArray(sensorID, pressureTotal, 
                                                   temperature, np.array([moleFractionRange[ii,:]]))

    # Compute the sensitive region for each sensor material in the array
    sensitiveRegion = np.zeros([arraySimResponse.shape[1],2])
    # Loop through all the materials
    firstDerivative = np.zeros([arraySimResponse.shape[0],arraySimResponse.shape[1]])
    firstDerivativeSimResponse_y1 = np.zeros([moleFractionRange.shape[0],arraySimResponse.shape[1]])
    firstDerivativeSimResponse_y2 = np.zeros([moleFractionRange.shape[0],arraySimResponse.shape[1]])
    for kk in range(arraySimResponse.shape[1]):
        firstDerivative[:,kk] = np.gradient(arraySimResponse[:,kk],moleFractionRange[1,0]-moleFractionRange[0,0])
        firstDerivativeSimResponse_y1[:,kk] = np.gradient(moleFractionRange[:,0],arraySimResponse[:,kk])
        firstDerivativeSimResponse_y2[:,kk] = np.gradient(moleFractionRange[:,1],arraySimResponse[:,kk])
        # Get the sign of the first derivative for increasing/decreasing
        if all(i >= 0. for i in firstDerivative[:,kk]):
            slopeDir = "increasing"
        elif all(i < 0. for i in firstDerivative[:,kk]):
            slopeDir = "decreasing"
        else:
            print("Dangerous! I should not be here!!!")
    
        # Compute the knee/elbow of the curve
        kneedle = KneeLocator(moleFractionRange[:,0], arraySimResponse[:,kk], 
                              direction=slopeDir)
        elbowPoint = list(kneedle.all_elbows)

        # Obtain coordinates to fill working region
        if slopeDir == "increasing":
            sensitiveRegion[kk,:] = [0,elbowPoint[0]]
        else:
            sensitiveRegion[kk,:] = [elbowPoint[0], 1.0]

    # Return the mole fraction, response and sensitive region for each 
    # material
    return moleFractionRange, arraySimResponse, sensitiveRegion

# fun: concatenateConcEstimate
# Concatenates concentration estimates into a panda dataframe and computes 
# the coefficient of variation
def concatenateConcEstimate(loadFileName,sensorText):
    import numpy as np
    from numpy import load
    import os
    import pandas as pd

    # Initialize x, y, and type for the plotting
    concatenatedX = []
    concatenatedY1 = []
    concatenatedY2 = []
    concatenatedType = []

    # Loop through the different files to generate the violin plot
    for kk in range(len(loadFileName)):
        # Initialize x, y, and type for the local loop
        xVar = []
        y1Var = []
        y2Var = []
        typeVar = []
    
        simResultsFile = os.path.join('..','simulationResults',loadFileName[kk]);
        resultOutput = load(simResultsFile)["arrayConcentration"]
        moleFrac = load(simResultsFile)["trueMoleFrac"]

        # Loop through all the molefractions
        for ii in range(resultOutput.shape[0]):
            if resultOutput.shape[2] == 3:
                counterInd = -1
            elif resultOutput.shape[2] == 4:
                counterInd = 0
            elif resultOutput.shape[2] == 5:
                counterInd = 1
                
            y1Var = np.concatenate((y1Var,resultOutput[ii,:,counterInd+2])) # y1
            y2Var = np.concatenate((y2Var,resultOutput[ii,:,counterInd+3])) # y2
            xVar = xVar + ([str(moleFrac[ii])] * len(resultOutput[ii,:,counterInd+2])) # x (true mole fraction)
            typeVar = typeVar+[sensorText[kk]] * len(resultOutput[ii,:,counterInd+2])
     
        # Concatenate all the data to form a data frame with x, y, and type
        concatenatedX = concatenatedX + xVar
        concatenatedY1 = np.concatenate((concatenatedY1,y1Var))
        concatenatedY2 = np.concatenate((concatenatedY2,y2Var))
        concatenatedType = concatenatedType + typeVar    
        # Reinitialize all the loaded values to empty variable
        simResultsFile = []
        resultOutput = []
        moleFrac = []
    
    # Generate panda data frame
    # x = molefraction (true)
    # y = molefraction (estimated)
    # dataType = either sensor id/comparison type
    df = pd.DataFrame({'x':concatenatedX,
                       'y1':concatenatedY1,
                       'y2':concatenatedY2,
                       'dataType':concatenatedType})

    # Compute the mean and standard deviation
    meanData = df.groupby(['dataType','x'], as_index=False, sort=False).mean() 
    stdData = df.groupby(['dataType','x'], as_index=False, sort=False).std()

    # Compute the relative error of the mean (non-negative)
    meanErr = stdData.copy()
    meanErr['y1'] = abs(meanData['x'].astype(float) - meanData['y1'])/meanData['x'].astype(float)
    meanErr['y2'] = abs((1.-meanData['x'].astype(float)) - meanData['y2'])/(1.-meanData['x'].astype(float))

    # Coefficient of variation
    cvData = stdData.copy()
    cvData['y1'] = stdData['y1']/meanData['y1']
    cvData['y2'] = stdData['y2']/meanData['y2']

    # Return the coefficient of variation
    return meanErr, cvData