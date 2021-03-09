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
            meanErr = plotForArticle_ResponseShape(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)

    # If graphical tool needs to be plotted
    if 'graphicalTool' in kwargs:
        if kwargs["graphicalTool"]:
            plotForArticle_GraphicalTool(gitCommitID, currentDT, 
                                       saveFlag, saveFileExtension)

    # If graphical tool needs to be plotted
    if 'threeMaterials' in kwargs:
        if kwargs["threeMaterials"]:
            plotForArticle_ThreeMaterials(gitCommitID, currentDT, 
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
                ax.text(0.56, 51, "Without Constraint", fontsize=10, 
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
                ax.text(0.56, 51, "Without Constraint", fontsize=10, 
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
    loadFileName = ["sensitivityAnalysis_17_20210212_1259_b02f8c3.npz", # No Noise
                    "sensitivityAnalysis_16_20210212_1300_b02f8c3.npz", # No Noise
                    "sensitivityAnalysis_6_20210212_1259_b02f8c3.npz"] # No Noise
                    # "sensitivityAnalysis_17_20210212_1355_b02f8c3.npz", # Noise
                    # "sensitivityAnalysis_16_20210212_1356_b02f8c3.npz" # Noise
                    # "sensitivityAnalysis_6_20210212_1355_b02f8c3.npz"] # Noise
    
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
                 palette = colorsForPlot[0:len(loadFileName)], linewidth = 0.5)
        
    ax2.set(xlabel='$y_1$ [-]', 
            ylabel='$\psi$ [-]',
            xlim = [0.,1.], ylim = [1e-10,100])
    ax2.locator_params(axis="x", nbins=4)
    ax2.set_yscale('log')
    plt.legend([],[], frameon=False)
    ax2.text(0.05, 8, "(b)", fontsize=10, 
            backgroundcolor = 'w')

    # Label for the materials         
    ax2.text(0.85, 6e-3, sensorText[0], fontsize=10, 
        backgroundcolor = 'w', color = colorsForPlot[0])
    ax2.text(0.3, 4e-4, sensorText[1], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[1])
    ax2.text(0.6, 5e-7, sensorText[2], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[2])

    # Label for the formula
    ax2.text(0.38, 7, "$\psi = |\mu - \hat{\mu}|/\mu$", fontsize=10, 
            backgroundcolor = 'w', color = '#0077b6')
    
    # CV - No noise 
    ax3 = plt.subplot(1,3,3)
    cvData["x"] = pd.to_numeric(cvData["x"], downcast="float")
    sns.lineplot(data=cvData, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(1,1),(1,1),(1,1)], markers = ['o','s','D'],
                 palette = colorsForPlot[0:len(loadFileName)], linewidth = 0.5)
        
    ax3.set(xlabel='$y_1$ [-]', 
            ylabel='$\chi$ [-]',
            xlim = [0.,1.], ylim = [1e-10,100])
    ax3.locator_params(axis="x", nbins=4)
    ax3.set_yscale('log')
    plt.legend([],[], frameon=False)
    ax3.text(0.05, 8, "(c)", fontsize=10, 
            backgroundcolor = 'w')

    # Label for the materials         
    ax3.text(0.85, 6e-2, sensorText[0], fontsize=10, 
        backgroundcolor = 'w', color = colorsForPlot[0])
    ax3.text(0.81, 4e-4, sensorText[1], fontsize=10, 
            backgroundcolor = 'w', color = colorsForPlot[1])
    ax3.text(0.6, 1e-6, sensorText[2], fontsize=10, 
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

# fun: plotForArticle_GraphicalTool
# Plots the histogram of gas compositions for a one and two material 
# sensor array
def plotForArticle_GraphicalTool(gitCommitID, currentDT,
                                 saveFlag, saveFileExtension):
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns 
    import matplotlib.pyplot as plt
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # File to be loaded for the simulation results
    loadFileName = ["sensitivityAnalysis_6-2_20210305_1109_b02f8c3.npz", # 6,2
                    "sensitivityAnalysis_17-16_20210305_1050_b02f8c3.npz"] # 17,16

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
                 palette = colorLeft, linewidth = 0.5)
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
                 palette = colorRight, linewidth = 0.5,
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
    ax2r.annotate("", xy=(0.5, 5e-7), xytext=(0.6, 5e-7), 
                  arrowprops=dict(arrowstyle="-|>", color = colorLeft[0]))
    ax2r.annotate("", xy=(0.95, 1e-5), xytext=(0.85, 1e-5), 
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
                 palette = colorLeft, linewidth = 0.5)
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
                 palette = colorRight, linewidth = 0.5,
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
    ax4r.annotate("", xy=(0.7, 1e-3), xytext=(0.6, 1e-3), 
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
# Plots the histogram of gas compositions for a one and two material 
# sensor array
def plotForArticle_ThreeMaterials(gitCommitID, currentDT,
                                  saveFlag, saveFileExtension):
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns 
    import matplotlib.pyplot as plt
    plt.style.use('doubleColumn2Row.mplstyle') # Custom matplotlib style file

    # File to be loaded for the simulation results
    loadFileName = ["sensitivityAnalysis_17-15-6_20210306_1515_b02f8c3.npz", # 17,15,6
                    "sensitivityAnalysis_17-15-16_20210306_1515_b02f8c3.npz", # 17,15,16
                    "sensitivityAnalysis_17-15_20210308_1002_b02f8c3.npz"] # 17,15

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
                 palette = colorLeft, linewidth = 0.5)
    meanErrRef["x"] = pd.to_numeric(meanErrRef["x"], downcast="float")
    sns.lineplot(data=meanErrRef, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(5,5)], markers = ['o'],
                 palette = colorLeft, linewidth = 0.5, alpha = 0.35)
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
                 palette = colorRight, linewidth = 0.5,
                 ax = ax2r)
    cvDataRef["x"] = pd.to_numeric(cvDataRef["x"], downcast="float")
    sns.lineplot(data=cvDataRef, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(5,5)], markers = ['D'],
                 palette = colorRight, linewidth = 0.5, alpha = 0.35,
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
    ax2r.annotate("", xy=(0.4, 2e-6), xytext=(0.5, 2e-6), 
                  arrowprops=dict(arrowstyle="-|>", color = colorLeft[0]))
    ax2r.annotate("", xy=(0.95, 5e-2), xytext=(0.85, 5e-2), 
                  arrowprops=dict(arrowstyle="-|>", color = colorLeft[1]))
    ax2r.text(0.025, 0.2, "(b)", fontsize=10)
    ax2r.spines["left"].set_color(colorLeft[0])
    ax2r.spines["right"].set_color(colorLeft[1])

    ax2r.text(0.78, 3e-6, "Array F", fontsize=10, 
             color = '#0077b6')    
    ax2r.text(0.4, 0.05, "Reference ($\gamma \zeta$)", fontsize=10, 
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
                 palette = colorLeft, linewidth = 0.5)
    sns.lineplot(data=meanErrRef, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(5,5)], markers = ['o'],
                 palette = colorLeft, linewidth = 0.5, alpha = 0.35)
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
                 palette = colorRight, linewidth = 0.5,
                 ax = ax4r)
    sns.lineplot(data=cvDataRef, x='x', y='y1', hue='dataType', style='dataType',
                 dashes = [(5,5)], markers = ['o'],
                 palette = colorRight, linewidth = 0.5, alpha = 0.35,
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

    ax4r.text(0.6, 1e-5, "Array G", fontsize=10, 
             color = '#0077b6')    
    ax4r.text(0.3, 0.1, "Reference ($\gamma \zeta$)", fontsize=10, 
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