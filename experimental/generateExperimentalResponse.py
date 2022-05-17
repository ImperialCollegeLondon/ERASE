############################################################################
#
# Imperial College London, United Kingdom
# Multifunctional Nanomaterials Laboratory
#
# Project:  ERASE
# Year:     2021
# Python:   Python 3.7
# Authors:  Ashwin Kumar Rajagopalan (AK)
#
# Purpose:
# Generates "true" experimetnal response for the ZLC+Dead volume setup
# and saves the time, mole fraction and the flow rate in the same fashion
# as is done by the the experimental setup. The output from this file is a
# .mat file that can then be fed to the parameter estimator.
#
# Last modified:
# - 2021-08-23, AK: Structure changes to reflect new kinetics
# - 2021-08-11, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

from simulateCombinedModel import simulateCombinedModel
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as sio

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

##### USER INPUT #####
# Experimental file name
# Note that one file name corresponds to one flow rate, one temperature
# Alphabets in the final file denotes the mole fraction
fileName = ['ZLC_ActivatedCarbon_Sim01',
            'ZLC_ActivatedCarbon_Sim02']

# Material isotherm parameters, kinetic rate constants, sorbent mass, density,
# and poroisty
# Note that the material isotherm parameters is obtained from the Quantachrome
# measurements
#### Activated Carbon (dimensional) ####
x = [4.65e-1, 1.02e-5, 2.51e4, 6.51, 3.51e-7, 2.57e4, 1.019, 16.787];
adsorbentDensity = 1680 # Skeletal density [kg/m3]
massSorbent = 0.0625  # Mass of sorbent [g]
particleEpsilon = 0.61  # Particle porosity [-]
#######################################

#### Boron Nitride (dimensional) ####
# x = [7.01, 2.32e-07, 2.49e4, 0.082, 302.962];
# adsorbentDensity = 3400 # Skeletal density [kg/m3]
# massSorbent = 0.0797  # Mass of sorbent [g]
# particleEpsilon = 0.88  # Particle porosity [-]
#######################################

#### Zeolite 13X (dimensional) ####
# x = [3.83, 1.33e-08, 40.0e3, 2.57, 4.88e-06, 35.16e3, 6.64e2, 7.61e1];
# adsorbentDensity = 4100 # Skeletal density [kg/m3]
# massSorbent = 0.0594 # Mass of sorbent [g]
# particleEpsilon = 0.79 # Particle porosity [-]
#######################################

# Temperature of the simulate experiment [K]
temperature = 308.15

# Inlet flow rate [ccm]
flowRate = [10, 60]

# Saturation mole fraction (works for a binary system)
initMoleFrac = np.array(([0.11, 0.94], [0.11, 0.73]))

# Dead volume file for the setup
deadVolumeFile = 'deadVolumeCharacteristics_20210810_1653_eddec53.npz'

############

# Integration time (set to 1000 s, default)
timeInt = (0.0,1000.0)

# Volume of sorbent material [m3]
volSorbent = (massSorbent/1000)/adsorbentDensity
# Volume of gas in pores [m3]
volGas = volSorbent/(1-particleEpsilon)*particleEpsilon

# Create the instance for the plots
fig = plt.figure
ax1 = plt.subplot(1,3,1)        
ax2 = plt.subplot(1,3,2)
ax3 = plt.subplot(1,3,3)
# Plot colors
colorsForPlot = ["#FE7F2D","#233D4D"]*2
markerForPlot = ["o"]*4

# Loop over all the conditions
for ii in range(len(flowRate)):
    for jj in range(np.size(initMoleFrac,1)):
        # Initialize the output dictionary
        experimentOutput = {}
        # Compute the composite response using the optimizer parameters
        timeElapsedSim , moleFracSim , resultMat = simulateCombinedModel(isothermModel = x[0:-2],
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
        lastIndThreshold = int(np.argwhere(moleFracSim<=1e-2)[0])

        # Cut the time, mole fraction and the flow rate to the last index
        # threshold
        timeExp = timeElapsedSim[0:lastIndThreshold] # Time elapsed [s]
        moleFrac = moleFracSim[0:lastIndThreshold] # Mole fraction [-]
        totalFlowRate = resultMat[3,0:lastIndThreshold]*1e6 # Total flow rate[ccs]
        
        # Save the output and git commit ID to .mat file (similar to experiments)
        experimentOutput = {'timeExp': timeExp.reshape(len(timeExp),1),
                            'moleFrac': moleFrac.reshape(len(moleFrac),1),
                            'totalFlowRate': totalFlowRate.reshape(len(totalFlowRate),1)}
        saveFileName = fileName[ii] + chr(65+jj) + '_Output.mat'
        sio.savemat('runData' + os.path.sep + saveFileName, 
                    {'experimentOutput': experimentOutput, # This is the only thing used for the parameter estimator (same as experiemnt)
                    # The fields below are saved only for checking purposes
                    'gitCommitID': gitCommitID,
                    'modelParameters': x,
                    'adsorbentDensity': adsorbentDensity,
                    'massSorbent': massSorbent,
                    'particleEpsilon': particleEpsilon,
                    'temperature': temperature,
                    'flowRate': flowRate,
                    'initMoleFrac': initMoleFrac,
                    'deadVolumeFile': deadVolumeFile})

        # Plot the responses for sanity check
        # y - Linear scale
        ax1.semilogy(timeExp,moleFrac,
                marker = markerForPlot[ii],linewidth = 0,
                color=colorsForPlot[ii],alpha=0.1) # Experimental response

        ax1.set(xlabel='$t$ [s]', 
                ylabel='$y_1$ [-]',
                xlim = [0,250], ylim = [1e-2, 1])    
        ax1.locator_params(axis="x", nbins=4)
        ax1.legend()

        # Ft - Log scale        
        ax2.semilogy(np.multiply(totalFlowRate,timeExp),moleFrac,
                      marker = markerForPlot[ii],linewidth = 0,
                      color=colorsForPlot[ii],alpha=0.1) # Experimental response
        ax2.set(xlabel='$Ft$ [cc]', 
                xlim = [0,60], ylim = [1e-2, 1])         
        ax2.locator_params(axis="x", nbins=4)
        
        # Flow rates
        ax3.plot(timeExp,totalFlowRate,
                marker = markerForPlot[ii],linewidth = 0,
                color=colorsForPlot[ii],alpha=0.1,label=str(round(np.mean(totalFlowRate),2))+" ccs") # Experimental response
        ax3.set(xlabel='$t$ [s]', 
                ylabel='$F$ [ccs]',
                xlim = [0,250], ylim = [0, 1.2])
        ax3.locator_params(axis="x", nbins=4)
        ax3.locator_params(axis="y", nbins=4)