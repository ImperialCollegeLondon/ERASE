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
# Script to perform a sensitivity analysis on the sensor response and 
# concentration estimate
#
# Last modified:
# - 2020-11-12, AK: Save arrayConcentration in output
# - 2020-11-12, AK: Bug fix for multipler error
# - 2020-11-11, AK: Add multipler nosie
# - 2020-11-10, AK: Improvements to run in HPC
# - 2020-11-10, AK: Add measurement nosie
# - 2020-11-06, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

import numpy as np
from numpy import savez
import multiprocessing # For parallel processing
from joblib import Parallel, delayed  # For parallel processing
import auxiliaryFunctions
import os
import sys
from tqdm import tqdm # To track progress of the loop
from estimateConcentration import estimateConcentration
import argparse

# For atgument parser if run through terminal. Sensor configuration provided
# as an input using --s and sorbent ids separated by a space
# e.g. python sensitivityAnalysis.py --s 6 2
parser = argparse.ArgumentParser()
parser.add_argument('--s', nargs='+', type=int)

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
simulationDT = auxiliaryFunctions.getCurrentDateTime()

# Find out the total number of cores available for parallel processing
num_cores = multiprocessing.cpu_count()

# Number of adsorbents
numberOfAdsorbents = 30

# Number of gases
numberOfGases = 2

# Sensor combination
# Check if argument provided (from terminal)
if len(sys.argv)>1:
    print("Sensor configuration provided!")
    for _, value in parser.parse_args()._get_kwargs():
        sensorID = value
# Use default values
else:
    print("\nSensor configuration not not provided. Default used!")
    sensorID = [6, 2]

# Measurement noise (Guassian noise)
meanError = 0. # [g/kg]
stdError = 0.1 # [g/kg]

# Multipler error for the sensor measurement
multiplierError = [1., 1.]

# Custom input mole fraction for gas 1
meanMoleFracG1 = [0.001, 0.01, 0.1, 0.25, 0.50, 0.75, 0.90]
# meanMoleFracG1 = [0.90]
        
diffMoleFracG1 = 0.00 # This plus/minus the mean is the bound for uniform dist.
numberOfIterations = 100

# Custom input mole fraction for gas 2 (for 3 gas system)
meanMoleFracG2 = 0.20

# Initialize mean and standard deviation of concentration estimates
meanConcEstimate = np.zeros([len(meanMoleFracG1),numberOfGases])
stdConcEstimate = np.zeros([len(meanMoleFracG1),numberOfGases])

# Initialize the arrayConcentration matrix
arrayConcentration = np.zeros([len(meanMoleFracG1),numberOfIterations,
                               numberOfGases+len(sensorID)])

for ii in range(len(meanMoleFracG1)):
    # Generate a uniform distribution of mole fractions
    if numberOfGases == 2:
        inputMoleFrac = np.zeros([numberOfIterations,2])
        inputMoleFrac[:,0] = np.random.uniform(meanMoleFracG1[ii]-diffMoleFracG1,
                                          meanMoleFracG1[ii]+diffMoleFracG1,
                                          numberOfIterations)
        inputMoleFrac[:,1] = 1. - inputMoleFrac[:,0]
    elif numberOfGases == 3:
        inputMoleFrac = np.zeros([numberOfIterations,3])
        inputMoleFrac[:,0] = meanMoleFracG1
        inputMoleFrac[:,1] = meanMoleFracG2
        inputMoleFrac[:,2] = 1 - meanMoleFracG1 - meanMoleFracG2

    
    # Loop over all the sorbents for a single material sensor
    # Using parallel processing to loop through all the materials
    arrayConcentrationTemp = Parallel(n_jobs=num_cores, prefer="threads")(delayed(estimateConcentration)
                                                                        (numberOfAdsorbents,numberOfGases,None,sensorID,
                                                                        moleFraction = inputMoleFrac[ii],
                                                                        multiplierError = multiplierError,
                                                                        addMeasurementNoise = [meanError,stdError])
                                                                        for ii in tqdm(range(inputMoleFrac.shape[0])))

    # Convert the output list to a matrix
    arrayConcentration[ii,:,:] = np.array(arrayConcentrationTemp)
    
    # Get the concentraiton mean and standard deviation for multiple concentrations
    # For single concentration, get the true sensor response
    # Compute the mean and the standard deviation of the concentration estimates
    if numberOfGases == 2 and len(sensorID) == 2:
        meanConcEstimate[ii,0] = np.mean(arrayConcentration[ii,:,2])
        meanConcEstimate[ii,1] = np.mean(arrayConcentration[ii,:,3])
        stdConcEstimate[ii,0] = np.std(arrayConcentration[ii,:,2])
        stdConcEstimate[ii,1] = np.std(arrayConcentration[ii,:,3])
    elif numberOfGases == 2 and len(sensorID) == 3:
        meanConcEstimate[ii,0] = np.mean(arrayConcentration[ii,:,3])
        meanConcEstimate[ii,1] = np.mean(arrayConcentration[ii,:,4])
        stdConcEstimate[ii,0] = np.std(arrayConcentration[ii,:,3])
        stdConcEstimate[ii,1] = np.std(arrayConcentration[ii,:,4])
    elif numberOfGases == 2 and len(sensorID) == 4:
        meanConcEstimate[ii,0] = np.mean(arrayConcentration[ii,:,4])
        meanConcEstimate[ii,1] = np.mean(arrayConcentration[ii,:,5])
        stdConcEstimate[ii,0] = np.std(arrayConcentration[ii,:,4])
        stdConcEstimate[ii,1] = np.std(arrayConcentration[ii,:,5])        


# Check if simulationResults directory exists or not. If not, create the folder
if not os.path.exists('simulationResults'):
    os.mkdir('simulationResults')
    
# Save the array concentration into a native numpy file
# The .npz file is saved in a folder called simulationResults (hardcoded)
filePrefix = "sensitivityAnalysis"
sensorText = str(sensorID).replace('[','').replace(']','').replace(' ','-').replace(',','')
saveFileName = filePrefix + "_" + sensorText + "_" + simulationDT + "_" + gitCommitID;
savePath = os.path.join('simulationResults',saveFileName)

# Save mean and std. for multiple concentraiton and array concentration 
# array for single concentration

# Save the mean, standard deviation, and molefraction array    
savez (savePath, numberOfGases = numberOfGases,
       numberOfIterations = numberOfIterations,
       moleFractionG1 = meanMoleFracG1,
       multiplierError = multiplierError,
       meanError = meanError,
       stdError = stdError,
       meanConcEstimate = meanConcEstimate,
       stdConcEstimate = stdConcEstimate,
       arrayConcentration = arrayConcentration)