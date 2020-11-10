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
# - 2020-11-11, AK: Add measurement nosie
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
from tqdm import tqdm # To track progress of the loop
from estimateConcentration import estimateConcentration
    
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
sensorID = [17, 6]

# Custom input mole fraction for gas 1
meanMoleFracG1 = [0.001, 0.01, 0.1, 0.25, 0.50, 0.75, 0.90]
diffMoleFracG1 = 0.00 # This plus/minus the mean is the bound for uniform dist.
numberOfIterations = 100

# Custom input mole fraction for gas 2 (for 3 gas system)
meanMoleFracG2 = 0.20

# Measurement noise (Guassian noise)
meanError = 0.
stdError = 0.1

# Initialize mean and standard deviation of concentration estimates
meanConcEstimate = np.zeros([len(meanMoleFracG1),numberOfGases])
stdConcEstimate = np.zeros([len(meanMoleFracG1),numberOfGases])

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
    arrayConcentration = np.zeros(numberOfAdsorbents)
    arrayConcentration = Parallel(n_jobs=num_cores, prefer="threads")(delayed(estimateConcentration)
                                                                        (numberOfAdsorbents,numberOfGases,None,sensorID,
                                                                        moleFraction = inputMoleFrac[ii],
                                                                        addMeasurementNoise = [meanError,stdError])
                                                                        for ii in tqdm(range(inputMoleFrac.shape[0])))

    # Convert the output list to a matrix
    arrayConcentration = np.array(arrayConcentration)
    
    # Compute the mean and the standard deviation of the concentration estimates
    if numberOfGases == 2 and len(sensorID) == 2:
        meanConcEstimate[ii,0] = np.mean(arrayConcentration[:,2])
        meanConcEstimate[ii,1] = np.mean(arrayConcentration[:,3])
        stdConcEstimate[ii,0] = np.std(arrayConcentration[:,2])
        stdConcEstimate[ii,1] = np.std(arrayConcentration[:,3])

# Save the array concentration into a native numpy file
# The .npz file is saved in a folder called simulationResults (hardcoded)
filePrefix = "sensitivityAnalysis"
sensorText = str(sensorID).replace('[','').replace(']','').replace(' ','-').replace(',','')
saveFileName = filePrefix + "_" + sensorText + "_" + simulationDT + "_" + gitCommitID;
savePath = os.path.join('simulationResults',saveFileName)

# Check if simulationResults directory exists or not. If not, create the folder
if not os.path.exists('simulationResults'):
    os.mkdir('simulationResults')

# Save the mean, standard deviation, and molefraction array    
savez (savePath, numberOfGases = numberOfGases,
       moleFractionG1 = meanMoleFracG1, 
       meanConcEstimate = meanConcEstimate,
       stdConcEstimate = stdConcEstimate)