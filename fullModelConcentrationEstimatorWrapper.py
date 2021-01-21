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
# Wrapper function to estimate the concentration given a time-resolved 
# measurement from the full model
#
# Last modified:
# - 2021-01-21, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

import auxiliaryFunctions
from simulateFullModel import simulateFullModel
from estimateConcentration import estimateConcentration
from tqdm import tqdm # To track progress of the loop
import numpy as np
from joblib import Parallel, delayed  # For parallel processing
import multiprocessing
import os 
from numpy import savez

# Find out the total number of cores available for parallel processing
num_cores = multiprocessing.cpu_count()

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Sensor ID
sensorID = [6,2] # [-]

# Number of Gases
numberOfGases = 2 # [-]

# Rate Constant
rateConstant = ([0.1,1.0,10000.0],
                [0.1,1.0,10000.0]) # [1/s]

# Feed mole fraction
feedMoleFrac = [0.5,0.5,0.0] # äö[-]

# Time span for integration [tuple with t0 and tf] [s]
timeInt = (0,500) # [s]

# Loop over all rate constants
outputStruct = {}
for ii in tqdm(range(len(sensorID))):
    # Call the full model with a given rate constant
    timeSim, _ , sensorFingerPrint, inputParameters = simulateFullModel(sensorID = sensorID[ii],
                                                                        rateConstant = rateConstant[ii],
                                                                        feedMoleFrac = feedMoleFrac,
                                                                        timeInt = timeInt)
    outputStruct[ii] = {'timeSim':timeSim,
                        'sensorFingerPrint':sensorFingerPrint,
                        'inputParameters':inputParameters} # Check simulateFullModel.py for entries
    
    
# Prepare time-resolved sendor finger print
timeSim = []
timeSim = outputStruct[0]["timeSim"]
sensorFingerPrint = np.zeros([len(timeSim),len(sensorID)])
arrayConcentration = np.zeros([len(timeSim),numberOfGases+len(sensorID)])
for ii in range(len(sensorID)):
    sensorFingerPrint[:,ii] = outputStruct[ii]["sensorFingerPrint"]

# Loop over all time instants and estimate the gas composition
arrayConcentrationTemp = Parallel(n_jobs=num_cores)(delayed(estimateConcentration)
                           (None,numberOfGases,None,sensorID,
                            fullModel = True,
                            fullModelResponse = sensorFingerPrint[ii,:])
                           for ii in tqdm(range(len(timeSim))))

arrayConcentration = np.array(arrayConcentrationTemp)

# Check if simulationResults directory exists or not. If not, create the folder
if not os.path.exists('simulationResults'):
    os.mkdir('simulationResults')
    
# Save the array concentration into a native numpy file
# The .npz file is saved in a folder called simulationResults (hardcoded)
filePrefix = "fullModelConcentrationEstimate"
sensorText = str(sensorID).replace('[','').replace(']','').replace(' ','-').replace(',','')
saveFileName = filePrefix + "_" + sensorText + "_" + currentDT + "_" + gitCommitID;
savePath = os.path.join('simulationResults',saveFileName)

savez (savePath, outputStruct = outputStruct,
        arrayConcentration = arrayConcentration)