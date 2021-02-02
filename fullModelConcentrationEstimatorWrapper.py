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
# - 2021-01-26, AK: Add noise to true measurement
# - 2021-01-25, AK: Integrate full model concentration estimator
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
from estimateConcentrationFullModel import estimateConcentrationFullModel
from tqdm import tqdm # To track progress of the loop
import numpy as np
from numpy import savez
from joblib import Parallel, delayed  # For parallel processing
import multiprocessing
import os 
import time
import socket

# Find out the total number of cores available for parallel processing
num_cores = multiprocessing.cpu_count()

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Get the current date and time for saving purposes    
currentDT = auxiliaryFunctions.getCurrentDateTime()

# Flag to determine whether concentration estimated accounted for kinetics 
# (False) or not (True)
flagIndTime = True

# Flag to determine whether constant pressure or constant flow rate model to 
# be used
# (False) or not (True)
flagIndTime = True

# Sensor ID
sensorID = [6,2] # [-]

# Number of Gases
numberOfGases = 2 # [-]

# Rate Constant
rateConstant = ([.01,.01,10000.0],
                [.01,.01,10000.0]) # [1/s]

# Feed mole fraction
feedMoleFrac = [0.1,0.9,0.0] # [-]

# Time span for integration [tuple with t0 and tf] [s]
timeInt = (0,1000) # [s]

# Volumetric flow rate [m3/s]
flowIn = 5e-7 # [s]

# Measurement noise characteristics
meanNoise = 0.0 # [g/kg]
stdNoise = 0.1 # [g/kg]

# Loop over all rate constants
outputStruct = {}
for ii in tqdm(range(len(sensorID))):
    # Call the full model with a given rate constant
    timeSim, _ , sensorFingerPrint, inputParameters = simulateFullModel(sensorID = sensorID[ii],
                                                                        rateConstant = rateConstant[ii],
                                                                        feedMoleFrac = feedMoleFrac,
                                                                        timeInt = timeInt,
                                                                        flowIn = flowIn)
    
    # Generate the noise data for all time instants and all materials
    measurementNoise = np.random.normal(meanNoise, stdNoise, len(timeSim))
    sensorFingerPrintRaw = sensorFingerPrint # Raw sensor finger print
    # Add measurement noise to the sensor finger print
    sensorFingerPrint = sensorFingerPrint + measurementNoise

    outputStruct[ii] = {'timeSim':timeSim,
                        'sensorFingerPrint':sensorFingerPrint,
                        'sensorFingerPrintRaw':sensorFingerPrintRaw, # Without noise
                        'measurementNoise':measurementNoise,
                        'inputParameters':inputParameters} # Check simulateFullModel.py for entries
        
# Prepare time-resolved sendor finger print
timeSim = []
timeSim = outputStruct[0]["timeSim"]
sensorFingerPrint = np.zeros([len(timeSim),len(sensorID)])
for ii in range(len(sensorID)):
    sensorFingerPrint[:,ii] = outputStruct[ii]["sensorFingerPrint"]

# flagIndTime - If true, each time instant evaluated without knowledge of 
# actual kinetics in the estimate of the concentration (accounted only in the
# generation of the true sensor response above) - This would lead to a 
# scenario of concentrations evaluated at each time instant
if flagIndTime:
    # Start time for time elapsed
    startTime = time.time()
    # Initialize output matrix
    arrayConcentration = np.zeros([len(timeSim),numberOfGases+len(sensorID)])
    # Loop over all time instants and estimate the gas composition
    arrayConcentrationTemp = Parallel(n_jobs=num_cores)(delayed(estimateConcentration)
                                (None,numberOfGases,None,sensorID,
                                fullModel = True,
                                fullModelResponse = sensorFingerPrint[ii,:])
                                for ii in tqdm(range(len(timeSim))))
    # Convert the list to array
    arrayConcentration = np.array(arrayConcentrationTemp)
    # Stop time for time elapsed
    stopTime = time.time()    
    # Time elapsed [s]
    timeElapsed = stopTime - startTime
# flagIndTime - If false, only one concentration estimate obtained. The full
# model is simulated in concentration estimator accounting for the kinetics.
else:
    # Start time for time elapsed
    startTime = time.time()
    # Initialize output matrix
    arrayConcentration = np.zeros([len(timeSim)-1,numberOfGases+1+len(sensorID)])
    # Loop over all time instants and estimate the gas composition
    # The concentration is estimated for all the time instants in timeSim. 
    # This simulates using the full model as and when a new measurement is 
    # available. This assumes that the estimator has knowledge of the past
    # measurements
    # The first time instant is not simulated. This is fixed by adding a dummy
    # row after the simulations are over
    arrayConcentrationTemp = Parallel(n_jobs=num_cores)(delayed(estimateConcentrationFullModel)
                                                        (numberOfGases,sensorID,
                                                          fullModelResponse = sensorFingerPrint[0:ii+1,:],
                                                          rateConstant = rateConstant,
                                                          timeInt = (0,timeSim[ii+1]),flowIn = flowIn)
                                                        for ii in tqdm(range(len(timeSim)-1)))

    # Convert the list to array
    arrayConcentration = np.array(arrayConcentrationTemp)
    # Add dummy row to be consistent (intialized to init condition)
    firstRow = np.concatenate((np.array(sensorID), inputParameters[5]))
    arrayConcentration = np.vstack([firstRow,arrayConcentration])
    # Stop time for time elapsed
    stopTime = time.time()    
    # Time elapsed [s]
    timeElapsed = stopTime - startTime

# Check if simulationResults directory exists or not. If not, create the folder
if not os.path.exists('simulationResults'):
    os.mkdir('simulationResults')
    
# Save the array concentration into a native numpy file
# The .npz file is saved in a folder called simulationResults (hardcoded)
filePrefix = "fullModelConcentrationEstimate"
sensorText = str(sensorID).replace('[','').replace(']','').replace(' ','-').replace(',','')
saveFileName = filePrefix + "_" + sensorText + "_" + currentDT + "_" + gitCommitID;
savePath = os.path.join('simulationResults',saveFileName)
savez (savePath, outputStruct = outputStruct, # True response
        arrayConcentration = arrayConcentration, # Estimated response
        eqbmModelFlag = flagIndTime, # Flag to tell whether eqbm. or full model used 
        noiseCharacteristics = [meanNoise, stdNoise], # Noise characteristics
        timeElapsed = timeElapsed, # Time elapsed for the simulation
        hostName = socket.gethostname()) # Hostname of the computer