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
# Generates hypothetical sorbents using latin hypercube sampling. The 
# sorbents are assumed to exhibit Langmuirian behavior.
#
# Last modified:
# - 2020-10-29, AK: Add 3 sorbent sensor
# - 2020-10-28, AK: Add auxiliary functions as a module
# - 2020-10-22, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

import numpy as np
from numpy import save
import auxiliaryFunctions
import multiprocessing # For parallel processing
from joblib import Parallel, delayed  # For parallel processing
from tqdm import tqdm # To track progress of the loop
from estimateConcentration import estimateConcentration
import os

# Number of sensors in the array
numSensors = 2

# Get the commit ID of the current repository
gitCommitID = auxiliaryFunctions.getCommitID()

# Find out the total number of cores available for parallel processing
num_cores = multiprocessing.cpu_count()

# Total number of sensor elements/gases simulated and generated using 
# generateHypotheticalAdsorbents.py function
numberOfAdsorbents = 10
numberOfGases = 2
    
# "True" gas composition that is exposed to the sensor array (0-4)
# Check generateTrueSensorResponse.py for the actual concentrations
moleFracID = 0

# Check for number of sensors in the array
if numSensors == 1:
    ##### FOR 1 SORBENT SENSOR ARRAY #####
    # Get the current date and time for saving purposes    
    simulationDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Loop over all the sorbents for a single material sensor
    # Using parallel processing to loop through all the materials
    arrayConcentration = np.zeros(numberOfAdsorbents)
    arrayConcentration = Parallel(n_jobs=num_cores, prefer="threads")(delayed(estimateConcentration)
                                                                      (numberOfAdsorbents,numberOfGases,moleFracID,[ii])
                                                                      for ii in tqdm(range(numberOfAdsorbents)))
    
    # Convert the output list to a matrix
    arrayConcentration = np.array(arrayConcentration)
    
    # Save the array concentration into a native numpy file
    # The .npy file is saved in a folder called simulationResults (hardcoded)
    filePrefix = "arrayConcentration"
    saveFileName = filePrefix + "_" + simulationDT + "_" + gitCommitID;
    savePath = os.path.join('simulationResults',saveFileName)
    
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists('simulationResults'):
        os.mkdir('simulationResults')
    
    # Save the array ceoncentration obtained from estimateConcentration
    save (savePath, arrayConcentration)

elif numSensors == 2:
    ##### FOR 2 SORBENT SENSOR ARRAY #####
    # Get the current date and time for saving purposes    
    simulationDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Loop over all the sorbents for a single material sensor
    # Using parallel processing to loop through all the materials
    arrayConcentration = np.zeros(numberOfAdsorbents)
    for jj in range(numberOfAdsorbents-1):
        arrayConcentrationTemp = Parallel(n_jobs=num_cores, prefer="threads")(delayed(estimateConcentration)
                                                                      (numberOfAdsorbents,numberOfGases,moleFracID,[ii,jj])
                                                                      for ii in tqdm(range(jj+1,numberOfAdsorbents)))
        # Convert the output list to a matrix
        arrayConcentrationTemp = np.array(arrayConcentrationTemp)
        if jj == 0:
            arrayConcentration = arrayConcentrationTemp
        else:
            arrayConcentration = np.append(arrayConcentration,arrayConcentrationTemp, axis=0)
    
    # Save the array concentration into a native numpy file
    # The .npy file is saved in a folder called simulationResults (hardcoded)
    filePrefix = "arrayConcentration"
    saveFileName = filePrefix + "_" + simulationDT + "_" + gitCommitID;
    savePath = os.path.join('simulationResults',saveFileName)
    
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists('simulationResults'):
        os.mkdir('simulationResults')
    
    # Save the array ceoncentration obtained from estimateConcentration
    save (savePath, arrayConcentration)

elif numSensors == 3:
    ##### FOR 3 SORBENT SENSOR ARRAY #####
    # Get the current date and time for saving purposes    
    simulationDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Loop over all the sorbents for a single material sensor
    # Using parallel processing to loop through all the materials
    arrayConcentration = np.zeros(numberOfAdsorbents)

    for kk in range(numberOfAdsorbents-1):    
        for jj in range(kk+1,numberOfAdsorbents-1):
            arrayConcentrationTemp = Parallel(n_jobs=num_cores, prefer="threads")(delayed(estimateConcentration)
                                                                          (numberOfAdsorbents,numberOfGases,moleFracID,[ii,jj,kk])
                                                                          for ii in tqdm(range(jj+1,numberOfAdsorbents)))
            # Convert the output list to a matrix
            arrayConcentrationTemp = np.array(arrayConcentrationTemp)
            if kk == 0 and jj == 1:
                arrayConcentration = arrayConcentrationTemp
            else:
                arrayConcentration = np.append(arrayConcentration,arrayConcentrationTemp, axis=0)
        
    # Delete entries that use the same materials for all three sensors
    # delRows = np.where(np.logical_and(arrayConcentration[:,0] == arrayConcentration[:,1],
    #                                   arrayConcentration[:,0] == arrayConcentration[:,2]))
    # arrayConcentration = np.delete(arrayConcentration,delRows,axis=0)
    
    # Save the array concentration into a native numpy file
    # The .npy file is saved in a folder called simulationResults (hardcoded)
    filePrefix = "arrayConcentration"
    saveFileName = filePrefix + "_" + simulationDT + "_" + gitCommitID;
    savePath = os.path.join('simulationResults',saveFileName)
    
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists('simulationResults'):
        os.mkdir('simulationResults')
    
    # Save the array ceoncentration obtained from estimateConcentration
    save (savePath, arrayConcentration)