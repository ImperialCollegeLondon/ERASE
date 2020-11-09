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
import multiprocessing # For parallel processing
from joblib import Parallel, delayed  # For parallel processing
from tqdm import tqdm # To track progress of the loop
from estimateConcentration import estimateConcentration

# Find out the total number of cores available for parallel processing
num_cores = multiprocessing.cpu_count()

# Number of adsorbents
numberOfAdsorbents = 30

# Number of gases
numberOfGases = 2

# Sensor combination
sensorID = [17, 15]

# Custom input mole fraction for gas 1
meanMoleFracG1 = 0.10
diffMoleFracG1 = 0.00 # This plus/minus the mean is the bound for uniform dist.
numberOfIterations = 50

# Custom input mole fraction for gas 2 (for 3 gas system)
meanMoleFracG2 = 0.20


# Generate a uniform distribution of mole fractions
if numberOfGases == 2:
    inputMoleFrac = np.zeros([numberOfIterations,2])
    inputMoleFrac[:,0] = np.random.uniform(meanMoleFracG1-diffMoleFracG1,
                                      meanMoleFracG1+diffMoleFracG1,
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
                                                                    moleFraction = inputMoleFrac[ii])
                                                                    for ii in tqdm(range(inputMoleFrac.shape[0])))

# Convert the output list to a matrix
arrayConcentration = np.array(arrayConcentration)

