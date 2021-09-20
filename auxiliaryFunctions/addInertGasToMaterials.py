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
# Adds an inert gas to the exisitng material property matrix for an n gas
# system. This is done to simulate the sensor array with a full model. The
# inert is used to clean the sensor array
#
# Last modified:
# - 2021-01-20, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def addInertGasToMaterials(numberOfGases):
    import numpy as np
    from numpy import load
    from numpy import savez
    import os
    import auxiliaryFunctions
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    simulationDT = auxiliaryFunctions.getCurrentDateTime()
    
    # For now load a given adsorbent isotherm material file
    if numberOfGases == 2:
        loadFileName = "isothermParameters_20201020_1756_5f263af.npz" # Two gases
    elif numberOfGases == 3:
        loadFileName = "isothermParameters_20201022_1056_782efa3.npz" # Three gases
    hypoAdsorbentFile = os.path.join('../inputResources',loadFileName);
    
    # Check if the file with the adsorbent properties exist 
    if os.path.exists(hypoAdsorbentFile):
        loadedFileContent = load(hypoAdsorbentFile)
        adsorbentIsothermTemp = loadedFileContent['adsIsotherm']
        adsorbentDensity = loadedFileContent['adsDensity']
        molecularWeightTemp = loadedFileContent['molWeight']
    else:
        errorString = "Adsorbent property file " + hypoAdsorbentFile + " does not exist."
        raise Exception(errorString)
    
    # Create adsorent isotherm matrix with the addition of an intert gas
    adsorbentIsotherm = np.zeros([numberOfGases+1,3,adsorbentIsothermTemp.shape[2]])
    adsorbentIsotherm[0:numberOfGases,:,:] = adsorbentIsothermTemp
    
    # Add the moleuclar weight of the inert (assumed to be helium)
    molecularWeight = np.concatenate((molecularWeightTemp,np.array([4.00])))
    
    # Save the adsorbent isotherm parameters into a native numpy file
    # The .npz file is saved in a folder called inputResources (hardcoded)
    filePrefix = "isothermParameters"
    saveFileName = filePrefix + "_" + simulationDT + "_" + gitCommitID + ".npz";
    savePath = os.path.join('../inputResources',saveFileName)
    
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists('../inputResources'):
        os.mkdir('../inputResources')
    
    # Save the adsorbent material array    
    savez (savePath, adsIsotherm = adsorbentIsotherm, 
            adsDensity = adsorbentDensity,
            molWeight = molecularWeight)