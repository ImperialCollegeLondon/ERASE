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
# - 2020-10-19, AK: Add adsorbent density and molecular weight
# - 2020-10-19, AK: Integrate git commit and save material properties
# - 2020-10-16, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def generateHypotheticalAdsorbents(numberOfGases, numberOfAdsorbents): 
    import numpy as np
    from numpy import savez
    from smt.sampling_methods import LHS
    import os # For OS related stuff (make directory, file separator, etc.)
    
    # Get the commit ID of the current repository
    from getCommitID import getCommitID
    gitCommitID = getCommitID()

    # Get the current date and time for saving purposes    
    from getCurrentDateTime import getCurrentDateTime
    simulationDT = getCurrentDateTime()
    
    # Define a range for single site Langmuir isotherm for one sensor material 
    # [qsat [mol/kg] b0 [m3/mol] delH [J/mol]]
    singleIsothermRange = np.array([[0.0, 10.0], [0.0, 3e-6],[0, -40000]])
    
    # Define a range for adsorbent densities for all the materials
    densityRange = np.array([[500.0, 1500.0]])
    
    # Define the molecular weight for the gases of interest
    # [CO2 N2 O2 SO2 NO2 H2O] - HARD CODED
    molecularWeight = np.array([44.01, 28.01, 15.99, 64.07, 46.01, 18.02])

    # Initialize the output matrix with zeros    
    adsorbentIsotherm = np.zeros((numberOfGases,3,numberOfAdsorbents))
    
    # Generate latin hypercube sampled hypothethical adsorbent materials. 
    # Isotherm parameters obtained as a matrix with dimensions [
    # numberOfAdsorbents*numberOfGases x 3]
    samplingLHS = LHS(xlimits=singleIsothermRange)
    allIsothermParameter = samplingLHS(numberOfAdsorbents*numberOfGases);
    
    # Also generate adsorbent densities that is latin hypercube sampled
    samplingLHS = LHS(xlimits=densityRange)
    adsorbentDensity = samplingLHS(numberOfAdsorbents)
    
    # Get the isotherms for each material for the predefined number of gases
    isothermCounter = 0;
    for ii in range(0,numberOfAdsorbents):
            for jj in range(0,numberOfGases):
                adsorbentIsotherm[jj,:,ii] = allIsothermParameter[isothermCounter,:]
                isothermCounter += 1

    # Save the adsorbent isotherm parameters into a native numpy file
    # The .npy file is saved in a folder called inputResources (hardcoded)
    filePrefix = "isothermParameters"
    saveFileName = filePrefix + "_" + simulationDT + "_" + gitCommitID + ".npz";
    savePath = os.path.join('inputResources',saveFileName)

    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists('inputResources'):
        os.mkdir('inputResources')

    # Save the adsorbent material array    
    savez (savePath, adsIsotherm = adsorbentIsotherm, 
           adsDensity = adsorbentDensity,
           molWeight = molecularWeight[0:numberOfGases])