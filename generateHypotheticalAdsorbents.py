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
# - 2020-10-16, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def generateHypotheticalAdsorbents():
    
    import numpy as np
    from smt.sampling_methods import LHS
    
    # Define the number of gases of interest
    numberOfGases = 2;
    
    # Define the number of materials that will be tested for the sensor array
    numberOfAdsorbents = 100;
    
    # Define a range for single site Langmuir isotherm for one sensor material 
    singleIsothermRange = np.array([[0.0, 10000.0], [0.0, 3e-6],[0, -40000]])
    
    adsorbentMaterial = np.zeros((numberOfGases,3,numberOfAdsorbents))
    
    
    # Generate latin hypercube sampled hypothethical adsorbent materials. 
    # Isotherm parameters obtained as a matrix with dimensions [
    # numberOfAdsorbents*numberOfGases x 3]
    samplingLHS = LHS(xlimits=singleIsothermRange)
    allIsothermParameter = samplingLHS(numberOfAdsorbents*numberOfGases);
    
    # Get the isotherms for each material for the predefined number of gases
    isothermCounter = 0;
    for ii in range(0,numberOfAdsorbents):
            for jj in range(0,numberOfGases):
                adsorbentMaterial[jj,:,ii] = allIsothermParameter[isothermCounter,:]
                isothermCounter += 1
                
    return adsorbentMaterial