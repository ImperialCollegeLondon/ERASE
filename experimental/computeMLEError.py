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
# Computes the MLE error for ZLC experiments. The function splits the 
# response curve to two distinct regions (high and low composition) to given
# equal weights. The cut off threshold for high and low composition is hard
# coded in this function. Note that the error is
#
# Last modified:
# - 2021-05-05, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def computeMLEError(moleFracExp,moleFracSim):
    import numpy as np
    
    # Objective function error
    # Find error for mole fraction below a given threshold
    thresholdFactor = 1e-2 
    lastIndThreshold = int(np.argwhere(np.array(moleFracExp)>thresholdFactor)[-1])
    # Do downsampling if the number of points in higher and lower
    # compositions does not match
    numPointsConc = np.zeros([2])
    numPointsConc[0] = len(moleFracExp[0:lastIndThreshold]) # High composition
    numPointsConc[1] = len(moleFracExp[lastIndThreshold:-1]) # Low composition            
    downsampleConc = numPointsConc/np.min(numPointsConc) # Downsampled intervals
    
    # Compute error for higher concentrations (accounting for downsampling)
    moleFracHighExp = moleFracExp[0:lastIndThreshold]
    moleFracHighSim = moleFracSim[0:lastIndThreshold]
    computedErrorHigh = np.log(np.sum(np.power(moleFracHighExp[::int(np.round(downsampleConc[0]))] 
                                                - moleFracHighSim[::int(np.round(downsampleConc[0]))],2)))
    
    # Find scaling factor for lower concentrations
    scalingFactor = int(1/thresholdFactor) # Assumes max composition is one
    # Compute error for lower concentrations
    moleFracLowExp = moleFracExp[lastIndThreshold:-1]*scalingFactor
    moleFracLowSim = moleFracSim[lastIndThreshold:-1]*scalingFactor

    # Compute error for low concentrations (accounting for downsampling)
    computedErrorLow = np.log(np.sum(np.power(moleFracLowExp[::int(np.round(downsampleConc[1]))] 
                                                - moleFracLowSim[::int(np.round(downsampleConc[1]))],2)))
    
    # Find the sum of computed error
    computedError = computedErrorHigh + computedErrorLow
    
    # Compute the number of points per experiment (accouting for down-
    # sampling in both experiments and high and low compositions
    numPoints = len(moleFracHighExp) + len(moleFracLowExp)

    return (numPoints/2)*(computedError)