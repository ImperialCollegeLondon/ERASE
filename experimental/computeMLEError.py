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
# Computes the MLE error for ZLC experiments.
#
# Last modified:
# - 2021-06-12, AK: Add pure data downsampling
# - 2021-05-24, AK: Add -inf input to avoid splitting compositions
# - 2021-05-13, AK: Add different modes for MLE error computations
# - 2021-05-05, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def computeMLEError(moleFracExp,moleFracSim,**kwargs):
    import numpy as np

    # Check if flag for downsampling data in low and high composition provided
    if 'downsampleData' in kwargs:
        downsampleData = kwargs["downsampleData"]
        # Threshold Factor is needed for pure downsampling
        if 'thresholdFactor' in kwargs:
            thresholdFactor = np.array(kwargs["thresholdFactor"])
        else:
            thresholdFactor = 0.5 # Default set to 0.5 (as data scaled between 0 and 1)
    # Default is false, uses either pure MLE or split MLE with scaling based 
    # on threshold
    else:
        downsampleData = False
        # Check if threshold is provided to split data to high and low 
        # compositions and scale for MLE computation    
        if 'thresholdFactor' in kwargs:
            thresholdFlag = True
            thresholdFactor = np.array(kwargs["thresholdFactor"])
            # If negative infinity provided as a threshold, do not split and uses 
            # all data with equal weights
            if np.isneginf(thresholdFactor):
                thresholdFlag = False            
        # Default is false, uses all data with equal weights
        else:
            thresholdFlag = False

    # If not pure downsampling of the data at different composition ranges
    if not downsampleData:
        # If no threshold provided just do normal MLE
        if not thresholdFlag:    
            computedError = np.log(np.sum(np.power(moleFracExp - moleFracSim, 2)))
            numPoints = len(moleFracExp)
        # If threshold provided, split the data to two and compute the error
        else:
            #### Quite a lot of bugs because of changing how error computed ###
            #### DO NOT USE THIS!!! ###
            # Objective function error
            # Find error for mole fraction below a given threshold
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
    # Pure downsampling of the data at different composition ranges
    else:
        # Objective function error
        # Concatenate the experiment and simulated data into an array
        concatenatedData = np.vstack((moleFracExp,moleFracSim)).T
        # Sort the data first (as everything is concatenated)
        sortedData = concatenatedData[np.argsort(concatenatedData[:,0]),:]
        # Find error for mole fraction below a given threshold
        lastIndThreshold = int(np.argwhere(sortedData[:,0]>thresholdFactor)[0])
        
        # Do downsampling if the number of points in higher and lower
        # compositions does not match
        numPointsConc = np.zeros([2])
        numPointsConc[0] = len(moleFracExp[0:lastIndThreshold]) # Low composition
        numPointsConc[1] = len(moleFracExp[lastIndThreshold:-1]) # High composition            
        downsampleConc = numPointsConc/np.min(numPointsConc) # Downsampled intervals
        
        # Compute error (accounting for downsampling)
        # Lower concentrations
        moleFracLowExp = moleFracExp[0:lastIndThreshold:int(np.round(downsampleConc[0]))]
        moleFracLowSim = moleFracSim[0:lastIndThreshold:int(np.round(downsampleConc[0]))]
        # Higher concentrations
        moleFracHighExp = moleFracExp[lastIndThreshold:-1:int(np.round(downsampleConc[1]))]
        moleFracHighSim = moleFracSim[lastIndThreshold:-1:int(np.round(downsampleConc[1]))]
        # Compute the error
        computedErrorLow = np.sum(np.power(moleFracLowExp - moleFracLowSim,2))
        computedErrorHigh = np.sum(np.power(moleFracHighExp - moleFracHighSim,2))
        computedError = np.log(computedErrorLow+computedErrorHigh)
        
        # Compute the number of points per experiment (accouting for down-
        # sampling in both experiments and high and low compositions
        numPoints = len(moleFracHighExp) + len(moleFracLowExp)

    return (numPoints/2)*(computedError)