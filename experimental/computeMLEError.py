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
# - 2021-08-10, AK: Fix for error computation
# - 2021-07-02, AK: Remove threshold factor
# - 2021-07-02, AK: Bug fix for data sorting
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
    # Default is false, uses pure MLE with no downsampling
    else:
        downsampleData = False

    # If not pure downsampling of the data at different composition ranges
    if not downsampleData:
        # If no threshold provided just do normal MLE
        computedError = np.log(np.sum(np.power(moleFracExp - moleFracSim, 2)))
        numPoints = len(moleFracExp)
    # Pure downsampling of the data at different composition ranges
    else:
        # Objective function error
        # Concatenate the experiment and simulated data into an array
        concatenatedData = np.vstack((moleFracExp,moleFracSim)).T
        # Sort the data first (as everything is concatenated)
        sortedData = concatenatedData[np.argsort(concatenatedData[:,0]),:]
        # Find error for mole fraction below a given threshold. The threshold 
        # corresponds to the median of the overall data (experimental)
        # This would enable equal weights to all the compositions
        lastIndThreshold = int(np.argwhere(sortedData[:,0]>np.median(sortedData[:,0]))[0])

        # Reinitialize mole fraction experimental and simulation based on sorted data        
        moleFracExp = sortedData[:,0] # Experimental
        moleFracSim = sortedData[:,1] # Simulation
    
        # Do downsampling if the number of points in higher and lower
        # compositions does not match
        numPointsConc = np.zeros([2])
        numPointsConc[0] = len(moleFracExp[0:lastIndThreshold]) # Low composition
        numPointsConc[1] = len(moleFracExp[lastIndThreshold:-1]) # High composition            
        downsampleConc = numPointsConc/np.min(numPointsConc) # Downsampled intervals     
        # downsampleConc = numPointsConc/numPointsConc # Downsampled intervals     
        
        # Compute error (accounting for downsampling)
        # Lower concentrations
        moleFracLowExp = moleFracExp[0:lastIndThreshold:int(np.round(downsampleConc[0]))]
        moleFracLowSim = moleFracSim[0:lastIndThreshold:int(np.round(downsampleConc[0]))]
        # Higher concentrations
        moleFracHighExp = moleFracExp[lastIndThreshold:-1:int(np.round(downsampleConc[1]))]
        moleFracHighSim = moleFracSim[lastIndThreshold:-1:int(np.round(downsampleConc[1]))]
        
        # Normalize the mole fraction by dividing it by maximum value to avoid
        # irregular weightings and scale it to the highest composition range
        # This is part of the downsampling/reweighting procedure 
        minExp = np.min(moleFracHighExp) # Compute the minimum from experiment
        normalizeFactorHigh = np.max(moleFracHighExp - minExp) # Compute the max from normalized data
        moleFracHighExp = (moleFracHighExp - minExp)
        moleFracHighSim = (moleFracHighSim - minExp)

        # Low concentrations
        minExp = np.min(moleFracLowExp) # Compute the minimum from experiment
        normalizeFactorLow = np.max(moleFracLowExp - minExp) # Compute the max from normalized data
        moleFracLowExp = (moleFracLowExp - minExp)/normalizeFactorLow*normalizeFactorHigh
        moleFracLowSim = (moleFracLowSim - minExp)/normalizeFactorLow*normalizeFactorHigh
        
        # Compute the error
        computedErrorLow = np.sum(np.power(moleFracLowExp - moleFracLowSim,2))
        computedErrorHigh = np.sum(np.power(moleFracHighExp - moleFracHighSim,2))
        # The higher and lower composition is treated as two independent
        # measurements
        computedError = np.log(computedErrorLow) + np.log(computedErrorHigh)
        
        # Compute the number of points per experiment (accouting for down-
        # sampling in both experiments and high and low compositions
        # The number of points in both the low and high compositions area 
        # similar, so the minimum number of points from the two is chosen
        # to be representative of the number of points in the experiments
        numPoints = min(len(moleFracHighExp),len(moleFracLowExp))
        
    return (numPoints/2)*(computedError)