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
# Simulates the dead volume of the ZLC setup. The function can either simualte
# one lumped dead volume or can simulate a cascade of dead volume and MS
#
# Last modified:
# - 2021-05-29, AK: Add optional arguments for combind model
# - 2021-05-28, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def deadVolumeWrapper(timeInt, flowRateDV, DV_p, flagMSDeadVolume, 
                      msDeadVolumeFile, **kwargs):
    import os
    from numpy import load
    from simulateDeadVolume import simulateDeadVolume
    
    # Initial Gas Mole Fraction [-]
    if 'initMoleFrac' in kwargs:
        initMoleFrac = kwargs["initMoleFrac"]
    else:
        initMoleFrac = [1.]
    
    # Feed Gas Mole Fraction [-]
    if 'feedMoleFrac' in kwargs:
        feedMoleFrac = kwargs["feedMoleFrac"]
    else:
        feedMoleFrac = [0.]
    
    # Simulates the tubings and fittings
    # Compute the dead volume response using the dead volume parameters input
    timeDV , _ , moleFracSim = simulateDeadVolume(deadVolume_1 = DV_p[0],
                                            deadVolume_2M = DV_p[1],
                                            deadVolume_2D = DV_p[2],                                      
                                            numTanks_1 = int(DV_p[3]),
                                            flowRate_D = DV_p[4],
                                            initMoleFrac = initMoleFrac,
                                            feedMoleFrac = feedMoleFrac,
                                            timeInt = timeInt,
                                            flowRate = flowRateDV,
                                            expFlag = True)
    
    # Simulates the MS response
    # Pass the mole fractions to the MS model (this uses a completely
    # different flow rate) and simulate the model
    if flagMSDeadVolume:
        # File with parameter estimates for the MS dead volume
        msDeadVolumeDir = '..' + os.path.sep + 'simulationResults/'
        modelOutputTemp = load(msDeadVolumeDir+str(msDeadVolumeFile), allow_pickle=True)["modelOutput"]
        # Parse out dead volume parameters
        msDV_p = modelOutputTemp[()]["variable"]
        # Get the MS flow rate
        msFlowRate = load(msDeadVolumeDir+str(msDeadVolumeFile))["msFlowRate"]
    
        _ , _ , moleFracMS = simulateDeadVolume(deadVolume_1 = msDV_p[0],
                                                deadVolume_2M = msDV_p[1],
                                                deadVolume_2D = msDV_p[2],                                      
                                                numTanks_1 = int(msDV_p[3]),
                                                flowRate_D = msDV_p[4],
                                                initMoleFrac = moleFracSim[0],
                                                feedMoleFrac = moleFracSim,
                                                timeInt = timeDV,
                                                flowRate = msFlowRate,
                                                expFlag = True)
     
        moleFracSim = [] # Initialize moleFracSim to empty
        moleFracSim = moleFracMS # Set moleFracSim to moleFrac MS for error computation
        
    # Return moleFracSim to the top function 
    return moleFracSim