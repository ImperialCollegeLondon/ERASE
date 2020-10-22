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
# Generates the single site Langmuir isotherms for different temperatures
# and concentrations.
#
# Last modified:
# - 2020-10-22, AK: Minor cosmetic changes
# - 2020-10-19, AK: Incorporate sorbent density and minor improvements
# - 2020-10-19, AK: Change functionality to work with a single sorbent input
# - 2020-10-16, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def simulateSSL(adsorbentIsotherm, adsorbentDensity, pressureTotal, 
                temperature, moleFraction):
    import numpy as np
    import math
    
    # Gas constant
    Rg = 8.314; # [J/mol K]

    # Get the number of gases
    numberOfGases = moleFraction.shape[1]

    if moleFraction.shape[1] != adsorbentIsotherm.shape[0]:
        raise Exception("The dimensions of the mole fraction and the number of gases in the adsorbent is not consistent!")

    # Initialize the equilibriumLoadings array with zeros
    equilibriumLoadings = np.zeros((moleFraction.shape[0],
                                  temperature.shape[0],numberOfGases))
    
    # Generate the isotherm
    for ii in range(numberOfGases):
        for jj in range(moleFraction.shape[0]):
            for kk in range(temperature.shape[0]):
                # Parse out the saturation capacity, equilibrium constant, and 
                # heat of adsorption
                qsat = adsorbentIsotherm[ii,0]
                b0 = adsorbentIsotherm[ii,1]
                delH = adsorbentIsotherm[ii,2]
                # Compute the concentraiton
                conc = pressureTotal*moleFraction[jj,ii]/(Rg*temperature[kk])
                # Compute the numerator and denominator for SSL
                loadingNum = adsorbentDensity*qsat*b0*math.exp(-delH/(Rg*temperature[kk]))*conc
                loadingDen = 1 + b0*math.exp(-delH/(Rg*temperature[kk]))*conc
                # Compute the equilibrium loading
                equilibriumLoadings[jj,kk,ii] = loadingNum/loadingDen # [mol/m3]
                
    # Return the equilibrium loadings
    return equilibriumLoadings