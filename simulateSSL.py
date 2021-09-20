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
# - 2020-10-23, AK: Replace the noncompetitive to competitive isotherm
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
                b0 = adsorbentIsotherm[:,1]
                delH = adsorbentIsotherm[:,2]
                b = np.multiply(b0,np.exp(-delH/(Rg*temperature[kk])))
                # Compute the concentraiton
                conc = pressureTotal*moleFraction[jj,:]/(Rg*temperature[kk])
                # Compute sum(b*c) for the multicomponet competitive eqbm (den.)
                bStarConc = np.multiply(b,conc)
                sumbStarConc = np.sum(bStarConc)
                # Compute the numerator and denominator for SSL
                loadingNum = adsorbentDensity*qsat*b[ii]*conc[ii]
                loadingDen = 1 + sumbStarConc
                # Compute the equilibrium loading
                equilibriumLoadings[jj,kk,ii] = loadingNum/loadingDen # [mol/m3]
                
    # Return the equilibrium loadings
    return equilibriumLoadings