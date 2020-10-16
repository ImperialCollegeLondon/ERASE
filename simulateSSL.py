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
# - 2020-10-16, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

# def simulateSSL():
import numpy as np
import matplotlib.pyplot as plt
import math

# Import the function generateHypotheticalAdsorbents
from generateHypotheticalAdsorbents import generateHypotheticalAdsorbents

# Gas constant
Rg = 8.314; # [J/mol K]

# Define total pressure
pressureTotal = 1.e5 # Pressure [Pa]

# Define mole fraction
moleFraction = np.linspace(0,1,num=101) # mole fraction [-]

# Define temperature
temperature = np.linspace(283,323,num=5) # mole fraction [-]

# Stack up the isotherm parameters for all the materials
adsorbentMaterial = generateHypotheticalAdsorbents()

# Get the number of gases and number of materials
numberOfGases = adsorbentMaterial.shape[0]
numberOfMaterials = adsorbentMaterial.shape[2]

equilibriumLoadings = np.zeros((moleFraction.shape[0],
                              temperature.shape[0],numberOfGases))

# Generate the isotherm
for ii in range(numberOfGases):
    for jj in range(moleFraction.shape[0]):
        for kk in range(temperature.shape[0]):
            # Parse out the saturation capacity, equilibrium constant, and 
            # heat of adsorption
            qsat = adsorbentMaterial[ii,0,0]
            b0 = adsorbentMaterial[ii,1,0]
            delH = adsorbentMaterial[ii,2,0]
            # Compute the concentraiton
            conc = pressureTotal*moleFraction[jj]/(Rg*temperature[kk])
            # Compute the numerator and denominator for SSL
            loadingNum = qsat*b0*math.exp(-delH/(Rg*temperature[kk]))*conc
            loadingDen = 1 + b0*math.exp(-delH/(Rg*temperature[kk]))*conc
            # Compute the equilibrium loading
            equilibriumLoadings[jj,kk,ii] = loadingNum/loadingDen
            

# Plot isotherm
for ii in range(numberOfGases):
    for kk in range(temperature.shape[0]):
        plt.plot(pressureTotal*moleFraction,equilibriumLoadings[:,kk,ii])