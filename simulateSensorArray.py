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
#
#
# Last modified:
# - 2020-10-20, AK: Obtain sensor array finger print
# - 2020-10-19, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

# def simulateSensorArray():
import numpy as np
from numpy import load
import os
from simulateSSL import simulateSSL

# Total pressure of the gas [Pa]
pressureTotal = np.array([1.e5]);

# Temperature of the gas [K]
# Can be a vector of temperatures
temperature = np.array([298.15]);

# Mole fraction of the gas [-]
# Can be [jxg], where j is the number of mole fractions for g gases
moleFraction = np.array([[0.6, 0.4]])

# Sensor combinations used in the array. This is a [gx1] vector that maps to
# the sorbent/material ID generated using the 
# generateHypotheticalAdsorbents.py function
sensorID = np.array([6,10,71])

# For now load a given adsorbent isotherm material file
loadFileName = "isothermParameters_20201020_1756_5f263af.npz"
hypoAdsorbentFile = os.path.join('inputResources',loadFileName);

# Check if the file with the adsorbent properties exist 
if os.path.exists(hypoAdsorbentFile):
    loadedFileContent = load(hypoAdsorbentFile)
    adsorbentIsothermTemp = loadedFileContent['adsIsotherm']
    adsorbentDensityTemp = loadedFileContent['adsDensity']
    molecularWeight = loadedFileContent['molWeight']
else:
    errorString = "Adsorbent property file " + hypoAdsorbentFile + " does not exist."
    raise Exception(errorString)

# Get the equilibrium loading for all the sensors for each gas
# This is a [nxg] matrix where n is the number of sensors and g the number
# of gases
sensorLoadingPerGasVol = np.zeros((sensorID.shape[0],moleFraction.shape[1])) # [mol/m3]
sensorLoadingPerGasMass = np.zeros((sensorID.shape[0],moleFraction.shape[1])) # [mol/kg]
for ii in range(sensorID.shape[0]): 
    adsorbentID = sensorID[ii]
    adsorbentIsotherm = adsorbentIsothermTemp[:,:,adsorbentID]
    adsorbentDensity = adsorbentDensityTemp[adsorbentID]
    equilibriumLoadings = simulateSSL(adsorbentIsotherm,adsorbentDensity,
                                      pressureTotal,temperature,moleFraction) # [mol/m3]
    sensorLoadingPerGasVol[ii,:] = equilibriumLoadings[0,0,:] # [mol/m3]
    sensorLoadingPerGasMass[ii,:] = equilibriumLoadings[0,0,:]/adsorbentDensity # [mol/kg]

# Obtain the sensor finger print # [g of total gas adsorbed/kg of sorbent]
sensorFingerPrint = np.dot(sensorLoadingPerGasMass,molecularWeight)