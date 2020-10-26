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
# Plots the single site Langmuir isotherms
#
# Last modified:
# - 2020-10-26, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

import numpy as np
import os
from numpy import load
from simulateSSL import simulateSSL
import matplotlib
import matplotlib.pyplot as plt

# Sensor ID to be plotted
sensorID = np.array([19])

# Total pressure of the gas [Pa]
pressureTotal = np.array([1.e5])

# Temperature of the gas [K]
# Can be a vector of temperatures
temperature = np.array([298.15])

# Molefraction
moleFraction = np.array([np.linspace(0,1,101)])

# For now load a given adsorbent isotherm material file
# loadFileName = "isothermParameters_20201020_1756_5f263af.npz" # Two gases
loadFileName = "isothermParameters_20201022_1056_782efa3.npz" # Three gases
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
    
###### TO DO: SERIOUS ISSUE WITH THE ISOTHERM PLOTTING
# Evaluate the isotherms
adsorbentID = sensorID
adsorbentIsotherm = adsorbentIsothermTemp[:,:,adsorbentID]
adsorbentDensity = adsorbentDensityTemp[adsorbentID]
equilibriumLoadings = np.zeros([moleFraction.shape[1],adsorbentIsotherm.shape[0]])
# Loop through all the gases so that the single component isotherm is 
# generated. If not multicomponent genretaed. Additionally, several 
# transpose operations are performed to be self-consistent with other codes
for ii in range(adsorbentIsotherm.shape[0]):
    equilibriumLoadings[:,ii] = np.squeeze(simulateSSL(adsorbentIsotherm[ii,:,:].T,adsorbentDensity,
                                      pressureTotal,temperature,moleFraction.T))/adsorbentDensity # [mol/m3]


# Plot the pure single component isotherm for the n gases
fig, ax = plt.subplots()
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)
# HARD CODED for 3 gases
ax.plot(pressureTotal*moleFraction.T/1.e5, equilibriumLoadings[:,0],'r')
ax.plot(pressureTotal*moleFraction.T/1.e5, equilibriumLoadings[:,1],'b')
ax.plot(pressureTotal*moleFraction.T/1.e5, equilibriumLoadings[:,2],'g')


ax.set(xlabel='Pressure P [bar]', 
       ylabel='q [mol kg$^{\mathregular{-1}}$]',
       xlim = [0, 1], ylim = [0, 10])
ax.grid()
plt.show()