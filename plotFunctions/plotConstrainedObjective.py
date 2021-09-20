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
# Plots the objective function used for concentration estimation for a mole 
# fraction grid. The constraint of mass conservation will show up as a line
# in the plot. The concept\visualization is similar to the one of Lagrange
# multipliers.
#
# Last modified:
# - 2021-01-08, AK: Code improvements
# - 2020-12-18, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################
def plotConstrainedObjective():    
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.style.use('doubleColumn.mplstyle') # Custom matplotlib style file
    import auxiliaryFunctions
    
    os.chdir("..")
    
    # Save flag for figure
    saveFlag = False
    
    # Number of molefractions
    numMolFrac= 1001
    numMesh = 21
    
    # Total pressure of the gas [Pa]
    pressureTotal = np.array([1.e5]);
    
    # Temperature of the gas [K]
    # Can be a vector of temperatures
    temperature = np.array([298.15]);
    
    # Number of Adsorbents
    numberOfAdsorbents = 20
    
    # Number of Gases
    numberOfGases = 3
    
    # Mole Fraction of interest
    moleFrac = np.array([[0.2,0.8,0.0], [0.6,0.4,0.0]])
    
    
    # Multiplier Error
    multiplierError = [1,1,1]
    
    # Sensor ID
    sensorID = np.array([3,4,1])
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # For two gases
    if numberOfGases == 2:
        # Obtain the objective functino for a mole fraction grid (constraint 
        # shown as a line int the plot) - similar to Lagrange multiplier
        objFunction = np.zeros([moleFrac.shape[0],numMesh,numMesh])
        for ii in range(moleFrac.shape[0]):
            moleFracTemp = moleFrac[ii,:]  
            x1m,x2m,objFunctionTemp = functionForMoleFrac(numberOfAdsorbents,numberOfGases,pressureTotal,
                                                          temperature,sensorID,moleFracTemp,numMolFrac,
                                                          numMesh,multiplierError)
            objFunction[ii,:,:] = objFunctionTemp[:,:]
            if ii == 0:
                minObj = np.max([-np.inf,np.min(objFunction[ii,:,:])])
                maxObj = np.min([np.inf,np.max(objFunction[ii,:,:])])
            else:
                minObj = np.max([minObj,np.min(objFunction[ii,:,:])])
                maxObj = np.min([maxObj,np.max(objFunction[ii,:,:])])


        fig = plt.figure
        ax = plt.subplot(1,2,1)
        # Plot obbjective function for first mole fraction
        plt.contourf(x1m,x2m,objFunction[0,:,:],levels = np.linspace(minObj,maxObj,200),cmap='RdYlBu')
        plt.colorbar()
        plt.plot([0.0,1.0],[1.0,0.0],linestyle = ':', linewidth = 1, color = 'k')
        ax.set(xlabel='$y_1$ [-]', 
                ylabel='$y_2$ [-]',
                xlim = [0,1], ylim = [0, 1.])     
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=4)
    
        ax = plt.subplot(1,2,2)
        # Plot obbjective function for second mole fraction
        plt.contourf(x1m,x2m,objFunction[1,:,:],levels = np.linspace(minObj,maxObj,200),cmap='RdYlBu')
        plt.colorbar()
        plt.plot([0.0,1.0],[1.0,0.0],linestyle = ':', linewidth = 1, color = 'k')
        ax.set(xlabel='$y_1$ [-]', 
                ylabel='$y_2$ [-]',
                xlim = [0,1], ylim = [0, 1.])     
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=4)

    # For three gases    
    elif numberOfGases == 3:
        # Obtain the objective functino for a mole fraction grid (constraint 
        # shown as a line int the plot) - similar to Lagrange multiplier
        objFunction = np.zeros([moleFrac.shape[0],numMesh,numMesh])
        for ii in range(moleFrac.shape[0]):
            moleFracTemp = moleFrac[ii,:]               
            x1m,x2m,x3m,objFunctionTemp = functionForMoleFrac(numberOfAdsorbents,numberOfGases,pressureTotal,
                                                                  temperature,sensorID,moleFracTemp,numMolFrac,
                                                                  numMesh,multiplierError)
            plotInd = np.where(x1m[:,0]==moleFracTemp[2])[0][0]
            objFunction[ii,:,:] = objFunctionTemp[:,:,plotInd]
            if ii == 0:
                minObj = np.max([-np.inf,np.min(objFunction[ii,:,:])])
                maxObj = np.min([np.inf,np.max(objFunction[ii,:,:])])
            else:
                minObj = np.max([minObj,np.min(objFunction[ii,:,:])])
                maxObj = np.min([maxObj,np.max(objFunction[ii,:,:])])

        # Plot obbjective function stacked on top of each other
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for ii in range(moleFrac.shape[0]):    
            ax.contourf(x1m[:,:,plotInd],x2m[:,:,plotInd],
                        objFunction[ii,:,:],zdir='z',offset = ii,
                        levels = np.linspace(minObj,maxObj,30),
                        cmap='RdYlBu')
        
        # Set 3D-axis-limits: 
        ax.set_xlim3d(0,1)
        ax.set_ylim3d(0,1)
        ax.set_zlim3d(0,moleFrac.shape[0])
        plt.show()
        
        fig = plt.figure
        ax = plt.subplot(1,2,1)
        # Plot obbjective function for first mole fraction
        plt.contourf(x1m[:,:,0],x2m[:,:,0],objFunction[0,:,:],levels = np.linspace(minObj,maxObj,200),cmap='RdYlBu')
        plt.colorbar()
        plt.plot([0.0,1-moleFrac[0,2]],[1-moleFrac[0,2],0.0],linestyle = ':', linewidth = 1, color = 'k')
        ax.set(xlabel='$y_1$ [-]', 
                ylabel='$y_2$ [-]',
                xlim = [0,1-moleFrac[0,2]], ylim = [0., 1.-moleFrac[0,2]])     
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=4)

        ax = plt.subplot(1,2,2)
        # Plot obbjective function for second mole fraction
        plt.contourf(x1m[:,:,0],x2m[:,:,0],objFunction[1,:,:],levels = np.linspace(minObj,maxObj,200),cmap='RdYlBu')
        plt.colorbar()
        plt.plot([0.0,1-moleFrac[1,2]],[1-moleFrac[1,2],0.0],linestyle = ':', linewidth = 1, color = 'k')
        ax.set(xlabel='$y_1$ [-]', 
                ylabel='$y_2$ [-]',
                xlim = [0,1-moleFrac[1,2]], ylim = [0, 1-moleFrac[1,2]])     
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=4)    
 
# Function to evaluate the objective function for a given mole fraction
def functionForMoleFrac(numberOfAdsorbents,numberOfGases,pressureTotal,
                                temperature,sensorID,moleFrac,numMolFrac,
                                numMesh,multiplierError):
    import numpy as np
    from generateTrueSensorResponse import generateTrueSensorResponse
    from simulateSensorArray import simulateSensorArray
    # Simulate the sensor response for all possible concentrations
    if numberOfGases == 2:
        x1 = np.linspace(0,1,numMesh)
        x2 = np.linspace(0,1,numMesh)
        x1m, x2m = np.meshgrid(x1, x2, sparse=False, indexing='ij')
        arraySimResponse = np.zeros([len(sensorID),numMesh,numMesh])
    elif numberOfGases == 3:
        x1 = np.linspace(0,1,numMesh)
        x2 = np.linspace(0,1,numMesh)
        x3 = np.linspace(0,1,numMesh)
        x1m, x2m, x3m = np.meshgrid(x1, x2, x3, sparse=False, indexing='ij')
        arraySimResponse = np.zeros([len(sensorID),numMesh,numMesh,numMesh])
    
    # Get simulated sensor response 
    if numberOfGases == 2:
        for ii in range(numMesh):
            for jj in range(numMesh):
                arraySimResponseTemp = simulateSensorArray(sensorID, pressureTotal, 
                                                           temperature, np.array([[x1m[ii,jj],x2m[ii,jj]]])) * multiplierError
                for kk in range(len(sensorID)):
                    arraySimResponse[kk,ii,jj] = arraySimResponseTemp[kk]
    elif numberOfGases == 3:
        for ii in range(numMesh):
            for jj in range(numMesh):
                for kk in range(numMesh):
                    arraySimResponseTemp = simulateSensorArray(sensorID, pressureTotal, 
                                                               temperature, np.array([[x1m[ii,jj,kk],x2m[ii,jj,kk],x3m[ii,jj,kk]]])) * multiplierError
                    for ll in range(len(sensorID)):
                        arraySimResponse[ll,ii,jj,kk] = arraySimResponseTemp[ll]
        
    
    # Get the individual sensor reponse for all the given "experimental/test" concentrations
    sensorTrueResponse = generateTrueSensorResponse(numberOfAdsorbents,numberOfGases,
                                                pressureTotal,temperature,moleFraction = moleFrac)
    # Parse out the true sensor response for the desired sensors in the array
    arrayTrueResponseTemp = np.zeros(len(sensorID))
    for ii in range(len(sensorID)):
        arrayTrueResponseTemp[ii] = sensorTrueResponse[sensorID[ii]]*multiplierError[ii]
    if numberOfGases == 2:
        arrayTrueResponse = np.zeros([len(sensorID),numMesh,numMesh])
        for kk in range(len(sensorID)):
            arrayTrueResponse[kk,:,:] = np.tile(arrayTrueResponseTemp[kk],(numMesh,numMesh))
    elif numberOfGases == 3:
        arrayTrueResponse = np.zeros([len(sensorID),numMesh,numMesh,numMesh])
        for ll in range(len(sensorID)):
            arrayTrueResponse[ll,:,:,:] = np.tile(arrayTrueResponseTemp[ll],(numMesh,numMesh,numMesh))
    
    # Compute the objective function over all the mole fractions
    if numberOfGases == 2:
        objFunction = np.zeros([numMesh,numMesh])
        for kk in range(len(sensorID)):
            objFunctionTemp = np.power(np.divide(arrayTrueResponse[kk,:,:] - arraySimResponse[kk,:,:],arrayTrueResponse[kk,:,:]),2)
            objFunction += objFunctionTemp
            objFunctionTemp = []
    elif numberOfGases == 3:
        objFunction = np.zeros([numMesh,numMesh,numMesh])
        for kk in range(len(sensorID)):
            objFunctionTemp = np.power(np.divide(arrayTrueResponse[kk,:,:,:] - arraySimResponse[kk,:,:,:],arrayTrueResponse[kk,:,:,:]),2)
            objFunction += objFunctionTemp
            objFunctionTemp = []

    # Function return
    if numberOfGases == 2:
        return x1m,x2m,objFunction
    elif numberOfGases == 3:
        return x1m,x2m,x3m,objFunction