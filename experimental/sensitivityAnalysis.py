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
# Performs the sensitivity analysis on the model parameters obtained by the
# ZLC parameter estimation routine. The theory behind this can be found
# in Nonlinear Parameter Estimation by Bard. Additional references are
# provided in the code
#
# Last modified:
# - 2021-08-20, AK: Change definition of rate constants
# - 2021-07-05, AK: Bug fix
# - 2021-07-01, AK: Change structure (to call from extractZLCParameters)
# - 2021-06-28, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def sensitivityAnalysis(fileParameter,alpha):
    import auxiliaryFunctions
    from extractDeadVolume import filesToProcess # File processing script
    import numpy as np
    from numpy import load
    from numpy import savez
    from numpy.linalg import multi_dot # Performs (multiple) matrix multiplication 
    from numpy.linalg import inv
    import os
    from scipy.stats import chi2
    import socket
    
    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Directory of raw data
    mainDir = 'runData'
    # File with parameter estimates
    simulationDir = os.path.join('..','simulationResults')
    # ZLC parameter path
    zlcParameterPath = os.path.join(simulationDir,fileParameter+'.npz')
    # Parse out the optimized model parameters
    # Note that this is nondimensional (reference value in the function)
    pOptTemp = load(zlcParameterPath, allow_pickle=True)["modelOutput"]
    pOpt = pOptTemp[()]["variable"]
    # Isotherm parameter reference
    pRef = load(zlcParameterPath)["parameterReference"]

    # Call the computeObjectiveFunction 
    _ , moleFracExpALL, moleFracSimALL = computeObjectiveFunction(mainDir, zlcParameterPath, pOpt, pRef)

    # Number of parameters
    Np = len(pOpt)
    # Number of time points
    Nt = len(moleFracExpALL)
    # Compute the approximate diagonal covariance matrix of the errors
    # See eq. 12: 10.1021/ie4031852
    # Here there is only one output, therefore V is a scalar and not a vector
    V = (1/Nt)*np.sum((moleFracExpALL-moleFracSimALL)**2)
    # Construct fancyV (eq. 15)
    fancyV = np.zeros((Nt,Nt)) # Initialize with a zero matrix
    np.fill_diagonal(fancyV, V) # Create diagnoal matrix with V (this changes directly in memory)

    # Compute the fancyW (eq. 15)
    # Define delp (delta of parameter to compute the Jacobian)
    delp = np.zeros((Np,Np))
    np.fill_diagonal(delp,np.multiply((np.finfo(float).eps**(1/3)),pOpt))
    # Parameter to the left
    pOpt_Left = pOpt - delp
    # Parameter to the right
    pOpt_Right = pOpt + delp
    # Initialize fancyW
    fancyW = np.zeros((Nt,Np))
    # Loop over all parameters to compute fancyW
    for ii in range(len(pOpt)):
        # Initialize model outputs
        modelOutput_Left = np.zeros((Nt,1))
        modelOutput_Right = np.zeros((Nt,1))
        # Compute the model output for the left side (derivative)
        _ , _ , modelOutput_Left = computeObjectiveFunction(mainDir, zlcParameterPath, pOpt_Left[ii,:], pRef)
        # Compute the model output for the left side (derivative)
        _ , _ , modelOutput_Right = computeObjectiveFunction(mainDir, zlcParameterPath, pOpt_Right[ii,:], pRef)
        # Compute the model Jacobian for the current parameter        
        fancyW[:,ii] = (modelOutput_Right - modelOutput_Left)/(2*delp[ii,ii])

    # Compute the covariance matrix for the multiplers (non dimensional)
    Vpinv = multi_dot([fancyW.T,inv(fancyV),fancyW]) # This is the inverse
    Vp = inv(Vpinv)

    # Transform the multiplier covariance matrix into parameter covariance 
    # matrix. Check eq. 7-20-2 in Nonlinear Parameter Estimation by Bard
    # Create a diagnol matrix for multipler references (pRef)
    T = np.zeros((Np,Np))
    np.fill_diagonal(T, pRef)
    Vx = multi_dot([T,Vp,T.T])
    
    # Obtain chi2 statistics for alpha confidence level and Np degrees
    # of freedom (inverse)
    chi2Statistics = chi2.ppf(alpha, Np)
    
    # Confidence intervals for actual model WITHOUT the linearization
    # assumption for the model equations. This method corresponds to the
    # intersection of one of the delp axes with the confidence
    # hyperellipsoid, i.e., to setting all other deltap to zero (see Bard 1974,
    # pp. 172-173)
    delpNonLinearized = np.sqrt(chi2Statistics/np.diag(inv(Vx)))
    
    # Compute the bounding box of the confidence hyperellipsoid. Note that
    # the matrix that defines the hyperellipsoid is given by
    # (chi2Statistics*Vx)^-1, and that the semiaxes of the bounding box are
    # given by the square roots of the diagonal elements of the inverse of 
    # this matrix.
    delpBoundingBox = np.sqrt(np.diag(chi2Statistics*Vx))
    
    # Print the parameters and the confidence intervals
    xOpt = np.multiply(pRef,pOpt)
    print("Confidence intervals (Nonlinearized):")
    for ii in range(Np):
        print('p' + str(ii+1) + ' : ' + str("{:.2e}".format(xOpt[ii])) 
              + ' +/- ' + str("{:.2e}".format(delpNonLinearized[ii])))
        
    print("Confidence intervals (Bounding Box):")
    for ii in range(Np):
        print('p' + str(ii+1) + ' : ' + str("{:.2e}".format(xOpt[ii])) 
              + ' +/- ' + str("{:.2e}".format(delpBoundingBox[ii])))
    
    
    # Save the sensitivity analysis output in .npz file
    # The .npz file is saved in a folder called simulationResults (hardcoded)
    filePrefix = "sensitivityAnalysis"
    saveFileName = filePrefix + "_" + fileParameter[0:-8] + "_" + gitCommitID;
    savePath = os.path.join('..','simulationResults',saveFileName)
    
    # Check if inputResources directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationResults')):
        os.mkdir(os.path.join('..','simulationResults'))
    
    # Save the output into a .npz file
    savez (savePath, parameterFile = fileParameter, # File name of parameter estimates
           pOpt = pOpt, # Optimized parameters (multipliers)
           pRef = pRef, # References for the multipliers
           xOpt = xOpt, # Optimized parameters (in actual units)
           confidenceLevel = alpha, # Confidence level for the parameters
           Np = Np, # Number of parameters
           Nt = Nt, # Number of data points
           Vp = Vp, # Covariance matrix of the multipliers
           Vx = Vx, # Covariance matrix of actual parameters
           chi2Statistics = chi2Statistics, # Inverse chi squared statistics
           delpNonLinearized = delpNonLinearized, # Confidence intervals (intersection with axis)
           delpBoundingBox = delpBoundingBox, # Confidence intervals (bounding box)
           hostName = socket.gethostname()) # Hostname of the computer

    # Remove all the .npy files genereated from the .mat
    # Load the names of the file to be used for estimating ZLC parameters
    filePath = filesToProcess(False,[],[],'ZLC')
    # Loop over all available files    
    for ii in range(len(filePath)):
        os.remove(filePath[ii])

# func: computeObjectiveFunction
# Computes the objective function and the model output for a given set of
# parameters
def computeObjectiveFunction(mainDir, zlcParameterPath, pOpt, pRef):
    import numpy as np
    from numpy import load
    from simulateCombinedModel import simulateCombinedModel
    from computeMLEError import computeMLEError
    from extractDeadVolume import filesToProcess # File processing script
    from scipy.io import loadmat
    import os
    # Parse out the experimental file names and temperatures
    rawFileName = load(zlcParameterPath)["fileName"]
    temperatureExp = load(zlcParameterPath)["temperature"]
    
    # Generate .npz file for python processing of the .mat file 
    filesToProcess(True,mainDir,rawFileName,'ZLC')
    # Get the processed file names
    fileName = filesToProcess(False,[],[],'ZLC')
    
    # Obtain the downsampling conditions
    downsampleData = load(zlcParameterPath)["downsampleFlag"]
    modelType = str(load(zlcParameterPath)["modelType"])
    
    # Adsorbent density, mass of sorbent and particle epsilon
    adsorbentDensity = load(zlcParameterPath)["adsorbentDensity"]
    particleEpsilon = load(zlcParameterPath)["particleEpsilon"]
    massSorbent = load(zlcParameterPath)["massSorbent"]
    rpore = load(zlcParameterPath)["rpore"]
    # Volume of sorbent material [m3]
    volSorbent = (massSorbent/1000)/adsorbentDensity
    # Volume of gas chamber (dead volume) [m3]
    volGas = volSorbent/(1-particleEpsilon)*particleEpsilon
    # Dead volume model
    deadVolumeFile = load(zlcParameterPath)["deadVolumeFile"]
    isothermDataFile = str(load(zlcParameterPath)["isothermDataFile"])
    # Get the parameter values (in actual units)
    xOpt = np.multiply(pOpt,pRef)
    isothermDir = '..' + os.path.sep + 'isothermFittingData/'
    modelOutputTemp = loadmat(isothermDir+isothermDataFile)["isothermData"]       
    # Convert the nDarray to list
    nDArrayToList = np.ndarray.tolist(modelOutputTemp)  
    tempListData = nDArrayToList[0][0]
    # Get the necessary variables
    isothermAll = tempListData[4]
    isothermTemp = isothermAll[:,0]
    if len(isothermTemp) == 6:
        idx = [0, 2, 4, 1, 3, 5]
        isothermTemp = isothermTemp[idx]
    if len(isothermTemp) == 13:
        idx = [0, 2, 4, 6, 7, 8, 9, 10, 11, 12]
        isothermTemp = isothermTemp[idx]
    # Compute the downsample intervals for the experiments
    # This is only to make sure that all experiments get equal weights
    numPointsExp = np.zeros(len(fileName))
    for ii in range(len(fileName)): 
        fileToLoad = fileName[ii]
        # Load experimental molefraction
        timeElapsedExp = load(fileToLoad)["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExp)
    # Downsample intervals
    downsampleInt = numPointsExp/np.min(numPointsExp)
    
    # Initialize variables
    computedError = 0
    moleFracExpALL = np.array([])
    moleFracSimALL = np.array([])
                
    # Loop over all available experiments    
    for ii in range(len(fileName)):
        fileToLoad = fileName[ii]   
        
        # Initialize simulation mole fraction
        moleFracSim = []  
        # Load experimental time, molefraction and flowrate (accounting for downsampling)
        timeElapsedExpTemp = load(fileToLoad)["timeElapsed"].flatten()
        moleFracExpTemp = load(fileToLoad)["moleFrac"].flatten()
        flowRateTemp = load(fileToLoad)["flowRate"].flatten()
        timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
        moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
        flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))]
        
        if moleFracExp[0] > 0.5:
            deadVolumeFlow = deadVolumeFile[1]
        else:
            deadVolumeFlow = deadVolumeFile[0]
        if len(deadVolumeFlow[0]) == 1: # 1 DV for 1 DV file
            deadVolumeFileTemp = str(deadVolumeFlow[0])
        elif len(deadVolumeFlow[0]) == 2: # 1 DV for 1 DV file
            if np.absolute(flowRateExp[-1] - 1) > 0.2: # for lowflowrate experiments!
                deadVolumeFileTemp =  str(deadVolumeFlow[0][0])
            else:
                deadVolumeFileTemp =  str(deadVolumeFlow[0][1])
        elif len(deadVolumeFlow[0]) == 3: # 1 DV for 1 DV file
            if np.absolute(flowRateExp[-1] - 1) < 0.1: # for lowflowrate experiments!
                deadVolumeFileTemp =  str(deadVolumeFlow[0][2])
            elif np.absolute(flowRateExp[-1] - 45/60) < 0.1: # for lowflowrate experiments!
                deadVolumeFileTemp =  str(deadVolumeFlow[0][1])
            else:
                deadVolumeFileTemp =  str(deadVolumeFlow[0][0])
                
                
        # Integration and ode evaluation time (check simulateZLC/simulateDeadVolume)
        timeInt = timeElapsedExp
    
        # Parse out the xOpt to the isotherm model and kinetic parameters
        # isothermModel = xOpt[0:-2]
        if modelType == 'Diffusion1T':
            isothermModel = isothermTemp[np.where(isothermTemp!=0)]        
            rateConstant_1 = xOpt[-2]
            rateConstant_2 = 0
            rateConstant_3 = xOpt[-1]
        elif modelType == 'Diffusion1TNI':
            isothermModel = isothermTemp[np.where(isothermTemp!=0)]        
            rateConstant_1 = xOpt[-2]
            rateConstant_2 = 0
            rateConstant_3 = xOpt[-1]
        elif modelType == 'Diffusion1Ttau':
            isothermModel = isothermTemp[np.where(isothermTemp!=0)]        
            rateConstant_1 = xOpt[-2]
            rateConstant_2 = 0
            rateConstant_3 = xOpt[-1]
        elif modelType == 'Diffusion1TNItau':
            isothermModel = isothermTemp[np.where(isothermTemp!=0)]        
            rateConstant_1 = xOpt[-2]
            rateConstant_2 = 0
            rateConstant_3 = xOpt[-1]
        elif modelType == 'KineticSB' or modelType == 'Kinetic' or modelType == 'KineticOld' :
            isothermModel = isothermTemp[np.where(isothermTemp!=0)]        
            rateConstant_1 = xOpt[-2]
            rateConstant_2 = xOpt[-1]
            rateConstant_3 = 0
        elif modelType == 'Diffusion' or modelType == 'KineticSBMacro':
            isothermModel = isothermTemp[np.where(isothermTemp!=0)]        
            rateConstant_1 = xOpt[-3]
            rateConstant_2 = xOpt[-2]
            rateConstant_3 = xOpt[-1]
        else:
            isothermModel = xOpt[0:-2]
            rateConstant_1 = xOpt[-2]
            rateConstant_2 = xOpt[-1]
            rateConstant_3 = 0
        
        if modelType == 'Diffusion1T':
            # Compute the model response using the optimized parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                        initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    isothermModel = isothermModel,
                                                    rateConstant_1 = rateConstant_1,
                                                    rateConstant_2 = 0,
                                                    rateConstant_3 = rateConstant_3,
                                                    rpore = rpore,
                                                    deadVolumeFile = str(deadVolumeFileTemp),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas,
                                                    temperature = temperatureExp[ii],
                                                    modelType = 'Diffusion')
        elif modelType == 'Diffusion1TNI':
            # Compute the model response using the optimized parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                        initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    isothermModel = isothermModel,
                                                    rateConstant_1 = rateConstant_1,
                                                    rateConstant_2 = 0,
                                                    rateConstant_3 = rateConstant_3,
                                                    rpore = rpore,
                                                    deadVolumeFile = str(deadVolumeFileTemp),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas,
                                                    temperature = temperatureExp[ii],
                                                    modelType = 'Diffusion1TNI')
        elif modelType == 'Diffusion1Ttau':
            # Compute the model response using the optimized parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                        initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    isothermModel = isothermModel,
                                                    rateConstant_1 = rateConstant_1,
                                                    rateConstant_2 = 0,
                                                    rateConstant_3 = rateConstant_3,
                                                    rpore = rpore,
                                                    deadVolumeFile = str(deadVolumeFileTemp),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas,
                                                    temperature = temperatureExp[ii],
                                                    modelType = 'Diffusion1Ttau')
        elif modelType == 'Diffusion1TNItau':
            # Compute the model response using the optimized parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                        initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    isothermModel = isothermModel,
                                                    rateConstant_1 = rateConstant_1,
                                                    rateConstant_2 = 0,
                                                    rateConstant_3 = rateConstant_3,
                                                    rpore = rpore,
                                                    deadVolumeFile = str(deadVolumeFileTemp),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas,
                                                    temperature = temperatureExp[ii],
                                                    modelType = 'Diffusion1TNItau')
        
        else:
                # Compute the model response using the optimized parameters
            _ , moleFracSim , resultMat = simulateCombinedModel(timeInt = timeInt,
                                                        initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-10:-1]*1e-6), # Flow rate for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    isothermModel = isothermModel,
                                                    rateConstant_1 = rateConstant_1,
                                                    rateConstant_2 = rateConstant_2,
                                                    rateConstant_3 = rateConstant_3,
                                                    rpore = rpore,
                                                    deadVolumeFile = str(deadVolumeFileTemp),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas,
                                                    temperature = temperatureExp[ii],
                                                    modelType = modelType)
       
        # Stack mole fraction from experiments and simulation for error 
        # computation
        minExp = np.min(moleFracExp) # Compute the minimum from experiment
        normalizeFactor = np.max(moleFracExp - np.min(moleFracExp)) # Compute the max from normalized data
        moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
        moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))
        
    # Compute the MLE error of the model for the given parameters
    computedError = computeMLEError(moleFracExpALL,moleFracSimALL,
                                    downsampleData=downsampleData)
    
    # Return the objective function value, experimental and simulated output
    return computedError, moleFracExpALL, moleFracSimALL