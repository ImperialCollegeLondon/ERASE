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
# Find the isotherm parameters and the kinetic rate constant by fitting
# the complete response curve from the ZLC experiment. Note that currently
# the isotherm can be SSL or DSL model. The rate constant is assumed to be a
# constant in the LDF model and is analogous to Gleuckauf approximation
# Reference: 10.1016/j.ces.2014.12.062
#
# Last modified:
# - 2021-08-20, AK: Change definition of rate constants
# - 2021-07-21, AK: Add adsorbent density as an input
# - 2021-07-02, AK: Remove threshold factor
# - 2021-07-01, AK: Add sensitivity analysis
# - 2021-06-16, AK: Add temperature dependence to kinetics
# - 2021-06-14, AK: More fixes for error computation
# - 2021-06-12, AK: Fix for error computation (major)
# - 2021-06-11, AK: Change normalization for error
# - 2021-06-02, AK: Add normalization for error
# - 2021-06-01, AK: Add temperature as an input
# - 2021-05-25, AK: Add kinetic mode for estimation
# - 2021-05-24, AK: Improve information passing (for output)
# - 2021-05-13, AK: Change structure to input mass of adsorbent
# - 2021-05-05, AK: Bug fix for MLE error computation
# - 2021-05-05, AK: Modify error computation for dead volume
# - 2021-04-28, AK: Add reference values for isotherm parameters
# - 2021-04-27, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def extractZLCParameters(**kwargs):
    import numpy as np
    from geneticalgorithm2 import geneticalgorithm2 as ga # GA
    from extractDeadVolume import filesToProcess # File processing script
    from sensitivityAnalysis import sensitivityAnalysis
    from smt.sampling_methods import LHS
    import auxiliaryFunctions
    import os
    from numpy import savez
    from numpy import load
    import multiprocessing # For parallel processing
    import socket
    from scipy.io import loadmat
    
    # Change path directory
    # Assumes either running from ERASE or from experimental. Either ways
    # this has to be run from experimental
    if not os.getcwd().split(os.path.sep)[-1] == 'experimental':
        os.chdir("experimental")

    # Get the commit ID of the current repository
    gitCommitID = auxiliaryFunctions.getCommitID()
    
    # Get the current date and time for saving purposes    
    currentDT = auxiliaryFunctions.getCurrentDateTime()
    
    # Find out the total number of cores available for parallel processing
    num_cores = 150

    #####################################
    ###### USER DEFINED PROPERTIES ######
    # If not passed to the function, default values used
    # Isotherm model type
    if 'modelType' in kwargs:
        modelType = kwargs["modelType"]
    else:
        modelType = 'SSL'

    # Number of times optimization repeated
    numOptRepeat = 7
    
    # Directory of raw data
    mainDir = 'runData'
    # File name of the experiments
    if 'fileName' in kwargs:
        fileName = kwargs["fileName"]
    else:
        fileName = ['ZLC_ActivatedCarbon_Exp43F_Output.mat',
                    'ZLC_ActivatedCarbon_Exp48F_Output.mat',
                    'ZLC_ActivatedCarbon_Exp55F_Output.mat',]
    
    # Temperature (for each experiment)
    if 'temperature' in kwargs:
        temperature = kwargs["temperature"]
    else:
        temperature = [306.47,317.18,339.14]
    
    # Parameters for genetic algorithm
    if 'algorithm_param' in kwargs:
        algorithm_param = kwargs["algorithm_param"]
    else:        
        algorithm_param = {'max_num_iteration':30,
                           'population_size':400,
                           'mutation_probability':0.25,
                           'crossover_probability': 0.55,
                           'parents_portion': 0.15,
                           'elit_ratio': 0.01,
                           'max_iteration_without_improv':None}
        
    # Sorbent Skeletal Density [kg/m3] (for each experiment)
    if 'adsorbentDensity' in kwargs:
        adsorbentDensity = kwargs["adsorbentDensity"]
    else:
        adsorbentDensity = 2000
        
    # Particle porosity [-] (for each experiment)
    if 'particleEpsilon' in kwargs:
        particleEpsilon = kwargs["particleEpsilon"]
    else:
        particleEpsilon = 0.7   
        
    # Sorbent Mass [g] (for each experiment)
    if 'massSorbent' in kwargs:
        massSorbent = kwargs["massSorbent"]
    else:
        massSorbent = 0.05
        
    # Dead volume model
    if 'deadVolumeFile' in kwargs:
        deadVolumeFile = kwargs["deadVolumeFile"]
    else:
        deadVolumeFile = [[['deadVolumeCharacteristics_20230321_1048_59cc206.npz', # 50A
                            'deadVolumeCharacteristics_20230321_1238_59cc206.npz']], # 51A
                          [['deadVolumeCharacteristics_20230321_1137_59cc206.npz', # 50B
                            'deadVolumeCharacteristics_20230321_1252_59cc206.npz']]] # 51B
        
    # Isotherm model (if fitting only kinetic constant)
    if 'isothermDataFile' in kwargs:
        isothermDataFile = kwargs["isothermDataFile"]
    else:
        isothermDataFile = 'Purolite_CO2_3F_290923.mat'  
        
    # Downsample the data at different compositions (this is done on 
    # normalized data) [High and low comp]
    if 'downsampleData' in kwargs:
        downsampleData = kwargs["downsampleData"]
    else:
        downsampleData = True  
        
    # Downsample the for each exp to have the same number of points
    if 'downsampleExp' in kwargs:
        downsampleExp = kwargs["downsampleExp"]
    else:
        downsampleExp = False
      
    # Downsample the for each exp to have the same number of points
    if 'rpore' in kwargs:
        rpore = kwargs["rpore"]
    else:
        rpore = 1e-9
        
    # Dummt file as placeholder
    isothermFile = 'zlcParameters_20210525_1610_a079f4a.npz'

    
    # Confidence interval for the sensitivity analysis
    alpha = 0.95
    
    popSize = algorithm_param["population_size"]
    
    #####################################
    #####################################
    
    # Generate .npz file for python processing of the .mat file 
    filesToProcess(True,mainDir,fileName,'ZLC')

    # Define the bounds and the type of the parameters to be optimized 
    # Parameters optimized: qs,b0,delU (for DSL: both sites), k0 and delE
    # (16.06.21: Arrhenius constant and activation energy)
    # Single-site Langmuir
    if modelType == 'SSL':
        optBounds = np.array(([0.01,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1]))
        optType=np.array(['real','real','real','real','real'])
        problemDimension = len(optType)
        isoRef = [10, 1e-5, 40e3, 1000, 1000] # Reference for parameters
        isothermFile = [] # Isotherm file is empty as it is fit
        paramIso = [] # Isotherm parameters is empty as it is fit
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)

    # Dual-site Langmuir
    elif modelType == 'DSL':
        optBounds = np.array(([0.01,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1], [0.01,1],
                              [np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1], [np.finfo(float).eps,1]))
        optType=np.array(['real','real','real','real','real','real','real','real'])
        problemDimension = len(optType)
        isoRef = [10, 1e-5, 40e3, 10, 1e-5, 40e3, 1000, 1000] # Reference for the parameters 
        isothermFile = [] # Isotherm file is empty as it is fit
        paramIso = [] # Isotherm parameters is empty as it is fit
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)
        
    # Single-site Sips
    elif modelType == 'SSS':
        optBounds = np.array(([np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1], [np.finfo(float).eps,1],
                              [np.finfo(float).eps,1], [np.finfo(float).eps,1]))
        optType=np.array(['real','real','real','real','real','real'])
        problemDimension = len(optType)
        isoRef = [10, 1e-5, 40e3, 2,  1000, 1000] # Reference for the parameters
        isothermFile = [] # Isotherm file is empty as it is fit
        paramIso = [] # Isotherm parameters is empty as it is fit
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)

    # Kinetic constants only
    # Note: This might be buggy for simulations performed before 20.08.21
    # This is because of the changes to the structure of the kinetic model
    elif modelType == 'Kinetic':
        optBounds = np.array(([np.finfo(float).eps,1], [np.finfo(float).eps,1]))
        optType=np.array(['real','real'])
        problemDimension = len(optType)
        isoRef = [1000, 1000] # Reference for the parameter (has to be a list)
        # File with parameter estimates for isotherm (ZLC)
        isothermDir = '..' + os.path.sep + 'isothermFittingData/'
        modelOutputTemp = loadmat(isothermDir+isothermDataFile)["isothermData"]       
        # Convert the nDarray to list
        nDArrayToList = np.ndarray.tolist(modelOutputTemp)
        # Unpack another time (due to the structure of loadmat)
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
        paramIso = isothermTemp[np.where(isothermTemp!=0)]
        paramIso = np.append(paramIso,[0,0])
        # modelNonDim = modelOutputTemp[()]["variable"]
        # parameterRefTemp = load(isothermDir+isothermFile, allow_pickle=True)["parameterReference"]
        # Get the isotherm parameters
        # paramIso = np.multiply(modelNonDim,parameterRefTemp)
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)
    
    # Kinetic constants only (Macropore control only)
    # Note: This might be buggy for simulations performed before 20.08.21
    # This is because of the changes to the structure of the kinetic model
    elif modelType == 'KineticMacro':
        optBounds = np.array(([np.finfo(float).eps,1], [np.finfo(float).eps,1]))
        optType=np.array(['real','real'])
        problemDimension = len(optType)
        isoRef = [1000, 1000] # Reference for the parameter (has to be a list)
        # File with parameter estimates for isotherm (ZLC)
        isothermDir = '..' + os.path.sep + 'isothermFittingData/'
        modelOutputTemp = loadmat(isothermDir+isothermDataFile)["isothermData"]       
        # Convert the nDarray to list
        nDArrayToList = np.ndarray.tolist(modelOutputTemp)
        # Unpack another time (due to the structure of loadmat)
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
        paramIso = isothermTemp[np.where(isothermTemp!=0)]
        paramIso = np.append(paramIso,[0,0])
        # modelNonDim = modelOutputTemp[()]["variable"]
        # parameterRefTemp = load(isothermDir+isothermFile, allow_pickle=True)["parameterReference"]
        # Get the isotherm parameters
        # paramIso = np.multiply(modelNonDim,parameterRefTemp)
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)

    elif modelType == 'KineticSB':
        optBounds = np.array(([np.finfo(float).eps,250], [np.finfo(float).eps,0.05]))
        optType=np.array(['real','real'])
        problemDimension = len(optType)
        isoRef = [1000, 1000] # Reference for the parameter (has to be a list)
        # File with parameter estimates for isotherm (ZLC)
        isothermDir = '..' + os.path.sep + 'isothermFittingData/'
        modelOutputTemp = loadmat(isothermDir+isothermDataFile)["isothermData"]       
        # Convert the nDarray to list
        nDArrayToList = np.ndarray.tolist(modelOutputTemp)
        # Unpack another time (due to the structure of loadmat)
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
        paramIso = isothermTemp[np.where(isothermTemp!=0)]
        paramIso = np.append(paramIso,[0,0])
        # modelNonDim = modelOutputTemp[()]["variable"]
        # parameterRefTemp = load(isothermDir+isothermFile, allow_pickle=True)["parameterReference"]
        # Get the isotherm parameters
        # paramIso = np.multiply(modelNonDim,parameterRefTemp)
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)
  
    elif modelType == 'KineticSBMacro':
        optBounds = np.array(([np.finfo(float).eps,250], [np.finfo(float).eps,0.05], [np.finfo(float).eps,3]))
        optType=np.array(['real','real','real'])
        problemDimension = len(optType)
        isoRef = [1000, 1000, 1000] # Reference for the parameter (has to be a list)
        # File with parameter estimates for isotherm (ZLC)
        isothermDir = '..' + os.path.sep + 'isothermFittingData/'
        modelOutputTemp = loadmat(isothermDir+isothermDataFile)["isothermData"]       
        # Convert the nDarray to list
        nDArrayToList = np.ndarray.tolist(modelOutputTemp)
        # Unpack another time (due to the structure of loadmat)
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
        paramIso = isothermTemp[np.where(isothermTemp!=0)]
        paramIso = np.append(paramIso,[0,0,0])
        # modelNonDim = modelOutputTemp[()]["variable"]
        # parameterRefTemp = load(isothermDir+isothermFile, allow_pickle=True)["parameterReference"]
        # Get the isotherm parameters
        # paramIso = np.multiply(modelNonDim,parameterRefTemp)
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)
        
    elif modelType == 'Diffusion':
        optBounds = np.array(([1000,10000], [0.023,0.040], [1.1e-3,6e-3]))
        optType=np.array(['real','real','real'])
        problemDimension = len(optType)
        isoRef = [1000, 1000, 1000] # Reference for the parameter (has to be a list)
        # File with parameter estimates for isotherm (ZLC)
        isothermDir = '..' + os.path.sep + 'isothermFittingData/'
        modelOutputTemp = loadmat(isothermDir+isothermDataFile)["isothermData"]       
        # Convert the nDarray to list
        nDArrayToList = np.ndarray.tolist(modelOutputTemp)
        # Unpack another time (due to the structure of loadmat)
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
        paramIso = isothermTemp[np.where(isothermTemp!=0)]
        paramIso = np.append(paramIso,[0,0,0])
        # modelNonDim = modelOutputTemp[()]["variable"]
        # parameterRefTemp = load(isothermDir+isothermFile, allow_pickle=True)["parameterReference"]
        # Get the isotherm parameters
        # paramIso = np.multiply(modelNonDim,parameterRefTemp)
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)            
    elif modelType == 'Diffusion1T':
        optBounds = np.array(([35.2e-3,40e-3],[0.8e-3,7e-3]))
        optType=np.array(['real','real'])
        problemDimension = len(optType)
        isoRef = [1000, 1000] # Reference for the parameter (has to be a list)
        # File with parameter estimates for isotherm (ZLC)
        isothermDir = '..' + os.path.sep + 'isothermFittingData/'
        modelOutputTemp = loadmat(isothermDir+isothermDataFile)["isothermData"]       
        # Convert the nDarray to list
        nDArrayToList = np.ndarray.tolist(modelOutputTemp)
        # Unpack another time (due to the structure of loadmat)
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
        paramIso = isothermTemp[np.where(isothermTemp!=0)]
        paramIso = np.append(paramIso,[0,0])
        # modelNonDim = modelOutputTemp[()]["variable"]
        # parameterRefTemp = load(isothermDir+isothermFile, allow_pickle=True)["parameterReference"]
        # Get the isotherm parameters
        # paramIso = np.multiply(modelNonDim,parameterRefTemp)
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)  
    elif modelType == 'Diffusion1Ttau':
        optBounds = np.array(([35.2e-3,40e-3],[1e-3,10e-3]))
        optType=np.array(['real','real'])
        problemDimension = len(optType)
        isoRef = [1000, 1000] # Reference for the parameter (has to be a list)
        # File with parameter estimates for isotherm (ZLC)
        isothermDir = '..' + os.path.sep + 'isothermFittingData/'
        modelOutputTemp = loadmat(isothermDir+isothermDataFile)["isothermData"]       
        # Convert the nDarray to list
        nDArrayToList = np.ndarray.tolist(modelOutputTemp)
        # Unpack another time (due to the structure of loadmat)
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
        paramIso = isothermTemp[np.where(isothermTemp!=0)]
        paramIso = np.append(paramIso,[0,0])
        # modelNonDim = modelOutputTemp[()]["variable"]
        # parameterRefTemp = load(isothermDir+isothermFile, allow_pickle=True)["parameterReference"]
        # Get the isotherm parameters
        # paramIso = np.multiply(modelNonDim,parameterRefTemp)
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)  
    elif modelType == 'Diffusion1TNItau':
        optBounds = np.array(([35.2e-3,40e-3],[1e-3,10e-3]))
        optType=np.array(['real','real'])
        problemDimension = len(optType)
        isoRef = [1000, 1000] # Reference for the parameter (has to be a list)
        # File with parameter estimates for isotherm (ZLC)
        isothermDir = '..' + os.path.sep + 'isothermFittingData/'
        modelOutputTemp = loadmat(isothermDir+isothermDataFile)["isothermData"]       
        # Convert the nDarray to list
        nDArrayToList = np.ndarray.tolist(modelOutputTemp)
        # Unpack another time (due to the structure of loadmat)
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
        paramIso = isothermTemp[np.where(isothermTemp!=0)]
        paramIso = np.append(paramIso,[0,0])
        # modelNonDim = modelOutputTemp[()]["variable"]
        # parameterRefTemp = load(isothermDir+isothermFile, allow_pickle=True)["parameterReference"]
        # Get the isotherm parameters
        # paramIso = np.multiply(modelNonDim,parameterRefTemp)
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)  
    elif modelType == 'Diffusion1TNI':
        optBounds = np.array(([35.2e-3,40e-3],[0.8e-3,7e-3]))
        optType=np.array(['real','real'])
        problemDimension = len(optType)
        isoRef = [1000, 1000] # Reference for the parameter (has to be a list)
        # File with parameter estimates for isotherm (ZLC)
        isothermDir = '..' + os.path.sep + 'isothermFittingData/'
        modelOutputTemp = loadmat(isothermDir+isothermDataFile)["isothermData"]       
        # Convert the nDarray to list
        nDArrayToList = np.ndarray.tolist(modelOutputTemp)
        # Unpack another time (due to the structure of loadmat)
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
        paramIso = isothermTemp[np.where(isothermTemp!=0)]
        paramIso = np.append(paramIso,[0,0])
        # modelNonDim = modelOutputTemp[()]["variable"]
        # parameterRefTemp = load(isothermDir+isothermFile, allow_pickle=True)["parameterReference"]
        # Get the isotherm parameters
        # paramIso = np.multiply(modelNonDim,parameterRefTemp)
        lhsPopulation = LHS(xlimits=optBounds)
        start_population = lhsPopulation(popSize)
    # Initialize the parameters used for ZLC fitting process
    fittingParameters(True,temperature,deadVolumeFile,adsorbentDensity,particleEpsilon,
                      massSorbent,isoRef,downsampleData,paramIso,downsampleExp,modelType,rpore)
    
    # Minimize an objective function to compute the equilibrium and kinetic 
    # parameters from ZLC experiments
    model = ga(function = ZLCObjectiveFunction, dimension=problemDimension, 
                               variable_type_mixed = optType,
                               variable_boundaries = optBounds,
                               algorithm_parameters=algorithm_param,
                               function_timeout = 300) # Timeout set to 300 s (change if code crashes)
    
    # Call the GA optimizer using multiple cores
    model.run(set_function=ga.set_function_multiprocess(ZLCObjectiveFunction,
                                                         n_jobs = num_cores),
              no_plot = True, start_generation= (start_population, None))
    # Repeat the optimization with the last generation repeated numOptRepeat
    # times (for better accuracy)
    for ii in range(numOptRepeat):
        model.run(set_function=ga.set_function_multiprocess(ZLCObjectiveFunction,
                                                             n_jobs = num_cores),
                  start_generation=model.output_dict['last_generation'], no_plot = True)
        
    # Save the zlc parameters into a native numpy file
    # The .npz file is saved in a folder called simulationResults (hardcoded)
    fileNameDummy = fileName[1]
    AdsName = fileNameDummy.split('_',3)[1]
    filePrefix = "zlcParameters"
    saveFileName = filePrefix + "_" + AdsName + "_" + currentDT + "_" + gitCommitID
    savePath = os.path.join('..','simulationResults',saveFileName)
    
    # Check if simulationResults directory exists or not. If not, create the folder
    if not os.path.exists(os.path.join('..','simulationResults')):
        os.mkdir(os.path.join('..','simulationResults'))
    
    # Save the output into a .npz file
    savez (savePath, modelOutput = model.output_dict, # Model output
           optBounds = optBounds, # Optimizer bounds
           algoParameters = algorithm_param, # Algorithm parameters
           numOptRepeat = numOptRepeat, # Number of times optimization repeated
           fileName = fileName, # Names of file used for fitting
           temperature = temperature, # Temperature [K]
           deadVolumeFile = deadVolumeFile, # Dead volume file used for parameter estimation
           isothermDataFile = isothermDataFile, # isotherm data file from matlab
           isothermFile = isothermFile, # Isotherm parameters file, if only kinetics estimated
           adsorbentDensity = adsorbentDensity, # Adsorbent density [kg/m3]
           particleEpsilon = particleEpsilon, # Particle voidage [-]
           massSorbent = massSorbent, # Mass of sorbent [g]
           paramIso = paramIso, # isotherm parameters used for fitting if fitting for kinetics only
           parameterReference = isoRef, # Parameter references [-]
           downsampleFlag = downsampleData, # Flag for downsampling data by conc [-]
           downsampleExp = downsampleExp, # Flag for downsampling data by number of points [-]
           hostName = socket.gethostname(),
           modelType = modelType,
           rpore = rpore) # Hostname of the computer
    
    # Remove all the .npy files genereated from the .mat
    # Load the names of the file to be used for estimating ZLC parameters
    filePath = filesToProcess(False,[],[],'ZLC')
    # Loop over all available files    
    for ii in range(len(filePath)):
        os.remove(filePath[ii])

    # Perform the sensitivity analysis with the optimized parameters
    sensitivityAnalysis(saveFileName, alpha)
    
    # Return the optimized values
    return model.output_dict
    
# func: deadVolObjectiveFunction
# For use with GA, the function accepts only one input (parameters from the 
# optimizer)
def ZLCObjectiveFunction(x):
    import numpy as np
    from numpy import load
    from extractDeadVolume import filesToProcess # File processing script
    from simulateCombinedModel import simulateCombinedModel
    from computeMLEError import computeMLEError
    import pdb 
    # Get the zlc parameters needed for the solver
    temperature, deadVolumeFile, adsorbentDensity, particleEpsilon, massSorbent, isoRef, downsampleData, paramIso, downsampleExp, modelType,rpore = fittingParameters(False,[],[],[],[],[],[],[],[],[],[],[])

    # Volume of sorbent material [m3]
    volSorbent = (massSorbent/1000)/adsorbentDensity
    # Volume of gas in pores [m3]
    volGas = volSorbent/(1-particleEpsilon)*particleEpsilon

    # Prepare isotherm model (the first n-1 parameters are for the isotherm model)
    if len(paramIso) != 0:
        if modelType == 'KineticSBMacro' or modelType == 'Diffusion':
            isothermModel = paramIso[0:-3] # Use this if isotherm parameter provided (for kinetics only: KineticSBMacro or Diffusion)
        else:
            isothermModel = paramIso[0:-2] # Use this if isotherm parameter provided (for kinetics only: Other kinetic models)
    else:        
        isothermModel = np.multiply(x[0:-2],isoRef[0:-2]) # Use this if both equilibrium and kinetics is fit

    # Load the names of the file to be used for estimating zlc parameters
    filePath = filesToProcess(False,[],[],'ZLC')
    
    # Parse out number of data points for each experiment (for downsampling)
    numPointsExp = np.zeros(len(filePath))
    for ii in range(len(filePath)): 
        # Load experimental molefraction
        timeElapsedExp = load(filePath[ii])["timeElapsed"].flatten()
        numPointsExp[ii] = len(timeElapsedExp)
        
    # Downsample intervals
    if downsampleExp:
        downsampleInt = numPointsExp/np.min(numPointsExp)
    else:
        downsampleInt = numPointsExp/numPointsExp
    
    # pdb.set_trace()
    # Initialize error for objective function
    computedError = 0
    moleFracExpALL = np.array([])
    moleFracSimALL = np.array([])    

    # Loop over all available files    
    for ii in range(len(filePath)):
        # Initialize outputs
        moleFracSim = []  
        # Load experimental time, molefraction and flowrate (accounting for downsampling)
        timeElapsedExpTemp = load(filePath[ii])["timeElapsed"].flatten()
        moleFracExpTemp = load(filePath[ii])["moleFrac"].flatten()
        flowRateTemp = load(filePath[ii])["flowRate"].flatten()
        timeElapsedExp = timeElapsedExpTemp[::int(np.round(downsampleInt[ii]))]
        moleFracExp = moleFracExpTemp[::int(np.round(downsampleInt[ii]))]
        flowRateExp = flowRateTemp[::int(np.round(downsampleInt[ii]))] # [cc/s]
            
        if moleFracExp[0] > 0.2:
            deadVolumeFlow = deadVolumeFile[1]
        else:
            deadVolumeFlow = deadVolumeFile[0]
            
            
        if len(deadVolumeFlow[0]) == 1: # 1 DV for 1 DV file
            deadVolumeFileTemp = str(deadVolumeFlow[0])
        elif len(deadVolumeFlow[0]) == 2: # 1 DV for 1 DV file
            if np.absolute(flowRateExp[-1] - 60/60) > 0.2: # for lowflowrate experiments!
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
        
        if modelType == 'KineticSBMacro' or modelType == 'Diffusion':
            # Compute the composite response using the optimizer parameters
            _ , moleFracSim , _ = simulateCombinedModel(isothermModel = isothermModel,
                                                    rateConstant_1 = x[-3]*isoRef[-3], # Last but one element is rate constant (Arrhenius constant)
                                                    rateConstant_2 = x[-2]*isoRef[-2], # Last element is activation energy
                                                    rateConstant_3 = x[-1]*isoRef[-1], # Last element is activation energy
                                                    rpore = rpore,
                                                    temperature = temperature[ii], # Temperature [K]
                                                    timeInt = timeInt,
                                                    initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-2:-1]*1e-6), # Flow rate [m3/s] for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    deadVolumeFile = str(deadVolumeFileTemp),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas,
                                                    adsorbentDensity = adsorbentDensity,
                                                    modelType = modelType)
        elif modelType == 'Diffusion1T':
            # Compute the composite response using the optimizer parameters
            _ , moleFracSim , _ = simulateCombinedModel(isothermModel = isothermModel,
                                                    rateConstant_1 = x[-2]*isoRef[-2], # Last but one element is rate constant (Arrhenius constant)
                                                    rateConstant_2 = 0, # Last element is activation energy
                                                    rateConstant_3 = x[-1]*isoRef[-1], # Last element is activation energy
                                                    rpore = rpore,
                                                    temperature = temperature[ii], # Temperature [K]
                                                    timeInt = timeInt,
                                                    initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-2:-1]*1e-6), # Flow rate [m3/s] for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    deadVolumeFile = str(deadVolumeFileTemp),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas,
                                                    adsorbentDensity = adsorbentDensity,
                                                    modelType = 'Diffusion')
        elif modelType == 'Diffusion1Ttau':
            # Compute the composite response using the optimizer parameters
            _ , moleFracSim , _ = simulateCombinedModel(isothermModel = isothermModel,
                                                    rateConstant_1 = x[-2]*isoRef[-2], # Last but one element is rate constant (Arrhenius constant)
                                                    rateConstant_2 = 0, # Last element is activation energy
                                                    rateConstant_3 = x[-1]*isoRef[-1], # Last element is activation energy
                                                    rpore = rpore,
                                                    temperature = temperature[ii], # Temperature [K]
                                                    timeInt = timeInt,
                                                    initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-2:-1]*1e-6), # Flow rate [m3/s] for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    deadVolumeFile = str(deadVolumeFileTemp),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas,
                                                    adsorbentDensity = adsorbentDensity,
                                                    modelType = 'Diffusion1Ttau')
        elif modelType == 'Diffusion1TNItau':
            # Compute the composite response using the optimizer parameters
            _ , moleFracSim , _ = simulateCombinedModel(isothermModel = isothermModel,
                                                    rateConstant_1 = x[-2]*isoRef[-2], # Last but one element is rate constant (Arrhenius constant)
                                                    rateConstant_2 = 0, # Last element is activation energy
                                                    rateConstant_3 = x[-1]*isoRef[-1], # Last element is activation energy
                                                    rpore = rpore,
                                                    temperature = temperature[ii], # Temperature [K]
                                                    timeInt = timeInt,
                                                    initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-2:-1]*1e-6), # Flow rate [m3/s] for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    deadVolumeFile = str(deadVolumeFileTemp),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas,
                                                    adsorbentDensity = adsorbentDensity,
                                                    modelType = 'Diffusion1TNItau')
        elif modelType == 'Diffusion1TNI':
            # Compute the composite response using the optimizer parameters
            _ , moleFracSim , _ = simulateCombinedModel(isothermModel = isothermModel,
                                                    rateConstant_1 = x[-2]*isoRef[-2], # Last but one element is rate constant (Arrhenius constant)
                                                    rateConstant_2 = 0, # Last element is activation energy
                                                    rateConstant_3 = x[-1]*isoRef[-1], # Last element is activation energy
                                                    rpore = rpore,
                                                    temperature = temperature[ii], # Temperature [K]
                                                    timeInt = timeInt,
                                                    initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-2:-1]*1e-6), # Flow rate [m3/s] for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    deadVolumeFile = str(deadVolumeFileTemp),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas,
                                                    adsorbentDensity = adsorbentDensity,
                                                    modelType = 'Diffusion1TNI')
        else:
            # Compute the composite response using the optimizer parameters
            _ , moleFracSim , _ = simulateCombinedModel(isothermModel = isothermModel,
                                                    rateConstant_1 = x[-2]*isoRef[-2], # Last but one element is rate constant (Arrhenius constant)
                                                    rateConstant_2 = x[-1]*isoRef[-1], # Last element is activation energy
                                                    rateConstant_3 = 0, # Last element is activation energy
                                                    rpore = rpore,
                                                    temperature = temperature[ii], # Temperature [K]
                                                    timeInt = timeInt,
                                                    initMoleFrac = [moleFracExp[0]], # Initial mole fraction assumed to be the first experimental point
                                                    flowIn = np.mean(flowRateExp[-1:-2:-1]*1e-6), # Flow rate [m3/s] for ZLC considered to be the mean of last 10 points (equilibrium)
                                                    expFlag = True,
                                                    deadVolumeFile = str(deadVolumeFileTemp),
                                                    volSorbent = volSorbent,
                                                    volGas = volGas,
                                                    adsorbentDensity = adsorbentDensity,
                                                    modelType = modelType)

        # Stack mole fraction from experiments and simulation for error 
        # computation
        # Normalize the mole fraction by dividing it by maximum value to avoid
        # irregular weightings for different experiment (at diff. scales)
        minExp = np.min(moleFracExp) # Compute the minimum from experiment
        normalizeFactor = np.max(moleFracExp - minExp) # Compute the max from normalized data
        moleFracExpALL = np.hstack((moleFracExpALL, (moleFracExp-minExp)/normalizeFactor))
        moleFracSimALL = np.hstack((moleFracSimALL, (moleFracSim-minExp)/normalizeFactor))
    
    # Compute the sum of the error for the difference between exp. and sim.
    computedError = computeMLEError(moleFracExpALL,moleFracSimALL,
                                    downsampleData=downsampleData)
    return computedError

# func: fittingParameters
# Parses dead volume calibration file, adsorbent density, voidage, mass to 
# be used for parameter estimation, parameter references and threshold for MLE
# This is done because the ga cannot handle additional user inputs
def fittingParameters(initFlag,temperature,deadVolumeFile,adsorbentDensity,
                      particleEpsilon,massSorbent,isoRef,downsampleData,
                      paramIso,downsampleExp,modelType, rpore):
    from numpy import savez
    from numpy import load
    # Process the data for python (if needed)
    if initFlag:
        # Save the necessary inputs to a temp file
        dummyFileName = 'tempFittingParametersZLC.npz'
        savez (dummyFileName, temperature = temperature,
               deadVolumeFile = deadVolumeFile,
               adsorbentDensity=adsorbentDensity,
               particleEpsilon=particleEpsilon,
               massSorbent=massSorbent,
               isoRef=isoRef,
               downsampleData=downsampleData,
               paramIso = paramIso, 
               downsampleExp = downsampleExp,
               modelType = modelType,
               rpore = rpore)
    # Returns the path of the .npz file to be used 
    else:
    # Load the dummy file with temperature, deadVolumeFile, adsorbent density, particle voidage,
    # and mass of sorbent
        dummyFileName = 'tempFittingParametersZLC.npz'
        temperature = load (dummyFileName)["temperature"]
        deadVolumeFile = load (dummyFileName)["deadVolumeFile"]
        adsorbentDensity = load (dummyFileName)["adsorbentDensity"]
        particleEpsilon = load (dummyFileName)["particleEpsilon"]
        massSorbent = load (dummyFileName)["massSorbent"]
        isoRef = load (dummyFileName)["isoRef"]
        downsampleData = load (dummyFileName)["downsampleData"]
        paramIso = load (dummyFileName)["paramIso"]        
        downsampleExp = load (dummyFileName)["downsampleExp"]
        modelType = load (dummyFileName)["modelType"]
        rpore = load (dummyFileName)["rpore"]
        return temperature, deadVolumeFile, adsorbentDensity, particleEpsilon, massSorbent, isoRef, downsampleData, paramIso, downsampleExp, modelType, rpore