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
# Computes the equilibrium loading at a given pressure, temperature, or mole 
# fraction using either single site Langmuir (SSL) or dual site Langmuir 
# model (DSL) for pure gases.
#
# Last modified:
# - 2021-04-26, AK: Initial creation
#
# Input arguments:
#
#
# Output arguments:
#
#
############################################################################

def computeEquilibriumLoading(**kwargs):
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    # Total pressure of the gas [Pa]
    if 'pressureTotal' in kwargs:
        pressureTotal = np.array(kwargs["pressureTotal"])
    else:
        pressureTotal = np.array([1e5])
        
    # Temperature of the gas [K]
    if 'temperature' in kwargs:
        temperature = np.array(kwargs["temperature"])
    else:
        temperature = np.array([298.15])
        
    # Mole fraction of the gas [-]
    if 'moleFrac' in kwargs:
        moleFrac = np.array(kwargs["moleFrac"])
    else:
        moleFrac = np.array([1.])
        
    # Isotherm Model (SSL or DSL)
    if 'isothermModel' in kwargs:
        isothermModel =  np.array(kwargs["isothermModel"])
        # If model has three parameters - SSL
        if len(isothermModel) == 3:
            modelType = 'SSL'
        # If model has six parameters - DSL
        elif len(isothermModel) == 6:
            modelType = 'DSL'
        # If model has four parameters - SSS
        elif len(isothermModel) == 4:
            modelType = 'SSS'
    else:
        # Default isotherm model is DSL and uses CO2 isotherm on AC8
        # Reference: 10.1007/s10450-020-00268-7
        isothermModel = np.array([0.44, 3.17e-6, 28.63e3, 6.10, 3.21e-6, 20.37e3])
        modelType = 'DSL'

    # Prepare the input parameters for the model
    inputParameters = (pressureTotal,temperature,moleFrac,isothermModel)
    
    try:
        # Call the function of the relevant model type
        if modelType == 'SSL':
            equilibriumLoading = simulateSSL(*inputParameters)
        elif modelType == 'DSL':
            equilibriumLoading = simulateDSL(*inputParameters)
        elif modelType == 'SSS':
            equilibriumLoading = simulateSSS(*inputParameters)
    except:
        equilibriumLoading = 0
    # Return the equilibrium loading
    return equilibriumLoading
    
# fun: simulateSSL
# Simulates the single site Langmuir model (pure) at a given pressure, 
# temperature and mole fraction
def simulateSSL(*inputParameters):
    import numpy as np
    
    # Gas constant
    Rg = 8.314; # [J/mol K]
    
    # Unpack the tuple with the inputs for the isotherm model
    pressureTotal,temperature,moleFrac,isothermModel = inputParameters
    
    # Compute the concentration at input pressure, temperature and mole fraction
    localConc = pressureTotal*moleFrac/(Rg*temperature)
    
    # Compute the adsorption affinity 
    isoAffinity = isothermModel[1]*np.exp(isothermModel[2]/(Rg*temperature))
    
    # Compute the numerator and denominator of a pure single site Langmuir
    isoNumerator = isothermModel[0]*isoAffinity*localConc
    isoDenominator = 1 + isoAffinity*localConc
    
    # Compute the equilibrium loading
    equilibriumLoading = isoNumerator/isoDenominator
    
    # Return the loading
    return equilibriumLoading
    
# fun: simulateDSL
# Simulates the dual site Langmuir model (pure) at a given pressure, 
# temperature and mole fraction
def simulateDSL(*inputParameters):
    import numpy as np

    # Gas constant
    Rg = 8.314; # [J/mol K]
    
    # Unpack the tuple with the inputs for the isotherm model
    pressureTotal,temperature,moleFrac,isothermModel = inputParameters
    
    # Compute the concentration at input pressure, temperature and mole fraction
    localConc = pressureTotal*moleFrac/(Rg*temperature)

    # Site 1
    # Compute the adsorption affinity 
    isoAffinity_1 = isothermModel[1]*np.exp(isothermModel[2]/(Rg*temperature))
    # Compute the numerator and denominator of a pure single site Langmuir
    isoNumerator_1 = isothermModel[0]*isoAffinity_1*localConc
    isoDenominator_1 = 1 + isoAffinity_1*localConc
    
    # Site 2
    # Compute the adsorption affinity 
    isoAffinity_2 = isothermModel[4]*np.exp(isothermModel[5]/(Rg*temperature))
    # Compute the numerator and denominator of a pure single site Langmuir
    isoNumerator_2 = isothermModel[3]*isoAffinity_2*localConc
    isoDenominator_2 = 1 + isoAffinity_2*localConc
    
    # Compute the equilibrium loading
    equilibriumLoading = isoNumerator_1/isoDenominator_1 + isoNumerator_2/isoDenominator_2

    # Return the loading
    return equilibriumLoading

def simulateSSS(*inputParameters):
    import numpy as np
    # import pdb
    # Gas constant
    Rg = 8.314; # [J/mol K]
    
    # Unpack the tuple with the inputs for the isotherm model
    pressureTotal,temperature,moleFrac,isothermModel = inputParameters
    if moleFrac < 0:
        moleFrac = 0
    # Compute the concentration at input pressure, temperature and mole fraction
    localConc = pressureTotal*moleFrac/(Rg*temperature)
    
    # Compute the adsorption affinity 
    isoAffinity = isothermModel[1]*np.exp(isothermModel[2]/(Rg*temperature))
    
    # Compute the numerator and denominator of a pure single site Langmuir
    isoNumerator = isothermModel[0]*(isoAffinity*localConc)**isothermModel[3]
    isoDenominator = 1 + (isoAffinity*localConc)**isothermModel[3]
    
    # Compute the equilibrium loading
    equilibriumLoading = isoNumerator/isoDenominator
    # pdb.set_trace()

    # Return the loading
    return equilibriumLoading