from extractZLCParameters import extractZLCParameters
from extractZLCParametersDA import extractZLCParametersDA
import sys
import numpy as np
import numpy.matlib
# import multiprocessing
# multiprocessing.set_start_method('fork')
sys.path.append('../ERASE/')
# Downsample the data at different compositions (this is done on 
# normalized data) [High and low comp]
downsampleData = True
downsampleExp = False # Same number of experimental points per curve



# modelType = 'Diffusion1Ttau'

modelType = 'Kinetic'


# ##############################################################################


# Adsorbent properties
# Adsorbent density [kg/m3]
# This has to be the skeletal density
# adsorbentDensity = 1020 # ZYH
# adsorbentDensity = 3200 # ZYNa
# adsorbentDensity = 2890 # ZYTMA
# adsorbentDensity = 2130 # ZYH MSB
# adsorbentDensity = 2410 # ZYNa MSB
# adsorbentDensity = 2310 # ZYTMA MSB
adsorbentDensity = 882

# Particle porosity

# particleEpsilon = 0.90 # ZYH
# particleEpsilon = 0.76 # ZYNa
# particleEpsilon = 0.47 # ZYNa excl micro
# particleEpsilon = 0.71 # ZYTMA
# particleEpsilon = 0.61 # CMS 3K
particleEpsilon = 0.37 # nominal value (does not matter for powder)

# Particle mass [g]
# massSorbent = 0.086 # ZYH real
#massSorbent = 0.062 # ZYNa
#massSorbent = 0.0125 # ZYNa
#massSorbent = 0.0123 # ZYNa
# massSorbent = 0.065 # ZYTMA
# massSorbent = 0.077 # CMS 3K
massSorbent = 0.0155

rpore = 166e-9
Dpvals = [5.6158e-05,5.9485e-05	,6.2905e-05] # DK
# Dpvals = [1.89156107809599e-05,	1.94990023798910e-05,	2.00786736020894e-05] # DUSTY

# Isotherm model (if fitting only kinetic constant)
# isothermDataFile = 'HCP-DETA-10min_DSL_052924.mat'
isothermDataFile = 'HCP-DETA-10min_TC_092024.mat'

# fileName  = ['ZLC_HCP_DETA_10min_DA_Exp01A_Output.mat', 
#               'ZLC_HCP_DETA_10min_DA_Exp01B_Output.mat', 
#               'ZLC_HCP_DETA_10min_DA_Exp02A_Output.mat',
#               'ZLC_HCP_DETA_10min_DA_Exp02B_Output.mat'] # Fitting all four curves

fileName  = ['ZLC_HCP_DETA_10min_DA_Exp01B_Output.mat',
              'ZLC_HCP_DETA_10min_DA_Exp02B_Output.mat'] # Fitting two curves

# fileName  = ['ZLC_HCP_DETA_10min_DA_Exp01A_Output.mat',
#               'ZLC_HCP_DETA_10min_DA_Exp02A_Output.mat',
#               'ZLC_HCP_DETA_10min_DA_Exp01A_Output.mat',
#               'ZLC_HCP_DETA_10min_DA_Exp02A_Output.mat',] # Fitting two curves (twice)

fileName  = ['ZLC_HCP_DETA_10min_DA_Exp01A_Output.mat'] # Fitting two curves

deadVolumeFile = [[['deadVolumeCharacteristics_20240521_1035_36_3_ed7d0dd.npz',  # low comp low flow
                    'deadVolumeCharacteristics_20240521_1041_19_3_ed7d0dd.npz']], # low comp high flow
                  [['deadVolumeCharacteristics_20240521_1037_53_3_ed7d0dd.npz', # high comp low flow
                    'deadVolumeCharacteristics_20240521_1043_32_3_ed7d0dd.npz']]] # high comp high flow

temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY
temperature = [ 298.15, ]*4 # ZY

print('fileName = '+str(fileName))
print('downsampleConc = '+str(downsampleData))
print('downsampleExp = '+str(downsampleExp))
print('massSorbent = '+str(massSorbent))
print('particleEpsilon = '+str(particleEpsilon))
print('adsorbentDensity = '+str(adsorbentDensity))
print('isothermDataFile = '+str(isothermDataFile))
print('deadVolumeFile = '+str(deadVolumeFile))
print('modelType = '+str(modelType))


for ii in range(1):
    # algorithm_param = {'max_num_iteration':50,
    #                     'population_size':200,
    #                     'mutation_probability':0.05, # generally (a lot) lower than crossover_probability
    #                     'crossover_probability': 0.3,
    #                     'parents_portion': 0.1,
    #                     'elit_ratio': 0.02,
    #                     'max_iteration_without_improv': 20}
    

    ### HASSAN'S SETTINGS ###
    algorithm_param = {'max_num_iteration':10,
                        'population_size':80,
                        'mutation_probability':0.25,
                        'crossover_probability': 0.55,
                        'parents_portion': 0.15,
                        'elit_ratio': 0.01,
                        'max_iteration_without_improv':None}
    
    # algorithm_param={'max_num_iteration':None,
    # 'population_size':100,
    # 'mutation_probability':0.1,
    # 'mutation_discrete_probability':None,
    # 'elit_ratio':0.01,
    # 'parents_portion':0.3,
    # 'crossover_type':'uniform',
    # 'mutation_type':'uniform_by_center',
    # 'mutation_discrete_type':'uniform_discrete',
    # 'selection_type':'roulette',
    # 'max_iteration_without_improv':None}

    extractZLCParametersDA(modelType = modelType,
                  fileName = fileName,
                  temperature = temperature,
                  algorithm_param = algorithm_param,
                  adsorbentDensity = adsorbentDensity,
                  particleEpsilon = particleEpsilon,
                  rpore = rpore,
                  Dpvals = Dpvals,
                  massSorbent = massSorbent,
                  deadVolumeFile = deadVolumeFile,
                  isothermDataFile = isothermDataFile,
                  downsampleData = downsampleData,
                  downsampleExp = downsampleExp)
    

# ##############################################################################
