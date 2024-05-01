from extractZLCParameters import extractZLCParameters
import sys
import numpy as np
import numpy.matlib

sys.path.append('../ERASE/')
# Downsample the data at different compositions (this is done on 
# normalized data) [High and low comp]
downsampleData = True
downsampleExp = False # Same number of experimental points per curve




##############################################################################


# Adsorbent properties
# Adsorbent density [kg/m3]
# This has to be the skeletal density
# adsorbentDensity = 1020 # ZYH
# adsorbentDensity = 3200 # ZYNa
# adsorbentDensity = 2890 # ZYTMA
# adsorbentDensity = 2130 # ZYH MSB
adsorbentDensity = 2410 # ZYNa MSB
# adsorbentDensity = 2310 # ZYTMA MSB

# Particle porosity

# particleEpsilon = 0.90 # ZYH
particleEpsilon = 0.76 # ZYNa
particleEpsilon = 0.47 # ZYNa excl micro
# particleEpsilon = 0.71 # ZYTMA
# particleEpsilon = 0.61 # CMS 3K

# Particle mass [g]
# massSorbent = 0.086 # ZYH real
massSorbent = 0.062 # ZYNa
massSorbent = 0.013 # ZYNa
# massSorbent = 0.065 # ZYTMA
# massSorbent = 0.077 # CMS 3K

rpore = 166e-9
Dpvals = [3.93229043563274e-05	,4.05543022340028e-05	,4.17735724713926e-05] # DK

# Isotherm model (if fitting only kinetic constant)
# isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# isothermDataFile = 'ZYH_DSL_QC_070523.mat'

modelType = 'Diffusion1Ttau'


# fileName  = ['ZLC_ZYNaCrush_Exp05A_Output.mat',
#             'ZLC_ZYNaCrush_Exp09A_Output.mat',
#             'ZLC_ZYNaCrush_Exp11A_Output.mat',
#             'ZLC_ZYNaCrush_Exp06A_Output.mat',
#             'ZLC_ZYNaCrush_Exp10A_Output.mat',
#             'ZLC_ZYNaCrush_Exp12A_Output.mat',
#             'ZLC_ZYNaCrush_Exp05B_Output.mat',
#             'ZLC_ZYNaCrush_Exp09B_Output.mat',
#             'ZLC_ZYNaCrush_Exp11B_Output.mat',
#             'ZLC_ZYNaCrush_Exp06B_Output.mat',
#             'ZLC_ZYNaCrush_Exp10B_Output.mat',
#             'ZLC_ZYNaCrush_Exp12B_Output.mat',] 

fileName  = ['ZLC_ZYNaCrush_Exp05A_Output.mat',
            'ZLC_ZYNaCrush_Exp09A_Output.mat',
            'ZLC_ZYNaCrush_Exp11A_Output.mat',
            'ZLC_ZYNaCrush_Exp06A_Output.mat',
            'ZLC_ZYNaCrush_Exp10A_Output.mat',
            'ZLC_ZYNaCrush_Exp12A_Output.mat',]


deadVolumeFile = [[['deadVolumeCharacteristics_20231122_1743_b571c46.npz', 
                    'deadVolumeCharacteristics_20231122_1757_b571c46.npz']], 
                  [['deadVolumeCharacteristics_20231122_1750_b571c46.npz', 
                    'deadVolumeCharacteristics_20231122_1804_b571c46.npz']]] 

temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY

print('fileName = '+str(fileName))
print('downsampleConc = '+str(downsampleData))
print('downsampleExp = '+str(downsampleExp))
print('massSorbent = '+str(massSorbent))
print('particleEpsilon = '+str(particleEpsilon))
print('adsorbentDensity = '+str(adsorbentDensity))
print('isothermDataFile = '+str(isothermDataFile))
print('deadVolumeFile = '+str(deadVolumeFile))
print('modelType = '+str(modelType))


for ii in range(5):
    algorithm_param = {'max_num_iteration':10,
                        'population_size':80,
                        'mutation_probability':0.25,
                        'crossover_probability': 0.55,
                        'parents_portion': 0.15,
                        'elit_ratio': 0.01,
                        'max_iteration_without_improv':None}

    extractZLCParameters(modelType = modelType,
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
    

##############################################################################

# Adsorbent properties
# Adsorbent density [kg/m3]
# This has to be the skeletal density
# adsorbentDensity = 1020 # ZYH
# adsorbentDensity = 3200 # ZYNa
# adsorbentDensity = 2890 # ZYTMA
# adsorbentDensity = 2130 # ZYH MSB
# adsorbentDensity = 2410 # ZYNa MSB
adsorbentDensity = 2310 # ZYTMA MSB

# Particle porosity

# particleEpsilon = 0.90 # ZYH
# particleEpsilon = 0.63 # ZYH excl micro
# particleEpsilon = 0.76 # ZYNa
particleEpsilon = 0.43 # ZYTMA
# particleEpsilon = 0.61 # CMS 3K

# Particle mass [g]
# massSorbent = 0.088 # ZYH real
# massSorbent = 0.025 # ZYH real
massSorbent = 0.029 # ZYTMA real
massSorbent = 0.02 # ZYTMA real
# massSorbent = 0.028 # ZYTMA real
# massSorbent = 0.07 # ZYNa
# massSorbent = 0.065 # ZYTMA
# massSorbent = 0.077 # CMS 3K

rpore = 162e-9
Dpvals = [5.20463922908315e-05,	5.38448006625916e-05,	5.56327545887979e-05]
Dpvals = [4.15320342568427e-05,	4.28284424375970e-05,	4.41121704678921e-05] # DK

# Isotherm model (if fitting only kinetic constant)
isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# isothermDataFile = 'ZYH_DSL_QC_070523.mat'

modelType = 'Diffusion1Ttau'


# fileName  = ['ZLC_ZYTMACrush_Exp09A_Output.mat',
#             'ZLC_ZYTMACrush_Exp05A_Output.mat',
#             'ZLC_ZYTMACrush_Exp03A_Output.mat',
#             'ZLC_ZYTMACrush_Exp10A_Output.mat',
#             'ZLC_ZYTMACrush_Exp06A_Output.mat',
#             'ZLC_ZYTMACrush_Exp04A_Output.mat',
#             'ZLC_ZYTMACrush_Exp09B_Output.mat',
#             'ZLC_ZYTMACrush_Exp05B_Output.mat',
#             'ZLC_ZYTMACrush_Exp03B_Output.mat',
#             'ZLC_ZYTMACrush_Exp10B_Output.mat',
#             'ZLC_ZYTMACrush_Exp06B_Output.mat',
#             'ZLC_ZYTMACrush_Exp04B_Output.mat',]

fileName  = ['ZLC_ZYTMACrush_Exp09A_Output.mat',
            'ZLC_ZYTMACrush_Exp05A_Output.mat',
            'ZLC_ZYTMACrush_Exp03A_Output.mat',
            'ZLC_ZYTMACrush_Exp10A_Output.mat',
            'ZLC_ZYTMACrush_Exp06A_Output.mat',
            'ZLC_ZYTMACrush_Exp04A_Output.mat',]


deadVolumeFile = [[['deadVolumeCharacteristics_20231122_1743_b571c46.npz', 
                    'deadVolumeCharacteristics_20231122_1757_b571c46.npz']], 
                  [['deadVolumeCharacteristics_20231122_1750_b571c46.npz', 
                    'deadVolumeCharacteristics_20231122_1804_b571c46.npz']]] 

temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY
# temperature = [ 288.15,]*4 # ZY

print('fileName = '+str(fileName))
print('downsampleConc = '+str(downsampleData))
print('downsampleExp = '+str(downsampleExp))
print('massSorbent = '+str(massSorbent))
print('particleEpsilon = '+str(particleEpsilon))
print('adsorbentDensity = '+str(adsorbentDensity))
print('isothermDataFile = '+str(isothermDataFile))
print('deadVolumeFile = '+str(deadVolumeFile))
print('modelType = '+str(modelType))


for ii in range(5):
    algorithm_param = {'max_num_iteration':10,
                        'population_size':80,
                        'mutation_probability':0.25,
                        'crossover_probability': 0.55,
                        'parents_portion': 0.15,
                        'elit_ratio': 0.01,
                        'max_iteration_without_improv':None}
    
    # for ii in range(1):
    extractZLCParameters(modelType = modelType,
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

# Adsorbent properties
# Adsorbent density [kg/m3]
# This has to be the skeletal density
# adsorbentDensity = 1020 # ZYH
# adsorbentDensity = 3200 # ZYNa
# adsorbentDensity = 2890 # ZYTMA
adsorbentDensity = 2130 # ZYH MSB
# adsorbentDensity = 2410 # ZYNa MSB
# adsorbentDensity = 2310 # ZYTMA MSB

# Particle porosity

particleEpsilon = 0.90 # ZYH
particleEpsilon = 0.57 # ZYH excl micro
# particleEpsilon = 0.76 # ZYNa
# particleEpsilon = 0.71 # ZYTMA
# particleEpsilon = 0.61 # CMS 3K

# Particle mass [g]
# massSorbent = 0.088 # ZYH real
# massSorbent = 0.025 # ZYH real
massSorbent = 0.0245 # ZYH real
# massSorbent = 0.07 # ZYNa
# massSorbent = 0.065 # ZYTMA
# massSorbent = 0.077 # CMS 3K

rpore = 107e-9
Dpvals = [2.35952892668521e-05	,2.42488804831046e-05,	2.48936504671912e-05] # DK

# Isotherm model (if fitting only kinetic constant)
# isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
isothermDataFile = 'ZYH_DSL_QC_070523.mat'


# fileName  = ['ZLC_ZYHCrush_Exp05A_Output_new.mat',
#             'ZLC_ZYHCrush_Exp07A_Output_new.mat',
#             'ZLC_ZYHCrush_Exp09A_Output_new.mat',
#             'ZLC_ZYHCrush_Exp06A_Output_new.mat',
#             'ZLC_ZYHCrush_Exp08A_Output_new.mat',
#             'ZLC_ZYHCrush_Exp10A_Output_new.mat',
#             'ZLC_ZYHCrush_Exp05B_Output_new.mat',
#             'ZLC_ZYHCrush_Exp07B_Output_new.mat',
#             'ZLC_ZYHCrush_Exp09B_Output_new.mat',
#             'ZLC_ZYHCrush_Exp06B_Output_new.mat',
#             'ZLC_ZYHCrush_Exp08B_Output_new.mat',
#             'ZLC_ZYHCrush_Exp10B_Output_new.mat',]

fileName  = ['ZLC_ZYHCrush_Exp05A_Output_new.mat',
            'ZLC_ZYHCrush_Exp07A_Output_new.mat',
            'ZLC_ZYHCrush_Exp09A_Output_new.mat',
            'ZLC_ZYHCrush_Exp06A_Output_new.mat',
            'ZLC_ZYHCrush_Exp08A_Output_new.mat',
            'ZLC_ZYHCrush_Exp10A_Output_new.mat',]


deadVolumeFile = [[['deadVolumeCharacteristics_20231122_1743_b571c46.npz', 
                    'deadVolumeCharacteristics_20231122_1757_b571c46.npz']], 
                  [['deadVolumeCharacteristics_20231122_1750_b571c46.npz', 
                    'deadVolumeCharacteristics_20231122_1804_b571c46.npz']]] 

temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY

print('fileName = '+str(fileName))
print('downsampleConc = '+str(downsampleData))
print('downsampleExp = '+str(downsampleExp))
print('massSorbent = '+str(massSorbent))
print('particleEpsilon = '+str(particleEpsilon))
print('adsorbentDensity = '+str(adsorbentDensity))
print('isothermDataFile = '+str(isothermDataFile))
print('deadVolumeFile = '+str(deadVolumeFile))
print('modelType = '+str(modelType))

for ii in range(5):
    algorithm_param = {'max_num_iteration':10,
                        'population_size':80,
                        'mutation_probability':0.25,
                        'crossover_probability': 0.55,
                        'parents_portion': 0.15,
                        'elit_ratio': 0.01,
                        'max_iteration_without_improv':None}
    
    # for ii in range(1):
    extractZLCParameters(modelType = modelType,
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
