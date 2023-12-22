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

# Isotherm model (if fitting only kinetic constant)
# isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# isothermDataFile = 'ZYH_DSL_QC_070523.mat'

# modelType = 'KineticSBMacro'
modelType = 'Diffusion1T'


# fileName  = ['ZLC_ZYNa_Exp05A_Output.mat',
#             'ZLC_ZYNa_Exp03A_Output.mat',
#             'ZLC_ZYNa_Exp09A_Output.mat',
#             'ZLC_ZYNa_Exp05B_Output.mat',
#             'ZLC_ZYNa_Exp03B_Output.mat',
#             'ZLC_ZYNa_Exp09B_Output.mat',
#             'ZLC_ZYNa_Exp06A_Output.mat',
#             'ZLC_ZYNa_Exp04A_Output.mat',
#             'ZLC_ZYNa_Exp10A_Output.mat',
#             'ZLC_ZYNa_Exp06B_Output.mat',
#             'ZLC_ZYNa_Exp04B_Output.mat',
#             'ZLC_ZYNa_Exp10B_Output.mat',]  

fileName  = ['ZLC_ZYNaCrush_Exp05A_Output.mat',
            'ZLC_ZYNaCrush_Exp09A_Output.mat',
            'ZLC_ZYNaCrush_Exp11A_Output.mat',
            'ZLC_ZYNaCrush_Exp06A_Output.mat',
            'ZLC_ZYNaCrush_Exp10A_Output.mat',
            'ZLC_ZYNaCrush_Exp12A_Output.mat',
            'ZLC_ZYNaCrush_Exp05B_Output.mat',
            'ZLC_ZYNaCrush_Exp09B_Output.mat',
            'ZLC_ZYNaCrush_Exp11B_Output.mat',
            'ZLC_ZYNaCrush_Exp06B_Output.mat',
            'ZLC_ZYNaCrush_Exp10B_Output.mat',
            'ZLC_ZYNaCrush_Exp12B_Output.mat',] 

fileName  = ['ZLC_ZYNaCrush_Exp05B_Output.mat',
            'ZLC_ZYNaCrush_Exp09B_Output.mat',
            'ZLC_ZYNaCrush_Exp11B_Output.mat',
            'ZLC_ZYNaCrush_Exp06B_Output.mat',
            'ZLC_ZYNaCrush_Exp10B_Output.mat',
            'ZLC_ZYNaCrush_Exp12B_Output.mat',] 

fileName  = ['ZLC_ZYNaCrush_Exp05A_Output.mat',
            'ZLC_ZYNaCrush_Exp09A_Output.mat',
            'ZLC_ZYNaCrush_Exp11A_Output.mat',
            'ZLC_ZYNaCrush_Exp06A_Output.mat',
            'ZLC_ZYNaCrush_Exp10A_Output.mat',
            'ZLC_ZYNaCrush_Exp12A_Output.mat',] 

# fileName  = ['ZLC_ZYNaCrush_Exp06A_Output.mat',
#             'ZLC_ZYNaCrush_Exp10A_Output.mat',
#             'ZLC_ZYNaCrush_Exp12A_Output.mat',] 
# fileName  = ['ZLC_ZYNa_Exp05B_Output.mat',
#             'ZLC_ZYNa_Exp06B_Output.mat',]  

deadVolumeFile = [[['deadVolumeCharacteristics_20231122_1743_b571c46.npz', 
                    'deadVolumeCharacteristics_20231122_1757_b571c46.npz']], 
                  [['deadVolumeCharacteristics_20231122_1750_b571c46.npz', 
                    'deadVolumeCharacteristics_20231122_1804_b571c46.npz']]] 

temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY
# temperature = [ 288.15]*4 # ZY

print('fileName = '+str(fileName))
print('downsampleConc = '+str(downsampleData))
print('downsampleExp = '+str(downsampleExp))
print('massSorbent = '+str(massSorbent))
print('particleEpsilon = '+str(particleEpsilon))
print('adsorbentDensity = '+str(adsorbentDensity))
print('isothermDataFile = '+str(isothermDataFile))
print('deadVolumeFile = '+str(deadVolumeFile))
print('modelType = '+str(modelType))


# for ii in range(5):
#     algorithm_param = {'max_num_iteration':30,
#                         'population_size':400,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}
for ii in range(2):
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

# Isotherm model (if fitting only kinetic constant)
isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# isothermDataFile = 'ZYH_DSL_QC_070523.mat'

# modelType = 'KineticSBMacro'
modelType = 'Diffusion1T'


fileName  = ['ZLC_ZYTMACrush_Exp09B_Output.mat',
            'ZLC_ZYTMACrush_Exp05B_Output.mat',
            'ZLC_ZYTMACrush_Exp03B_Output.mat',
            'ZLC_ZYTMACrush_Exp10B_Output.mat',
            'ZLC_ZYTMACrush_Exp06B_Output.mat',
            'ZLC_ZYTMACrush_Exp04B_Output.mat',]

fileName  = ['ZLC_ZYTMACrush_Exp09A_Output.mat',
            'ZLC_ZYTMACrush_Exp05A_Output.mat',
            'ZLC_ZYTMACrush_Exp03A_Output.mat',
            'ZLC_ZYTMACrush_Exp10A_Output.mat',
            'ZLC_ZYTMACrush_Exp06A_Output.mat',
            'ZLC_ZYTMACrush_Exp04A_Output.mat',]

# fileName  = ['ZLC_ZYTMACrush_Exp10A_Output.mat',
#             'ZLC_ZYTMACrush_Exp06A_Output.mat',
#             'ZLC_ZYTMACrush_Exp04A_Output.mat',]

# fileName  = ['ZLC_ZYHCrush_Exp06B_Output.mat',
#             'ZLC_ZYHCrush_Exp08B_Output.mat',
#             'ZLC_ZYHCrush_Exp10B_Output.mat',] 
# fileName  = ['ZLC_ZYH_Exp09B_Output.mat',
#             'ZLC_ZYH_Exp10B_Output.mat',]


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


# for ii in range(5):
#     algorithm_param = {'max_num_iteration':30,
#                         'population_size':400,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}
for ii in range(2):
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
massSorbent = 0.026 # ZYH real
# massSorbent = 0.07 # ZYNa
# massSorbent = 0.065 # ZYTMA
# massSorbent = 0.077 # CMS 3K

# Isotherm model (if fitting only kinetic constant)
# isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
isothermDataFile = 'ZYH_DSL_QC_070523.mat'

# modelType = 'KineticSBMacro'
modelType = 'Diffusion1T'


# fileName  = ['ZLC_ZYH_Exp09A_Output.mat',
#             'ZLC_ZYH_Exp11A_Output.mat',
#             'ZLC_ZYH_Exp13A_Output.mat',
#             'ZLC_ZYH_Exp09B_Output.mat',
#             'ZLC_ZYH_Exp11B_Output.mat',
#             'ZLC_ZYH_Exp13B_Output.mat',
#             'ZLC_ZYH_Exp10A_Output.mat',
#             'ZLC_ZYH_Exp12A_Output.mat',
#             'ZLC_ZYH_Exp14A_Output.mat',
#             'ZLC_ZYH_Exp10B_Output.mat',
#             'ZLC_ZYH_Exp12B_Output.mat',
#             'ZLC_ZYH_Exp14B_Output.mat',]

fileName  = ['ZLC_ZYHCrush_Exp05A_Output.mat',
            'ZLC_ZYHCrush_Exp07A_Output.mat',
            'ZLC_ZYHCrush_Exp09A_Output.mat',
            'ZLC_ZYHCrush_Exp06A_Output.mat',
            'ZLC_ZYHCrush_Exp08A_Output.mat',
            'ZLC_ZYHCrush_Exp10A_Output.mat',
            'ZLC_ZYHCrush_Exp05B_Output.mat',
            'ZLC_ZYHCrush_Exp07B_Output.mat',
            'ZLC_ZYHCrush_Exp09B_Output.mat',
            'ZLC_ZYHCrush_Exp06B_Output.mat',
            'ZLC_ZYHCrush_Exp08B_Output.mat',
            'ZLC_ZYHCrush_Exp10B_Output.mat',] 

fileName  = ['ZLC_ZYHCrush_Exp05B_Output.mat',
            'ZLC_ZYHCrush_Exp07B_Output.mat',
            'ZLC_ZYHCrush_Exp09B_Output.mat',
            'ZLC_ZYHCrush_Exp06B_Output.mat',
            'ZLC_ZYHCrush_Exp08B_Output.mat',
            'ZLC_ZYHCrush_Exp10B_Output.mat',] 

fileName  = ['ZLC_ZYHCrush_Exp05A_Output.mat',
            'ZLC_ZYHCrush_Exp07A_Output.mat',
            'ZLC_ZYHCrush_Exp09A_Output.mat',
            'ZLC_ZYHCrush_Exp06A_Output.mat',
            'ZLC_ZYHCrush_Exp08A_Output.mat',
            'ZLC_ZYHCrush_Exp10A_Output2.mat',]

fileName  = ['ZLC_ZYHCrush_Exp05A_Output_new.mat',
            'ZLC_ZYHCrush_Exp07A_Output_new.mat',
            'ZLC_ZYHCrush_Exp09A_Output_new.mat',
            'ZLC_ZYHCrush_Exp06A_Output_new.mat',
            'ZLC_ZYHCrush_Exp08A_Output_new.mat',
            'ZLC_ZYHCrush_Exp10A_Output_new.mat',]

# fileName  = ['ZLC_ZYHCrush_Exp06A_Output_new.mat',
#             'ZLC_ZYHCrush_Exp08A_Output_new.mat',
#             'ZLC_ZYHCrush_Exp10A_Output_new.mat',]

# fileName  = ['ZLC_ZYHCrush_Exp06A_Output.mat',
#             'ZLC_ZYHCrush_Exp08A_Output.mat',
#             'ZLC_ZYHCrush_Exp10A_Output.mat',] 

# fileName  = ['ZLC_ZYHCrush_Exp06B_Output.mat',
#             'ZLC_ZYHCrush_Exp08B_Output.mat',
#             'ZLC_ZYHCrush_Exp10B_Output.mat',] 
# fileName  = ['ZLC_ZYH_Exp09B_Output.mat',
#             'ZLC_ZYH_Exp10B_Output.mat',]


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


# for ii in range(5):
#     algorithm_param = {'max_num_iteration':30,
#                         'population_size':400,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}
for ii in range(2):
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
                  massSorbent = massSorbent,
                  deadVolumeFile = deadVolumeFile,
                  isothermDataFile = isothermDataFile,
                  downsampleData = downsampleData,
                  downsampleExp = downsampleExp)


# ##############################################################################

# # Adsorbent properties
# # Adsorbent density [kg/m3]
# # This has to be the skeletal density
# # adsorbentDensity = 1020 # ZYH
# # adsorbentDensity = 3200 # ZYNa
# # adsorbentDensity = 2890 # ZYTMA
# # adsorbentDensity = 2130 # ZYH MSB
# adsorbentDensity = 2410 # ZYNa MSB
# # adsorbentDensity = 2310 # ZYTMA MSB

# # Particle porosity

# # particleEpsilon = 0.90 # ZYH
# particleEpsilon = 0.76 # ZYNa
# particleEpsilon = 0.49 # ZYNa excl micro
# # particleEpsilon = 0.71 # ZYTMA
# # particleEpsilon = 0.61 # CMS 3K

# # Particle mass [g]
# # massSorbent = 0.086 # ZYH real
# massSorbent = 0.062 # ZYNa
# # massSorbent = 0.065 # ZYTMA
# # massSorbent = 0.077 # CMS 3K

# # Isotherm model (if fitting only kinetic constant)
# # isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# # isothermDataFile = 'ZYH_DSL_QC_070523.mat'

# # modelType = 'KineticSBMacro'
# modelType = 'Diffusion'


# # fileName  = ['ZLC_ZYNa_Exp05A_Output.mat',
# #             'ZLC_ZYNa_Exp03A_Output.mat',
# #             'ZLC_ZYNa_Exp09A_Output.mat',
# #             'ZLC_ZYNa_Exp05B_Output.mat',
# #             'ZLC_ZYNa_Exp03B_Output.mat',
# #             'ZLC_ZYNa_Exp09B_Output.mat',
# #             'ZLC_ZYNa_Exp06A_Output.mat',
# #             'ZLC_ZYNa_Exp04A_Output.mat',
# #             'ZLC_ZYNa_Exp10A_Output.mat',
# #             'ZLC_ZYNa_Exp06B_Output.mat',
# #             'ZLC_ZYNa_Exp04B_Output.mat',
# #             'ZLC_ZYNa_Exp10B_Output.mat',]  

# fileName  = ['ZLC_ZYNa_Exp05B_Output.mat',
#             'ZLC_ZYNa_Exp03B_Output.mat',
#             'ZLC_ZYNa_Exp09B_Output.mat',
#             'ZLC_ZYNa_Exp06B_Output.mat',
#             'ZLC_ZYNa_Exp04B_Output.mat',
#             'ZLC_ZYNa_Exp10B_Output.mat',]  

# # fileName  = ['ZLC_ZYNa_Exp05B_Output.mat',
# #             'ZLC_ZYNa_Exp06B_Output.mat',]  

# deadVolumeFile =[[['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']], 
#                   [['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']]] 

# temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY
# # temperature = [ 288.15]*4 # ZY

# print('fileName = '+str(fileName))
# print('downsampleConc = '+str(downsampleData))
# print('downsampleExp = '+str(downsampleExp))
# print('massSorbent = '+str(massSorbent))
# print('particleEpsilon = '+str(particleEpsilon))
# print('adsorbentDensity = '+str(adsorbentDensity))
# print('isothermDataFile = '+str(isothermDataFile))
# print('deadVolumeFile = '+str(deadVolumeFile))
# print('modelType = '+str(modelType))


# # for ii in range(5):
# #     algorithm_param = {'max_num_iteration':30,
# #                         'population_size':400,
# #                         'mutation_probability':0.25,
# #                         'crossover_probability': 0.55,
# #                         'parents_portion': 0.15,
# #                         'elit_ratio': 0.01,
# #                         'max_iteration_without_improv':None}
# for ii in range(2):
#     algorithm_param = {'max_num_iteration':10,
#                         'population_size':80,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}

#     extractZLCParameters(modelType = modelType,
#                   fileName = fileName,
#                   temperature = temperature,
#                   algorithm_param = algorithm_param,
#                   adsorbentDensity = adsorbentDensity,
#                   particleEpsilon = particleEpsilon,
#                   massSorbent = massSorbent,
#                   deadVolumeFile = deadVolumeFile,
#                   isothermDataFile = isothermDataFile,
#                   downsampleData = downsampleData,
#                   downsampleExp = downsampleExp)
    
# # ##############################################################################

# # Adsorbent properties
# # Adsorbent density [kg/m3]
# # This has to be the skeletal density
# # adsorbentDensity = 1020 # ZYH
# # adsorbentDensity = 3200 # ZYNa
# # adsorbentDensity = 2890 # ZYTMA
# adsorbentDensity = 2130 # ZYH MSB
# # adsorbentDensity = 2410 # ZYNa MSB
# # adsorbentDensity = 2310 # ZYTMA MSB

# # Particle porosity

# particleEpsilon = 0.90 # ZYH
# particleEpsilon = 0.63 # ZYH excl micro
# # particleEpsilon = 0.76 # ZYNa
# # particleEpsilon = 0.71 # ZYTMA
# # particleEpsilon = 0.61 # CMS 3K

# # Particle mass [g]
# massSorbent = 0.088 # ZYH real
# # massSorbent = 0.07 # ZYNa
# # massSorbent = 0.065 # ZYTMA
# # massSorbent = 0.077 # CMS 3K

# # Isotherm model (if fitting only kinetic constant)
# # isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# # isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# isothermDataFile = 'ZYH_DSL_QC_070523.mat'

# # modelType = 'KineticSBMacro'
# modelType = 'Diffusion'


# # fileName  = ['ZLC_ZYH_Exp09A_Output.mat',
# #             'ZLC_ZYH_Exp11A_Output.mat',
# #             'ZLC_ZYH_Exp13A_Output.mat',
# #             'ZLC_ZYH_Exp09B_Output.mat',
# #             'ZLC_ZYH_Exp11B_Output.mat',
# #             'ZLC_ZYH_Exp13B_Output.mat',
# #             'ZLC_ZYH_Exp10A_Output.mat',
# #             'ZLC_ZYH_Exp12A_Output.mat',
# #             'ZLC_ZYH_Exp14A_Output.mat',
# #             'ZLC_ZYH_Exp10B_Output.mat',
# #             'ZLC_ZYH_Exp12B_Output.mat',
# #             'ZLC_ZYH_Exp14B_Output.mat',]

# fileName  = ['ZLC_ZYH_Exp09B_Output2.mat',
#             'ZLC_ZYH_Exp11B_Output2.mat',
#             'ZLC_ZYH_Exp13B_Output2.mat',
#             'ZLC_ZYH_Exp10B_Output2.mat',
#             'ZLC_ZYH_Exp12B_Output2.mat',
#             'ZLC_ZYH_Exp14B_Output2.mat',]

# # fileName  = ['ZLC_ZYH_Exp09B_Output.mat',
# #             'ZLC_ZYH_Exp10B_Output.mat',]


# deadVolumeFile =[[['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']], 
#                   [['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']]] 

# temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY
# # temperature = [ 288.15,]*4 # ZY

# print('fileName = '+str(fileName))
# print('downsampleConc = '+str(downsampleData))
# print('downsampleExp = '+str(downsampleExp))
# print('massSorbent = '+str(massSorbent))
# print('particleEpsilon = '+str(particleEpsilon))
# print('adsorbentDensity = '+str(adsorbentDensity))
# print('isothermDataFile = '+str(isothermDataFile))
# print('deadVolumeFile = '+str(deadVolumeFile))
# print('modelType = '+str(modelType))


# # for ii in range(5):
# #     algorithm_param = {'max_num_iteration':30,
# #                         'population_size':400,
# #                         'mutation_probability':0.25,
# #                         'crossover_probability': 0.55,
# #                         'parents_portion': 0.15,
# #                         'elit_ratio': 0.01,
# #                         'max_iteration_without_improv':None}
# for ii in range(1):
#     algorithm_param = {'max_num_iteration':10,
#                         'population_size':80,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}
    
#     # for ii in range(1):
#     extractZLCParameters(modelType = modelType,
#                   fileName = fileName,
#                   temperature = temperature,
#                   algorithm_param = algorithm_param,
#                   adsorbentDensity = adsorbentDensity,
#                   particleEpsilon = particleEpsilon,
#                   massSorbent = massSorbent,
#                   deadVolumeFile = deadVolumeFile,
#                   isothermDataFile = isothermDataFile,
#                   downsampleData = downsampleData,
#                   downsampleExp = downsampleExp)

   
# # # #############################################################################

# # Adsorbent properties
# # Adsorbent density [kg/m3]
# # This has to be the skeletal density
# # adsorbentDensity = 1020 # ZYH
# # adsorbentDensity = 3200 # ZYNa
# # adsorbentDensity = 2890 # ZYTMA
# # adsorbentDensity = 2130 # ZYH MSB
# adsorbentDensity = 2410 # ZYNa MSB
# # adsorbentDensity = 2310 # ZYTMA MSB

# # Particle porosity

# # particleEpsilon = 0.90 # ZYH
# particleEpsilon = 0.76 # ZYNa
# particleEpsilon = 0.49 # ZYNa excl micro
# # particleEpsilon = 0.71 # ZYTMA
# # particleEpsilon = 0.61 # CMS 3K

# # Particle mass [g]
# # massSorbent = 0.086 # ZYH real
# massSorbent = 0.068 # ZYNa
# # massSorbent = 0.065 # ZYTMA
# # massSorbent = 0.077 # CMS 3K

# # Isotherm model (if fitting only kinetic constant)
# # isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# # isothermDataFile = 'ZYH_DSL_QC_070523.mat'

# # modelType = 'KineticSBMacro'
# modelType = 'Diffusion1T'


# # fileName  = ['ZLC_ZYNa_Exp05A_Output.mat',
# #             'ZLC_ZYNa_Exp03A_Output.mat',
# #             'ZLC_ZYNa_Exp09A_Output.mat',
# #             'ZLC_ZYNa_Exp05B_Output.mat',
# #             'ZLC_ZYNa_Exp03B_Output.mat',
# #             'ZLC_ZYNa_Exp09B_Output.mat',
# #             'ZLC_ZYNa_Exp06A_Output.mat',
# #             'ZLC_ZYNa_Exp04A_Output.mat',
# #             'ZLC_ZYNa_Exp10A_Output.mat',
# #             'ZLC_ZYNa_Exp06B_Output.mat',
# #             'ZLC_ZYNa_Exp04B_Output.mat',
# #             'ZLC_ZYNa_Exp10B_Output.mat',]  

# # fileName  = ['ZLC_ZYNa_Exp05B_Output.mat',
# #             'ZLC_ZYNa_Exp03B_Output.mat',
# #             'ZLC_ZYNa_Exp09B_Output.mat',
# #             'ZLC_ZYNa_Exp06B_Output.mat',
# #             'ZLC_ZYNa_Exp04B_Output.mat',
# #             'ZLC_ZYNa_Exp10B_Output.mat',]  

# fileName  = ['ZLC_ZYNa_Exp03B_Output.mat',
#              'ZLC_ZYNa_Exp04B_Output.mat',]  

# deadVolumeFile =[[['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']], 
#                   [['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']]] 

# # temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY
# temperature = [ 298.15]*4 # ZY

# print('fileName = '+str(fileName))
# print('downsampleConc = '+str(downsampleData))
# print('downsampleExp = '+str(downsampleExp))
# print('massSorbent = '+str(massSorbent))
# print('particleEpsilon = '+str(particleEpsilon))
# print('adsorbentDensity = '+str(adsorbentDensity))
# print('isothermDataFile = '+str(isothermDataFile))
# print('deadVolumeFile = '+str(deadVolumeFile))
# print('modelType = '+str(modelType))


# # for ii in range(5):
# #     algorithm_param = {'max_num_iteration':30,
# #                         'population_size':400,
# #                         'mutation_probability':0.25,
# #                         'crossover_probability': 0.55,
# #                         'parents_portion': 0.15,
# #                         'elit_ratio': 0.01,
# #                         'max_iteration_without_improv':None}
# for ii in range(1):
#     algorithm_param = {'max_num_iteration':10,
#                         'population_size':80,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}

#     extractZLCParameters(modelType = modelType,
#                   fileName = fileName,
#                   temperature = temperature,
#                   algorithm_param = algorithm_param,
#                   adsorbentDensity = adsorbentDensity,
#                   particleEpsilon = particleEpsilon,
#                   massSorbent = massSorbent,
#                   deadVolumeFile = deadVolumeFile,
#                   isothermDataFile = isothermDataFile,
#                   downsampleData = downsampleData,
#                   downsampleExp = downsampleExp)

# # # ##############################################################################

# # Adsorbent properties
# # Adsorbent density [kg/m3]
# # This has to be the skeletal density
# # adsorbentDensity = 1020 # ZYH
# # adsorbentDensity = 3200 # ZYNa
# # adsorbentDensity = 2890 # ZYTMA
# # adsorbentDensity = 2130 # ZYH MSB
# adsorbentDensity = 2410 # ZYNa MSB
# # adsorbentDensity = 2310 # ZYTMA MSB

# # Particle porosity

# # particleEpsilon = 0.90 # ZYH
# particleEpsilon = 0.76 # ZYNa
# particleEpsilon = 0.49 # ZYNa excl micro
# # particleEpsilon = 0.71 # ZYTMA
# # particleEpsilon = 0.61 # CMS 3K

# # Particle mass [g]
# # massSorbent = 0.086 # ZYH real
# massSorbent = 0.068 # ZYNa
# # massSorbent = 0.065 # ZYTMA
# # massSorbent = 0.077 # CMS 3K

# # Isotherm model (if fitting only kinetic constant)
# # isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# # isothermDataFile = 'ZYH_DSL_QC_070523.mat'

# # modelType = 'KineticSBMacro'
# modelType = 'Diffusion1T'


# # fileName  = ['ZLC_ZYNa_Exp05A_Output.mat',
# #             'ZLC_ZYNa_Exp03A_Output.mat',
# #             'ZLC_ZYNa_Exp09A_Output.mat',
# #             'ZLC_ZYNa_Exp05B_Output.mat',
# #             'ZLC_ZYNa_Exp03B_Output.mat',
# #             'ZLC_ZYNa_Exp09B_Output.mat',
# #             'ZLC_ZYNa_Exp06A_Output.mat',
# #             'ZLC_ZYNa_Exp04A_Output.mat',
# #             'ZLC_ZYNa_Exp10A_Output.mat',
# #             'ZLC_ZYNa_Exp06B_Output.mat',
# #             'ZLC_ZYNa_Exp04B_Output.mat',
# #             'ZLC_ZYNa_Exp10B_Output.mat',]  

# # fileName  = ['ZLC_ZYNa_Exp05B_Output.mat',
# #             'ZLC_ZYNa_Exp03B_Output.mat',
# #             'ZLC_ZYNa_Exp09B_Output.mat',
# #             'ZLC_ZYNa_Exp06B_Output.mat',
# #             'ZLC_ZYNa_Exp04B_Output.mat',
# #             'ZLC_ZYNa_Exp10B_Output.mat',]  

# fileName  = ['ZLC_ZYNa_Exp09B_Output.mat',
#              'ZLC_ZYNa_Exp10B_Output.mat',]  

# deadVolumeFile =[[['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']], 
#                   [['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']]] 

# # temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY
# temperature = [ 308.15]*4 # ZY

# print('fileName = '+str(fileName))
# print('downsampleConc = '+str(downsampleData))
# print('downsampleExp = '+str(downsampleExp))
# print('massSorbent = '+str(massSorbent))
# print('particleEpsilon = '+str(particleEpsilon))
# print('adsorbentDensity = '+str(adsorbentDensity))
# print('isothermDataFile = '+str(isothermDataFile))
# print('deadVolumeFile = '+str(deadVolumeFile))
# print('modelType = '+str(modelType))


# # for ii in range(5):
# #     algorithm_param = {'max_num_iteration':30,
# #                         'population_size':400,
# #                         'mutation_probability':0.25,
# #                         'crossover_probability': 0.55,
# #                         'parents_portion': 0.15,
# #                         'elit_ratio': 0.01,
# #                         'max_iteration_without_improv':None}
# for ii in range(1):
#     algorithm_param = {'max_num_iteration':10,
#                         'population_size':80,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}

#     extractZLCParameters(modelType = modelType,
#                   fileName = fileName,
#                   temperature = temperature,
#                   algorithm_param = algorithm_param,
#                   adsorbentDensity = adsorbentDensity,
#                   particleEpsilon = particleEpsilon,
#                   massSorbent = massSorbent,
#                   deadVolumeFile = deadVolumeFile,
#                   isothermDataFile = isothermDataFile,
#                   downsampleData = downsampleData,
#                   downsampleExp = downsampleExp)

# # ##############################################################################

# # Adsorbent properties
# # Adsorbent density [kg/m3]
# # This has to be the skeletal density
# # adsorbentDensity = 1020 # ZYH
# # adsorbentDensity = 3200 # ZYNa
# # adsorbentDensity = 2890 # ZYTMA
# # adsorbentDensity = 2130 # ZYH MSB
# # adsorbentDensity = 2410 # ZYNa MSB
# adsorbentDensity = 2310 # ZYTMA MSB

# # Particle porosity

# # particleEpsilon = 0.90 # ZYH
# # particleEpsilon = 0.76 # ZYNa
# particleEpsilon = 0.71 # ZYTMA
# particleEpsilon = 0.43 # ZYTMA excl micro
# # particleEpsilon = 0.61 # CMS 3K

# # Particle mass [g]
# # massSorbent = 0.086 # ZYH real
# # massSorbent = 0.075 # ZYNa
# massSorbent = 0.065 # ZYTMA
# # massSorbent = 0.077 # CMS 3K

# # Isotherm model (if fitting only kinetic constant)
# isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# # isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# # isothermDataFile = 'ZYH_DSL_QC_070523.mat'

# # modelType = 'KineticSBMacro'
# modelType = 'Diffusion'


# # fileName  = ['ZLC_ZYTMA_Exp01A_Output.mat',
#             # 'ZLC_ZYTMA_Exp03A_Output.mat',
#             # 'ZLC_ZYTMA_Exp05A_Output.mat',
#             # 'ZLC_ZYTMA_Exp01B_Output.mat',
#             # 'ZLC_ZYTMA_Exp03B_Output.mat',
#             # 'ZLC_ZYTMA_Exp05B_Output.mat',
#             # 'ZLC_ZYTMA_Exp02A_Output.mat',
#             # 'ZLC_ZYTMA_Exp04A_Output.mat',
#             # 'ZLC_ZYTMA_Exp06A_Output.mat',
#             # 'ZLC_ZYTMA_Exp02B_Output.mat',
#             # 'ZLC_ZYTMA_Exp04B_Output.mat',
#             # 'ZLC_ZYTMA_Exp06B_Output.mat',] 

# fileName  = ['ZLC_ZYTMA_Exp01B_Output.mat',
#             'ZLC_ZYTMA_Exp03B_Output.mat',
#             'ZLC_ZYTMA_Exp05B_Output.mat',
#             'ZLC_ZYTMA_Exp02B_Output.mat',
#             'ZLC_ZYTMA_Exp04B_Output.mat',
#             'ZLC_ZYTMA_Exp06B_Output.mat',] 

# # fileName  = ['ZLC_ZYTMA_Exp01B_Output.mat',
# #             'ZLC_ZYTMA_Exp02B_Output.mat',] 


# deadVolumeFile =[[['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']], 
#                   [['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']]] 

# temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY
# # temperature = [ 288.15, ]*4 # ZY


# print('fileName = '+str(fileName))
# print('downsampleConc = '+str(downsampleData))
# print('downsampleExp = '+str(downsampleExp))
# print('massSorbent = '+str(massSorbent))
# print('particleEpsilon = '+str(particleEpsilon))
# print('adsorbentDensity = '+str(adsorbentDensity))
# print('isothermDataFile = '+str(isothermDataFile))
# print('deadVolumeFile = '+str(deadVolumeFile))
# print('modelType = '+str(modelType))


# # for ii in range(5):
# #     algorithm_param = {'max_num_iteration':30,
# #                         'population_size':400,
# #                         'mutation_probability':0.25,
# #                         'crossover_probability': 0.55,
# #                         'parents_portion': 0.15,
# #                         'elit_ratio': 0.01,
# #                         'max_iteration_without_improv':None}
# for ii in range(3):
#     algorithm_param = {'max_num_iteration':10,
#                         'population_size':80,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}
    
#     # for ii in range(1):
#     extractZLCParameters(modelType = modelType,
#                   fileName = fileName,
#                   temperature = temperature,
#                   algorithm_param = algorithm_param,
#                   adsorbentDensity = adsorbentDensity,
#                   particleEpsilon = particleEpsilon,
#                   massSorbent = massSorbent,
#                   deadVolumeFile = deadVolumeFile,
#                   isothermDataFile = isothermDataFile,
#                   downsampleData = downsampleData,
#                   downsampleExp = downsampleExp)


# #############################################################################

# # Adsorbent properties
# # Adsorbent density [kg/m3]
# # This has to be the skeletal density
# # adsorbentDensity = 1020 # ZYH
# # adsorbentDensity = 3200 # ZYNa
# # adsorbentDensity = 2890 # ZYTMA
# # adsorbentDensity = 2130 # ZYH MSB
# # adsorbentDensity = 2410 # ZYNa MSB
# adsorbentDensity = 2310 # ZYTMA MSB

# # Particle porosity

# # particleEpsilon = 0.90 # ZYH
# # particleEpsilon = 0.76 # ZYNa
# particleEpsilon = 0.71 # ZYTMA
# particleEpsilon = 0.43 # ZYTMA excl micro
# # particleEpsilon = 0.61 # CMS 3K

# # Particle mass [g]
# # massSorbent = 0.086 # ZYH real
# # massSorbent = 0.075 # ZYNa
# massSorbent = 0.065 # ZYTMA
# # massSorbent = 0.077 # CMS 3K

# # Isotherm model (if fitting only kinetic constant)
# isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# # isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# # isothermDataFile = 'ZYH_DSL_QC_070523.mat'

# # modelType = 'KineticSBMacro'
# modelType = 'Diffusion1T'


# # fileName  = ['ZLC_ZYTMA_Exp01A_Output.mat',
#             # 'ZLC_ZYTMA_Exp03A_Output.mat',
#             # 'ZLC_ZYTMA_Exp05A_Output.mat',
#             # 'ZLC_ZYTMA_Exp01B_Output.mat',
#             # 'ZLC_ZYTMA_Exp03B_Output.mat',
#             # 'ZLC_ZYTMA_Exp05B_Output.mat',
#             # 'ZLC_ZYTMA_Exp02A_Output.mat',
#             # 'ZLC_ZYTMA_Exp04A_Output.mat',
#             # 'ZLC_ZYTMA_Exp06A_Output.mat',
#             # 'ZLC_ZYTMA_Exp02B_Output.mat',
#             # 'ZLC_ZYTMA_Exp04B_Output.mat',
#             # 'ZLC_ZYTMA_Exp06B_Output.mat',] 

# # fileName  = ['ZLC_ZYTMA_Exp01B_Output.mat',
# #             'ZLC_ZYTMA_Exp03B_Output.mat',
# #             'ZLC_ZYTMA_Exp05B_Output.mat',
# #             'ZLC_ZYTMA_Exp02B_Output.mat',
# #             'ZLC_ZYTMA_Exp04B_Output.mat',
# #             'ZLC_ZYTMA_Exp06B_Output.mat',] 

# fileName  = ['ZLC_ZYTMA_Exp03B_Output.mat',
#              'ZLC_ZYTMA_Exp04B_Output.mat',] 


# deadVolumeFile =[[['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']], 
#                   [['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']]] 

# # temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY
# temperature = [ 298.15, ]*4 # ZY


# print('fileName = '+str(fileName))
# print('downsampleConc = '+str(downsampleData))
# print('downsampleExp = '+str(downsampleExp))
# print('massSorbent = '+str(massSorbent))
# print('particleEpsilon = '+str(particleEpsilon))
# print('adsorbentDensity = '+str(adsorbentDensity))
# print('isothermDataFile = '+str(isothermDataFile))
# print('deadVolumeFile = '+str(deadVolumeFile))
# print('modelType = '+str(modelType))


# # for ii in range(5):
# #     algorithm_param = {'max_num_iteration':30,
# #                         'population_size':400,
# #                         'mutation_probability':0.25,
# #                         'crossover_probability': 0.55,
# #                         'parents_portion': 0.15,
# #                         'elit_ratio': 0.01,
# #                         'max_iteration_without_improv':None}
# for ii in range(1):
#     algorithm_param = {'max_num_iteration':10,
#                         'population_size':80,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}
    
#     # for ii in range(1):
#     extractZLCParameters(modelType = modelType,
#                   fileName = fileName,
#                   temperature = temperature,
#                   algorithm_param = algorithm_param,
#                   adsorbentDensity = adsorbentDensity,
#                   particleEpsilon = particleEpsilon,
#                   massSorbent = massSorbent,
#                   deadVolumeFile = deadVolumeFile,
#                   isothermDataFile = isothermDataFile,
#                   downsampleData = downsampleData,
#                   downsampleExp = downsampleExp)

# ##############################################################################

# # Adsorbent properties
# # Adsorbent density [kg/m3]
# # This has to be the skeletal density
# # adsorbentDensity = 1020 # ZYH
# # adsorbentDensity = 3200 # ZYNa
# # adsorbentDensity = 2890 # ZYTMA
# # adsorbentDensity = 2130 # ZYH MSB
# # adsorbentDensity = 2410 # ZYNa MSB
# adsorbentDensity = 2310 # ZYTMA MSB

# # Particle porosity

# # particleEpsilon = 0.90 # ZYH
# # particleEpsilon = 0.76 # ZYNa
# particleEpsilon = 0.71 # ZYTMA
# particleEpsilon = 0.43 # ZYTMA excl micro
# # particleEpsilon = 0.61 # CMS 3K

# # Particle mass [g]
# # massSorbent = 0.086 # ZYH real
# # massSorbent = 0.075 # ZYNa
# massSorbent = 0.065 # ZYTMA
# # massSorbent = 0.077 # CMS 3K

# # Isotherm model (if fitting only kinetic constant)
# isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# # isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# # isothermDataFile = 'ZYH_DSL_QC_070523.mat'

# # modelType = 'KineticSBMacro'
# modelType = 'Diffusion1T'


# # fileName  = ['ZLC_ZYTMA_Exp01A_Output.mat',
#             # 'ZLC_ZYTMA_Exp03A_Output.mat',
#             # 'ZLC_ZYTMA_Exp05A_Output.mat',
#             # 'ZLC_ZYTMA_Exp01B_Output.mat',
#             # 'ZLC_ZYTMA_Exp03B_Output.mat',
#             # 'ZLC_ZYTMA_Exp05B_Output.mat',
#             # 'ZLC_ZYTMA_Exp02A_Output.mat',
#             # 'ZLC_ZYTMA_Exp04A_Output.mat',
#             # 'ZLC_ZYTMA_Exp06A_Output.mat',
#             # 'ZLC_ZYTMA_Exp02B_Output.mat',
#             # 'ZLC_ZYTMA_Exp04B_Output.mat',
#             # 'ZLC_ZYTMA_Exp06B_Output.mat',] 

# # fileName  = ['ZLC_ZYTMA_Exp01B_Output.mat',
# #             'ZLC_ZYTMA_Exp03B_Output.mat',
# #             'ZLC_ZYTMA_Exp05B_Output.mat',
# #             'ZLC_ZYTMA_Exp02B_Output.mat',
# #             'ZLC_ZYTMA_Exp04B_Output.mat',
# #             'ZLC_ZYTMA_Exp06B_Output.mat',] 

# fileName  = ['ZLC_ZYTMA_Exp05B_Output.mat',
#              'ZLC_ZYTMA_Exp06B_Output.mat',] 


# deadVolumeFile =[[['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']], 
#                   [['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']]] 

# temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY
# temperature = [ 308.15, ]*4 # ZY


# print('fileName = '+str(fileName))
# print('downsampleConc = '+str(downsampleData))
# print('downsampleExp = '+str(downsampleExp))
# print('massSorbent = '+str(massSorbent))
# print('particleEpsilon = '+str(particleEpsilon))
# print('adsorbentDensity = '+str(adsorbentDensity))
# print('isothermDataFile = '+str(isothermDataFile))
# print('deadVolumeFile = '+str(deadVolumeFile))
# print('modelType = '+str(modelType))


# # for ii in range(5):
# #     algorithm_param = {'max_num_iteration':30,
# #                         'population_size':400,
# #                         'mutation_probability':0.25,
# #                         'crossover_probability': 0.55,
# #                         'parents_portion': 0.15,
# #                         'elit_ratio': 0.01,
# #                         'max_iteration_without_improv':None}
# for ii in range(1):
#     algorithm_param = {'max_num_iteration':10,
#                         'population_size':80,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}
    
#     # for ii in range(1):
#     extractZLCParameters(modelType = modelType,
#                   fileName = fileName,
#                   temperature = temperature,
#                   algorithm_param = algorithm_param,
#                   adsorbentDensity = adsorbentDensity,
#                   particleEpsilon = particleEpsilon,
#                   massSorbent = massSorbent,
#                   deadVolumeFile = deadVolumeFile,
#                   isothermDataFile = isothermDataFile,
#                   downsampleData = downsampleData,
#                   downsampleExp = downsampleExp)

# ##############################################################################

# modelType = 'KineticSB'

# # Isotherm model (if fitting only kinetic constant)
# isothermDataFile = 'CMS3K_DSL_082123.mat'

# ## Adsorbent properties
# # Adsorbent density [kg/m3]
# # This has to be the skeletal density
# adsorbentDensity = 1680 # CMS 3K

# # Particle porosity
# particleEpsilon = 0.61 # CMS 3K

# # Particle mass [g]
# massSorbent = 0.077 # CMS 3K

# # fileName  = ['ZLC_CMS3K_Exp13A_Output.mat',
# #             'ZLC_CMS3K_Exp17A_Output.mat',
# #             'ZLC_CMS3K_Exp15A_Output.mat',
# #             'ZLC_CMS3K_Exp14A_Output.mat',
# #             'ZLC_CMS3K_Exp18A_Output.mat',
# #             'ZLC_CMS3K_Exp16A_Output.mat',
# #             'ZLC_CMS3K_Exp13B_Output.mat',
# #             'ZLC_CMS3K_Exp17B_Output.mat',
# #             'ZLC_CMS3K_Exp15B_Output.mat',
# #             'ZLC_CMS3K_Exp14B_Output.mat',
# #             'ZLC_CMS3K_Exp18B_Output.mat',
# #             'ZLC_CMS3K_Exp16B_Output.mat',]

# fileName  = ['ZLC_CMS3K_Exp13B_Output.mat',
#             'ZLC_CMS3K_Exp17B_Output.mat',
#             'ZLC_CMS3K_Exp15B_Output.mat',
#             'ZLC_CMS3K_Exp14B_Output.mat',
#             'ZLC_CMS3K_Exp18B_Output.mat',
#             'ZLC_CMS3K_Exp16B_Output.mat',]

# # fileName  = ['ZLC_CMS3K_Exp13A_Output.mat',]

# deadVolumeFile =[[['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']], 
#                   [['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']]]    

# temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY


# print('fileName = '+str(fileName))
# print('downsampleConc = '+str(downsampleData))
# print('downsampleExp = '+str(downsampleExp))
# print('massSorbent = '+str(massSorbent))
# print('particleEpsilon = '+str(particleEpsilon))
# print('adsorbentDensity = '+str(adsorbentDensity))
# print('isothermDataFile = '+str(isothermDataFile))
# print('deadVolumeFile = '+str(deadVolumeFile))
# print('modelType = '+str(modelType))


# for ii in range(5):
#     algorithm_param = {'max_num_iteration':30,
#                         'population_size':400,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}
    
#     # for ii in range(1):
#     extractZLCParameters(modelType = modelType,
#                   fileName = fileName,
#                   temperature = temperature,
#                   algorithm_param = algorithm_param,
#                   adsorbentDensity = adsorbentDensity,
#                   particleEpsilon = particleEpsilon,
#                   massSorbent = massSorbent,
#                   deadVolumeFile = deadVolumeFile,
#                   isothermDataFile = isothermDataFile,
#                   downsampleData = downsampleData,
#                   downsampleExp = downsampleExp)

# ##############################################################################

# modelType = 'KineticSB'

# # Isotherm model (if fitting only kinetic constant)
# isothermDataFile = 'CMS3K_DSL_082123.mat'

# ## Adsorbent properties
# # Adsorbent density [kg/m3]
# # This has to be the skeletal density
# adsorbentDensity = 1680 # CMS 3K

# # Particle porosity
# particleEpsilon = 0.61 # CMS 3K

# # Particle mass [g]
# massSorbent = 0.077 # CMS 3K

# # fileName = ['ZLC_CMS3KAr_Exp01A_Output.mat',
# #             'ZLC_CMS3KAr_Exp03A_Output.mat',
# #             'ZLC_CMS3KAr_Exp05A_Output.mat',
# #             'ZLC_CMS3KAr_Exp02A_Output.mat',
# #             'ZLC_CMS3KAr_Exp04A_Output.mat',
# #             'ZLC_CMS3KAr_Exp06A_Output.mat',
# #             'ZLC_CMS3KAr_Exp01B_Output.mat',
# #             'ZLC_CMS3KAr_Exp03B_Output.mat',
# #             'ZLC_CMS3KAr_Exp05B_Output.mat',
# #             'ZLC_CMS3KAr_Exp02B_Output.mat',
# #             'ZLC_CMS3KAr_Exp04B_Output.mat',
# #             'ZLC_CMS3KAr_Exp06B_Output.mat',]

# fileName = ['ZLC_CMS3KAr_Exp01B_Output.mat',
#             'ZLC_CMS3KAr_Exp03B_Output.mat',
#             'ZLC_CMS3KAr_Exp05B_Output.mat',
#             'ZLC_CMS3KAr_Exp02B_Output.mat',
#             'ZLC_CMS3KAr_Exp04B_Output.mat',
#             'ZLC_CMS3KAr_Exp06B_Output.mat',]

# # Dead volume model
# deadVolumeFile = [[['deadVolumeCharacteristics_20230821_1803_b571c46.npz', #lowflow
#                     'deadVolumeCharacteristics_20230821_1849_b571c46.npz']],
#                   [['deadVolumeCharacteristics_20230821_1813_b571c46.npz', #lowflow
#                     'deadVolumeCharacteristics_20230821_1909_b571c46.npz']]] #highflow CMS Ar

# temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY


# print('fileName = '+str(fileName))
# print('downsampleConc = '+str(downsampleData))
# print('downsampleExp = '+str(downsampleExp))
# print('massSorbent = '+str(massSorbent))
# print('particleEpsilon = '+str(particleEpsilon))
# print('adsorbentDensity = '+str(adsorbentDensity))
# print('isothermDataFile = '+str(isothermDataFile))
# print('deadVolumeFile = '+str(deadVolumeFile))
# print('modelType = '+str(modelType))

# for ii in range(5):
#     algorithm_param = {'max_num_iteration':30,
#                         'population_size':400,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}
    
#     # for ii in range(1):
#     extractZLCParameters(modelType = modelType,
#                   fileName = fileName,
#                   temperature = temperature,
#                   algorithm_param = algorithm_param,
#                   adsorbentDensity = adsorbentDensity,
#                   particleEpsilon = particleEpsilon,
#                   massSorbent = massSorbent,
#                   deadVolumeFile = deadVolumeFile,
#                   isothermDataFile = isothermDataFile,
#                   downsampleData = downsampleData,
#                   downsampleExp = downsampleExp)

##############################################################################

# # Adsorbent properties
# # Adsorbent density [kg/m3]
# # This has to be the skeletal density
# # adsorbentDensity = 1020 # ZYH
# # adsorbentDensity = 3200 # ZYNa
# # adsorbentDensity = 2890 # ZYTMA
# # adsorbentDensity = 2130 # ZYH MSB
# # adsorbentDensity = 2410 # ZYNa MSB
# # adsorbentDensity = 2310 # ZYTMA MSB
# adsorbentDensity = 4100 # 13X

# # Particle porosity

# # particleEpsilon = 0.90 # ZYH
# # particleEpsilon = 0.76 # ZYNa
# # particleEpsilon = 0.71 # ZYTMA
# # particleEpsilon = 0.61 # CMS 3K
# particleEpsilon = 0.79 # 13X

# # Particle mass [g]
# # massSorbent = 0.086 # ZYH real
# # massSorbent = 0.075 # ZYNa
# # massSorbent = 0.065 # ZYTMA
# # massSorbent = 0.077 # CMS 3K
# massSorbent = 0.0594 # 13X
# massSorbent = 0.057 # 13X

# # Isotherm model (if fitting only kinetic constant)
# # isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# # isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# # isothermDataFile = 'ZYH_DSL_QC_070523.mat'
# isothermDataFile = '13X_CO2_L_QC_120923.mat'

# modelType = 'KineticSBMacro'

# temperature = [ 306.32, 325.66, 345.36,]*4 



# fileName  =['ZLC_Zeolite13X_Exp62B_Output.mat',
#                 'ZLC_Zeolite13X_Exp70B_Output.mat',
#                 'ZLC_Zeolite13X_Exp68B_Output.mat',
#                 'ZLC_Zeolite13X_Exp63B_Output.mat',
#                 'ZLC_Zeolite13X_Exp71B_Output.mat',
#                 'ZLC_Zeolite13X_Exp69B_Output.mat',]


# deadVolumeFile = [[['deadVolumeCharacteristics_20230924_1133_b571c46.npz', # 
#                     'deadVolumeCharacteristics_20230924_1207_b571c46.npz']], # 
#                   [['deadVolumeCharacteristics_20230924_1133_b571c46.npz', # 
#                     'deadVolumeCharacteristics_20230924_1207_b571c46.npz']]] # 13X old setup 

# print('fileName = '+str(fileName))
# print('downsampleConc = '+str(downsampleData))
# print('downsampleExp = '+str(downsampleExp))
# print('massSorbent = '+str(massSorbent))
# print('particleEpsilon = '+str(particleEpsilon))
# print('adsorbentDensity = '+str(adsorbentDensity))
# print('isothermDataFile = '+str(isothermDataFile))
# print('deadVolumeFile = '+str(deadVolumeFile))
# print('modelType = '+str(modelType))


# for ii in range(5):
#     algorithm_param = {'max_num_iteration':30,
#                         'population_size':400,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}
    
#     # for ii in range(1):
#     extractZLCParameters(modelType = modelType,
#                   fileName = fileName,
#                   temperature = temperature,
#                   algorithm_param = algorithm_param,
#                   adsorbentDensity = adsorbentDensity,
#                   particleEpsilon = particleEpsilon,
#                   massSorbent = massSorbent,
#                   deadVolumeFile = deadVolumeFile,
#                   isothermDataFile = isothermDataFile,
#                   downsampleData = downsampleData,
#                   downsampleExp = downsampleExp)


# ##############################################################################

# ## Adsorbent properties
# # Adsorbent density [kg/m3]
# # This has to be the skeletal density
# adsorbentDensity = 1680 # Activated carbon skeletal density [kg/m3]

# # Particle porosity
# particleEpsilon = 0.61 # AC

# # Particle mass [g]
# massSorbent = 0.0625  # AC

# # Isotherm model (if fitting only kinetic constant)
# # isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# # isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# # isothermDataFile = 'ZYH_DSL_QC_070523.mat'
# # isothermDataFile = '13X_CO2_L_QC_120923.mat'
# isothermDataFile = 'AC_RB3_CO2_QC_120923.mat'

# modelType = 'KineticSBMacro'

# temperature = [308.15, 328.15, 348.15]*4



# # fileName = ['ZLC_ActivatedCarbon_Exp84A_Output.mat',
# #             'ZLC_ActivatedCarbon_Exp86A_Output.mat',
# #             'ZLC_ActivatedCarbon_Exp88A_Output.mat',
# #             'ZLC_ActivatedCarbon_Exp84B_Output.mat',
# #             'ZLC_ActivatedCarbon_Exp86B_Output.mat',
# #             'ZLC_ActivatedCarbon_Exp88B_Output.mat',
# #             'ZLC_ActivatedCarbon_Exp85A_Output.mat',
# #             'ZLC_ActivatedCarbon_Exp87A_Output.mat',
# #             'ZLC_ActivatedCarbon_Exp89A_Output.mat',
# #             'ZLC_ActivatedCarbon_Exp85B_Output.mat',
# #             'ZLC_ActivatedCarbon_Exp87B_Output.mat',
# #             'ZLC_ActivatedCarbon_Exp89B_Output.mat',]

# fileName = ['ZLC_ActivatedCarbon_Exp84B_Output.mat',
#             'ZLC_ActivatedCarbon_Exp86B_Output.mat',
#             'ZLC_ActivatedCarbon_Exp88B_Output.mat',
#             'ZLC_ActivatedCarbon_Exp85B_Output.mat',
#             'ZLC_ActivatedCarbon_Exp87B_Output.mat',
#             'ZLC_ActivatedCarbon_Exp89B_Output.mat',]

# deadVolumeFile =[[['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']], 
#                   [['deadVolumeCharacteristics_20230925_1741_b571c46.npz', 
#                     'deadVolumeCharacteristics_20230925_1810_b571c46.npz']]]   

# print('fileName = '+str(fileName))
# print('downsampleConc = '+str(downsampleData))
# print('downsampleExp = '+str(downsampleExp))
# print('massSorbent = '+str(massSorbent))
# print('particleEpsilon = '+str(particleEpsilon))
# print('adsorbentDensity = '+str(adsorbentDensity))
# print('isothermDataFile = '+str(isothermDataFile))
# print('deadVolumeFile = '+str(deadVolumeFile))
# print('modelType = '+str(modelType))


# for ii in range(5):
#     algorithm_param = {'max_num_iteration':30,
#                         'population_size':400,
#                         'mutation_probability':0.25,
#                         'crossover_probability': 0.55,
#                         'parents_portion': 0.15,
#                         'elit_ratio': 0.01,
#                         'max_iteration_without_improv':None}
    
#     # for ii in range(1):
#     extractZLCParameters(modelType = modelType,
#                   fileName = fileName,
#                   temperature = temperature,
#                   algorithm_param = algorithm_param,
#                   adsorbentDensity = adsorbentDensity,
#                   particleEpsilon = particleEpsilon,
#                   massSorbent = massSorbent,
#                   deadVolumeFile = deadVolumeFile,
#                   isothermDataFile = isothermDataFile,
#                   downsampleData = downsampleData,
#                   downsampleExp = downsampleExp)


# #############################################################################