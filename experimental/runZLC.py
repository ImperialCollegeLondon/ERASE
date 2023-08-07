from extractZLCParameters import extractZLCParameters
import sys
import numpy as np
import numpy.matlib

sys.path.append('../ERASE/')

# fileName =         ['ZLC_Lewatit_DA_Exp05A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp07A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp09A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp11A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp13A_Output.mat',
#                     'ZLC_Lewatit_DA_Exp05B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp07B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp09B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp11B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp13B_Output.mat',
#                     'ZLC_Lewatit_DA_Exp06A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp08A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp10A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp12A_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp14A_Output.mat',
#                     'ZLC_Lewatit_DA_Exp06B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp08B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp10B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp12B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp14B_Output.mat',]

# fileName =         ['ZLC_Lewatit_DA_Exp13B_Output.mat',]

# fileName =         ['ZLC_Lewatit_DA_Exp05B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp07B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp09B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp11B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp13B_Output.mat',
#                     'ZLC_Lewatit_DA_Exp06B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp08B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp10B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp12B_Output.mat',
# 		            'ZLC_Lewatit_DA_Exp14B_Output.mat',]

# fileName =         ['ZLC_Lewatit_DA_Exp05B_Output.mat',
#                     'ZLC_Lewatit_DA_Exp09B_Output.mat',
#                     'ZLC_Lewatit_DA_Exp13B_Output.mat',
#                     'ZLC_Lewatit_DA_Exp06B_Output.mat',
#                     'ZLC_Lewatit_DA_Exp10B_Output.mat',
#                     'ZLC_Lewatit_DA_Exp14B_Output.mat',]

# fileName =         ['ZLC_BNFASp_Exp35B_Output.mat',
# 		            'ZLC_BNFASp_Exp37B_Output.mat',
# 		            'ZLC_BNFASp_Exp33B_Output.mat',
# 		            'ZLC_BNFASp_Exp36B_Output.mat',
# 		            'ZLC_BNFASp_Exp38B_Output.mat',
# 		            'ZLC_BNFASp_Exp34B_Output.mat',]
# fileName = ['ZLC_ZIF8_MT_Exp01A_Output.mat',
#             'ZLC_ZIF8_MT_Exp03A_Output.mat',
#             'ZLC_ZIF8_MT_Exp07A_Output.mat',
#             'ZLC_ZIF8_MT_Exp01B_Output.mat',
#             'ZLC_ZIF8_MT_Exp03B_Output.mat',
#             'ZLC_ZIF8_MT_Exp07B_Output.mat',
#             'ZLC_ZIF8_MT_Exp02A_Output.mat',
#             'ZLC_ZIF8_MT_Exp04A_Output.mat',
#             'ZLC_ZIF8_MT_Exp08A_Output.mat',
#             'ZLC_ZIF8_MT_Exp02B_Output.mat',
#             'ZLC_ZIF8_MT_Exp04B_Output.mat',
#             'ZLC_ZIF8_MT_Exp08B_Output.mat',]

# fileName =  ['ZLC_ZYH_Exp09A_Output.mat',
# 	            'ZLC_ZYH_Exp11A_Output.mat',
# 	            'ZLC_ZYH_Exp13A_Output.mat',
# 	            'ZLC_ZYH_Exp09B_Output.mat',
# 	            'ZLC_ZYH_Exp11B_Output.mat',
# 	            'ZLC_ZYH_Exp13B_Output.mat',
# 	            'ZLC_ZYH_Exp10A_Output.mat',
# 	            'ZLC_ZYH_Exp12A_Output.mat',
# 	            'ZLC_ZYH_Exp14A_Output.mat',
# 	            'ZLC_ZYH_Exp10B_Output.mat',
# 	            'ZLC_ZYH_Exp12B_Output.mat',
#                 'ZLC_ZYH_Exp14B_Output.mat',]

# fileName =     ['ZLC_ZYH_Exp09B_Output.mat',
#  	            'ZLC_ZYH_Exp11B_Output.mat',
#  	            'ZLC_ZYH_Exp13B_Output.mat',
#  	            'ZLC_ZYH_Exp10B_Output.mat',
#  	            'ZLC_ZYH_Exp12B_Output.mat',
#                 'ZLC_ZYH_Exp14B_Output.mat',]

# fileName = ['ZLC_ZYNa_Exp05B_Output.mat',
#             'ZLC_ZYNa_Exp03B_Output.mat',
#             'ZLC_ZYNa_Exp09B_Output.mat',
#             'ZLC_ZYNa_Exp06B_Output.mat',
#             'ZLC_ZYNa_Exp04B_Output.mat',
#             'ZLC_ZYNa_Exp10B_Output.mat',]

fileName = ['ZLC_ZYTMA_Exp01B_Output.mat',
            'ZLC_ZYTMA_Exp03B_Output.mat',
            'ZLC_ZYTMA_Exp05B_Output.mat',
            'ZLC_ZYTMA_Exp02B_Output.mat',
            'ZLC_ZYTMA_Exp04B_Output.mat',
            'ZLC_ZYTMA_Exp06B_Output.mat',]  

# Temperature (for each experiment)
# temperature = [308.15, 328.15, 348.15]*4 # AC sim
# temperature = [ 345.63, 325.85, 306.32,]*4 # Boron Nitride (S2)
# temperature = [ 344.6, 325.49, 306.17,]*4 # Boron Nitride (2 pellets)
# temperature = [ 325.66, 306.32, 345.36,]*4 # Zeolite 13X
# temperature = [ 283.15, 293.15, 303.15,]*4 # BNFASp and BNpFAS
# temperature = [ 308.15, 328.15, 348.15,]*4 # AC new setup
# temperature = [ 363.15, 348.15, 333.15, 318.15, 303.15,]*2 # lewatit new setup
# temperature = [303.15,]*2
# temperature = [ 363.15, 333.15, 303.15,]*2 # lewatit new setup
# temperature = [ 303.15, 293.15, 283.15, ]*4 # ZIF8 MT
temperature = [ 288.15, 298.15, 308.15, ]*4 # ZY

# Adsorbent properties
# Adsorbent density [kg/m3]
# This has to be the skeletal density
# adsorbentDensity = 1680 # Activated carbon skeletal density [kg/m3]
# adsorbentDensity = 4100 # Zeolite 13X H 
# adsorbentDensity = 1250 # BNFASp skeletal density [kg/m3]
# adsorbentDensity = 2320 # BNpFAS skeletal density [kg/m3]
# adsorbentDensity = 1060 # lewatit skeletal density [kg/m3]
# adsorbentDensity = 988 # Lewatit skeletal density [kg/m3]
# adsorbentDensity = 1555 # ZIF-8 MT
# adsorbentDensity = 2400 # ZIF-8 MCB20
# adsorbentDensity = 2100 # ZIF-8 MCB30
# adsorbentDensity = 1020 # ZYH
# adsorbentDensity = 3200 # ZYNa
# adsorbentDensity = 2890 # ZYTMA
# adsorbentDensity = 2130 # ZYH MSB
adsorbentDensity = 2410 # ZYNa MSB
adsorbentDensity = 2310 # ZYTMA MSB

# Particle porosity
# particleEpsilon = 0.61 # AC
# particleEpsilon = 0.79 # Zeolite 13X H
# particleEpsilon = 0.64 # BNFASp
# particleEpsilon = 0.67 # BNpFAS
# particleEpsilon = 0.337 # lewatit
# particleEpsilon = 0.44 # Lewatit
# particleEpsilon = 0.67 # BNpFAS
# particleEpsilon = 0.47 # ZIF-8 MT
# particleEpsilon = 0.62 # ZIF-8 MCB20
# particleEpsilon = 0.59 # ZIF-8 MCB30
# particleEpsilon = 0.90 # ZYH
# particleEpsilon = 0.76 # ZYNa
particleEpsilon = 0.71 # ZYTMA

# Particle mass [g]
# massSorbent = 0.0625  # AC
# massSorbent = 0.0594 # Zeolite 13X H
# massSorbent = 0.069  # BNFASp
# massSorbent = 0.1  # BNFASp
# massSorbent = 0.0262  # lewatit
# massSorbent = 0.0262  # Lewatit
# massSorbent = 0.1295  # BNpFAS
# massSorbent = 0.059 # ZIF-8 MT
# massSorbent = 0.09 # ZIF-8 MCB20
# massSorbent = 0.102 # ZIF-8 MCB30
# massSorbent = 0.08 # ZYH (v1)
# massSorbent = 0.07 # ZYNa
massSorbent = 0.065 # ZYTMA

    
# Dead volume model
# deadVolumeFile = ['deadVolumeCharacteristics_20230220_1813_7e5a5aa.npz', #lowflow
#                   'deadVolumeCharacteristics_20230220_1752_7e5a5aa.npz'] #highflow ZIF

# deadVolumeFile = ['deadVolumeCharacteristics_20230309_1626_7e5a5aa.npz', #lowflow
#                   'deadVolumeCharacteristics_20230309_1908_7e5a5aa.npz'] #highflow ZEOLITES RUN

deadVolumeFile = [[['deadVolumeCharacteristics_20230321_1048_59cc206.npz', # 50A
                  'deadVolumeCharacteristics_20230321_1238_59cc206.npz']], # 51A
                  [['deadVolumeCharacteristics_20230321_1137_59cc206.npz', # 50B
                  'deadVolumeCharacteristics_20230321_1252_59cc206.npz']]] # 51B

# Isotherm model (if fitting only kinetic constant)
isothermDataFile = 'ZYTMA_DSL_QC_070523.mat'
# isothermDataFile = 'ZYNa_DSL_QC_070323.mat'
# isothermDataFile = 'ZYH_DSL_QC_070523.mat'

# Downsample the data at different compositions (this is done on 
# normalized data) [High and low comp]
downsampleData = False
downsampleExp = True # Change downsampleInt in extractZLCParameters

print('fileName = '+str(fileName))
print('downsampleConc = '+str(downsampleData))
print('downsampleExp = '+str(downsampleExp))
print('massSorbent = '+str(massSorbent))
print('particleEpsilon = '+str(particleEpsilon))
print('adsorbentDensity = '+str(adsorbentDensity))
print('adsorbentDensity = '+str(adsorbentDensity))
print('isothermDataFile = '+str(isothermDataFile))
print('deadVolumeFile = '+str(deadVolumeFile))


for ii in range(5):
    algorithm_param = {'max_num_iteration':30,
                        'population_size':400,
                        'mutation_probability':0.25,
                        'crossover_probability': 0.55,
                        'parents_portion': 0.15,
                        'elit_ratio': 0.01,
                        'max_iteration_without_improv':None}
    
    # for ii in range(1):
    extractZLCParameters(modelType = 'Kinetic',
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
    