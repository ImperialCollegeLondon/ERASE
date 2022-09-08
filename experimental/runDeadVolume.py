from extractDeadVolume import extractDeadVolume
import sys
import os
sys.path.append('../ERASE/')

for ii in range(5):

    fileName = ['ZLC_Empty_DA_Exp02B_Output.mat',
                'ZLC_Empty_DA_Exp03B_Output.mat',
                'ZLC_Empty_DA_Exp04B_Output.mat',
                'ZLC_Empty_DA_Exp05B_Output.mat',]
    
    extractDeadVolume(fileName=fileName)