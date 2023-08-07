from extractDeadVolume import extractDeadVolume
import sys
sys.path.append('../ERASE/')

# for ii in range(5):

    # fileName = ['ZLC_Empty_DA_Exp02B_Output.mat',
    #             'ZLC_Empty_DA_Exp03B_Output.mat',
    #             'ZLC_Empty_DA_Exp04B_Output.mat',
    #             'ZLC_Empty_DA_Exp05B_Output.mat',]
# fileName = ['ZLC_Empty_Exp17A_Output.mat',
#             'ZLC_Empty_Exp17B_Output.mat']
    
# extractDeadVolume(fileName=fileName)
    
for ii in range(5):

    fileName = ['ZLC_Empty_Exp50A_Output.mat',
                'ZLC_Empty_Exp50B_Output.mat']
    
    extractDeadVolume(fileName=fileName)
    
    
for ii in range(5):

    fileName = ['ZLC_Empty_Exp51A_Output.mat',
                'ZLC_Empty_Exp51B_Output.mat']
    
    extractDeadVolume(fileName=fileName)
    
    