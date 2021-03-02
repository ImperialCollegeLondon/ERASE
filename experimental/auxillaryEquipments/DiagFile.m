clc
clear
MFM = struct('portName','COM5','baudRate',19200,'terminator','CR');

gasName = 'He';
gasID = checkGasName(gasName);
flagAlicat = checkManufacturer(MFM);

massflowout = controlAuxiliaryEquipments(MFM,generateSerialCommand('pollData',flagAlicat),flagAlicat,gasID);
