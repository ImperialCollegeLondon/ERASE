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
# Obtain the current date and time to be used either for saving in file
# name or to enhance traceability of the simulation
#
# Last modified:
# - 2020-10-19, AK: Initial creation
#
# Input arguments:
# - N/A
#
# Output arguments:
# - simulationDT: Current date and time in YYYYmmdd_HHMM format
#
############################################################################

def getCurrentDateTime():
    # Get the current date and time for saving purposes    
    from datetime import datetime
    now = datetime.now()
    simulationDT = now.strftime("%Y%m%d_%H%M")

    # Return the current date and time
    return simulationDT