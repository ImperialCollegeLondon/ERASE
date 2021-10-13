			      ___           ___           ___           ___           ___     
			     /\  \         /\  \         /\  \         /\  \         /\  \    
			    /::\  \       /::\  \       /::\  \       /::\  \       /::\  \   
			   /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/\ \  \     /:/\:\  \  
			  /::\~\:\  \   /::\~\:\  \   /::\~\:\  \   _\:\~\ \  \   /::\~\:\  \ 
			 /:/\:\ \:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ /\ \:\ \ \__\ /:/\:\ \:\__\
			 \:\~\:\ \/__/ \/_|::\/:/  / \/__\:\/:/  / \:\ \:\ \/__/ \:\~\:\ \/__/
			  \:\ \:\__\      |:|::/  /       \::/  /   \:\ \:\__\    \:\ \:\__\  
			   \:\ \/__/      |:|\/__/        /:/  /     \:\/:/  /     \:\ \/__/  
			    \:\__\        |:|  |         /:/  /       \::/  /       \:\__\    
			     \/__/         \|__|         \/__/         \/__/         \/__/    

			     	EMISSION REDUCTION BY ARRAY OF SENSOR ELECTRONICS
                               

				   Copyright (C) 2021 Ashwin Kumar Rajagopalan
				     Imperial College London, United Kingdom

  				       Multifunctional Nanomaterials Group
 						Dr. Camille Petit
 						 Project: ERASE 
				  Readme last updated: 13th October 2021, AK

## INTRODUCTION
This folder contains the final simulation result files for the work performed under the ERASE project. The folder has results from both the computational and experimental work performed during ERASE. To plot the figures that appear in the corresponding manuscripts, please use either `plotsForArticle_Simulation.py` or `plotsForArticle_Experiment.py` under the `plotFunctions` folder.


## FILES FOR THE COMPUTATIONAL ARTICLE (1)

### Sensor array combinations
* With constraint on mole fraction (Figure 2)
- 2 gases, 1 sensor: ﻿arrayConcentration_20210212_1050_b02f8c3.npz
- 2 gases, 2 sensors: ﻿arrayConcentration_20210211_1822_b02f8c3.npz
* Without constraint on mole fraction
- 2 gases, 1 sensor: ﻿arrayConcentration_20210211_1818_b02f8c3.npz
- 2 gases, 2 sensors: ﻿arrayConcentration_20210212_1055_b02f8c3.npz

### Material response shape (Figure 3)
* A: sensitivityAnalysis_17_20210706_2258_ecbbb3e.npz
* B: sensitivityAnalysis_6_20210707_1125_ecbbb3e.npz
* C: sensitivityAnalysis_16_20210707_0842_ecbbb3e.npz

### Graphical Tool (Figure 4)
* D: sensitivityAnalysis_6-2_20210719_1117_ecbbb3e.npz
* E: sensitivityAnalysis_17-16_20210706_2120_ecbbb3e.npz

### Impact of adsorption capacity (Figure 5)
* D: sensitivityAnalysis_6-2_20210719_1117_ecbbb3e.npz
* D3: sensitivityAnalysis_6-2_20210719_2145_ecbbb3e.npz
* D10: sensitivityAnalysis_6-2_20210719_1458_ecbbb3e.npz

### Three Materials (Figure 6)
* F: sensitivityAnalysis_17-15-6_20210707_2036_ecbbb3e.npz
* G: sensitivityAnalysis_17-15-16_20210708_0934_ecbbb3e.npz
* REF: sensitivityAnalysis_17-15_20210709_1042_ecbbb3e.npz

### Importance of Kinetics (Figure 7)
* Equilibrium: fullModelConcentrationEstimate_6-2_20210320_1336_4b80775.npz
* Full Model: fullModelConcentrationEstimate_6-2_20210320_1338_4b80775.npz

### Factors influencing response (Figure 8)
* Rate constant: fullModelSensitivity_rateConstant_20210324_0957_c9e8179.npz
* Void fraction: fullModelSensitivity_voidFrac_20210324_1006_c9e8179.npz
* Total volume: fullModelSensitivity_volTotal_20210324_1013_c9e8179.npz
﻿* Flow rate: fullModelSensitivity_flowIn_20210407_1656_c9e8179.npz


## REFERENCES

(1) Rajagopalan, A. K.; Petit, C. Material Screening for Gas Sensing Using an Electronic Nose: Gas Sorption Thermodynamic and Kinetic Considerations. *ACS Sensors* **2021**, 15, 38. https://doi.org/10.1021/ACSSENSORS.1C01807.