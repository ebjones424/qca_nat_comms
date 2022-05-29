# qca_nat_comms
This repository holds code used to generate results for the publication "Small-world complex network generation on a digital quantum processor". For the corresponding data used to support the conclusions of the publication, please see the Dryad repository: https://doi.org/10.5061/dryad.fbg79cnxd

To use the data hosted there, read the README_Dryad.txt document and dowload and unzip the 1d.zip file. All the python files used in this github repository to interact with this data should be placed one directory level up from the unzipped /1d/... folder.

There are four main python files used to generate and process the data used in the manuscript:

1.) QCA_1d_Weber_batch_optimizaed.py
    - This script was used to generate raw data by constructing QCA circuits and either numerically emulating them (classically) or sending them to the Weber processor.       All relevant circuit parameters are initialized in the main() function and should either be commented on or self-explanatory. Note that without access to the
      Google cloud service, you will not be able to take quantum computer data. Classical emulation will still work under sim_mode='cirq'.
      
2.) QCA_1d_Rainbow_batch_optimized.py
    - Nearly identical as script #1 above, but intended for the Rainbow processor.
    
3.) plots_1d_Weber.py
    - Plotting and data processing utility for e.g., calculating population dynamics and complex network observables. Circuit parameters and main functionality are in
      function main().

4.) paper_analysis.py
    - Plotting and data processing utility for making publication-quality plots and comparing classical to quantum data. Circuit parameters and main functionality are in       function main().
    
For questions regarding the use of this code and data, please reach out to Eric Jones at ericbjones14@gmail.com or eric.jones@coldquanta.com.
