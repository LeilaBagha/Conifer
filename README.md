# Conifer

## Input files 

single-cell matrix, which is ternary matrix D of dimension nxm, where n denotes number of mutations and m denotes number of single cells obtained in sequencing experiment
Single cell matrix contains 0,1 and 3, which shows absence, presence or missing value for mutation calls, respectively. 
Bulk sequencing derived matrix containing VAFs for each of n mutations in different bulk samples.
Folder input contains an example input of bulk file (bulk.csv) and single-cell matrix (singlecell.csv).

## Running Conifer
Simple Python script main.py for running Conifer is provided . Description of the parameters is also provided inside this file. In order to run B-SCITE it suffices to adjust related parameters in main.py and run command "python main.py". Please note that the values of number of iterations and repetitions provided in main.py are set to very small value in order to facilitate verifying the correctness of installation, variable assignments etc. For the real application, these numbers should be significantly higher (we recommend at least 3 repretitions and at least several hundred-thousands of iterations).

In main.py two fuctions conifer and clone_identifier are provided. the former is for providing tree based on 

Repository for paper Conifer: Clonal Tree Inference for Tumor Heterogeneity with Single-Cell and Bulk Sequencing Data 
(code revision in progress).

This is a package uses the distance-dependent Chinese Restaurant Process (dd-CRP) 
It is based roughly on code originally written by 
[Christopher Baldassano](https://github.com/cbaldassano/Parcellating-connectivity).
