Code repository for Project 4: 'Ising model' in course FYS4150.
--------------------------------------------------------------

This repo is supposed to act as a code base for project 4 in course FYS4150. Figures and data used have been included for convenience, but the included code is intended to be able to reproduce all results. The main simulation function part of class MCMC uses parallelized code, which uses thread number as seeds for number generation. Therefore the program would have to be run for 4 threads to correctly recreate data.

main.cpp
----------
Main C++ program is contained in main.cpp, but is based on classes contained in IsingModel.cpp and MCMC.cpp. Contains functions which can be used to reproduce results used in project report (with same seed set as default), but by default does nothing as the total runtime would be many hours. 

To compile/link and run main program:
- g++ main.cpp src/IsingModel.cpp src/MCMC.cpp -o main.exe -larmadillo -fopenmp -O3 -I include/
- ./main.exe

Optimization flag -O3 is optional but highly recommended. Flags -larmadillo and -fopenmp are not optional.

src/IsingModel.cpp
----------
File for class IsingModel used as container for Ising model grid, temperature, energy, and magnetization, with supporting functions for calculation, initialization, and spin flips.

Compiled as part of main program

src/MCMC.cpp
----------
File for class MCMC intended for sampling and processing of Ising model data. Takes simulation and model parameters as initialization input, which then are used as part of class function to generate samples. Two functions are included, one for main simulation which samples all values and one for sampling for histogram generation.

Compiled as part of main program

plot.py
----------
A secondary python3 program plot.py is included for plotting results. When run it automatically recreates figures used in report and saves them to folder 'fig' by reading data from folder 'data'.

To compile and run plotting program:
- python3 plot.py
