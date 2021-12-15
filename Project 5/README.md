Code repository for Project 5: 'The Schrödinger equation' in course FYS4150.
--------------------------------------------------------------

This repo is supposed to act as a code base for project 5 in course FYS4150. Figures have been included for convenience, but the included code is intended to be able to reproduce all results. Data was not included due to size limitations, but can be recreated with the main program.

main.cpp
----------
Main C++ program is contained in main.cpp, but is based on class Packet contained in Packet.cpp. Running the main program recreates data used in project and stores it in folder 'data'.

To compile/link and run main program:
- g++ main.cpp src/Packet.cpp -o main.exe -larmadillo -O3 -I include/
- ./main.exe

Optimization flag -O3 is optional but highly recommended. Flag -larmadillo is not optional.

src/Packet.cpp
----------
File containing class Packet which is intended to create a wave packet, a potential acting as environment, and a step function for progressing the system in time. Used in main program main.cpp to perform simulations.

Compiled as part of main program

plot.py
----------
A secondary python3 program plot.py is included for plotting results. When run it automatically recreates figures (both used and unused) as well as animations and saves them to folder 'fig' by reading data from folder 'data'.

To compile and run plotting program:
- python3 plot.py
