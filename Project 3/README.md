Code repository for Project 3: 'Penning trap' in course FYS4150.
--------------------------------------------------------------

main.cpp
----------
Main C++ program is contained in main.cpp, but is based on classes contained in PenningTrap.cpp and Particle.cpp. It includes a function Simulate() for simple interfacing with the classes, and by default recreates data used in project report. Data is saved to folder 'data' as armadillo matrices/cubes, and sample data used in project is included for convenience. Program for default settings has expected runtime of ~10 hours for a modern home laptop.

To compile/link and run main program:
- g++ main.cpp src/PenningTrap.cpp src/Particle.cpp -I include/ -o main.exe -larmadillo -O3
- ./main.exe

Optimization flag -O3 is optional but highly recommended. Flag -larmadillo is not optional.

plot.py
----------
A secondary python3 program plot.py is included for plotting results. It includes several functions for loading and plotting results generated from main program, but is mainly intended as a support program for project. As such, user friendliness has not been prioritized. By default recreates figures (but not animation due to long runtime) used in project report when run and saves them to folder 'fig'. Figures and animations used are included for convenience.

To compile and run plotting program:
- python3 plot.py

animations
----------
To view animations mentioned in program report, please see folder 'fig'. Currently two animations are included, visualizing the motion of a single and two particles. With runtime of 166 seconds and simulation time 100 µs, the animation rate is roughly 0.6 µs per second.

Current animations:
- single_3D.mp4
  - Shows motion of a single particle in 3D
- duo_3D_coulomb.mp4
  - Shows motion of two particles in 3D with Coulomb forces enabled
