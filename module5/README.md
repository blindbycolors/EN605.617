# Program Notes
The program can be configured in both <u>Debug</u> and <u>Release</u> mode.

# Running in Debug
Running the program in this mode will allow the kernel functions to print the values as they are being processed. Note that this will affect the run time of the function since I/O takes a longer time than kernel functions without any I/O.

1. Enter `make DEBUG=1` into the terminal
2. Entering `make run_default` into the terminal will run the program with built in default values for the Caeser cipher offset (3) and random seed (100);
3. Entering `make run_options` into the terminal will run the program with a Caeser cipher offset of 25 and random seed of 350. These values can be configured via the make file.

# Running in Release
Running the program in this mode will not execute the print calls in the kernel code.

1. Enter `make` into the terminal. This will set the DEBUG flag to 0.
2. Entering `make run_default` into the terminal will run the program with built in default values for the Caeser cipher offset (3) and random seed (100);
3. Entering `make run_options` into the terminal will run the program with a Caeser cipher offset of 25 and random seed of 350. These values can be configured via the make file.

# Cleaning Program
1. Enter `make clean` will remove all object files and the `assignment` executable.

# Stretch Problem
Although I see why the code uses a run_gpu kernel function to switch between the different operations (add, subtract, multiply, mod), I think it would be better avoid switch statements since it will cause branching in the GPU. For complicated calculations, the branching may not be a bottle neck but since the functions are simple with one operation only, the branching could slow down the code. On a positive note, the author avoided making branch calls in a for loop which would have a much larger impact than a one-off branch. One other thing that stood out is the call to copyToShared in each case (assuming this was called iside all other cases as it shows copToShowed inside case 5:). Since the function call to copyToShared does the same thing for all cases, the call can simply be done outside the switch statement. 
