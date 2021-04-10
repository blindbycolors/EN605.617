# Program Notes
The program can be configured in both <u>Debug</u> and <u>Release</u> mode.

# Running in Debug
Running the program in this mode will allow the kernel functions to print the values as they are being processed. Note that this will affect the run time of the function since I/O takes a longer time than kernel functions without any I/O.

1. Enter `make DEBUG=1` into the terminal
2. Entering `make run` into the terminal will run

# Running in Release
Running the program in this mode will not execute the print calls in the kernel code.

1. Enter `make` into the terminal. This will set the DEBUG flag to 0.
2. Entering `make run` into the terminal will run the program

# Cleaning Program
1. Enter `make clean` will remove all object files and the `assignment` executable.
