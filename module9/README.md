# Program Notes
The program can be configured in both <u>Debug</u> and <u>Release</u> mode. Additionally, the program accepts one optional input:
  - {number of elements}

# Default values
- Number of elements: 1 << 20

# Running in Debug
Running the program in `Debug` mode will print the contents of the arrays (i.e. the original arrays, A and B, and the arrays containing the result of the mathematical operations).

1. Enter `make DEBUG=1` into the terminal.
2. Entering `make run_default` into the terminal will run the program with built in default values.
3. Entering `make run_zero` into the terminal will run the program with 100 elements.
4. Entering `make run_one` into the terminal will run the program with 1024 elements.
5. Entering `make run_two` into the terminal will run the program with 4096 elements.

# Running in Release
Running the program in this mode will not execute the print calls for the arrays.

1. Enter `make` into the terminal. This will set the DEBUG flag to 0.
2. The same options are available as running in Debug mode. The difference is the program will execute print statements in the kernel, which will significantly slow down the run time. However, it will demonstrate the asynchronous nature of streams.

# Cleaning Program
1. Enter `make clean` will remove all object files and the `assignment` executable.
