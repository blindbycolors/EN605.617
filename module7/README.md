# Program Notes
The program can be configured in both <u>Debug</u> and <u>Release</u> mode. Additionally, the program accepts three optional inputs:
  - --seed {integer}
  - --num_stream {integer}
  - --block_size {integer}

# Default values
- Random seed: 100
- Block size: 256
- Total threads: 2048

# Running in Debug
Running the program in this mode will allow the kernel functions to print the values as they are being processed. Note that this will affect the run time of the function since I/O takes a longer time than kernel functions without any I/O.

1. Enter `make DEBUG=1` into the terminal
2. Entering `make run_default` into the terminal will run the program with built in default values;
3. Entering `make run_zero` into the terminal will run the program with all default values except the block size which is set to 2048 via the command line interface.
4. Entering `make run_one` into the terminal will run the program with a random seed of 10, total threads of 1048576, and the default block size;
5. Entering `make run_two` will run the program with a random seed of 10, total threads of 1048576, and block size of 512.
6. Entering `make run_three` will run the program with total threads of 41943404 and default values for block size and random seed;
7. Entering `make run_four` will run the program with total threads of 4194304, a block size of 512, and default random seed value.

# Running in Release
Running the program in this mode will not execute the print calls in the kernel code.

1. Enter `make` into the terminal. This will set the DEBUG flag to 0.
2. The same options are available as running in Debug mode. The difference is the program will execute print statements in the kernel, which will significantly slow down the run time. However, it will demonstrate the asynchronous nature of streams.

# Cleaning Program
1. Enter `make clean` will remove all object files and the `assignment` executable.

# Stretch Problem
The program seems to be attempting to run a stream with asynchronous memory copy. It initializes a CUDA stream on lines 151 and 152. However, a couple issues exist. First, the program uses `cudaMemcpy` instead of `cudaMemcpyAsync`. The `cudaMemcpy` function uses a default stream so the program did not necessarily need to create a stream. If the goal was to implement asynchronous streams, the program should have used `cudaMemcpyAsync` to initialize the kernel. The same issue applies to lines 208/209 where the stream initialization code would have been.
