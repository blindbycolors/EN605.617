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
One thing that stood out in the code is the magic number 16 that appeared throughout the function test_register_mem_cpy and test_shared_mem_cpy. From the code, 16 is the total number of threads and total elements being allocated for the output array. However, I think a more effective way of using the constant 16 is to set it as a constant variable via #define or const key word at the top of the file since the value 16 is used in multiple functions. Using a descriptive name for the variable and removing the magic number 16 will make the code easier to modify and much easier for an individual who did not write the code to review and quickly understand. Additionally, the checks for the success status from the kernel is a great practice. The timing check to include the copying of data between host and device memory is different than the previous examples and what seems to be a typical practice of timing the memory calculations only. However, I think this is a great way to gauge the actual speedup of using the GPU vs if everything was computed in the CPU especially since the data transfer between host and device can be a significant bottleneck to overcome. Lastly, the consistency with the naming convention is great. However, I believe C/C++ standards call for camel case / upper camel case as a best practice. Just the fact that the developer remained consistent with his/her naming convention is great.
