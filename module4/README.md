# Running Program
1. Enter `make` in the terminal
2. Preset options in:
  - `make run_default` : This option will run the program with built in defaults. The built in defaults will run both mathematical operations and the Caeser cipher. Math operations have a default total number of threads of 1 << 20 and block sizes of 512. The Caeser cipher operations have a default string length of 64 and a set block size of 32.
  - `make run_cipher` : This option will run only the Caeser cipher algorithm. The input must be in the format:
  `./assignment --cipher {offset} {string length}`
  where `{offset}` and `{string length}` are specifiable by the user. The `make run_cipher` option uses -10 as the offset and a string length of 512.
  - `make run_operations` : This option will run only the mathematical operations. The input must be in the format:
  `./assignment --do_opt {total threads} {block size} {...}`
  where `{total threads}` and `{block size}` are specifiable by the user. The option uses thread sizes 512 and 1024 and block sizes of 64 and 32. This option is set to accept an arbitrary number of pairs of thread sizes and block sizes; however, the program will identify the unique combination of thread sizes and block sizes.

# Stretch Problem
One thing that pops out to me line 51 in `pageable_transfer_execution` where the calculation for the variable `num_threads` is being done. I believe that variable is not needed and the `threads_per_block` variable can be used to initialize the `encrypt<<<>>>` kernel function. After I read through some nvidia blogs about best practices such as using a utility function to check that the CUDA calls don't return any errors. Additionally, the code could have used the CUDA utility functions `cudaEventCreate` to initialize the start/stop timers. Since this code is utilizing C++ along with CUDA, it's best to implement the functions and variables with C++ best practices (i.e. the naming convention of camel case), although I understand that nvidia CUDA examples use a form of snake case combined with camel case to differentiate between host and device arrays (i.e. `h_aPageable` and `d_aPageable`). 
