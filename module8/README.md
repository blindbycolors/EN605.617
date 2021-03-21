# Program Notes
The program can be configured in both <u>Debug</u> and <u>Release</u> mode. Additionally, the program accepts three optional inputs in any order:
  --a_row {integer} : Defines the number of rows in first matrix for matrix multiplication
  --a_col {integer} : Defines the number of columns in first matrix for matrix multiplication
  --b_row {integer} : Defines the number of rows in second matrix for matrix multiplication
  --b_col {integer} : Defines the number of columns in second matrix for matrix multiplication
  --eigen {integer} : Defines the number of rows and columns for a randomly generated symmetric matrix

# Default values
- a_row: 1
- a_col: 5
- b_row: 5
- b_col: 3
- eigen: 3

# Running Program
Running the program in this mode will allow the kernel functions to print the values as they are being processed. Note that this will affect the run time of the function since I/O takes a longer time than kernel functions without any I/O.

1. Enter `make` into the terminal
2. Enter `make run_default` into terminal to run program with built in default values.
3. Enter `make run_zero` into terminal to run program with:
    - a_row 8
    - a_col 5
    - b_row 5
    - b_col 2
    - eigen 10
4. Enter `make run_one` into the terminal to run program with:
    - a_row 100
    - a_col 200
    - b_row 200
    - b_col 150
    - eigen 1024

    # Cleaning Program
    1. Enter `make clean` will remove all object files and the `assignment` executable.
