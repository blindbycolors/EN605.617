The assignment.exe executable processes up to six inputs. Any additional inputs thereafter are ignored by the program.

For example:

./assignment.exe 256 64 128 32 64 16 

will generate three different values for the total number of threads (64, 32, 16) and three different values for the block size (256,128,64). 

Default total number of threads are: 1024, 32768, 1048576
Default block sizes are: 64, 128, 256

The program will generate calculations for addition, subtraction, multiplication, and modulus for each combination of total threads and block sizes for a total of 8 different combinations of threads and block sizes.