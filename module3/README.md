The assignment.exe executable processes up to six inputs. Any additional inputs thereafter are ignored by the program.

For example:

./assignment.exe 256 64 128 32 64 16 

will generate three different values for the total number of threads (64, 32, 16) and three different values for the block size (256,128,64). 

Default total number of threads are: 1024, 32768, 1048576<br>
Default block sizes are: 64, 128, 256 <br>
<b>Note:</b> The default values will cause a <br><br>

<span style="color:red">Warning - the output rate '187445.73353293413 byptes/s' has exceeded the allowed value '6000 bytes/s': attempting to kill the process. Please use Ctrl-C to return to the prompt</span> <br><br>
message to appear. If this is run on a Linux terminal (I ran this on Ubuntu 20.04 LTS), no warning should occur.

The program will generate calculations for addition, subtraction, multiplication, and modulus for each combination of total threads and block sizes for a total of 8 different combinations of threads and block sizes.


<h2>Stretch Problem</h2>

If the surrounding script only executes the main function once, a bad thing
about the submission is that it does not provide an accurate image of the CPU 
runtime efficiency versus the GPU runtime efficiency. For example, depending on
number of block size and number of threads, as mentioned in the lecture videos,
the efficiency of the GPU processing may be effected (i.e. a larger block size
may increase the GPU efficiency because of the large number of threads provided
to ther kernel). Additionally, if the number of threads is not divisible by the
block size (i.e. totalThreads % blockSize != 0), this can affect the kernel
configuration and the way the SM handles the threads. For example, one run with
thread blocks size set to 576 and total threads set to 2048 would cause the thread
blocks to no longer fit on each SM so the will need to run with 3 thread blocks
(1728 threads instead of 2048), effectively decreasing the efficiency of the GPU.
This example also points to possible effects of not usineg a multiple of the warp
size for the number of threads per block.

Another bad thing, if using the submission to show the comparison between the
GPU and CPU, having one data size is not enough to provide any meaningful insights
into efficiency of the GPU vs. CPU. For example, both CPU and GPU may have
similar execution times for small datasets but as the data size increases,
the execution times may start to diverge.

One good thing about running the submission once is getting insight into the GPU
unit (if the individual did not know the details about his/her GPU already).
The hardware can be a bottleneck for both a GPU and a CPU; however, since GPUs are
fashioned to be solutions to parallelizing data processing of large data sets,
GPU hardware limitations may cause more latency to occur. On the other hand, the
CPU hardware limitations can also cause latency issues as well.