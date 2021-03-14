CC = nvcc
ifeq ($(DEBUG), 1)
	CFLAGS = -DDEBUG -std=c++11
else
	CFLAGS = -DNDEBUG -std=c++11
endif

OBJS = helper.o assignment.o

all: assignment

assignment: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o assignment

helper.o: HelperFunctions.cu HelperFunctions.cuh
	$(CC) $(CFLAGS) -c HelperFunctions.cu -o helper.o

assignment.o: assignment.cu
	$(CC) $(CFLAGS) -c assignment.cu -o assignment.o

clean:
	rm -f *~ *.o assignment

run_default:
	./assignment

run_zero:
	./assignment --blockSize 2048

run_one:
	./assignment --seed 10 --num_stream 1048576

run_two:
	./assignment --seed 10 --num_stream 1048576 --block_size 512

run_three:
	./assignment --num_stream 4194304

run_four:
	./assignment --num_stream 4194304 --block_size 512
