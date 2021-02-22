CC = nvcc
CFLAGS =
OBJS = operations.o assignment.o

all: assignment

assignment: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o assignment

operations.o: operations.cu operations.cuh
	$(CC) $(CFLAGS) -c operations.cu

assignment.o: assignment.cu assignment.cuh
	$(CC) $(CFLAGS) -c assignment.cu

clean:
	rm -f *~ *.o assignment

run_default:
	./assignment

run_cipher:
	./assignment --cipher -10 512

run_operations:
	./assignment --do_opt 512 64 1024 32