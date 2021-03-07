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

run_options:
	./assignment 25 350
