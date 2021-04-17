CC = nvcc

CFLAGS += -std=c++11 -Wno-deprecated-gpu-targets

OBJS = assignment.o

all: assignment

assignment: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lOpenCL -o assignment

assignment.o: assignment.cpp
	$(CC) $(CFLAGS) -c assignment.cpp -lOpenCL -o assignment.o

clean:
	rm -f *~ *.o assignment

run:
	./assignment
