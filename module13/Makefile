CC = nvcc
ifeq ($(DEBUG), 1)
	CFLAGS = -DDEBUG
else
	CFLAGS = -DNDEBUG
endif

CFLAGS += -std=c++11 -Wno-deprecated-gpu-targets

OBJS = assignment.o

all: assignment

assignment: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lOpenCL -o assignment

assignment.o: assignment.cpp
	$(CC) $(CFLAGS) -c assignment.cpp -lOpenCL -o assignment.o

clean:
	rm -f *~ *.o assignment

run_square:
	./assignment square

run_cube:
	./assignment cube

run_double:
	./assignment bitwiseDouble

run_half:
	./assignment bitwiseHalf

run_zero:
	./assignment square bitwiseDouble bitwiseHalf

run_one:
	./assignment cube bitwiseHalf square

run_two:
	./assignment square bitwiseHalf bitwiseHalf bitwiseHalf cube bitwiseDouble
