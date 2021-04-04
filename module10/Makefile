CC = g++
ifeq ($(DEBUG), 1)
	CFLAGS = -DDEBUG
else
	CFLAGS = -DNDEBUG
endif

CFLAGS += -std=c++11 -w -fpermissive

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
