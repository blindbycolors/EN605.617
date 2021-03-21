CC = nvcc
CFLAGS = -std=c++11 -lcudart -lcuda -lcusparse -lcusolver -lcublas
OBJS = helper.o assignment.o

all: assignment

assignment: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o assignment

helper.o: HelperFunctions.cu  HelperFunctions.cuh
	$(CC) $(CFLAGS) -c HelperFunctions.cu -o helper.o

assignment.o: assignment.cu
	$(CC) $(CFLAGS) -c assignment.cu -o assignment.o

clean:
	rm -f *~ *.o assignment

run_default:
	./assignment

run_zero:
	./assignment --a_row 8 --a_col 5 --b_row 5 --b_col 2 --eigen 10

run_one:
	./assignment --a_row 100 --a_col 200 --b_row 200 --b_col 150 --eigen 1024
