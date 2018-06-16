CC=nvcc
CFLAGS=-c -O3 -arch=sm_30
LDFLAGS=-lm -O3 -lcurand -lcublas -lcudart -lcufft -arch=sm_30
SOURCES=functions.cu kernels.cu main.cu read_files.cu distance.cu randgen.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=lgcp

all: $(SOURCES) $(EXECUTABLE)

$(OBJECTS): $(SOURCES) 
	$(CC) $(CFLAGS) $(SOURCES)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cu.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o 
