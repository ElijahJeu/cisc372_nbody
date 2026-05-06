FLAGS = -DDEBUG
LIBS = -lm
NVCC = nvcc

all: nbody

nbody: nbody.o compute.o
	$(NVCC) $(FLAGS) nbody.o compute.o -o nbody $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h compute.h
	$(NVCC) $(FLAGS) -c nbody.c

compute.o: compute.cu config.h vector.h compute.h
	$(NVCC) $(FLAGS) -c compute.cu

clean:
	rm -f *.o nbody 
