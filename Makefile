CC=gcc
CCFLAGS= -I
DEPS = MZC3D64.h khash.h

ODIR=obj
LDIR=lib

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

clean :
	rm app worker.o L.o 
