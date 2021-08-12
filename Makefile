CC=gcc

IDIR=include 
ODIR=obj
LDIR=lib
CCFLAGS= -I$(IDIR)

SRC = SRC = $(shell find . -name *.cc)

LIBS= -lm -lgsl

_DEPS = sph_data_types.h sph_linked_list.h sph_compute.h MZC3D64.h khash.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = sph_compute.o sph_linked_list.o comptue_density_3d_test.o sph_linked_list_test.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

sph_linked_list_test: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

all: $(BIN)

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~