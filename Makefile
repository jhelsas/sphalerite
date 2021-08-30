# Makefile based on the link below:
# https://stackoverflow.com/questions/23854810/makefile-for-multiple-executables-with-folders
#
# It is not obvious at all how to create a makefile for this kind of project, which should be 
# quite common, or at least simple, but it wasn't. Finding the right source took me several days.
#
CC = gcc
#CC = clang
MPICC = mpicc
CFLAGS = -std=c11 
#CFLAGS += -Wall 
#CFLAGS += -O2
CFLAGS += -O3 
CFLAGS += -fPIE
CFLAGS += -fopenmp #=libomp
#CFLAGS += -fopt-info-vec-missed 
CFLAGS += -march=native
CFLAGS += -ffast-math 
#CFLAGS += -DAVX256
#CFLAGS += -funroll-loops
#CFLAGS += -fprofile-generate=profile/
#CFLAGS += -fopt-info
#CFLAGS += -fsave-optimization-record
#CFLAGS += -ftree-vectorize
CFLAGS += -I/usr/lib/gcc/x86_64-linux-gnu/10/include/
LDFLAGS = -lm -lgsl 
#LDFLAGS += -openmp
LDFLAGS += -fopenmp
#LDFLAGS += -fprofile-generate=profile/

#############################################

SRC_DIR_LIB = src
SRC_DIR_EXE = mains
SRC_DIR_TST = test
SRC_DIR_MED = medium
OBJ_DIR = obj
OBJ_DIR_LIB = $(OBJ_DIR)/lib
OBJ_DIR_EXE = $(OBJ_DIR)/exe
OBJ_DIR_TST = $(OBJ_DIR)/test
OBJ_DIR_MED = $(OBJ_DIR)/medium
BIN_DIR = bin
BIN_DIR_EXE = $(BIN_DIR)
BIN_DIR_TST = $(BIN_DIR)
BIN_DIR_MED = $(BIN_DIR)
HEAD_DIR = include

MPI_SRC_DIR_LIB = mpi/src
MPI_SRC_DIR_EXE = mpi/mains
MPI_SRC_DIR_TST = mpi/test
MPI_OBJ_DIR = obj
MPI_OBJ_DIR_LIB = $(MPI_OBJ_DIR)/lib
MPI_OBJ_DIR_EXE = $(MPI_OBJ_DIR)/exe
MPI_OBJ_DIR_TST = $(MPI_OBJ_DIR)/test
MPI_BIN_DIR = bin
MPI_BIN_DIR_EXE = $(MPI_BIN_DIR)
MPI_BIN_DIR_TST = $(MPI_BIN_DIR)
MPI_HEAD_DIR = mpi/include

#############################################

SRC_FILES_LIB = $(wildcard $(SRC_DIR_LIB)/*.c)
SRC_FILES_EXE = $(wildcard $(SRC_DIR_EXE)/*.c)
SRC_FILES_TST = $(wildcard $(SRC_DIR_TST)/*.c)
SRC_FILES_MED = $(wildcard $(SRC_DIR_MED)/*.c)
HEAD_FILES = $(wildcard $(HEAD_DIR)/*.h) # maybe it is not necessary because of INC_DIRS and _FLAGS

OBJ_FILES_LIB = $(patsubst $(SRC_DIR_LIB)/%.c,$(OBJ_DIR_LIB)/%.o,$(SRC_FILES_LIB))
OBJ_FILES_EXE = $(patsubst $(SRC_DIR_EXE)/%.c,$(OBJ_DIR_EXE)/%.o,$(SRC_FILES_EXE))
OBJ_FILES_TST = $(patsubst $(SRC_DIR_TST)/%.c,$(OBJ_DIR_TST)/%.o,$(SRC_FILES_TST))
OBJ_FILES_MED = $(patsubst $(SRC_DIR_MED)/%.c,$(OBJ_DIR_MED)/%.o,$(SRC_FILES_MED))

EXEC_FILES = $(patsubst $(SRC_DIR_EXE)/%.c,$(BIN_DIR_EXE)/%,$(SRC_FILES_EXE))
TEST_FILES = $(patsubst $(SRC_DIR_TST)/%.c,$(BIN_DIR_TST)/%,$(SRC_FILES_TST))
MED_FILES = $(patsubst $(SRC_DIR_MED)/%.c,$(BIN_DIR_MED)/%,$(SRC_FILES_MED))

# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS = $(shell find $(HEAD_DIR) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS = $(addprefix -I,$(INC_DIRS))

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
CPPFLAGS = $(INC_FLAGS)

###############

MPI_SRC_FILES_LIB = $(wildcard $(MPI_SRC_DIR_LIB)/*.c)
MPI_SRC_FILES_EXE = $(wildcard $(MPI_SRC_DIR_EXE)/*.c)
MPI_SRC_FILES_TST = $(wildcard $(MPI_SRC_DIR_TST)/*.c)
MPI_HEAD_FILES = $(wildcard $(MPI_HEAD_DIR)/*.h) # maybe it is not necessary because of INC_DIRS and _FLAGS

MPI_OBJ_FILES_LIB = $(patsubst $(MPI_SRC_DIR_LIB)/%.c,$(MPI_OBJ_DIR_LIB)/%.o,$(MPI_SRC_FILES_LIB))
MPI_OBJ_FILES_EXE = $(patsubst $(MPI_SRC_DIR_EXE)/%.c,$(MPI_OBJ_DIR_EXE)/%.o,$(MPI_SRC_FILES_EXE))
MPI_OBJ_FILES_TST = $(patsubst $(MPI_SRC_DIR_TST)/%.c,$(MPI_OBJ_DIR_TST)/%.o,$(MPI_SRC_FILES_TST))

MPI_EXEC_FILES = $(patsubst $(MPI_SRC_DIR_EXE)/%.c,$(MPI_BIN_DIR_EXE)/%,$(MPI_SRC_FILES_EXE))
MPI_TEST_FILES = $(patsubst $(MPI_SRC_DIR_TST)/%.c,$(MPI_BIN_DIR_TST)/%,$(MPI_SRC_FILES_TST))

# Every folder in ./src will need to be passed to GCC so that it can find header files
MPI_INC_DIRS = $(shell find $(MPI_HEAD_DIR) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
MPI_INC_FLAGS = $(addprefix -I,$(MPI_INC_DIRS))

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
MPICPPFLAGS = $(MPI_INC_FLAGS)

############################################

$(OBJ_DIR_LIB)/%.o: $(SRC_DIR_LIB)/%.c $(HEAD_FILES)
	mkdir -p $(dir $@)
	$(CC) -o $@ -c $<  $(CFLAGS) $(CPPFLAGS)

$(OBJ_DIR_EXE)/%.o: $(SRC_DIR_EXE)/%.c $(HEAD_FILES)
	mkdir -p $(dir $@)
	$(CC) -o $@ -c $<  $(CFLAGS) $(CPPFLAGS)

$(OBJ_DIR_TST)/%.o: $(SRC_DIR_TST)/%.c $(HEAD_FILES)
	mkdir -p $(dir $@)
	$(CC) -o $@ -c $<  $(CFLAGS) $(CPPFLAGS)

$(OBJ_DIR_MED)/%.o: $(SRC_DIR_MED)/%.c $(HEAD_FILES)
	mkdir -p $(dir $@)
	$(CC) -o $@ -c $<  $(CFLAGS) $(CPPFLAGS)

###########

$(BIN_DIR_EXE)/%: $(OBJ_DIR_EXE)/%.o $(OBJ_FILES_LIB)
	mkdir -p $(dir $@)
	$(CC) -o $@ -s $(subst $(BIN_DIR_EXE)/,$(OBJ_DIR_EXE)/,$@).o $(OBJ_FILES_LIB) $(LDFLAGS)

$(BIN_DIR_TST)/%: $(OBJ_DIR_TST)/%.o $(OBJ_FILES_LIB)
	mkdir -p $(dir $@)
	$(CC) -o $@ -s $(subst $(BIN_DIR_TST)/,$(OBJ_DIR_TST)/,$@).o $(OBJ_FILES_LIB) $(LDFLAGS)

$(BIN_DIR_MED)/medium/%: $(OBJ_DIR_MED)/%.o $(OBJ_FILES_LIB)
	mkdir -p $(dir $@)
	$(CC) -o $@ -s $(subst $(BIN_DIR_MED)/,$(OBJ_DIR_MED)/,$@).o $(OBJ_FILES_LIB) $(LDFLAGS)

###########################################

$(MPI_OBJ_DIR_LIB)/%.o: $(MPI_SRC_DIR_LIB)/%.c $(HEAD_FILES) $(MPI_HEAD_FILES)
	mkdir -p $(dir $@)
	$(MPICC) -o $@ -c $<  $(CFLAGS) $(CPPFLAGS) $(MPICPPFLAGS)

$(MPI_OBJ_DIR_EXE)/%.o: $(MPI_SRC_DIR_EXE)/%.c $(HEAD_FILES) $(MPI_HEAD_FILES)
	mkdir -p $(dir $@)
	$(MPICC) -o $@ -c $<  $(CFLAGS) $(CPPFLAGS) $(MPICPPFLAGS)

$(MPI_OBJ_DIR_TST)/%.o: $(MPI_SRC_DIR_TST)/%.c $(HEAD_FILES) $(MPI_HEAD_FILES)
	mkdir -p $(dir $@)
	$(MPICC) -o $@ -c $<  $(CFLAGS) $(CPPFLAGS) $(MPICPPFLAGS)

###########

$(MPI_BIN_DIR_EXE)/%: $(MPI_OBJ_DIR_EXE)/%.o $(OBJ_FILES_LIB) $(MPI_OBJ_FILES_LIB)
	mkdir -p $(dir $@)
	$(MPICC) -o $@ -s $(subst $(MPI_BIN_DIR_EXE)/,$(MPI_OBJ_DIR_EXE)/,$@).o $(OBJ_FILES_LIB) $(MPI_OBJ_FILES_LIB) $(LDFLAGS)

$(MPI_BIN_DIR_TST)/%: $(MPI_OBJ_DIR_TST)/%.o $(OBJ_FILES_LIB) $(MPI_OBJ_FILES_LIB)
	mkdir -p $(dir $@)
	$(MPICC) -o $@ -s $(subst $(MPI_BIN_DIR_TST)/,$(MPI_OBJ_DIR_TST)/,$@).o $(OBJ_FILES_LIB) $(MPI_OBJ_FILES_LIB) $(LDFLAGS)

############################################

tests: $(TEST_FILES) 

all: $(TEST_FILES) $(EXEC_FILES)

mpi_tests: $(MPI_TEST_FILES)

mpi_all: $(MPI_TEST_FILES) $(MPI_TEST_FILES)

medium: $(MEDIUM)

show: 
	@echo "Non MPI files:"
	@echo "SRC_DIR_LIB=$(SRC_DIR_LIB)"
	@echo "SRC_DIR_EXE=$(SRC_DIR_EXE)"
	@echo "SRC_DIR_TST=$(SRC_DIR_TST)"
	@echo "SRC_DIR_MED=$(SRC_DIR_MED)"

	@echo "\nOBJ_DIR=$(OBJ_DIR)"
	@echo "OBJ_DIR_LIB=$(OBJ_DIR_LIB)"
	@echo "OBJ_DIR_EXE=$(OBJ_DIR_EXE)"
	@echo "OBJ_DIR_TST=$(OBJ_DIR_TST)"
	@echo "OBJ_DIR_MED=$(OBJ_DIR_MED)"
	
	@echo "\nBIN_DIR=$(BIN_DIR)"
	@echo "BIN_DIR_EXE=$(BIN_DIR_EXE)"
	@echo "BIN_DIR_TST=$(BIN_DIR_TST)"
	@echo "BIN_DIR_MED=$(BIN_DIR_MED)"

	@echo "\nHEAD_DIR=$(HEAD_DIR)"

	@echo "\nSRC_FILES_LIB=$(SRC_FILES_LIB)"
	@echo "SRC_FILES_EXE=$(SRC_FILES_EXE)"
	@echo "SRC_FILES_TST=$(SRC_FILES_TST)"
	@echo "SRC_FILES_MED=$(SRC_FILES_MED)"

	@echo "\nHEAD_FILES=$(HEAD_FILES)"

	@echo "\nOBJ_FILES_LIB=$(OBJ_FILES_LIB)"
	@echo "OBJ_FILES_EXE=$(OBJ_FILES_EXE)"
	@echo "OBJ_FILES_TST=$(OBJ_FILES_TST)"
	@echo "OBJ_FILES_MED=$(OBJ_FILES_MED)"

	@echo "\nEXEC_FILES=$(EXEC_FILES)"
	@echo "TEST_FILES=$(TEST_FILES)"
	@echo "MED_FILES=$(MED_FILES)"

	@echo "\n-----------------\n"

	@echo "MPI related files:"
	@echo "MPI_SRC_DIR_LIB=$(MPI_SRC_DIR_LIB)"
	@echo "MPI_SRC_DIR_EXE=$(MPI_SRC_DIR_EXE)"
	@echo "MPI_SRC_DIR_TST=$(MPI_SRC_DIR_TST)"

	@echo "\nMPI_OBJ_DIR=$(MPI_OBJ_DIR)"
	@echo "MPI_OBJ_DIR_LIB=$(MPI_OBJ_DIR_LIB)"
	@echo "MPI_OBJ_DIR_EXE=$(MPI_OBJ_DIR_EXE)"
	@echo "MPI_OBJ_DIR_TST=$(MPI_OBJ_DIR_TST)"
	
	@echo "\nMPI_BIN_DIR=$(MPI_BIN_DIR)"
	@echo "MPI_BIN_DIR_EXE=$(MPI_BIN_DIR_EXE)"
	@echo "MPI_BIN_DIR_TST=$(MPI_BIN_DIR_TST)"

	@echo "\nMPI_HEAD_DIR=$(MPI_HEAD_DIR)"

	@echo "\nMPI_SRC_FILES_LIB=$(MPI_SRC_FILES_LIB)"
	@echo "MPI_SRC_FILES_EXE=$(MPI_SRC_FILES_EXE)"
	@echo "MPI_SRC_FILES_TST=$(MPI_SRC_FILES_TST)"

	@echo "\nMPI_HEAD_FILES=$(MPI_HEAD_FILES)"

	@echo "\nMPI_OBJ_FILES_LIB=$(MPI_OBJ_FILES_LIB)"
	@echo "MPI_OBJ_FILES_EXE=$(MPI_OBJ_FILES_EXE)"
	@echo "MPI_OBJ_FILES_TST=$(MPI_OBJ_FILES_TST)"

	@echo "\nMPI_EXEC_FILES=$(MPI_EXEC_FILES)"
	@echo "MPI_TEST_FILES=$(MPI_TEST_FILES)"

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)