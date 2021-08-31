# Makefile based on the link below:
# https://stackoverflow.com/questions/23854810/makefile-for-multiple-executables-with-folders
#
# It is not obvious at all how to create a makefile for this kind of project, which should be 
# quite common, or at least simple, but it wasn't. Finding the right source took me several days.
#
CC = gcc
MPICC = mpicc
CFLAGS0 = -std=c11 -Wall -O0
CFLAGS1 = -std=c11 -Wall -O2 
CFLAGS2 = -std=c11 -Wall -O3 
CFLAGS3 = -std=c11 -Wall -O3 -fopenmp
CFLAGS4 = -std=c11 -Wall -O3 -fopenmp -ffast-math
CFLAGS5 = -std=c11 -Wall -O3 -fopenmp -ffast-math -march=native
CFLAGS6 = -std=c11 -Wall -O3 -march=native
CFLAGS = $(CFLAGS5)

LDFLAGS = -lm -lgsl 
LDFLAGS += -fopenmp

#############################################

SRC_DIR_LIB = src
SRC_DIR_EXE = mains
SRC_DIR_TST = test
OBJ_DIR = obj
OBJ_DIR_LIB = $(OBJ_DIR)/lib
OBJ_DIR_EXE = $(OBJ_DIR)/exe
OBJ_DIR_TST = $(OBJ_DIR)/test
BIN_DIR = bin
BIN_DIR_EXE = $(BIN_DIR)
BIN_DIR_TST = $(BIN_DIR)
HEAD_DIR = include

#############################################

SRC_FILES_LIB = $(wildcard $(SRC_DIR_LIB)/*.c)
SRC_FILES_EXE = $(wildcard $(SRC_DIR_EXE)/*.c)
SRC_FILES_TST = $(wildcard $(SRC_DIR_TST)/*.c)

HEAD_FILES = $(wildcard $(HEAD_DIR)/*.h) # maybe it is not necessary because of INC_DIRS and _FLAGS

OBJ_FILES_LIB = $(patsubst $(SRC_DIR_LIB)/%.c,$(OBJ_DIR_LIB)/%.o,$(SRC_FILES_LIB))
OBJ_FILES_EXE = $(patsubst $(SRC_DIR_EXE)/%.c,$(OBJ_DIR_EXE)/%.o,$(SRC_FILES_EXE))
OBJ_FILES_TST = $(patsubst $(SRC_DIR_TST)/%.c,$(OBJ_DIR_TST)/%.o,$(SRC_FILES_TST))

EXEC_FILES = $(patsubst $(SRC_DIR_EXE)/%.c,$(BIN_DIR_EXE)/%,$(SRC_FILES_EXE))
TEST_FILES = $(patsubst $(SRC_DIR_TST)/%.c,$(BIN_DIR_TST)/%,$(SRC_FILES_TST))

# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS = $(shell find $(HEAD_DIR) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS = $(addprefix -I,$(INC_DIRS))

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
CPPFLAGS = $(INC_FLAGS)

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

###########

$(BIN_DIR_EXE)/%: $(OBJ_DIR_EXE)/%.o $(OBJ_FILES_LIB)
	mkdir -p $(dir $@)
	$(CC) -o $@ -s $(subst $(BIN_DIR_EXE)/,$(OBJ_DIR_EXE)/,$@).o $(OBJ_FILES_LIB) $(LDFLAGS)

$(BIN_DIR_TST)/%: $(OBJ_DIR_TST)/%.o $(OBJ_FILES_LIB)
	mkdir -p $(dir $@)
	$(CC) -o $@ -s $(subst $(BIN_DIR_TST)/,$(OBJ_DIR_TST)/,$@).o $(OBJ_FILES_LIB) $(LDFLAGS)

###########################################

main: $(EXEC_FILES) 

tests: $(TEST_FILES) 

all: $(TEST_FILES) $(EXEC_FILES)

show: 
	@echo "Non MPI files:"
	@echo "SRC_DIR_LIB=$(SRC_DIR_LIB)"
	@echo "SRC_DIR_EXE=$(SRC_DIR_EXE)"
	@echo "SRC_DIR_TST=$(SRC_DIR_TST)"

	@echo "\nOBJ_DIR=$(OBJ_DIR)"
	@echo "OBJ_DIR_LIB=$(OBJ_DIR_LIB)"
	@echo "OBJ_DIR_EXE=$(OBJ_DIR_EXE)"
	@echo "OBJ_DIR_TST=$(OBJ_DIR_TST)"
	
	@echo "\nBIN_DIR=$(BIN_DIR)"
	@echo "BIN_DIR_EXE=$(BIN_DIR_EXE)"
	@echo "BIN_DIR_TST=$(BIN_DIR_TST)"

	@echo "\nHEAD_DIR=$(HEAD_DIR)"

	@echo "\nSRC_FILES_LIB=$(SRC_FILES_LIB)"
	@echo "SRC_FILES_EXE=$(SRC_FILES_EXE)"
	@echo "SRC_FILES_TST=$(SRC_FILES_TST)"

	@echo "\nHEAD_FILES=$(HEAD_FILES)"

	@echo "\nOBJ_FILES_LIB=$(OBJ_FILES_LIB)"
	@echo "OBJ_FILES_EXE=$(OBJ_FILES_EXE)"
	@echo "OBJ_FILES_TST=$(OBJ_FILES_TST)"

	@echo "\nEXEC_FILES=$(EXEC_FILES)"
	@echo "TEST_FILES=$(TEST_FILES)"

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)