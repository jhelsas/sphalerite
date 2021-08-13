CC = gcc#
BUILD_DIR = ./build#
SRC_DIR = ./src#
TEST_DIR = ./src/tests#
CFLAGS = -Wall#
LDFLAGS = -lz -lm -lgsl -lcurl -pthread#

# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS = $(shell find $(SRC_DIR) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS = $(addprefix -I,$(INC_DIRS))

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
CPPFLAGS = $(INC_FLAGS) -MMD -MP

# Find all the C and C++ files we want to compile
#SRCS = $(shell find $(SRC_DIRS) -name *.c)

BASE = $(wildcard $(SRC_DIR)/*.c)
BASE_OBJS = $(BASE:%.c=$(BUILD_DIR)/%.o)

TESTS = $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS = $(TESTS:%.c=$(BUILD_DIR)/%.o) #$(TSTS:%.c=$(BUILD_DIR)/%.o)
#TSTS_EXE = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR),$(TSTS))

# Build step for C source
$(BUILD_DIR)/%.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

#$(BUILD_DIR)/$(TEST_DIR)/%.o: $(TEST_DIR)/%.c
#	mkdir -p $(dir $@)
#	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/sph_linked_list_test: $(BASE_OBJS) #$(BUILD_DIR)/$(TEST_DIR)/sph_linked_list.o
	echo $(TESTS)
	#echo $(TSTS)
	#$(CC) $(BASE_OBJS) $(BUILD_DIR)/$(TEST_DIR)/sph_linked_list.o -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)