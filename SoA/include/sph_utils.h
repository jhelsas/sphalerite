#ifndef SPH_UTILS_H
#define SPH_UTILS_H

#include <stdbool.h>
#include "sph_data_types.h"

int arg_parse(int argc, char **argv, int64_t *N, double *h, long int *seed, int *runs, bool *run_seed, linkedListBox *box);

#endif
