#include <stdio.h>

#ifdef __linux__
#include <stdint.h>
#endif

int read_nblock(char *, int *, double *, int, int *, int, int64_t *);
int read_eblock(char *, int *, int *, int, int, int64_t *);
int read_nblock_from_nwrite(char *, int *, double *, int);
int write_array_ascii(const char *, const double *, int nvalues);
int read_eblock_cfile(FILE *cfile, int *elem_off, int *elem, int nelem);
int read_nblock_cfile(
    FILE *cfile, int *nnum, double *nodes, int nnodes, int *d_size, int f_size);
