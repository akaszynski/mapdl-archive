#include <stdint.h>

#if defined(__linux__) || defined(__APPLE__)
typedef int64_t vtk_int;
#else
typedef int32_t vtk_int;
#endif

int ans_to_vtk(
    const int,
    const int *,
    const int *,
    const int *,
    const int,
    const int *,
    vtk_int *,
    vtk_int *,
    uint8_t *);
