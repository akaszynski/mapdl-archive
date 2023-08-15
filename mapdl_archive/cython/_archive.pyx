# cython: language_level=3
# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: embedsignature=True

import ctypes

from libc.math cimport abs
from libc.stdio cimport FILE, fclose, fdopen, fopen, fwrite

import numpy as np

cimport numpy as np

ctypedef unsigned char uint8_t

cdef extern from 'archive.h' nogil:
    int write_nblock(FILE*, const int, const int, const int*, const double*,
                     const double*, int)
    int write_nblock_float(FILE*, const int, const int, const int*, const float*,
                           const float*, int)
    int write_eblock(FILE*, const int, const int*, const int*, const int*,
                     const int*, const int*, const uint8_t*, const int*,
                     const int*, const int*, const int*);

from libc.stdint cimport int32_t, int64_t

# VTK celltypes
ctypedef unsigned char uint8
cdef uint8 VTK_HEXAHEDRON = 12
cdef uint8 VTK_PYRAMID = 14
cdef uint8 VTK_TETRA = 10
cdef uint8 VTK_WEDGE = 13
cdef uint8 VTK_QUADRATIC_TRIANGLE = 22
cdef uint8 VTK_QUADRATIC_QUAD = 23
cdef uint8 VTK_QUADRATIC_HEXAHEDRON = 25
cdef uint8 VTK_QUADRATIC_PYRAMID = 27
cdef uint8 VTK_QUADRATIC_TETRA = 24
cdef uint8 VTK_QUADRATIC_WEDGE = 26


cdef extern from "stdio.h":
    FILE *fdopen(int, const char *)


def py_write_nblock(filename, const int [::1] node_id, int max_node_id,
                    const double [:, ::1] pos, const double [:, ::1] angles,
                    mode='w'):
    """Write a node block to a file.

    Parameters
    ----------
    fid : _io.TextIOWrapper
        Opened Python file object.

    node_id : np.ndarray
        Array of node ids.

    pos : np.ndarray
        Double array of node coordinates

    angles : np.ndarray, optional


    """
    # attach the stream to the python file
    cdef FILE* cfile = fopen(filename.encode(), mode.encode())

    cdef int n_nodes = pos.shape[0]

    cdef int has_angles = 0
    if angles.size == pos.size:
        has_angles = 1
    else:
        angles = np.zeros((1, 1), np.double)
    write_nblock(cfile, n_nodes, max_node_id, &node_id[0], &pos[0, 0],
                 &angles[0, 0], has_angles);
    fclose(cfile)


def py_write_nblock_float(filename, const int [::1] node_id, int max_node_id,
                          const float [:, ::1] pos, const float [:, ::1] angles,
                          mode='w'):
    """Write a float32 node block to a file.

    Parameters
    ----------
    fid : _io.TextIOWrapper
        Opened Python file object.

    node_id : np.ndarray
        Array of node ids.

    pos : np.float32 np.ndarray
        Double array of node coordinates

    angles : np.ndarray, optional

    """
    # attach the stream to the python file
    cdef FILE* cfile = fopen(filename.encode(), mode.encode())

    cdef int n_nodes = pos.shape[0]
    cdef int has_angles = 0
    if angles.size == pos.size:
        has_angles = 1
    else:
        angles = np.zeros((1, 1), np.float32)
    write_nblock_float(cfile, n_nodes, max_node_id, &node_id[0], &pos[0, 0],
                       &angles[0, 0], has_angles);
    fclose(cfile)


def py_write_eblock(
    filename,
    const int [::1] elem_id,
    const int [::1] etype,
    const int [::1] mtype,
    const int [::1] rcon,
    const int [::1] elem_nnodes,
    const int [::1] cells,
    const int [::1] offset,
    const uint8_t [::1] celltypes,
    const int [::1] typenum,
    const int [::1] nodenum,
    mode='w'):
    cdef FILE* cfile = fopen(filename.encode(), mode.encode())
    write_eblock(
        cfile,
        celltypes.size,
        &elem_id[0],
        &etype[0],
        &mtype[0],
        &rcon[0],
        &elem_nnodes[0],
        &celltypes[0],
        &offset[0],
        &cells[0],
        &typenum[0],
        &nodenum[0]
    )
    fclose(cfile)


def cmblock_items_from_array(int [::1] array):
    """Given a list of items, convert to a ANSYS formatted CMBLOCK.

    For example ``1, 2, 3, 4, 8``

    will be converted to

    ``1, -4, 8``

    Where the -4 indicates all the items between 1 and -4.
    """

    # first, verify items in array are sorted
    cdef int i
    cdef is_sorted = 1
    for i in range(array.size - 1):
        if array[i] > array[i + 1]:
            is_sorted = 0
            break

    if not is_sorted:
        array = np.unique(array)

    cdef int [::1] items = np.empty_like(array)
    items[0] = array[0]

    cdef int c = 1
    cdef int in_list = 0
    for i in range(array.size - 1):
        # check if part of a range
        if array[i + 1] - array[i] == 1:
            in_list = 1
        elif array[i + 1] - array[i] > 1:
            if in_list:
                items[c] = -array[i]; c += 1
                items[c] = array[i + 1]; c += 1
            else:
                items[c] = array[i + 1]; c += 1
            in_list = 0

    # catch if last item is part of a list
    if items[c - 1] != abs(array[array.size - 1]):
        items[c] = -array[array.size - 1]; c += 1

    return np.array(items[:c])


cdef inline void relax_mid_tri(int64_t [::1] cellarr, int c, double [:, ::1] pts,
                               double rfac):
    """
    Resets the midside nodes of the quadratic quad starting at index c.

    relaxation factor rfac

    The ordering of the three points defining the cell is point ids (0-2,3-5)
    where id #3 is the midedge node between points (0,1); id #4 is the midedge
    node between points (1,2); and id #5 is the midedge node between points
    (2,0).

    """
    cdef int ind0 = cellarr[c + 0]
    cdef int ind1 = cellarr[c + 1]
    cdef int ind2 = cellarr[c + 2]
    cdef int ind3 = cellarr[c + 3]
    cdef int ind4 = cellarr[c + 4]
    cdef int ind5 = cellarr[c + 5]

    cdef int j

    for j in range(3):
        pts[ind3, j] = pts[ind3, j]*(1 - rfac) + (pts[ind0, j] + pts[ind1, j])*0.5*rfac
        pts[ind4, j] = pts[ind4, j]*(1 - rfac) + (pts[ind1, j] + pts[ind2, j])*0.5*rfac
        pts[ind5, j] = pts[ind5, j]*(1 - rfac) + (pts[ind2, j] + pts[ind0, j])*0.5*rfac


cdef inline void relax_mid_quad(int64_t [::1] cellarr, int c, double [:, ::1] pts,
                                double rfac):
    """
    Resets the midside nodes of the quadratic quad starting at index c.

    relaxation factor rfac

    midedge nodes between
    (0,1), (1,2), (1,2), (0,3)

    """
    cdef int ind0 = cellarr[c + 0]
    cdef int ind1 = cellarr[c + 1]
    cdef int ind2 = cellarr[c + 2]
    cdef int ind3 = cellarr[c + 3]
    cdef int ind4 = cellarr[c + 4]
    cdef int ind5 = cellarr[c + 5]
    cdef int ind6 = cellarr[c + 6]
    cdef int ind7 = cellarr[c + 7]

    cdef int j

    for j in range(3):
        pts[ind4, j] = pts[ind4, j]*(1 - rfac) + (pts[ind0, j] + pts[ind1, j])*0.5*rfac
        pts[ind5, j] = pts[ind5, j]*(1 - rfac) + (pts[ind1, j] + pts[ind2, j])*0.5*rfac
        pts[ind6, j] = pts[ind6, j]*(1 - rfac) + (pts[ind2, j] + pts[ind3, j])*0.5*rfac
        pts[ind7, j] = pts[ind7, j]*(1 - rfac) + (pts[ind0, j] + pts[ind3, j])*0.5*rfac


cdef inline void relax_mid_tet(int64_t [::1] cellarr, int c, double [:, ::1] pts,
                               double rfac):
    """
    Resets the midside nodes of the tetrahedral starting at index c

    relaxation factor rfac

    midedge nodes between
    (0,1), (1,2), (2,0), (0,3), (1,3), and (2,3)

    """

    cdef int ind0 = cellarr[c + 0]
    cdef int ind1 = cellarr[c + 1]
    cdef int ind2 = cellarr[c + 2]
    cdef int ind3 = cellarr[c + 3]
    cdef int ind4 = cellarr[c + 4]
    cdef int ind5 = cellarr[c + 5]
    cdef int ind6 = cellarr[c + 6]
    cdef int ind7 = cellarr[c + 7]
    cdef int ind8 = cellarr[c + 8]
    cdef int ind9 = cellarr[c + 9]

    cdef int j

    for j in range(3):
        pts[ind4, j] = pts[ind4, j]*(1 - rfac) + (pts[ind0, j] + pts[ind1, j])*0.5*rfac
        pts[ind5, j] = pts[ind5, j]*(1 - rfac) + (pts[ind1, j] + pts[ind2, j])*0.5*rfac
        pts[ind6, j] = pts[ind6, j]*(1 - rfac) + (pts[ind2, j] + pts[ind0, j])*0.5*rfac
        pts[ind7, j] = pts[ind7, j]*(1 - rfac) + (pts[ind0, j] + pts[ind3, j])*0.5*rfac
        pts[ind8, j] = pts[ind8, j]*(1 - rfac) + (pts[ind1, j] + pts[ind3, j])*0.5*rfac
        pts[ind9, j] = pts[ind9, j]*(1 - rfac) + (pts[ind2, j] + pts[ind3, j])*0.5*rfac


cdef inline void relax_mid_pyr(int64_t [::1] cellarr, int c, double [:, ::1] pts,
                               double rfac):
    """

    5 (0, 1)
    6 (1, 2)
    7 (2, 3)
    8 (3, 0)
    9 (0, 4)
    10(1, 4)
    11(2, 4)
    12(3, 4)

    """

    cdef int ind0 = cellarr[c + 0]
    cdef int ind1 = cellarr[c + 1]
    cdef int ind2 = cellarr[c + 2]
    cdef int ind3 = cellarr[c + 3]
    cdef int ind4 = cellarr[c + 4]
    cdef int ind5 = cellarr[c + 5]
    cdef int ind6 = cellarr[c + 6]
    cdef int ind7 = cellarr[c + 7]
    cdef int ind8 = cellarr[c + 8]
    cdef int ind9 = cellarr[c + 9]
    cdef int ind10= cellarr[c + 10]
    cdef int ind11= cellarr[c + 11]
    cdef int ind12= cellarr[c + 12]

    cdef int j

    for j in range(3):
        pts[ind5, j]  = pts[ind5,  j]*(1 - rfac) + (pts[ind0, j] + pts[ind1, j])*0.5*rfac
        pts[ind6, j]  = pts[ind6,  j]*(1 - rfac) + (pts[ind1, j] + pts[ind2, j])*0.5*rfac
        pts[ind7, j]  = pts[ind7,  j]*(1 - rfac) + (pts[ind2, j] + pts[ind3, j])*0.5*rfac
        pts[ind8, j]  = pts[ind8,  j]*(1 - rfac) + (pts[ind3, j] + pts[ind0, j])*0.5*rfac
        pts[ind9, j]  = pts[ind9,  j]*(1 - rfac) + (pts[ind0, j] + pts[ind4, j])*0.5*rfac
        pts[ind10, j] = pts[ind10, j]*(1 - rfac) + (pts[ind1, j] + pts[ind4, j])*0.5*rfac
        pts[ind11, j] = pts[ind11, j]*(1 - rfac) + (pts[ind2, j] + pts[ind4, j])*0.5*rfac
        pts[ind12, j] = pts[ind12, j]*(1 - rfac) + (pts[ind3, j] + pts[ind4, j])*0.5*rfac


cdef inline void relax_mid_weg(int64_t [::1] cellarr, int c, double [:, ::1] pts,
                               double rfac):
    """

    6  (0,1)
    7  (1,2)
    8  (2,0)
    9  (3,4)
    10 (4,5)
    11 (5,3)
    12 (0,3)
    13 (1,4)
    14 (2,5)
    """
    cdef int ind0 = cellarr[c + 0]
    cdef int ind1 = cellarr[c + 1]
    cdef int ind2 = cellarr[c + 2]
    cdef int ind3 = cellarr[c + 3]
    cdef int ind4 = cellarr[c + 4]
    cdef int ind5 = cellarr[c + 5]
    cdef int ind6 = cellarr[c + 6]
    cdef int ind7 = cellarr[c + 7]
    cdef int ind8 = cellarr[c + 8]
    cdef int ind9 = cellarr[c + 9]
    cdef int ind10= cellarr[c + 10]
    cdef int ind11= cellarr[c + 11]
    cdef int ind12= cellarr[c + 12]
    cdef int ind13= cellarr[c + 13]
    cdef int ind14= cellarr[c + 14]

    cdef int j

    for j in range(3):
        pts[ind6, j]  = pts[ind6,  j]*(1 - rfac) + (pts[ind0, j] + pts[ind1, j])*0.5*rfac
        pts[ind7, j]  = pts[ind7,  j]*(1 - rfac) + (pts[ind1, j] + pts[ind2, j])*0.5*rfac
        pts[ind8, j]  = pts[ind8,  j]*(1 - rfac) + (pts[ind2, j] + pts[ind0, j])*0.5*rfac
        pts[ind9, j]  = pts[ind9,  j]*(1 - rfac) + (pts[ind3, j] + pts[ind4, j])*0.5*rfac
        pts[ind10, j] = pts[ind10, j]*(1 - rfac) + (pts[ind4, j] + pts[ind5, j])*0.5*rfac
        pts[ind11, j] = pts[ind11, j]*(1 - rfac) + (pts[ind5, j] + pts[ind3, j])*0.5*rfac
        pts[ind12, j] = pts[ind12, j]*(1 - rfac) + (pts[ind0, j] + pts[ind3, j])*0.5*rfac
        pts[ind13, j] = pts[ind13, j]*(1 - rfac) + (pts[ind1, j] + pts[ind4, j])*0.5*rfac
        pts[ind14, j] = pts[ind14, j]*(1 - rfac) + (pts[ind2, j] + pts[ind5, j])*0.5*rfac


cdef inline void relax_mid_hex(int64_t [::1] cellarr, int c, double [:, ::1] pts,
                               double rfac):

    """

    8  (0,1)
    9  (1,2)
    10 (2,3)
    11 (3,0)
    12 (4,5)
    13 (5,6)
    14 (6,7)
    15 (7,4)
    16 (0,4)
    17 (1,5)
    18 (2,6)
    19 (3,7)

    """

    cdef int ind0 = cellarr[c + 0]
    cdef int ind1 = cellarr[c + 1]
    cdef int ind2 = cellarr[c + 2]
    cdef int ind3 = cellarr[c + 3]
    cdef int ind4 = cellarr[c + 4]
    cdef int ind5 = cellarr[c + 5]
    cdef int ind6 = cellarr[c + 6]
    cdef int ind7 = cellarr[c + 7]
    cdef int ind8 = cellarr[c + 8]
    cdef int ind9 = cellarr[c + 9]
    cdef int ind10= cellarr[c + 10]
    cdef int ind11= cellarr[c + 11]
    cdef int ind12= cellarr[c + 12]
    cdef int ind13= cellarr[c + 13]
    cdef int ind14= cellarr[c + 14]
    cdef int ind15= cellarr[c + 15]
    cdef int ind16= cellarr[c + 16]
    cdef int ind17= cellarr[c + 17]
    cdef int ind18= cellarr[c + 18]
    cdef int ind19= cellarr[c + 19]

    cdef int j

    for j in range(3):
        pts[ind8, j]  = pts[ind8,  j]*(1 - rfac) + (pts[ind0, j] + pts[ind1, j])*0.5*rfac
        pts[ind9, j]  = pts[ind9,  j]*(1 - rfac) + (pts[ind1, j] + pts[ind2, j])*0.5*rfac
        pts[ind10, j] = pts[ind10, j]*(1 - rfac) + (pts[ind2, j] + pts[ind3, j])*0.5*rfac
        pts[ind11, j] = pts[ind11, j]*(1 - rfac) + (pts[ind3, j] + pts[ind0, j])*0.5*rfac
        pts[ind12, j] = pts[ind12, j]*(1 - rfac) + (pts[ind4, j] + pts[ind5, j])*0.5*rfac
        pts[ind13, j] = pts[ind13, j]*(1 - rfac) + (pts[ind5, j] + pts[ind6, j])*0.5*rfac
        pts[ind14, j] = pts[ind14, j]*(1 - rfac) + (pts[ind6, j] + pts[ind7, j])*0.5*rfac
        pts[ind15, j] = pts[ind15, j]*(1 - rfac) + (pts[ind7, j] + pts[ind4, j])*0.5*rfac
        pts[ind16, j] = pts[ind16, j]*(1 - rfac) + (pts[ind0, j] + pts[ind4, j])*0.5*rfac
        pts[ind17, j] = pts[ind17, j]*(1 - rfac) + (pts[ind1, j] + pts[ind5, j])*0.5*rfac
        pts[ind18, j] = pts[ind18, j]*(1 - rfac) + (pts[ind2, j] + pts[ind6, j])*0.5*rfac
        pts[ind19, j] = pts[ind19, j]*(1 - rfac) + (pts[ind3, j] + pts[ind7, j])*0.5*rfac


def reset_midside(int64_t [::1] cellarr, uint8 [::1] celltypes,
                  int64_t [::1] offset, double [:, ::1] pts):
    """Resets positions of midside nodes to directly between edge nodes.

    Parameters
    ----------
    cellarr : int64_t [::1]
        VTK formatted cell array

    celltypes : uint8 [::1]
        VTK formatted celltype array

    pts : double [:, ::1]
        3D double point array.

    """
    cdef int i
    cdef int ncells = offset.size
    cdef uint8 celltype

    for i in range(ncells):
        celltype = celltypes[i]
        if celltype == VTK_QUADRATIC_TRIANGLE:
            relax_mid_tri(cellarr, offset[i] + 1, pts, 1)
        if celltype == VTK_QUADRATIC_QUAD:
            relax_mid_quad(cellarr, offset[i] + 1, pts, 1)
        elif celltype == VTK_QUADRATIC_TETRA:
            relax_mid_tet(cellarr, offset[i] + 1, pts, 1)
        elif celltype == VTK_QUADRATIC_PYRAMID:
            relax_mid_pyr(cellarr, offset[i] + 1, pts, 1)
        elif celltype == VTK_QUADRATIC_WEDGE:
            relax_mid_weg(cellarr, offset[i] + 1, pts, 1)
        elif celltype == VTK_QUADRATIC_HEXAHEDRON:
            relax_mid_hex(cellarr, offset[i] + 1, pts, 1)
