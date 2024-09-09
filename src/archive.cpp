#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "array_support.h"

// necessary for ubuntu build on azure
#ifdef __linux__
#include <stdint.h>
#endif

// VTK cell types
#define VTK_TRIANGLE 5
#define VTK_QUAD 9
#define VTK_TETRA 10
#define VTK_VOXEL 11
#define VTK_HEXAHEDRON 12
#define VTK_WEDGE 13
#define VTK_PYRAMID 14
#define VTK_QUADRATIC_TETRA 24
#define VTK_QUADRATIC_HEXAHEDRON 25
#define VTK_QUADRATIC_WEDGE 26
#define VTK_QUADRATIC_PYRAMID 27

// character to fill when the number width overflows specified width
char const OVERFLOW_FILL_CHAR = '*';

char *create_format_str(int fields, int sig_digits) {
    static char format_str[64]; // Buffer for our format string
    char field_format[16];      // Buffer for a single field format string

    // Leaves one whitespace, sign, one before the decimal, decimal, and four
    // for the scientific notation
    // For example: " -1.233333333333E+09"
    int total_char = sig_digits + 7;

    // Create a single field format string
    sprintf(field_format, "%%%d.%dE", total_char, sig_digits);

    // Start the format string with the fixed part
    strcpy(format_str, "%8d       0       0");

    // Add the field format string for each field
    for (int i = 0; i < fields; i++) {
        strcat(format_str, field_format);
    }

    // Add a newline to the end of the format string
    strcat(format_str, "\n");

    return format_str;
}

// Write node IDs, node coordinates, and angles to file as a NBLOCK
template <typename T>
void WriteNblock(
    std::string &filename,
    const int max_node_id,
    const NDArray<const int, 1> node_id_arr,
    const NDArray<const double, 2> nodes_arr,
    const NDArray<const double, 2> angles_arr,
    int sig_digits,
    std::string &mode) {

    FILE *file = fopen(filename.c_str(), mode.c_str());
    if (file == nullptr) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    const int n_nodes = node_id_arr.size();
    bool has_angles = angles_arr.size() > 0;

    // Header
    // Tell ANSYS to start reading the node block with 6 fields,
    // associated with a solid, the maximum node number and the number
    // of lines in the node block
    fprintf(file, "/PREP7\n");
    fprintf(file, "NBLOCK,6,SOLID,%10d,%10d\n", max_node_id, n_nodes);
    fprintf(file, "(3i8,6e%d.%d)\n", sig_digits + 7, sig_digits);

    char *format_str;
    const int *node_id = node_id_arr.data();
    const double *nodes = nodes_arr.data();
    const double *angles = angles_arr.data();

    int i;
    if (has_angles) {
        format_str = create_format_str(6, sig_digits);
        for (i = 0; i < n_nodes; i++) {
            fprintf(
                file,
                format_str,
                node_id[i],
                nodes[i * 3 + 0],
                nodes[i * 3 + 1],
                nodes[i * 3 + 2],
                angles[i * 3 + 0],
                angles[i * 3 + 1],
                angles[i * 3 + 2]);
        }
    } else {
        format_str = create_format_str(3, sig_digits);
        for (i = 0; i < n_nodes; i++) {
            fprintf(
                file,
                format_str,
                node_id[i],
                nodes[i * 3 + 0],
                nodes[i * 3 + 1],
                nodes[i * 3 + 2]);
        }
    }

    fprintf(file, "N,R5.3,LOC,       -1,\n");

    fclose(file);
}

// Write an ANSYS EBLOCK to file
void WriteEblock(
    const std::string &filename,
    const int n_elem,                              // number of elements
    const NDArray<const int, 1> elem_id_arr,       // element ID array
    const NDArray<const int, 1> etype_arr,         // element type ID array
    const NDArray<const int, 1> mtype_arr,         // material ID array
    const NDArray<const int, 1> rcon_arr,          // real constant ID array
    const NDArray<const int, 1> elem_nnodes_arr,   // number of nodes per element
    const NDArray<const uint8_t, 1> celltypes_arr, // VTK celltypes array
    const NDArray<const int64_t, 1> offset_arr,    // VTK offset array
    const NDArray<const int64_t, 1> cells_arr,     // VTK cell connectivity array
    const NDArray<const int, 1> typenum_arr, // ANSYS type number (e.g. 187 for SOLID187)
    const NDArray<const int, 1> nodenum_arr, // ANSYS node numbering
    std::string &mode) {

    FILE *file = fopen(filename.c_str(), mode.c_str());

    const int *elem_id = elem_id_arr.data();
    const int *etype = etype_arr.data();
    const int *mtype = mtype_arr.data();
    const int *rcon = rcon_arr.data();
    const int *elem_nnodes = elem_nnodes_arr.data();
    const uint8_t *celltypes = celltypes_arr.data();
    const int64_t *offset = offset_arr.data();
    const int64_t *cells = cells_arr.data();
    const int *typenum = typenum_arr.data();
    const int *nodenum = nodenum_arr.data();

    // Write header
    fprintf(file, "EBLOCK,19,SOLID,%10d,%10d\n", elem_id[n_elem - 1], n_elem);
    fprintf(file, "(19i8)\n");

    int c; // position within offset array
    for (int i = 0; i < n_elem; i++) {
        c = offset[i];

        // Write cell info
        fprintf(
            file,
            "%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d",
            mtype[i],       // Field 1: material reference number
            etype[i],       // Field 2: element type number
            rcon[i],        // Field 3: real constant reference number
            1,              // Field 4: section number
            0,              // Field 5: element coordinate system
            0,              // Field 6: Birth/death flag
            0,              // Field 7:
            0,              // Field 8:
            elem_nnodes[i], // Field 9: Number of nodes
            0,              // Field 10: Not Used
            elem_id[i]);    // Field 11: Element number

        // Write element nodes
        switch (celltypes[i]) {
        case VTK_QUADRATIC_TETRA:
            if (typenum[i] == 187) {
                fprintf(
                    file,
                    "%8d%8d%8d%8d%8d%8d%8d%8d\n%8d%8d\n",
                    nodenum[cells[c + 0]],  // 0, I
                    nodenum[cells[c + 1]],  // 1, J
                    nodenum[cells[c + 2]],  // 2, K
                    nodenum[cells[c + 3]],  // 3, L
                    nodenum[cells[c + 4]],  // 4, M
                    nodenum[cells[c + 5]],  // 5, N
                    nodenum[cells[c + 6]],  // 6, O
                    nodenum[cells[c + 7]],  // 7, P
                    nodenum[cells[c + 8]],  // 8, Q
                    nodenum[cells[c + 9]]); // 9, R
            } else {
                // Using SOLID186-like format
                fprintf(
                    file,
                    "%8d%8d%8d%8d%8d%8d%8d%8d\n%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d\n",
                    nodenum[cells[c + 0]],  // 0,  I
                    nodenum[cells[c + 1]],  // 1,  J
                    nodenum[cells[c + 2]],  // 2,  K
                    nodenum[cells[c + 2]],  // 3,  L (duplicate of K)
                    nodenum[cells[c + 3]],  // 4,  M
                    nodenum[cells[c + 3]],  // 5,  N (duplicate of M)
                    nodenum[cells[c + 3]],  // 6,  O (duplicate of M)
                    nodenum[cells[c + 3]],  // 7,  P (duplicate of M)
                    nodenum[cells[c + 4]],  // 8,  Q
                    nodenum[cells[c + 5]],  // 9,  R
                    nodenum[cells[c + 3]],  // 10, S (duplicate of K)
                    nodenum[cells[c + 6]],  // 11, T
                    nodenum[cells[c + 3]],  // 12, U (duplicate of M)
                    nodenum[cells[c + 3]],  // 13, V (duplicate of M)
                    nodenum[cells[c + 3]],  // 14, W (duplicate of M)
                    nodenum[cells[c + 3]],  // 15, X (duplicate of M)
                    nodenum[cells[c + 7]],  // 16, Y
                    nodenum[cells[c + 8]],  // 17, Z
                    nodenum[cells[c + 9]],  // 18, A
                    nodenum[cells[c + 9]]); // 19, B (duplicate of A)
            }
            break;
        case VTK_TETRA: // point
            fprintf(
                file,
                "%8d%8d%8d%8d%8d%8d%8d%8d\n",
                nodenum[cells[c + 0]],  // 0,  I
                nodenum[cells[c + 1]],  // 1,  J
                nodenum[cells[c + 2]],  // 2,  K
                nodenum[cells[c + 2]],  // 3,  L (duplicate of K)
                nodenum[cells[c + 3]],  // 4,  M
                nodenum[cells[c + 3]],  // 5,  N (duplicate of M)
                nodenum[cells[c + 3]],  // 6,  O (duplicate of M)
                nodenum[cells[c + 3]]); // 7,  P (duplicate of M)
            break;
        case VTK_WEDGE:
            fprintf(
                file,
                "%8d%8d%8d%8d%8d%8d%8d%8d\n",
                nodenum[cells[c + 2]],  // 0,  I
                nodenum[cells[c + 1]],  // 1,  J
                nodenum[cells[c + 0]],  // 2,  K
                nodenum[cells[c + 0]],  // 3,  L (duplicate of K)
                nodenum[cells[c + 5]],  // 4,  M
                nodenum[cells[c + 4]],  // 5,  N
                nodenum[cells[c + 3]],  // 6,  O
                nodenum[cells[c + 3]]); // 7,  P (duplicate of O)
            break;
        case VTK_QUADRATIC_WEDGE:
            fprintf(
                file,
                "%8d%8d%8d%8d%8d%8d%8d%8d\n%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d\n",
                nodenum[cells[c + 2]],   // 0,  I
                nodenum[cells[c + 1]],   // 1,  J
                nodenum[cells[c + 0]],   // 2,  K
                nodenum[cells[c + 0]],   // 3,  L (duplicate of K)
                nodenum[cells[c + 5]],   // 4,  M
                nodenum[cells[c + 4]],   // 5,  N
                nodenum[cells[c + 3]],   // 6,  O
                nodenum[cells[c + 3]],   // 7,  P (duplicate of O)
                nodenum[cells[c + 7]],   // 8,  Q
                nodenum[cells[c + 6]],   // 9,  R
                nodenum[cells[c + 0]],   // 10, S   (duplicate of K)
                nodenum[cells[c + 8]],   // 11, T
                nodenum[cells[c + 10]],  // 12, U
                nodenum[cells[c + 9]],   // 13, V
                nodenum[cells[c + 3]],   // 14, W (duplicate of O)
                nodenum[cells[c + 11]],  // 15, X
                nodenum[cells[c + 14]],  // 16, Y
                nodenum[cells[c + 13]],  // 17, Z
                nodenum[cells[c + 12]],  // 18, A
                nodenum[cells[c + 12]]); // 19, B (duplicate of A)
            break;
        case VTK_QUADRATIC_PYRAMID:
            fprintf(
                file,
                "%8d%8d%8d%8d%8d%8d%8d%8d\n%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d\n",
                nodenum[cells[c + 0]],   // 0,  I
                nodenum[cells[c + 1]],   // 1,  J
                nodenum[cells[c + 2]],   // 2,  K
                nodenum[cells[c + 3]],   // 3,  L
                nodenum[cells[c + 4]],   // 4,  M
                nodenum[cells[c + 4]],   // 5,  N (duplicate of M)
                nodenum[cells[c + 4]],   // 6,  O (duplicate of M)
                nodenum[cells[c + 4]],   // 7,  P (duplicate of M)
                nodenum[cells[c + 5]],   // 8,  Q
                nodenum[cells[c + 6]],   // 9,  R
                nodenum[cells[c + 7]],   // 10, S
                nodenum[cells[c + 8]],   // 11, T
                nodenum[cells[c + 4]],   // 12, U (duplicate of M)
                nodenum[cells[c + 4]],   // 13, V (duplicate of M)
                nodenum[cells[c + 4]],   // 14, W (duplicate of M)
                nodenum[cells[c + 4]],   // 15, X (duplicate of M)
                nodenum[cells[c + 9]],   // 16, Y
                nodenum[cells[c + 10]],  // 17, Z
                nodenum[cells[c + 11]],  // 18, A
                nodenum[cells[c + 12]]); // 19, B (duplicate of A)
            break;
        case VTK_PYRAMID:
            fprintf(
                file,
                "%8d%8d%8d%8d%8d%8d%8d%8d\n",
                nodenum[cells[c + 0]],  // 0,  I
                nodenum[cells[c + 1]],  // 1,  J
                nodenum[cells[c + 2]],  // 2,  K
                nodenum[cells[c + 3]],  // 3,  L
                nodenum[cells[c + 4]],  // 4,  M
                nodenum[cells[c + 4]],  // 5,  N (duplicate of M)
                nodenum[cells[c + 4]],  // 6,  O (duplicate of M)
                nodenum[cells[c + 4]]); // 7,  P (duplicate of M)
            break;
        case VTK_VOXEL:
            // note the flipped order for nodes (K, L) and (O, P)
            fprintf(
                file,
                "%8d%8d%8d%8d%8d%8d%8d%8d\n",
                nodenum[cells[c + 0]],  // 0, I
                nodenum[cells[c + 1]],  // 1, J
                nodenum[cells[c + 3]],  // 2, K
                nodenum[cells[c + 2]],  // 3, L
                nodenum[cells[c + 4]],  // 4, M
                nodenum[cells[c + 5]],  // 5, N
                nodenum[cells[c + 7]],  // 6, O
                nodenum[cells[c + 6]]); // 7, P
            break;
        case VTK_HEXAHEDRON:
            fprintf(
                file,
                "%8d%8d%8d%8d%8d%8d%8d%8d\n",
                nodenum[cells[c + 0]],
                nodenum[cells[c + 1]],
                nodenum[cells[c + 2]],
                nodenum[cells[c + 3]],
                nodenum[cells[c + 4]],
                nodenum[cells[c + 5]],
                nodenum[cells[c + 6]],
                nodenum[cells[c + 7]]);
            break;
        case VTK_QUADRATIC_HEXAHEDRON:
            fprintf(
                file,
                "%8d%8d%8d%8d%8d%8d%8d%8d\n%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d\n",
                nodenum[cells[c + 0]],
                nodenum[cells[c + 1]],
                nodenum[cells[c + 2]],
                nodenum[cells[c + 3]],
                nodenum[cells[c + 4]],
                nodenum[cells[c + 5]],
                nodenum[cells[c + 6]],
                nodenum[cells[c + 7]],
                nodenum[cells[c + 8]],
                nodenum[cells[c + 9]],
                nodenum[cells[c + 10]],
                nodenum[cells[c + 11]],
                nodenum[cells[c + 12]],
                nodenum[cells[c + 13]],
                nodenum[cells[c + 14]],
                nodenum[cells[c + 15]],
                nodenum[cells[c + 16]],
                nodenum[cells[c + 17]],
                nodenum[cells[c + 18]],
                nodenum[cells[c + 19]]);
            break;
        case VTK_TRIANGLE:
            fprintf(
                file,
                "%8d%8d%8d%8d\n",
                nodenum[cells[c + 0]],  // 0,  I
                nodenum[cells[c + 1]],  // 1,  J
                nodenum[cells[c + 2]],  // 2,  K
                nodenum[cells[c + 2]]); // 3,  L (duplicate of K)
            break;
        case VTK_QUAD:
            fprintf(
                file,
                "%8d%8d%8d%8d\n",
                nodenum[cells[c + 0]],  // 0,  I
                nodenum[cells[c + 1]],  // 1,  J
                nodenum[cells[c + 2]],  // 2,  K
                nodenum[cells[c + 3]]); // 3,  L
            break;
        }
    }
    fprintf(file, "      -1\n");

    // TODO: consider using C++ ifstream
    fclose(file);
}

NDArray<int, 1> CmblockItems(const NDArray<const int, 1> array) {

    const int *data = array.data();

    // Use vector for initial operations
    std::vector<int> temp(data, data + array.size());
    std::sort(temp.begin(), temp.end());
    temp.erase(std::unique(temp.begin(), temp.end()), temp.end());

    std::vector<int> items;
    items.push_back(temp[0]);

    int last_index = 0; // Tracks the index of the last unique range start
    for (size_t i = 0; i < temp.size() - 1; ++i) {
        // Check if the next item forms a continuous range
        if (temp[i + 1] - temp[i] != 1) {
            if (temp[i] != items[last_index]) {
                items.push_back(-temp[i]); // End the current range
            }
            items.push_back(temp[i + 1]); // Start a new range
            last_index = items.size() - 1;
        }
    }

    // End the last range if it was continuous
    if (temp.back() != items.back()) {
        items.push_back(-temp.back());
    }

    // Copy to ndarray
    NDArray<int, 1> items_arr = MakeNDArray<int, 1>({(int)items.size()});
    int *items_data = items_arr.data();
    std::copy(items.begin(), items.end(), items_data);

    return items_arr;
}

// Resets the midside nodes of the tetrahedral starting at index c.
// midside nodes between
// (0,1), (1,2), (2,0), (0,3), (1,3), and (2,3)
template <typename T> inline void ResetMidTet(const int64_t *cells, T *points) {
    int64_t ind0 = cells[0] * 3;
    int64_t ind1 = cells[1] * 3;
    int64_t ind2 = cells[2] * 3;
    int64_t ind3 = cells[3] * 3;
    int64_t ind4 = cells[4] * 3;
    int64_t ind5 = cells[5] * 3;
    int64_t ind6 = cells[6] * 3;
    int64_t ind7 = cells[7] * 3;
    int64_t ind8 = cells[8] * 3;
    int64_t ind9 = cells[9] * 3;

    for (size_t j = 0; j < 3; j++) {
        points[ind4 + j] = (points[ind0 + j] + points[ind1 + j]) * 0.5;
        points[ind5 + j] = (points[ind1 + j] + points[ind2 + j]) * 0.5;
        points[ind6 + j] = (points[ind2 + j] + points[ind0 + j]) * 0.5;
        points[ind7 + j] = (points[ind0 + j] + points[ind3 + j]) * 0.5;
        points[ind8 + j] = (points[ind1 + j] + points[ind3 + j]) * 0.5;
        points[ind9 + j] = (points[ind2 + j] + points[ind3 + j]) * 0.5;
    }
}

// Reset pyr midside nodes between:
// 5 (0, 1)
// 6 (1, 2)
// 7 (2, 3)
// 8 (3, 0)
// 9 (0, 4)
// 10(1, 4)
// 11(2, 4)
// 12(3, 4)
template <typename T> inline void ResetMidPyr(const int64_t *cells, T *points) {
    int64_t ind0 = cells[0] * 3;
    int64_t ind1 = cells[1] * 3;
    int64_t ind2 = cells[2] * 3;
    int64_t ind3 = cells[3] * 3;
    int64_t ind4 = cells[4] * 3;
    int64_t ind5 = cells[5] * 3;
    int64_t ind6 = cells[6] * 3;
    int64_t ind7 = cells[7] * 3;
    int64_t ind8 = cells[8] * 3;
    int64_t ind9 = cells[9] * 3;
    int64_t ind10 = cells[10] * 3;
    int64_t ind11 = cells[11] * 3;
    int64_t ind12 = cells[12] * 3;

    for (size_t j = 0; j < 3; j++) {
        points[ind5 + j] = (points[ind0 + j] + points[ind1 + j]) * 0.5;
        points[ind6 + j] = (points[ind1 + j] + points[ind2 + j]) * 0.5;
        points[ind7 + j] = (points[ind2 + j] + points[ind3 + j]) * 0.5;
        points[ind8 + j] = (points[ind3 + j] + points[ind0 + j]) * 0.5;
        points[ind9 + j] = (points[ind0 + j] + points[ind4 + j]) * 0.5;
        points[ind10 + j] = (points[ind1 + j] + points[ind4 + j]) * 0.5;
        points[ind11 + j] = (points[ind2 + j] + points[ind4 + j]) * 0.5;
        points[ind12 + j] = (points[ind3 + j] + points[ind4 + j]) * 0.5;
    }
}

template <typename T> inline void ResetMidWeg(const int64_t *cells, T *points) {
    // Reset midside nodes of a wedge cell:
    // 6  (0,1)
    // 7  (1,2)
    // 8  (2,0)
    // 9  (3,4)
    // 10 (4,5)
    // 11 (5,3)
    // 12 (0,3)
    // 13 (1,4)
    // 14 (2,5)
    int64_t ind0 = cells[0] * 3;
    int64_t ind1 = cells[1] * 3;
    int64_t ind2 = cells[2] * 3;
    int64_t ind3 = cells[3] * 3;
    int64_t ind4 = cells[4] * 3;
    int64_t ind5 = cells[5] * 3;
    int64_t ind6 = cells[6] * 3;
    int64_t ind7 = cells[7] * 3;
    int64_t ind8 = cells[8] * 3;
    int64_t ind9 = cells[9] * 3;
    int64_t ind10 = cells[10] * 3;
    int64_t ind11 = cells[11] * 3;
    int64_t ind12 = cells[12] * 3;
    int64_t ind13 = cells[13] * 3;
    int64_t ind14 = cells[14] * 3;

    for (size_t j = 0; j < 3; j++) {
        points[ind6 + j] = (points[ind0 + j] + points[ind1 + j]) * 0.5;
        points[ind7 + j] = (points[ind1 + j] + points[ind2 + j]) * 0.5;
        points[ind8 + j] = (points[ind2 + j] + points[ind0 + j]) * 0.5;
        points[ind9 + j] = (points[ind3 + j] + points[ind4 + j]) * 0.5;
        points[ind10 + j] = (points[ind4 + j] + points[ind5 + j]) * 0.5;
        points[ind11 + j] = (points[ind5 + j] + points[ind3 + j]) * 0.5;
        points[ind12 + j] = (points[ind0 + j] + points[ind3 + j]) * 0.5;
        points[ind13 + j] = (points[ind1 + j] + points[ind4 + j]) * 0.5;
        points[ind14 + j] = (points[ind2 + j] + points[ind5 + j]) * 0.5;
    }
}

// Reset midside nodes of a hexahedral cell
// 8  (0,1)
// 9  (1,2)
// 10 (2,3)
// 11 (3,0)
// 12 (4,5)
// 13 (5,6)
// 14 (6,7)
// 15 (7,4)
// 16 (0,4)
// 17 (1,5)
// 18 (2,6)
// 19 (3,7)
template <typename T> inline void ResetMidHex(const int64_t *cells, T *points) {
    int64_t ind0 = cells[0] * 3;
    int64_t ind1 = cells[1] * 3;
    int64_t ind2 = cells[2] * 3;
    int64_t ind3 = cells[3] * 3;
    int64_t ind4 = cells[4] * 3;
    int64_t ind5 = cells[5] * 3;
    int64_t ind6 = cells[6] * 3;
    int64_t ind7 = cells[7] * 3;
    int64_t ind8 = cells[8] * 3;
    int64_t ind9 = cells[9] * 3;
    int64_t ind10 = cells[10] * 3;
    int64_t ind11 = cells[11] * 3;
    int64_t ind12 = cells[12] * 3;
    int64_t ind13 = cells[13] * 3;
    int64_t ind14 = cells[14] * 3;
    int64_t ind15 = cells[15] * 3;
    int64_t ind16 = cells[16] * 3;
    int64_t ind17 = cells[17] * 3;
    int64_t ind18 = cells[18] * 3;
    int64_t ind19 = cells[19] * 3;

    for (size_t j = 0; j < 3; j++) {
        points[ind8 + j] = (points[ind0 + j] + points[ind1 + j]) * 0.5;
        points[ind9 + j] = (points[ind1 + j] + points[ind2 + j]) * 0.5;
        points[ind10 + j] = (points[ind2 + j] + points[ind3 + j]) * 0.5;
        points[ind11 + j] = (points[ind3 + j] + points[ind0 + j]) * 0.5;
        points[ind12 + j] = (points[ind4 + j] + points[ind5 + j]) * 0.5;
        points[ind13 + j] = (points[ind5 + j] + points[ind6 + j]) * 0.5;
        points[ind14 + j] = (points[ind6 + j] + points[ind7 + j]) * 0.5;
        points[ind15 + j] = (points[ind7 + j] + points[ind4 + j]) * 0.5;
        points[ind16 + j] = (points[ind0 + j] + points[ind4 + j]) * 0.5;
        points[ind17 + j] = (points[ind1 + j] + points[ind5 + j]) * 0.5;
        points[ind18 + j] = (points[ind2 + j] + points[ind6 + j]) * 0.5;
        points[ind19 + j] = (points[ind3 + j] + points[ind7 + j]) * 0.5;
    }
}

template <typename T>
void ResetMidside(
    NDArray<const uint8_t, 1> celltypes_arr,
    NDArray<const int64_t, 1> cells_arr,
    NDArray<const int64_t, 1> offset_arr,
    NDArray<T, 2> points_arr) {

    int n_cells = celltypes_arr.size();
    const int64_t *cells = cells_arr.data();
    const int64_t *offset = offset_arr.data();
    const uint8_t *celltypes = celltypes_arr.data();
    T *points = points_arr.data();

    for (int i = 0; i < n_cells; i++) {
        int64_t c = offset[i];

        switch (celltypes[i]) {
        case VTK_QUADRATIC_TETRA:
            ResetMidTet(&cells[c], points);
            continue;
        case VTK_QUADRATIC_PYRAMID:
            ResetMidPyr(&cells[c], points);
            continue;
        case VTK_QUADRATIC_WEDGE:
            ResetMidWeg(&cells[c], points);
            continue;
        case VTK_QUADRATIC_HEXAHEDRON:
            ResetMidHex(&cells[c], points);
        }
    }
}

double fast_10pow(size_t const exponent) {
    static double pow10[10] = {
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
        10000000.0,
        100000000.0,
        1000000000.0};

    if (exponent <= 9) {
        // fast, look-up
        return pow10[exponent];
    } else {
        // generic
        double powered = 1.0;
        for (size_t n = 0; n < exponent; ++n) {
            powered = powered * 10.0;
        }
        return powered;
    }
}

size_t integer_str_length(uint64_t const value) {
    // general solution (using it for big numbers)
    if (value >= 10000000)
        return floor(log10(value)) + 1;
    // ugly, but optimal
    // credits: https://stackoverflow.com/a/3069580
    if (value >= 100000)
        return 6;
    if (value >= 10000)
        return 5;
    if (value >= 1000)
        return 4;
    if (value >= 100)
        return 3;
    if (value >= 10)
        return 2;
    return 1;
}

void write_integer(
    char *put, uint64_t const value, bool const rounded_last_digit, bool const insert_nul) {

    uint64_t int_value = value;
    size_t const digits = integer_str_length(int_value);

    for (size_t pos = 0; pos < digits; ++pos) {
        size_t power = static_cast<size_t>(fast_10pow(digits - pos - 1));
        unsigned int newvalue = static_cast<int>(int_value / power);

        // round last digit
        if (rounded_last_digit && pos == digits - 1) {
            double lastval = round(int_value - static_cast<int>(int_value / 10.0) * 10.0);
            newvalue = static_cast<int>(lastval);
        }

        put[pos] = newvalue + '0';

        int_value = int_value - newvalue * power;
    }
    if (insert_nul) {
        put[digits] = '\0';
    }
}

inline bool is_negative(double const value) {
    // a function that handles 0.0 and -0.0 (the test -0.0 < 0.0 fails)
    return std::signbit(value);
}

void fill_with_char(char *put, char const fill, size_t const width) {
    for (size_t n = 0; n < width; ++n) {
        put[n] = fill;
    }
}

void format_i(
    char *put,
    uint64_t const value,
    size_t const width,
    size_t const fill,
    bool const plus_sign) {
    size_t len = integer_str_length(value);

    bool require_sign = is_negative(value) || plus_sign;
    bool fill_chars = len < fill;

    size_t maxlen = len + require_sign + fill_chars * (fill - len);

    size_t pos = 0;
    if (maxlen > width) {
        fill_with_char(put, OVERFLOW_FILL_CHAR, width);
        pos = pos + width;
    } else {
        // right-alignment whitespace
        for (; pos < width - maxlen; ++pos) {
            put[pos] = ' ';
        }

        // minus sign
        if (require_sign) {
            if (is_negative(value)) {
                put[pos] = '-';
            } else {
                put[pos] = '+';
            }

            pos = pos + 1;
        }

        // leading zeroes
        if (fill_chars) {
            for (size_t fillcount = 0; fillcount < fill - len; ++fillcount, ++pos) {
                put[pos] = '0';
            }
        }

        // integer
        bool const NOT_ROUNDED = false;
        bool const INSERT_NUL = false;
        write_integer(put + pos, value, NOT_ROUNDED, INSERT_NUL);
    }
    put[width] = '\0';
}

int base10_exponent(double const value) {
    double const absvalue = fabs(value);

    // general case, absvalue >= 1E+6 or < 1E-6
    if (absvalue >= 1000000.0 || absvalue < 0.000001) {
        return floor(log10(absvalue));
    } else {
        if (absvalue >= 100000.0)
            return 5;
        if (absvalue >= 10000.0)
            return 4;
        if (absvalue >= 1000.0)
            return 3;
        if (absvalue >= 100.0)
            return 2;
        if (absvalue >= 10.0)
            return 1;
        if (absvalue >= 1.0)
            return 0;
        if (absvalue >= 0.1)
            return -1;
        if (absvalue >= 0.01)
            return -2;
        if (absvalue >= 0.001)
            return -3;
        if (absvalue >= 0.0001)
            return -4;
        if (absvalue >= 0.00001)
            return -5;
        return -6;
    }
}

void format_e(
    char *put,
    double const value,
    size_t const width,
    size_t const precision,
    char const expchar,
    size_t const exponent_width,
    bool const plus_sign) {
    assert(exponent_width > 0);

    if (value == 0.0) {
        strcpy(put, " 0.");
        for (int i = 0; i < precision; ++i) {
            put[3 + i] = '0';
        }
        strcpy(put + 3 + precision, "E+000");
        return;
    }

    double absvalue = fabs(value);
    bool require_sign = is_negative(value) || plus_sign;

    // minimum length without leading zero
    // . precision D+00, ., D, and + = 3
    size_t const FORMATCHARS = 3;
    size_t const MINLEN = require_sign + FORMATCHARS + precision + exponent_width;

    // check whether there's space for a leading zero
    bool include_leading_zero = width > MINLEN;
    size_t const LEN = MINLEN + include_leading_zero;

    size_t pos = 0;
    if (LEN > width) {
        fill_with_char(put, OVERFLOW_FILL_CHAR, width);
        pos = pos + width;
    } else {
        // calculate the fractional part and the exponent
        double finalnumber;
        int const exponent = base10_exponent(value);
        // print exponent, since the number starts with 0.
        int const expfort = exponent + 1;
        // power of 10 to extract the values to print in the space given the precision
        double power;
        if (exponent >= 0) {
            if (expfort >= static_cast<int>(precision)) {
                power = 1.0 / static_cast<double>(fast_10pow(expfort - precision));
            } else {
                power = static_cast<double>(fast_10pow(precision - expfort));
            }
        } else {
            // -exponent + 1 = numbers of zeroes in the fractional part
            int const zeroes = -(exponent + 1);
            power = static_cast<double>(fast_10pow(zeroes + precision));
        }

        // final number after the decimal separator
        finalnumber = absvalue * power;
        uint64_t intval = static_cast<uint64_t>(finalnumber);

        // right-alignment whitespace
        if (width > LEN) {
            for (; pos < width - LEN; ++pos) {
                put[pos] = ' ';
            }
        }

        // minus sign
        if (require_sign) {
            if (is_negative(value)) {
                put[pos] = '-';
            } else {
                put[pos] = '+';
            }

            pos = pos + 1;
        }

        // leading zero
        if (include_leading_zero) {
            put[pos] = '0';
            pos = pos + 1;
        }
        put[pos] = '.';
        pos = pos + 1;

        // number value
        format_i(put + pos, intval, precision, 0, false);
        pos = pos + precision;

        // exponent sign
        put[pos] = expchar;
        pos = pos + 1;
        if (expfort >= 0) {
            put[pos] = '+';
        } else {
            put[pos] = '-';
        }
        pos = pos + 1;

        // exponent value
        format_i(put + pos, abs(expfort), exponent_width, exponent_width, false);
        pos = pos + exponent_width;
    }
    put[pos] = '\0';
}

void OverwriteNblock(
    const std::string &filename_in,
    const std::string &filename_out,
    const NDArray<const double, 2> coord,
    int nblock_start,
    int ilen, // Number of characters to preserve at the start of each line
    int l,    // Total length of each number field
    int d,    // Number of digits after the decimal
    int e     // Number of digits in the exponent notation
) {
    std::fstream file_in(filename_in, std::ios::in | std::ios::binary);
    if (!file_in.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename_in);
    }

    std::fstream file_out(filename_out, std::ios::in | std::ios::out | std::ios::binary);
    if (!file_in.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename_out);
    }

    const double *coord_data = coord.data();
    const int n_nodes = coord.size() / 3;

    // and the end of the new file
    file_out.seekg(0, std::ios::end);

    // Read the first line to detect the line ending
    file_in.seekg(nblock_start, std::ios::beg);
    std::string line_ending;
    char ch;
    while (file_in.get(ch)) {
        if (ch == '\n' || ch == '\r') {
            line_ending += ch;
            if (ch == '\r' && file_in.peek() == '\n') {
                file_in.get(ch);
                line_ending += ch;
            }
            break;
        }
    }

    if (line_ending.empty()) {
        line_ending = "\n"; // Default to LF if no line ending is detected
    }

    // Reset the file pointer to the start of the node block
    file_in.seekg(nblock_start, std::ios::beg);

    std::string buffer;
    buffer.reserve(1024);

    std::string line;
    for (int i = 0; i < n_nodes; ++i) {
        std::getline(file_in, line);

        // Prepare buffer with the initial characters preserved
        buffer.assign(line.substr(0, ilen));

        // Append formatted coordinates
        for (int j = 0; j < 3; ++j) { // Assuming each node has three coordinates (x, y, z)
            char coord_buffer[100];   // Enough for one coordinate

            format_e(coord_buffer, coord_data[i * 3 + j], l, d, 'e', e, false);
            buffer.append(coord_buffer);
        }

        // add in the remainder of the line
        if (line.size() > buffer.size()) {
            buffer.append(line.substr(buffer.size()));
            buffer.append(line_ending);
        } else {
            buffer.append(line_ending);
        }

        file_out.write(buffer.c_str(), buffer.size());
    }

    // write last line noting end of node block
    buffer.assign("N,R5.3,LOC,       -1,");
    buffer.append(line_ending);
    file_out.write(buffer.c_str(), buffer.size());

    file_in.close();
    file_out.close();
}

NB_MODULE(_archive, m) {
    m.def("write_nblock", &WriteNblock<float>);
    m.def("write_nblock", &WriteNblock<double>);
    m.def("write_eblock", &WriteEblock);
    m.def("overwrite_nblock", &OverwriteNblock);
    m.def("cmblock_items_from_array", &CmblockItems);
    m.def("reset_midside", &ResetMidside<float>);
    m.def("reset_midside", &ResetMidside<double>);
}
