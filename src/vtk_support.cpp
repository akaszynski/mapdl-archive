#include <stdint.h>
#include <stdlib.h>

#include "vtk_support.h"

/* VTK numbering for vtk cells */
uint8_t VTK_EMPTY_CELL = 0;
uint8_t VTK_VERTEX = 1;
uint8_t VTK_LINE = 3;
uint8_t VTK_TRIANGLE = 5;
uint8_t VTK_QUAD = 9;
uint8_t VTK_QUADRATIC_TRIANGLE = 22;
uint8_t VTK_QUADRATIC_QUAD = 23;
uint8_t VTK_HEXAHEDRON = 12;
uint8_t VTK_PYRAMID = 14;
uint8_t VTK_TETRA = 10;
uint8_t VTK_WEDGE = 13;
uint8_t VTK_QUADRATIC_EDGE = 21;
uint8_t VTK_QUADRATIC_TETRA = 24;
uint8_t VTK_QUADRATIC_PYRAMID = 27;
uint8_t VTK_QUADRATIC_WEDGE = 26;
uint8_t VTK_QUADRATIC_HEXAHEDRON = 25;

// Contains data for VTK UnstructuredGrid
struct VtkData {
    int *offset;
    int *cells;
    uint8_t *celltypes;
    int loc;   // current position within cells
    int *nref; // conversion between ansys and vtk numbering
};
struct VtkData vtk_data;

// Populate offset, cell type, and prepare the cell array for a cell
static inline void add_cell(int n_points, uint8_t celltype) {
    vtk_data.offset[0] = vtk_data.loc;
    vtk_data.offset++;

    vtk_data.celltypes[0] = celltype;
    vtk_data.celltypes++;

    /* printf("Finished adding cell\n"); */
    return;
}

/* ============================================================================
 * Store hexahedral element in vtk arrays.  ANSYS elements are
 * ordered in the same manner as VTK.
 *
 * VTK DOCUMENTATION
 * Linear Hexahedral
 * The hexahedron is defined by the eight points (0-7) where
 * (0,1,2,3) is the base of the hexahedron which, using the right
 * hand rule, forms a quadrilaterial whose normal points in the
 * direction of the opposite face (4,5,6,7).
 *
 * Quadradic Hexahedral
 * The ordering of the twenty points defining the cell is point ids
 * (0-7, 8-19) where point ids 0-7 are the eight corner vertices of
 * the cube; followed by twelve midedge nodes (8-19)
 * Note that these midedge nodes correspond lie on the edges defined by:
 * Midside   Edge nodes
 * 8         (0, 1)
 * 9         (1, 2)
 * 10        (2, 3)
 * 11        (3, 0)
 * 12        (4, 5)
 * 13        (5, 6)
 * 14        (6, 7)
 * 15        (7, 4)
 * 16        (0, 4)
 * 17        (1, 5)
 * 18        (2, 6)
 * 19        (3, 7)
 */
static inline void add_hex(const int *elem, int nnode) {
    int i;
    bool quad = nnode > 8;
    if (quad) {
        add_cell(20, VTK_QUADRATIC_HEXAHEDRON);
    } else {
        add_cell(8, VTK_HEXAHEDRON);
    }

    // Always add linear
    for (i = 0; i < 8; i++) {
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[i]];
    }

    // translate connectivity
    if (quad) {
        for (i = 8; i < nnode; i++) {
            vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[i]];
            /* printf("added %d at %d using node %d\n", vtk_data.nref[elem[i]],
             * vtk_data.loc - 1, elem[i]); */
        }
        // ANSYS sometimes doesn't write 0 at the end of the element block
        // and quadratic cells always contain 10 nodes
        for (i = nnode; i < 20; i++) {
            vtk_data.cells[vtk_data.loc++] = -1;
        }
    }

    return;
}

/* ============================================================================
Store wedge element in vtk arrays.  ANSYS elements are ordered
differently than vtk elements.  ANSYS orders counter-clockwise and
VTK orders clockwise

VTK DOCUMENTATION
Linear Wedge
The wedge is defined by the six points (0-5) where (0,1,2) is the
base of the wedge which, using the right hand rule, forms a
triangle whose normal points outward (away from the triangular
face (3,4,5)).

Quadradic Wedge
The ordering of the fifteen points defining the
cell is point ids (0-5,6-14) where point ids 0-5 are the six
corner vertices of the wedge, defined analogously to the six
points in vtkWedge (points (0,1,2) form the base of the wedge
which, using the right hand rule, forms a triangle whose normal
points away from the triangular face (3,4,5)); followed by nine
midedge nodes (6-14). Note that these midedge nodes correspond lie
on the edges defined by :
(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (0,3), (1,4), (2,5)
*/
static inline void add_wedge(const int *elem, int nnode) {
    bool quad = nnode > 8;
    if (quad) {
        add_cell(15, VTK_QUADRATIC_WEDGE);
    } else {
        add_cell(6, VTK_WEDGE);
    }

    // [0, 1, 2, 2, 3, 4, 5, 5]
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[2]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[1]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[0]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[6]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[5]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[4]];

    if (quad) { // todo: check if index > nnode - 1
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[9]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[8]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[11]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[13]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[12]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[15]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[18]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[17]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[16]];
    }

    return;
}

static inline void add_pyr(const int *elem, int nnode) {
    int i; // counter
    bool quad = nnode > 8;
    if (quad) {
        add_cell(13, VTK_QUADRATIC_PYRAMID);
    } else {
        add_cell(5, VTK_PYRAMID);
    }

    // [0, 1, 2, 3, 4, X, X, X]
    for (i = 0; i < 5; i++) {
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[i]];
    }

    if (quad) { // todo: check if index > nnode - 1
        for (i = 8; i < 12; i++) {
            vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[i]];
        }

        for (i = 16; i < 20; i++) {
            vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[i]];
        }
    }
    return;
}

/* ============================================================================
 * Stores tetrahedral element in vtk arrays.  ANSYS elements are
 * ordered the same as vtk elements.
 *
 * VTK DOCUMENTATION:
 * Linear
 * The tetrahedron is defined by the four points (0-3); where (0,1,2)
 * is the base of the tetrahedron which, using the right hand rule,
 * forms a triangle whose normal points in the direction of the
 * fourth point.
 *
 * Quadradic
 * The cell includes a mid-edge node on each of the size edges of the
 * tetrahedron. The ordering of the ten points defining the cell is
 * point ids (0-3,4-9) where ids 0-3 are the four tetra vertices; and
 * point ids 4-9 are the midedge nodes between:
 * (0,1), (1,2), (2,0), (0,3), (1,3), and (2,3)
============================================================================ */
static inline void add_tet(const int *elem, int nnode) {
    bool quad = nnode > 8;
    if (quad) {
        add_cell(10, VTK_QUADRATIC_TETRA);
    } else {
        add_cell(4, VTK_TETRA);
    }

    // edge nodes
    // [0, 1, 2, 2, 3, 3, 3, 3]
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[0]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[1]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[2]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[4]];

    if (quad) { // todo: check if index > nnode - 1
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[8]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[9]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[11]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[16]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[17]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[18]];
    }

    return;
}

// ANSYS Tetrahedral style [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
static inline void add_tet10(const int *elem, int nnode) {
    int i; // counter
    bool quad = nnode > 4;
    if (quad) {
        add_cell(10, VTK_QUADRATIC_TETRA);
    } else {
        add_cell(4, VTK_TETRA);
    }

    // edge nodes
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[0]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[1]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[2]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[3]];

    if (quad) {
        for (i = 4; i < nnode; i++) {
            vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[i]];
        }
        // ANSYS sometimes doesn't write 0 at the end of the element block
        // and quadratic cells always contain 10 nodes
        for (i = nnode; i < 10; i++) {
            vtk_data.cells[vtk_data.loc++] = -1;
        }
    }

    return;
}

static inline void add_quad(const int *elem, bool is_quad) {
    int i;
    int n_points;
    if (is_quad) {
        n_points = 8;

        if (elem[6] == elem[7]) { // edge case: check if repeated
            n_points = 4;
            /* printf("Duplicate midside points found...\nCelltype - Linear\n"); */
            add_cell(n_points, VTK_QUAD);
        } else {
            /* printf("Celltype - Quadratic\n"); */
            add_cell(n_points, VTK_QUADRATIC_QUAD);
        }
    } else {
        n_points = 4;
        /* printf("Celltype - Linear\n"); */
        add_cell(n_points, VTK_QUAD);
    }

    for (i = 0; i < n_points; i++) {
        /* printf("(%i) %i --> ", i, elem[i]); */
        /* printf("%i, \n", vtk_data.nref[elem[i]]); */
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[i]];
    }
    /* printf("\n"); */

    return;
}

void add_tri(const int *elem, bool quad) {
    if (quad) {
        add_cell(6, VTK_QUADRATIC_TRIANGLE);
    } else {
        add_cell(3, VTK_TRIANGLE);
    }

    // edge nodes
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[0]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[1]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[2]];

    if (quad) {
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[4]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[5]];
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[7]];
    }

    return;
}

void add_tri_missing_midside(const int *elem, const int n_mid) {
    add_cell(6, VTK_QUADRATIC_TRIANGLE);

    int i;
    int elem_indices[] = {4, 5, 7};

    // edge nodes
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[0]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[1]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[2]];

    // add midside nodes and populate any missing midside with -1
    for (i = 0; i < n_mid && i < 3; i++) {
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[elem_indices[i]]];
    }
    for (i = 3; i > n_mid; i--) {
        vtk_data.cells[vtk_data.loc++] = -1;
    }

    return;
}

static inline void add_line(const int *elem, int nnode) {
    bool is_quad;
    if (nnode > 2) {
        is_quad = elem[2] > 0;
    } else {
        is_quad = false;
    }

    /* printf("is_quad, %i\n", is_quad); */
    if (is_quad) {
        add_cell(3, VTK_QUADRATIC_EDGE);
    } else {
        add_cell(2, VTK_LINE);
    }

    // edge nodes
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[0]];
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[1]];
    if (is_quad) {
        vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[2]];
    }

    return;
}

static inline void add_point(const int *elem) {
    add_cell(1, VTK_VERTEX);
    vtk_data.cells[vtk_data.loc++] = vtk_data.nref[elem[0]];
    return;
}

/*
Stores wedge element in vtk arrays.  ANSYS elements are ordered
differently than vtk elements.  ANSYS orders counter-clockwise and
VTK orders clockwise

VTK DOCUMENTATION
Linear Wedge
The wedge is defined by the six points (0-5) where (0,1,2) is the
base of the wedge which, using the right hand rule, forms a
triangle whose normal points outward (away from the triangular
face (3,4,5)).

Quadradic Wedge
The ordering of the fifteen points defining the
cell is point ids (0-5,6-14) where point ids 0-5 are the six
corner vertices of the wedge, defined analogously to the six
points in vtkWedge (points (0,1,2) form the base of the wedge
which, using the right hand rule, forms a triangle whose normal
points away from the triangular face (3,4,5)); followed by nine
midedge nodes (6-14). Note that these midedge nodes correspond lie
on the edges defined by :
(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (0,3), (1,4), (2,5)
*/

/* ============================================================================
 * function: ans_to_vtk
 * Convert raw ANSYS elements to a VTK UnstructuredGrid format.
 *
 * Parameters
 * ----------
 * nelem : Number of elements.
 *
 * elem: Array of elements
 *   Each element contains 10 items plus the nodes belonging to the
 *   element.  The first 10 items are:
 *     mat    - material reference number
 *     type   - element type number
 *     real   - real constant reference number
 *     secnum - section number
 *     esys   - element coordinate system
 *     death  - death flag (0 - alive, 1 - dead)
 *     solidm - solid model reference
 *     shape  - coded shape key
 *     elnum  - element number
 *     baseeid- base element number (applicable to reinforcing elements only
 *     nodes  - The nodes belonging to the element in ANSYS numbering.
 *
 * elem_off : Indices of the start of each element in ``elem``.
 *
 * type_ref : Maps an element type number of an element type to a
 *            corresponding basic VTK element type:
 #       - TYPE 0 : Skip
 #       - TYPE 1 : Point
 #       - TYPE 2 : Line
 #       - TYPE 3 : Shell
 #       - TYPE 4 : 3D Solid (Hexahedral, wedge, pyramid, tetrahedral)
 #       - TYPE 5 : Tetrahedral
 *
 * nnode : Number of nodes.
 *
 * nnum : ANSYS Node numbering
 *
 * Returns (Given as as parameters)
 * -------
 * offset : VTK offset array to populate
 *
 * cells : VTK cell connectivity
 *
 * celltypes: VTK cell types
 *

 * ==========================================================================*/
int ans_to_vtk(
    const int nelem,
    const int *elem,
    const int *elem_off,
    const int *type_ref,
    const int nnode,
    const int *nnum,
    int *offset,
    int *cells,
    uint8_t *celltypes) {
    bool is_quad;
    int i;          // counter
    int nnode_elem; // number of nodes belonging to the element
    int off;        // location of the element nodes
    int etype;      // ANSYS element type number

    // index ansys node number to VTK C based compatible indexing
    // max node number should be last node
    // Consider using a hash table here instead
    int *nref = (int *)malloc((nnum[nnode - 1] + 1) * sizeof(int));
    nref[0] = -1; // for missing midside nodes ANSYS uses a node number of 0
    for (i = 0; i < nnode; i++) {
        nref[nnum[i]] = i;
    }

    // populate global vtk data
    vtk_data.offset = offset;
    vtk_data.cells = cells;
    vtk_data.celltypes = celltypes;
    vtk_data.nref = nref;
    vtk_data.loc = 0;

    // Convert each element from ANSYS connectivity to VTK connectivity
    for (i = 0; i < nelem; i++) {
        // etype
        etype = elem[elem_off[i] + 1];
        off = elem_off[i] + 10;             // to the start of the nodes
        nnode_elem = elem_off[i + 1] - off; // number of nodes in element

        switch (type_ref[etype]) {
        case 0: // not supported or not set
            add_cell(0, VTK_EMPTY_CELL);
            break;
        case 1: // point
            add_point(&elem[off]);
            break;
        case 2: // line
            add_line(&elem[off], nnode_elem);
            break;
        case 3: // shell
            if (nnode_elem == 3) {
                /*General triangular elements*/
                add_tri(&elem[off], false);

            } else if (nnode_elem == 4 && elem[off + 2] == elem[off + 3]) {
                /* Degenerated linear quad elements */
                add_tri(&elem[off], false);

            } else if (nnode_elem == 4) {
                /* General linear quadrangle */
                add_quad(&elem[off], false);

            } else if (nnode_elem == 6) {
                /* Quadratic tri elements */
                add_tri(&elem[off], true);

            } else if (nnode_elem == 8 && elem[off + 2] == elem[off + 3]) {
                /* Degenerated quadratic quadrangle elements */
                add_tri(&elem[off], true);

            } else if (nnode_elem == 8) {
                /* General quadratic quadrangle */
                add_quad(&elem[off], true);

            } else if (nnode_elem == 5) {
                // For shell/plane elements with one extra node.
                // Check element SURF 152, with KEYOPT 5,1.
                add_quad(&elem[off], false);

            } else if (nnode_elem == 10) {
                // For quadratic shell/plane elements with two extra nodes.
                // Check element TARGET170.
                add_quad(&elem[off], true);

            } else {
                // Any other case. Possible when missing midside nodes.
                is_quad = nnode_elem > 5;

                // Check degenerate triangle
                if (elem[off + 2] == elem[off + 3]) {
                    if (is_quad) {
                        add_tri_missing_midside(&elem[off], 3 - (8 - nnode_elem));
                    } else {
                        add_tri(&elem[off], false);
                    }
                } else {
                    // printf(" The type could not be identified. Check vtk_support.c
                    // file"); printf("Number of elements is %d\n" , nnode_elem);

                    // Assume quad
                    add_quad(&elem[off], is_quad);
                }
            }
            break;
        case 4: // solid
            /* printf("Adding solid "); */
            if (elem[off + 6] != elem[off + 7]) { // hexahedral
                /* printf(" subtype hexahedral\n"); */
                add_hex(&elem[off], nnode_elem);
            } else if (elem[off + 5] != elem[off + 6]) { // wedge
                /* printf(" subtype wedge\n"); */
                add_wedge(&elem[off], nnode_elem);
            } else if (elem[off + 2] != elem[off + 3]) { // pyramid
                /* printf(" subtype pyramid\n"); */
                add_pyr(&elem[off], nnode_elem);
            } else { // tetrahedral
                     /* printf(" subtype tetrahedral\n"); */
                add_tet(&elem[off], nnode_elem);
            }
            break;
        case 5: // tetrahedral
            /* printf("Adding tetrahedral\n"); */
            add_tet10(&elem[off], nnode_elem);
            break;
        case 6: // linear line
            add_line(&elem[off], false);
            // should never reach here
        } // end of switch
    } // end of loop

    /* printf("Done\n"); */
    offset[nelem] = vtk_data.loc;

    free(nref);
    return vtk_data.loc;
}
