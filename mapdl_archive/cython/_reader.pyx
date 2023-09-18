# cython: language_level=2
# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: embedsignature=True

""" Cython implementation of a CDB reader """
from libc.stdint cimport int64_t
from libc.stdio cimport (
    EOF,
    FILE,
    SEEK_CUR,
    SEEK_END,
    SEEK_SET,
    fclose,
    fgetc,
    fgets,
    fopen,
    fread,
    fscanf,
    fseek,
    ftell,
    printf,
    sscanf,
    ungetc,
)
from libc.stdlib cimport atof, atoi, free, malloc
from libc.string cimport memcmp, strcmp, strncpy

ctypedef unsigned char uint8_t

import ctypes

import numpy as np

cimport numpy as np


def safe_int(value):
    """Safely convert a value to int.

    Return int if can be converted, None if otherwise
    """
    try:
        return int(value)
    except:
        return


cdef extern from "reader.h":
    int read_nblock_from_nwrite(char*, int*, double*, int)
    int write_array_ascii(const char*, const double*, int);
    int read_eblock_cfile(FILE *cfile, int *elem_off, int *elem, int nelem)
    int read_nblock_cfile(FILE *cfile, int *nnum, double *nodes, int nnodes, int* d_size, int f_size)

cdef extern from 'vtk_support.h':
    int ans_to_vtk(const int, const int*, const int*, const int*, const int,
    const int*, int64_t*, int64_t*, uint8_t*, const int)


def read(filename, read_parameters=False, debug=False, read_eblock=True):
    """Read blocked ansys archive file."""
    badstr = 'Badly formatted cdb file'
    filename_byte_string = filename.encode("UTF-8")
    cdef char* fname = filename_byte_string
    parameters = {}
    # Check file exists
    cdef FILE* cfile
    cfile = fopen(fname, 'r')

    if cfile == NULL:
        raise Exception("No such file or directory: '%s'" % filename)

    # Define variables
    cdef size_t l = 0
    cdef ssize_t read
    cdef int[5] blocksz
    cdef int i, j, linelen, isz, tempint
    cdef float tempflt

    # Size temp char array
    cdef char line[500]
    cdef char tempstr[100]

    # Get element types
    elem_type = []
    rnum = []
    rdat = []

    # NBLOCK
    cdef int nnodes = 0
    cdef int [::1] nnum = np.empty(0, ctypes.c_int)
    cdef double [:, ::1] nodes = np.empty((0, 0))
    cdef int d_size[3]
    cdef int f_size, nfields, nblock_start, nblock_end, _start

    # EBLOCK
    cdef int nelem = 0
    cdef int [::1] elem = np.empty(0, dtype=ctypes.c_int)
    cdef int [::1] elem_off = np.empty(0, dtype=ctypes.c_int)
    cdef int elem_sz

    # CMBLOCK
    cdef int ncomp
    cdef int [::1] component
    cdef int nblock

    node_comps = {}
    elem_comps = {}

    nodes_read = False
    eblock_read = False

    # parameters
    cdef char *end_marker = b"END PREAD"
    cdef list parm_lines = []
    cdef long position

    # keyopt
    keyopt = {}

    cdef int first_char, next_char;

    while 1:
        first_char = fgetc(cfile)
        if first_char == EOF:
            break
        elif first_char == '\r' or first_char == '\n':
            ungetc(first_char, cfile)
            fgets(line, sizeof(line), cfile)
        elif first_char == 'E' or first_char == 'e':
            ungetc(first_char, cfile)
            fgets(line, sizeof(line), cfile)

            # Record element types
            if b'ET,' == line[:3] or b'et,' == line[:3]:
                if debug:
                    print('reading ET')

                # element number and element type
                et_val = line.decode().split(',')
                try:
                    int(et_val[1])
                    elem_type.append([int(et_val[1]), int(et_val[2])])
                except:
                    if debug:
                        print('Invalid "ET" command %s' % line.decode())
                    continue

            # read in new ETBLOCK (replacement for ET)
            elif b'ETBLOCK' in line or b'etblock' in line:
                # read the number of items in the block
                set_dat = [safe_int(value) for value in line.split(b',')[1:]]
                n_items = set_dat[0]

                # Skip Format1 (2i9,19a9)
                fgets(line, sizeof(line), cfile)

                for item in range(n_items):
                    fgets(line, sizeof(line), cfile)

                    # Only read in the first two items (ELEM index and TYPE)
                    # NOTE: Remaining contents of the block are unknown
                    et_val = [safe_int(value) for value in line.split()[:2]]
                    elem_type.append(et_val)

            elif b'EBLOCK,' == line[:7] or b'eblock,' == line[:7] and read_eblock:
                if eblock_read:
                    # Sometimes, DAT files contain two EBLOCKs.  Read
                    # only the first block.
                    if debug:
                        print('EBLOCK already read, skipping...')
                    continue
                if debug:
                    print('reading EBLOCK...')

                # only read entries with SOLID
                if b'SOLID' in line or b'solid' in line:

                    # Get size of EBLOCK from the last item in the line
                    # Example: "EBLOCK,19,SOLID,,3588"
                    nelem = int(line[line.rfind(b',') + 1:])
                    if nelem == 0:
                        raise RuntimeError('Unable to read element block')

                    # Populate element field data and connectivity
                    elem = np.empty(nelem*30, dtype=ctypes.c_int)
                    elem_off = np.empty(nelem + 1, dtype=ctypes.c_int)
                    elem_sz = read_eblock_cfile(cfile, &elem_off[0], &elem[0], nelem)

                    eblock_read = True
                    if debug:
                        print('finished')

        elif first_char == 'K' or first_char == 'k':
            ungetc(first_char, cfile);  # Put the character back into the stream
            fgets(line, sizeof(line), cfile)

            if debug:
                print('Hit "K"')

            if b'KEYOP' in line or b'keyop' in line:
                if debug:
                    print('reading KEYOP')

                try:
                    entry = []
                    for item in line.split(b',')[1:]:
                        entry.append(int(item))
                except:
                    continue

                key_num = int(entry[0])
                if key_num in keyopt:
                    keyopt[key_num].append(entry[1:])
                else:
                    keyopt[key_num] = [entry[1:]]

        elif first_char == 'R' or first_char == 'r':
            ungetc(first_char, cfile);  # Put the character back into the stream
            fgets(line, sizeof(line), cfile)

            if debug:
                print('Hit "R"')

            if b'RLBLOCK' in line or b'rlblock' in line:
                if debug:
                    print('reading RLBLOCK')

                # The RLBLOCK command defines a real constant
                # set. The real constant sets follow each set,
                # starting with

                # Format1 and followed by one or more Format2's, as
                # needed. The command format is:
                # RLBLOCK,NUMSETS,MAXSET,MAXITEMS,NPERLINE
                # Format1
                # Format2
                #
                # where:
                # Format1 - Data descriptor defining the format of the
                # first line. For the RLBLOCK command, this is always
                # (2i8,6g16.9).  The first i8 is the set number, the
                # second i8 is the number of values in this set,
                # followed by up to 6 real
                #
                # Format2 - Data descriptors defining the format of
                # the subsequent lines (as needed); this is always
                # (7g16.9).
                #
                # - NUMSETS : The number of real constant sets defined
                # - MAXSET : Maximum real constant set number
                # - MAXITEMS : Maximum number of reals in any one set
                # - NPERLINE : Number of reals defined on a line

                # Get number of sets from the RLBLOCK line
                set_dat = [safe_int(value) for value in line.split(b',')[1:]]
                nset, maxset, maxitems, nperline = set_dat

                # Skip Format1 and Format2 (always 2i8,6g16.9 and 7g16.9)
                if fgets(line, sizeof(line), cfile) == NULL: raise RuntimeError(badstr)
                if fgets(line, sizeof(line), cfile) == NULL: raise RuntimeError(badstr)

                # Read data
                for _ in range(nset):
                    rcon = [] # real constants

                    if fgets(line, sizeof(line), cfile) == NULL: raise RuntimeError(badstr)

                    # Get real constant number
                    rnum.append(int(line[:8]))

                    # Number of constants
                    ncon = int(line[8:16])

                    # Get constant data
                    if ncon > 6: # if multiple lines
                        for i in range(6):
                            rcon.append(float(line[16 + 16*i:32 + 16*i]))
                            ncon -= 1

                        # advance line
                        if fgets(line, sizeof(line), cfile) == NULL:
                            raise RuntimeError(badstr)

                        # read next line
                        while True:
                            if ncon > 7:
                                for i in range(7):
                                    rcon.append(float(line[16*i:16*(i + 1)]))
                                    ncon -= 1
                                # advance
                                if fgets(line, sizeof(line), cfile) == NULL:
                                    raise RuntimeError(badstr)

                            else:
                                for i in range(ncon):
                                    try:
                                        rcon.append(float(line[16*i:16 + 16*i]))
                                    # account for empty 0 values
                                    except:
                                        rcon.append(0.0)

                                break

                    # If only one in constant data
                    else:
                        for i in range(ncon):
                            rcon.append(float(line[16 + 16*i:32 + 16*i]))

                    rdat.append(rcon)

        elif first_char == 'N' or first_char == 'n':
            ungetc(first_char, cfile);  # Put the character back into the stream
            _start = ftell(cfile)  # used for NBLOCK
            fgets(line, sizeof(line), cfile)

            if debug:
                print('Hit "N"')

            # if line contains the start of the node block
            if line[:6] == b'NBLOCK' or line[:6] == b'nblock':
                if nodes_read:
                    if debug:
                        print('Skipping additional NBLOCK')
                    continue
                if debug:
                    print('reading NBLOCK due to ', line.decode())

                # Before reading NBLOCK, save where the nblock started
                nblock_start = _start

                # Get size of NBLOCK
                nnodes = int(line[line.rfind(b',') + 1:])
                nnum = np.empty(nnodes, dtype=ctypes.c_int)
                nodes = np.empty((nnodes, 6))

                # Get format of nblock
                fgets(line, sizeof(line), cfile)
                d_size_py, f_size, _, _ = node_block_format(line)
                d_size[0] = d_size_py[0]
                d_size[1] = d_size_py[1]
                d_size[2] = d_size_py[2]

                nnodes_read = read_nblock_cfile(cfile, &nnum[0], &nodes[0, 0], nnodes, d_size, f_size)
                nodes_read = True

                # read final line
                fgets(line, sizeof(line), cfile)
                nblock_end = ftell(cfile)

                if nnodes_read != nnodes:
                    nnodes = nnodes_read
                    nodes = nodes[:nnodes]
                    nnum = nnum[:nnodes]

                if debug:
                    print('Read', nnodes_read, 'nodes')

        elif first_char == 'C' or first_char == 'c':
            ungetc(first_char, cfile);  # Put the character back into the stream
            fgets(line, sizeof(line), cfile)

            if debug:
                print('Hit "C"')

            if line[:8] == b'CMBLOCK,' or line[:8] == b'cmblock,':  # component block
                if debug:
                    print('reading CMBLOCK')

                try:
                    line.decode()
                except:
                    print('Poor formatting of CMBLOCK: %s' % line)
                    continue

                split_line = line.split(b',')
                if len(split_line) < 3:
                    print('Poor formatting of CMBLOCK: %s' % line)
                    continue

                # Get number of items
                ncomp = int(split_line[3].split(b'!')[0].strip())
                component = np.empty(ncomp, ctypes.c_int)

                # Get integer size
                fgets(line, sizeof(line), cfile)
                isz = int(line[line.find(b'i') + 1:line.find(b')')])
                tempstr[isz] = '\0'

                # Number of integers per line
                nblock = int(line[line.find(b'(') + 1:line.find(b'i')])

                # Extract nodes
                for i in range(ncomp):

                    # Read new line if at the end of the line
                    if i%nblock == 0:
                        fgets(line, sizeof(line), cfile)

                    strncpy(tempstr, line + isz*(i%nblock), isz)
                    component[i] = atoi(tempstr)

                # Convert component to array and store
                comname = split_line[1].decode().strip()
                line_comp_type = split_line[2]
                if b'NODE' in line_comp_type:
                    node_comps[comname] = component_interperter(component)
                elif b'ELEM' in line_comp_type:
                    elem_comps[comname] = component_interperter(component)

        elif read_parameters and first_char == '*':  # maybe *DIM
            ungetc(first_char, cfile);  # Put the character back into the stream
            fgets(line, sizeof(line), cfile)

            if debug:
                print('Hit "*"')

            if b'DIM' in line:
                items = line.decode().split(',')
                if len(items) < 3:
                    continue

                name = items[1]
                if items[2].lower() == 'string':
                    fgets(line, sizeof(line), cfile)
                    string_items = line.decode().split('=')
                    if len(string_items) > 1:
                        parameters[name] = string_items[1].replace("'", '').strip()
                    else:
                        parameters[name] = line.decode()
                elif items[2].lower() == 'array':
                    fgets(line, sizeof(line), cfile)
                    if b'PREAD' in line:
                        if debug:
                            print('reading PREAD')

                        _, name, arr_size = line.decode().split(',')
                        name = name.strip()
                        position = ftell(cfile)

                        while True:
                            if fgets(line, sizeof(line), cfile) is NULL:
                                # EOF or error: Reset the file position
                                if fseek(cfile, position, SEEK_SET) != 0:
                                    # Error in fseek
                                    raise IOError("Error setting file position")
                                # EOF or error
                                break

                            # Check for the end marker and end if reached it
                            if memcmp(line, end_marker, sizeof(end_marker) - 1) == 0:
                                break

                            parm_lines.append(line)

                        parameters[name] = np.genfromtxt((b''.join(parm_lines)).split())

        else:
            # no match, simply read remainder of line

            fgets(line, sizeof(line), cfile)

    # # if the node block was not read for some reason
    # if not nodes_read:
    #     if debug:
    #         print('Did not read nodes block. Rereading from start.')

    if debug:
        print('Returning arrays')

    return {
        'rnum': np.asarray(rnum),
        'rdat': rdat,
        'ekey': np.asarray(elem_type, ctypes.c_int),
        'nnum': np.asarray(nnum),
        'nodes': np.asarray(nodes),
        'elem': np.array(elem[:elem_sz]),
        'elem_off': np.array(elem_off),
        'node_comps': node_comps,
        'elem_comps': elem_comps,
        'keyopt': keyopt,
        'parameters': parameters,
        'nblock_start': nblock_start,
        'nblock_end': nblock_end,
    }


def node_block_format(string):
    """Get the node block format.

    Example formats:
    (3i9,6e21.13e3)
    3 ints, all 9 digits wide followed by 6 floats

    (1i7,2i9,6e21.13)
    1 int 7 digits wide, 2 ints, 9 digits wide, 6 floats
    """
    string = string.decode().replace('(', '').replace(')', '')
    fields = string.split(',')

    # double and float size
    d_size = np.zeros(3, np.int32)
    nexp = 2  # default when missing
    nfields = 6
    f_size = 21
    c = 0
    for field in fields:
        if 'i' in field:
            items = field.split('i')
            for n in range(int(items[0])):
                d_size[c] = int(items[1])
                c += 1
        elif 'e' in field:
            f_size = int(field.split('e')[1].split('.')[0])

            # get number of possible integers in the float scientific notation
            if 'e' in field.split('.')[1]:
                nexp = int(field.split('.')[1].split('e')[1])

            nfields = int(field.split('e')[0])

    return d_size, f_size, nfields, nexp


def component_interperter(component):
    """If a node is negative, it is describing a list from the
    previous node.  This is ANSYS's way of saving file size when
    writing components.

    This function has not been optimized.

    """
    f_new = []
    for i in range(len(component)):
        if component[i] > 0: # Append if positive
            f_new.append(component[i])
        else: # otherwise, append list
            f_new.append(range(abs(component[i - 1]) + 1, abs(component[i]) + 1))

    return np.hstack(f_new).astype(ctypes.c_int)


def ans_vtk_convert(const int [::1] elem, const int [::1] elem_off,
                    const int [::1] type_ref,
                    const int [::1] nnum, int build_offset):
    """Convert ansys style connectivity to VTK connectivity"""
    cdef int nelem = elem_off.size - 1
    cdef int64_t [::1] offset = np.empty(nelem, ctypes.c_int64)
    cdef uint8_t [::1] celltypes = np.empty(nelem, dtype='uint8')

    # Allocate connectivity
    # max cell size is 20 (VTK_HEXAHEDRAL) and cell header is 1
    cdef int64_t [::1] cells = np.empty(nelem*21, ctypes.c_int64)
    cdef int loc = ans_to_vtk(nelem, &elem[0], &elem_off[0],
                              &type_ref[0], nnum.size, &nnum[0],
                              &offset[0], &cells[0], &celltypes[0],
                              build_offset)

    return np.asarray(offset), np.asarray(celltypes), np.asarray(cells[:loc])


def read_from_nwrite(filename, int nnodes):
    """Read the node coordinates from the output from the MAPDL NWRITE command"""
    cdef int [::1] nnum = np.empty(nnodes, ctypes.c_int)
    cdef double [:, ::1] nodes = np.empty((nnodes, 3), np.double)

    read_nblock_from_nwrite(filename, &nnum[0], &nodes[0, 0], nnodes)
    return np.array(nnum), np.array(nodes)


def write_array(filename, const double [::1] arr):
    cdef int nvalues = arr.size
    write_array_ascii(filename, &arr[0], nvalues)
