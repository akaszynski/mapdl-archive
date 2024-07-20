#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "array_support.h"
#include "vtk_support.h"

using namespace nb::literals;

#if defined(_WIN32) || defined(_WIN64)
/* We are on Windows */
#define strtok_r strtok_s
#endif

// #define DEBUG

static const double DIV_OF_TEN[] = {
    1.0e-0,  1.0e-1,  1.0e-2,  1.0e-3,  1.0e-4,  1.0e-5,  1.0e-6,  1.0e-7,  1.0e-8,  1.0e-9,
    1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13, 1.0e-14, 1.0e-15, 1.0e-16, 1.0e-17, 1.0e-18, 1.0e-19,
    1.0e-20, 1.0e-21, 1.0e-22, 1.0e-23, 1.0e-24, 1.0e-25, 1.0e-26, 1.0e-27,
};

static inline double power_of_ten(int exponent) {
    double result = 1.0;
    double base = (exponent < 0) ? 0.1 : 10.0;
    int abs_exponent = abs(exponent);
    for (int i = 0; i < abs_exponent; ++i) {
        result *= base;
    }
    return result;
}

//=============================================================================
// Fast string to integer convert to ANSYS formatted integers
//=============================================================================
static inline int fast_atoi(const char *raw, const int intsz) {
    int val = 0;
    int c;
    char current_char;

    for (c = 0; c < intsz; ++c) {
        current_char = *raw++;

        // Multiply by 10 only if current_char is a digit
        if (current_char >= '0' && current_char <= '9') {
            val = val * 10 + (current_char - '0');
        }
    }

    return val;
}

//=============================================================================
// Checks for negative character
//=============================================================================
static inline int checkneg(char *raw, int intsz) {
    int c;
    for (c = 0; c < intsz; ++c) {
        if (raw[0] == '-') {
            return 1;
        }
        ++raw;
    }
    return 0;
}

// Reads various ansys float formats in the form of
// "3.7826539829200E+00"
// "1.0000000000000E-001"
// "        -6.01203 "
//
// fltsz : Number of characters to read in a floating point number
static inline int ans_strtod(char *raw, int fltsz, double *arr) {
    char *end = raw + fltsz;
    double sign = 1;

    while (raw < end) {
        if (*raw == '\0') {
            // section empty, value is zero
            *arr = 0;
            return 1;
        } else if (*raw != ' ') { // always skip whitespace
            break;
        }
        raw++;
    }

    // either a number or a sign
    if (*raw == '-') {
        sign = -1;
        ++raw;
    }

    // next value is always a number
    // Use integer arithmetric and then convert to a float
    uint64_t val_int = *raw++ - '0';
    raw++; // next value is always a "."

    // Read through the rest of the number
    int decimal_digits = 0;
    while (raw < end) {
        if (*raw == 'e' || *raw == 'E') { // incredibly, can be lowercase
            break;
        } else if (*raw >= '0' && *raw <= '9') {
            val_int = val_int * 10 + (*raw++ - '0');
            decimal_digits++;
        }
    }

    // Compute the floating-point value
    double val;
    if (decimal_digits < 27) {
        val = (double)val_int * DIV_OF_TEN[decimal_digits];
    } else {
        val = (double)val_int * power_of_ten(10);
    }

    // Might have scientific notation remaining, for example:
    // 1.0000000000000E-001
    int evalue = 0;
    int esign = 1;
    if (*raw == 'e' || *raw == 'E') {
        raw++; // skip "E"
        // always a sign of some sort
        if (*raw == '-') {
            esign = -1;
        }
        raw++;

        while (raw < end) {
            // read to whitespace or end of the line
            if (*raw == ' ' || *raw == '\0') {
                break;
            }
            evalue = evalue * 10 + (*raw++ - '0');
        }
        if (esign == 1) {
            val *= power_of_ten(evalue);
        } else {
            val /= power_of_ten(evalue);
        }
    }

    // seek through end of float value
    if (sign == -1) {
        *arr = -val;
    } else {
        *arr = val;
    }

    // check if at end of line
    return 0;
}

int ReadEBlockCfile(std::ifstream &cfile, int *elem_off, int *elem, const int nelem) {
    int i, j, n_node;

    // set to start of the NBLOCK
    char line[256]; // maximum of 19 values, at most 13 char per int is 228
    char *cursor = line;

    if (!cfile.getline(line, sizeof(line))) {
        return 0;
    }

    char *i_pos = strchr(line, 'i');
    char *close_paren_pos = strchr(line, ')');
    if (i_pos == NULL || close_paren_pos == NULL || i_pos > close_paren_pos) {
        fprintf(stderr, "Invalid line format\n");
        return 0;
    }

    int isz, n_char, n_values;
    sscanf(i_pos + 1, "%d", &isz);

    // Loop through elements
    int c = 0;
    for (i = 0; i < nelem; ++i) {
        // store start of each element
        elem_off[i] = c;

        // Read the line and determine the number of values. MAPDL does not write all the
        // values on a single line
        cfile.getline(line, sizeof(line));
        n_values = cfile.gcount() / isz;

        cursor = line; // is this necessary, why not just use line?

        // It's possible that less nodes are written to the record than
        // indicated.  In this case the line starts with a -1

        // Check if at end of the block
        if (checkneg(cursor, isz)) {
            cursor += isz;
            break;
        }

        // ANSYS archive format:
        // Field 1: material reference number
        elem[c++] = fast_atoi(cursor, isz);
        cursor += isz;

        // Field 2: element type number
        elem[c++] = fast_atoi(cursor, isz);
        cursor += isz;

        // Field 3: real constant reference number
        elem[c++] = fast_atoi(cursor, isz);
        cursor += isz;

        // Field 4: section number
        elem[c++] = fast_atoi(cursor, isz);
        cursor += isz;

        // Field 5: element coordinate system
        elem[c++] = fast_atoi(cursor, isz);
        cursor += isz;

        // Field 6: Birth/death flag
        elem[c++] = fast_atoi(cursor, isz);
        cursor += isz;

        // Field 7: Solid model reference
        elem[c++] = fast_atoi(cursor, isz);
        cursor += isz;

        // Field 8: Coded shape key
        elem[c++] = fast_atoi(cursor, isz);
        cursor += isz;

        // Field 9: Number of nodes, don't store this
        n_node = fast_atoi(cursor, isz);
        cursor += isz;

        //   /* // sanity check */
        //   /* if (n_node > 20){ */
        //   /*   printf("Element %d\n", i); */
        //   /*   perror("Greater than 20 nodes\n"); */
        //   /*   exit(1); */
        //   /* } */

        // Field 10: Not Used
        cursor += isz;

        // Field 11: Element number
        elem[c++] = fast_atoi(cursor, isz);
        cursor += isz;
        /* printf("reading element %d\n", elem[c - 1]); */

        // Need an additional value for consistency with other formats
        elem[c++] = 0;

        // Read the node indices in this line
        int n_read = n_values - 11;
        for (j = 0; j < n_read; j++) {
            elem[c++] = fast_atoi(cursor, isz);
            cursor += isz;
        }

        // There's a second line if we've not read all the nodes
        if (n_read < n_node) {
            cfile.getline(line, sizeof(line));
            cursor = line;
            n_read = cfile.gcount() / isz;
            for (j = 0; j < n_read; j++) {
                elem[c++] = fast_atoi(cursor, isz);
                cursor += isz;
            }
        }

        // Edge case where missing midside nodes are not written (because
        // MAPDL refuses to write zeros at the end of a line)
        if (n_node < 20 && n_node > 10) {
            for (j = n_node; j < 20; j++) {
                elem[c++] = 0;
            }
        }
    }

    // Return total data read
    elem_off[nelem] = c;
    return c;
}

// Simply write an array to disk as ASCII
int write_array_ascii(const char *filename, const double *arr, const int nvalues) {
    FILE *stream = fopen(filename, "w");
    int i;

    for (i = 0; i < nvalues; i++) {
        fprintf(stream, "%20.12E\n", arr[i]);
    }

    fclose(stream);

    return 0;
}

int ReadNBlockCfile(
    std::ifstream &cfile,
    int *nnum,
    double *nodes,
    double *node_angles,
    const int nnodes,
    std::array<int, 3> d_size,
    const int f_size) {

    int i, j, i_val, eol;
    char line[256];

    for (i = 0; i < nnodes; i++) {
        // Read a line from the file
        cfile.getline(line, sizeof(line));

        // It's possible that less nodes are written to the record than
        // indicated.  In this case the line starts with a -1
        if (line[0] == '-') {
            break;
        }

        std::streamsize count = cfile.gcount();
        char *cursor = line;

        i_val = fast_atoi(cursor, d_size[0]);
#ifdef DEBUG
        printf("%8d    \n", i_val);
#endif
        nnum[i] = i_val;

        cursor += d_size[0];
        cursor += d_size[1];
        cursor += d_size[2];

        // read nodes
        int n_read = (count - d_size[0] * 3) / f_size;
        int n_read_nodes = (n_read < 3) ? n_read : 3;
        for (j = 0; j < n_read_nodes; j++) {
            ans_strtod(cursor, f_size, &nodes[3 * i + j]);
            cursor += f_size;
        }
        // fill in unread nodes with zeros
        for (; j < 3; j++) {
            nodes[3 * i + j] = 0;
        }

        // read in node angles if applicable
        for (; j < n_read; j++) {
            eol = ans_strtod(cursor, f_size, &node_angles[3 * i + (j - 3)]);
            if (eol)
                break;
            cursor += f_size;
        }
    }

    return i;
}

int safe_int(const std::string &value) {
    try {
        return std::stoi(value);
    } catch (...) {
        return -1;
    }
}

int getSizeOfEBLOCK(const std::string &line) {
    size_t lastComma = line.rfind(',');
    if (lastComma == std::string::npos) {
        throw std::runtime_error("No comma found in the input line");
    }

    int nelem = std::stoi(line.substr(lastComma + 1)); // Extract and convert
    if (nelem == 0) {
        throw std::runtime_error("Unable to read element block");
    }

    return nelem;
}

struct NodeBlockFormat {
    std::array<int, 3> d_size;
    int f_size;
    int nfields;
    int nexp;
};

NodeBlockFormat GetNodeBlockFormat(const std::string &str) {
    std::string cleaned_str = str;
    cleaned_str.erase(remove(cleaned_str.begin(), cleaned_str.end(), '('), cleaned_str.end());
    cleaned_str.erase(remove(cleaned_str.begin(), cleaned_str.end(), ')'), cleaned_str.end());

    std::stringstream ss(cleaned_str);
    std::string field;
    std::vector<std::string> fields;

    while (getline(ss, field, ',')) {
        fields.push_back(field);
    }

    NodeBlockFormat format;
    format.d_size.fill(0);
    format.nexp = 2; // default when missing
    format.nfields = 6;
    format.f_size = 21;
    int c = 0;

    for (const auto &field : fields) {
        if (field.find('i') != std::string::npos) {
            std::stringstream field_ss(field);
            std::string item;
            getline(field_ss, item, 'i');
            int count = std::stoi(item);
            getline(field_ss, item);
            int size = std::stoi(item);

            for (int n = 0; n < count; ++n) {
                format.d_size[c++] = size;
            }
        } else if (field.find('e') != std::string::npos) {
            std::stringstream field_ss(field);
            std::string item;
            getline(field_ss, item, 'e');
            format.nfields = std::stoi(item);

            getline(field_ss, item, '.');
            format.f_size = std::stoi(item);

            if (getline(field_ss, item, 'e')) {
                format.nexp = std::stoi(item);
            }
        }
    }

    return format;
}

// Expect:
// [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 48]
// or:
// [25, -39, 48]
// where the negative number indicates a contigious set of integers between it and the
// previous number
NDArray<int, 1> InterpretComponent(const std::vector<int> &component) {
    size_t size = 0;

    // First pass: determine the total size
    for (size_t i = 0; i < component.size(); ++i) {
        if (component[i] > 0) {
            ++size;
        } else {
            size += std::abs(component[i]) - std::abs(component[i - 1]);
        }
    }

    // Allocate the array
    NDArray<int, 1> comp_arr = MakeNDArray<int, 1>({(int)size});
    int *comp_data = comp_arr.data();

    // Second pass: populate the array
    size_t index = 0;
    for (size_t i = 0; i < component.size(); ++i) {
        if (component[i] > 0) {
            comp_data[index++] = component[i];
        } else {
            int start = std::abs(component[i - 1]) + 1;
            int end = std::abs(component[i]) + 1;
            for (int j = start; j < end; ++j) {
                comp_data[index++] = j;
            }
        }
    }

    return comp_arr;
}

class Archive {

  public:
    std::string filename;
    bool read_parameters;
    bool debug;
    bool read_eblock;
    bool eblock_is_read = false;
    bool nblock_is_read = false;
    std::ifstream cfile;

    std::vector<std::vector<int>> elem_type;
    std::unordered_map<int, std::vector<std::vector<int>>> keyopt;

    // Element block
    int n_elem = 0;
    NDArray<int, 1> elem_arr;
    NDArray<int, 1> elem_off_arr;

    // Node block
    int n_nodes = 0;
    int nblock_start = 0;
    int nblock_end = 0;
    NDArray<double, 2> nodes_arr;
    NDArray<double, 2> node_angles_arr;
    NDArray<int, 1> nnum_arr;

    // real constants
    std::vector<int> rnum;                 // real constant number
    std::vector<std::vector<double>> rdat; // real constant data

    // components
    std::unordered_map<std::string, NDArray<int, 1>> elem_comps;
    std::unordered_map<std::string, NDArray<int, 1>> node_comps;

    Archive(
        const std::string &fname,
        bool readParams = false,
        bool dbg = false,
        bool readEblock = true)
        : filename(fname), read_parameters(readParams), debug(dbg), read_eblock(readEblock),
          cfile(fname) {

        // Likely bogus leak warnings. See:
        // https://nanobind.readthedocs.io/en/latest/faq.html#why-am-i-getting-errors-about-leaked-functions-and-types
        nb::set_leak_warnings(false);

        std::ifstream cfile(filename);

        if (!cfile.is_open()) {
            throw std::runtime_error("No such file or directory: '" + filename + "'");
        }

        if (read_parameters) {
            throw std::runtime_error("Read parameters has been deprecated");
        }
    }

    // EXAMPLE: ET, 4, 186
    // Element type, element number, element type
    void ReadETLine() {
        if (debug) {
            std::cout << "reading ET" << std::endl;
        }

        std::string line;
        std::getline(cfile, line);
        std::istringstream iss(line);
        std::string token;
        std::vector<int> et_vals;

        // Must start with ET (simply skip first token)
        std::getline(iss, token, ',');
        // if (token.compare(0, 2, "ET") != 0 && token.compare(0, 2, "et") != 0) {
        //     return;
        // }

        while (std::getline(iss, token, ',')) {
            try {
                et_vals.push_back(std::stoi(token));
            } catch (const std::invalid_argument &) {
                return;
            }
        }
        if (et_vals.size() == 2) {
            elem_type.push_back(et_vals);
        }
    }

    // Read ETBLOCK line
    void ReadETBlock() {
        std::string line;
        std::string token;
        std::vector<int> values;

        // start by reading in the first line, which must be an ETBLOCK
        std::getline(cfile, line);
        std::istringstream iss(line);

        // skip first item (ETBLOCK)
        std::getline(iss, token, ',');

        // read the number of items in the block
        std::getline(iss, token, ',');
        int n_items = 0;
        try {
            n_items = std::stoi(token);
        } catch (...) {
            std::cerr << "Failed to read ETBLOCK n_items" << std::endl;
            return;
        }

        // Skip format line (2i9,19a9)
        std::getline(cfile, line);

        // Read each item
        int elem_id, elem_type_id;
        for (int i = 0; i < n_items; i++) {
            std::getline(cfile, line);
            iss.clear();
            iss.str(line);

            // We only care about the first two integers in the block, the rest are keyopts
            if (!(iss >> elem_id >> elem_type_id)) {
                // allow a soft return
                std::cerr << "Failed to read ETBLOCK entry at line: " << line << std::endl;
                return;
            }
            std::vector<int> entry;
            entry.push_back(elem_id);
            entry.push_back(elem_type_id);
            elem_type.push_back(entry);
        }
    }

    void ReadEBlock() {

        // start by reading in the first line, which must be an EBLOCK
        std::string line;
        std::getline(cfile, line);

        // Sometimes, DAT files contain two EBLOCKs. Read only the first block.
        if (eblock_is_read) {
            return;
        }

        std::istringstream iss(line);

        // Only read in SOLID eblocks
        std::transform(line.begin(), line.end(), line.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        if (line.find("solid") == std::string::npos) {
            return;
        }

        // Get size of EBLOCK from the last item in the line
        // Example: "EBLOCK,19,SOLID,,3588"
        n_elem = getSizeOfEBLOCK(line);

        // we have to allocate memory for the maximum size since we don't know that a priori
        int *elem_data = AllocateArray<int>(n_elem * 30);
        elem_off_arr = MakeNDArray<int, 1>({n_elem + 1});
        int elem_sz = ReadEBlockCfile(cfile, elem_off_arr.data(), elem_data, n_elem);

        // wrap the raw data but limit it to the size of the number of elements read
        // Note: This is faster than reallocating a new array but will use more memory
        elem_arr = WrapNDarray<int, 1>(elem_data, {elem_sz});

        // must mark eblock is read since we can only read one eblock per archive file
        eblock_is_read = true;
    }

    // Read:
    // KEYOPT, ITYPE, KNUM, VALUE
    void ReadKEYOPTLine() {
        if (debug) {
            std::cout << "reading KEYOPT" << std::endl;
        }

        std::string line;
        std::getline(cfile, line);
        std::istringstream iss(line);
        std::string token;

        // Skip the first token (KEYOPT)
        std::getline(iss, token, ',');

        // element number is the first item
        int etype;
        std::getline(iss, token, ',');
        try {
            etype = std::stoi(token);
        } catch (const std::invalid_argument &) {
            // soft error
            std::cerr << "Invalid format in KEYOPT line for elem_num:" << line << std::endl;
            return;
        }

        // Read the rest of the values (Element Type, Key Option, Value)
        std::vector<int> keyopt_vals;
        while (std::getline(iss, token, ',')) {
            try {
                keyopt_vals.push_back(std::stoi(token));
            } catch (const std::invalid_argument &) {
                // soft error
                std::cerr << "Invalid format in KEYOPT line:" << line << std::endl;
                return;
            }
        }

        if (keyopt_vals.size() == 2) {
            keyopt[etype].push_back(keyopt_vals);
        }
    }

    void ReadRLBLOCK() {
        std::string line;
        std::getline(cfile, line);
        if (debug) {
            std::cout << "reading RLBLOCK" << std::endl;
        }

        std::vector<int> set_dat;
        std::istringstream iss(line);
        std::string token;
        // while (std::getline(iss, token, ',')) {
        //     set_dat.push_back(std::stoi(token));
        // }

        // skip starting RLBLOCK command
        std::getline(iss, token, ',');

        int nset, maxset, maxitems, nperline;
        try {
            std::getline(iss, token, ',');
            nset = std::stoi(token);
        } catch (...) {
            std::cerr << "Error RLBLOCK line when reading nset:" << line << std::endl;
            return;
        }
        try {
            std::getline(iss, token, ',');
            maxset = std::stoi(token);
        } catch (...) {
            std::cerr << "Error RLBLOCK line when reading maxset:" << line << std::endl;
            return;
        }
        try {
            std::getline(iss, token, ',');
            maxitems = std::stoi(token);
        } catch (...) {
            std::cerr << "Error RLBLOCK line when reading maxitems:" << line << std::endl;
            return;
        }
        try {
            std::getline(iss, token, ',');
            nperline = std::stoi(token);
        } catch (...) {
            std::cerr << "Error RLBLOCK line when reading nperline:" << line << std::endl;
            return;
        }

        // Skip next two lines
        std::getline(cfile, line); // (2i8,6g16.9)
        std::getline(cfile, line); // (7g16.9)

        for (int i = 0; i < nset; i++) {
            // read real constant data in the form of:
            //        2       6  1.00000000     0.566900000E-07  0.00000000      0.000...
            std::getline(cfile, line);
            std::vector<double> rcon;

            // real constant number
            try {
                rnum.push_back(std::stoi(line.substr(0, 8)));
            } catch (...) {
                std::cerr << "Failed to read real constant ID when reading:" << line
                          << std::endl;
                return;
            }

            int ncon = std::stoi(line.substr(8, 8));
            if (ncon > 6) {
                for (int j = 0; j < 6; j++) {
                    rcon.push_back(std::stod(line.substr(16 + 16 * j, 16)));
                    ncon--;
                }

                std::getline(cfile, line);

                while (ncon > 0) {
                    for (int j = 0; j < 7 && ncon > 0; j++) {
                        rcon.push_back(std::stod(line.substr(16 * j, 16)));
                        ncon--;
                    }
                    if (ncon > 0) {
                        std::getline(cfile, line);
                    }
                }
            } else {
                for (int j = 0; j < ncon; j++) {
                    rcon.push_back(std::stod(line.substr(16 + 16 * j, 16)));
                }
            }

            rdat.push_back(rcon);
        }
    }

    void ReadNBlock() {

        // Before reading NBLOCK, save where the nblock started
        nblock_start = cfile.tellg();

        // start by reading in the first line, which must be a NBLOCK
        std::string line;
        std::getline(cfile, line);

        // Sometimes, DAT files contains multiple node blocks. Read only the first block.
        if (nblock_is_read) {
            return;
        }

        // Get size of NBLOCK
        try {
            n_nodes = std::stoi(line.substr(line.rfind(',') + 1));
        } catch (...) {
            std::cerr << "Failed to read number of nodes when reading:" << line << std::endl;
            return;
        }
        int *nnum_data = AllocateArray<int>(n_nodes);
        double *nodes_data = AllocateArray<double>(n_nodes * 3);

        // often angles aren't written, so it makes sense to initialize this to
        // zero
        double *node_angles_data = AllocateArray<double>(n_nodes * 3, true);

        // Get format of nblock
        std::getline(cfile, line);
        NodeBlockFormat nfmt = GetNodeBlockFormat(line);
        if (debug) {
            std::cout << "Reading " << n_nodes << " nodes" << std::endl;
        }

        // Return actual number of nodes read and wrap the raw data
        n_nodes = ReadNBlockCfile(
            cfile,
            nnum_data,
            nodes_data,
            node_angles_data,
            n_nodes,
            nfmt.d_size,
            nfmt.f_size);
        nnum_arr = WrapNDarray<int, 1>(nnum_data, {n_nodes});
        nodes_arr = WrapNDarray<double, 2>(nodes_data, {n_nodes, 3});
        node_angles_arr = WrapNDarray<double, 2>(node_angles_data, {n_nodes, 3});

        // Read final line, this is always "N,R5.3,LOC, -1," and store file
        // position. This is used for later access (or rewrite) of the node
        // block.
        std::getline(cfile, line);
        nblock_end = cfile.tellg();

        // Must mark nblock is read since we can only read one nblock per archive file.
        nblock_is_read = true;
        if (debug) {
            std::cout << "Last line is: " << line << std::endl;
            std::cout << "NBLOCK complete, read " << n_nodes << " nodes." << std::endl;
        }
    }

    void ReadCMBlock() {
        std::string line;
        std::getline(cfile, line);

        if (line.compare(0, 8, "CMBLOCK,") != 0 && line.compare(0, 8, "cmblock,") != 0) {
            return;
        }

        if (debug) {
            std::cout << "reading CMBLOCK" << std::endl;
        }

        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> split_line;

        while (std::getline(iss, token, ',')) {
            split_line.push_back(token);
        }

        if (split_line.size() < 4) {
            std::cerr << "Poor formatting of CMBLOCK: " << line << std::endl;
            return;
        }

        int ncomp;
        try {
            ncomp = std::stoi(split_line[3].substr(0, split_line[3].find('!')).c_str());
        } catch (...) {
            std::cerr << "Poor formatting of CMBLOCK: " << line << std::endl;
            return;
        }

        std::getline(cfile, line);
        int isz;
        try {
            isz = std::stoi(
                line.substr(line.find('i') + 1, line.find(')') - line.find('i') - 1));
        } catch (...) {
            std::cerr << "Failed to read integer size in CMBLOCK" << std::endl;
            return;
        }

        int nblock;
        try {
            nblock = std::stoi(
                line.substr(line.find('(') + 1, line.find('i') - line.find('(') - 1));
        } catch (...) {
            std::cerr << "Failed to read number of integers per line in CMBLOCK" << std::endl;
            return;
        }

        std::vector<int> component(ncomp);
        std::string tempstr(isz, '\0');

        for (int i = 0; i < ncomp; ++i) {
            if (i % nblock == 0) {
                std::getline(cfile, line);
            }
            try {
                component[i] = std::stoi(line.substr(isz * (i % nblock), isz));
            } catch (...) {
                std::cerr << "Failed to parse integer from CMBLOCK line: " << line
                          << std::endl;
                return;
            }
        }

        // get component name
        std::string comname = split_line[1];
        comname.erase(comname.find_last_not_of(' ') + 1, std::string::npos);

        std::string line_comp_type = split_line[2];

        if (line_comp_type.find("NODE") != std::string::npos) {
            node_comps[comname] = InterpretComponent(component);
        } else if (line_comp_type.find("ELEM") != std::string::npos) {
            elem_comps[comname] = InterpretComponent(component);
        }
    }

    void Read() {
        int first_char, next_char;
        std::string line;

        int position_start, position_end;
        while (true) {
            // Parse based on the first character rather than reading the entire
            // line. It's faster and the parsing logic is always based on the first
            // character
            first_char = cfile.peek();
            // if (debug) {
            //   std::cout << "Read character: " << static_cast<char>(first_char) <<
            //   std::endl;
            // }
            if (cfile.eof()) {
                break;

            } else if (first_char == 'E' || first_char == 'e') {
                // E commands (ET or ETBLOCK)

                if (debug) {
                    std::cout << "Read E" << std::endl;
                }

                // get line but do not advance
                position_start = cfile.tellg();
                std::getline(cfile, line);
                position_end = cfile.tellg();
                cfile.seekg(position_start);

                // Record element type
                if (line.compare(0, 3, "ET,") == 0 || line.compare(0, 3, "et,") == 0) {
                    ReadETLine();
                } else if (
                    line.compare(0, 7, "ETBLOCK") == 0 ||
                    line.compare(0, 7, "etblock") == 0) {
                    ReadETBlock();
                } else if (
                    line.compare(0, 6, "EBLOCK") == 0 ||
                    line.compare(0, 6, "eblock") == 0 && read_eblock) {
                    ReadEBlock();
                } else {
                    cfile.seekg(position_end);
                }
            } else if (first_char == 'K' || first_char == 'k') {
                if (debug) {
                    std::cout << "Read K" << std::endl;
                }

                // get line but do not advance
                position_start = cfile.tellg();
                std::getline(cfile, line);
                position_end = cfile.tellg();
                cfile.seekg(position_start);

                // Record keyopt
                if (line.compare(0, 5, "KEYOP") == 0 || line.compare(0, 5, "keyop") == 0) {
                    ReadKEYOPTLine();
                } else {
                    cfile.seekg(position_end);
                }

            } else if (first_char == 'R' || first_char == 'r') {
                // test for RLBLOCK
                if (debug) {
                    std::cout << "Read R" << std::endl;
                }

                // get line but do not advance
                position_start = cfile.tellg();
                std::getline(cfile, line);
                position_end = cfile.tellg();
                cfile.seekg(position_start);

                // Record keyopt
                if (line.compare(0, 5, "RLBLO") == 0 || line.compare(0, 5, "rlblo") == 0) {
                    ReadRLBLOCK();
                } else {
                    cfile.seekg(position_end);
                }

            } else if (first_char == 'N' || first_char == 'n') {
                // test for NBLOCK

                if (debug) {
                    std::cout << "Read N" << std::endl;
                }

                // get line but do not advance
                position_start = cfile.tellg();
                std::getline(cfile, line);
                position_end = cfile.tellg();
                cfile.seekg(position_start);

                // Record node block
                if (line.compare(0, 5, "NBLOC") == 0 || line.compare(0, 5, "nbloc") == 0) {
                    ReadNBlock();
                } else {
                    cfile.seekg(position_end);
                }
            } else if (first_char == 'C' || first_char == 'c') {
                if (debug) {
                    std::cout << "Read C" << std::endl;
                }

                // get line but do not advance
                position_start = cfile.tellg();
                std::getline(cfile, line);
                position_end = cfile.tellg();
                cfile.seekg(position_start);

                // Record component block
                if (line.compare(0, 5, "CMBLO") == 0 || line.compare(0, 5, "cmblo") == 0) {
                    ReadCMBlock();
                } else {
                    cfile.seekg(position_end);
                }
            } else {
                if (debug) {
                    std::cout << "No match, continuing..." << std::endl;
                }
                // Skip remainder of the line
                cfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
        }
    }

    // Convert ansys style connectivity to VTK connectivity
    // type_ref is a mapping between ansys element types and VTK element types
    nb::tuple ToVTK(NDArray<int, 1> type_ref) {
        NDArray<int64_t, 1> offset_arr = MakeNDArray<int64_t, 1>({n_elem + 1});
        NDArray<uint8_t, 1> celltypes_arr = MakeNDArray<uint8_t, 1>({n_elem});

        // Allocate connectivity: max cell size 20 for VTK_HEXAHEDRAL
        int64_t *cell_data = new int64_t[n_elem * 20];

        int loc = ans_to_vtk(
            n_elem,
            elem_arr.data(),
            elem_off_arr.data(),
            type_ref.data(),
            n_nodes,
            nnum_arr.data(),
            offset_arr.data(),
            cell_data,
            celltypes_arr.data());

        NDArray<int64_t, 1> cells_arr = WrapNDarray<int64_t, 1>(cell_data, {loc});
        return nb::make_tuple(offset_arr, celltypes_arr, cells_arr);
    }

    ~Archive() { cfile.close(); }

}; // Archive

// Convert ansys style connectivity to VTK connectivity
// type_ref is a mapping between ansys element types and VTK element types
// def ans_vtk_convert(
//     elem: NDArray[np.int32],
//     elem_off: NDArray[np.int32],
//     type_ref: NDArray[np.int32],
//     nnum: NDArray[np.int32],
// ) -> Tuple[NDArray[U], NDArray[np.uint8], NDArray[U]]: ...

nb::tuple ConvertToVTK(
    NDArray<int, 1> elem_arr,
    NDArray<int, 1> elem_off_arr,
    NDArray<int, 1> type_ref,
    NDArray<int, 1> nnum_arr) {

    int n_elem = elem_off_arr.size() - 1;
    int n_nodes = nnum_arr.size();

    NDArray<int64_t, 1> offset_arr = MakeNDArray<int64_t, 1>({n_elem + 1});
    NDArray<uint8_t, 1> celltypes_arr = MakeNDArray<uint8_t, 1>({n_elem});

    // Allocate connectivity: max cell size 20 for VTK_HEXAHEDRAL
    int64_t *cell_data = new int64_t[n_elem * 20];

    int loc = ans_to_vtk(
        n_elem,
        elem_arr.data(),
        elem_off_arr.data(),
        type_ref.data(),
        n_nodes,
        nnum_arr.data(),
        offset_arr.data(),
        cell_data,
        celltypes_arr.data());

    NDArray<int64_t, 1> cells_arr = WrapNDarray<int64_t, 1>(cell_data, {loc});
    return nb::make_tuple(offset_arr, celltypes_arr, cells_arr);
}

NB_MODULE(_reader, m) {
    m.def("ans_to_vtk", &ConvertToVTK);
    nb::class_<Archive>(m, "Archive")
        .def(
            nb::init<const std::string &, bool, bool, bool>(),
            "fname"_a,
            "read_params"_a = false,
            "debug"_a = false,
            "read_eblock"_a = true,
            "This class represents an Ansys CDB Archive.")
        .def_ro("n_elem", &Archive::n_elem)
        .def_ro("elem_type", &Archive::elem_type)
        .def_ro("elem", &Archive::elem_arr, nb::rv_policy::automatic)
        .def_ro("elem_off", &Archive::elem_off_arr, nb::rv_policy::automatic)
        .def_ro("keyopt", &Archive::keyopt)
        .def_ro("rdat", &Archive::rdat)
        .def_ro("rnum", &Archive::rnum)
        .def_ro("elem_comps", &Archive::elem_comps, nb::rv_policy::automatic)
        .def_ro("node_comps", &Archive::node_comps, nb::rv_policy::automatic)
        .def_ro("nodes", &Archive::nodes_arr, nb::rv_policy::automatic)
        .def_ro("node_angles", &Archive::node_angles_arr, nb::rv_policy::automatic)
        .def_ro("nnum", &Archive::nnum_arr, nb::rv_policy::automatic)
        .def_ro("n_nodes", &Archive::n_nodes)
        .def_ro("nblock_start", &Archive::nblock_start)
        .def_ro("nblock_end", &Archive::nblock_end)
        .def_ro("nblock_end", &Archive::nblock_end)
        .def("to_vtk", &Archive::ToVTK)
        .def("read", &Archive::Read)
        .def("read_nblock", &Archive::ReadNBlock)
        .def("read_rlblock", &Archive::ReadRLBLOCK)
        .def("read_keyopt_line", &Archive::ReadKEYOPTLine)
        .def("read_et_line", &Archive::ReadETLine)
        .def("read_etblock", &Archive::ReadETBlock)
        .def("read_cmblock", &Archive::ReadCMBlock)
        .def("read_eblock", &Archive::ReadEBlock);
}
