#include <algorithm>
#include <climits>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
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

/* We are on Windows */
#if defined(_WIN32) || defined(_WIN64)
#define strtok_r strtok_s
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
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

//=============================================================================
// Block header parsing
//=============================================================================
// NBLOCK and EBLOCK headers are comma delimited and their trailing count fields
// are optional:
//
//   NBLOCK,NUMFIELD,Solkey,NDMAX,NDSEL
//   EBLOCK,NUMNODES,Solkey,NDMAX,NDSEL
//
// Real decks write all of "NBLOCK,6,SOLID,70631,10504", "NBLOCK,6,SOLID,105",
// "EBLOCK,19,SOLID      ,80" and "EBLOCK,19,SOLID,". Splitting on the last
// comma therefore reads "SOLID" or "" as the count, so the fields have to be
// addressed by position instead.
static std::vector<std::string> SplitHeaderFields(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, ',')) {
        size_t first = field.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            fields.push_back(std::string());
        } else {
            size_t last = field.find_last_not_of(" \t\r\n");
            fields.push_back(field.substr(first, last - first + 1));
        }
    }
    return fields;
}

// Read one header field as a positive count. Returns -1 when the field is
// missing, blank, or not a plain integer, which the caller treats as "the deck
// did not tell us, go and count the records".
static int ParseOptionalCount(const std::vector<std::string> &fields, size_t index) {
    if (index >= fields.size() || fields[index].empty()) {
        return -1;
    }
    const std::string &field = fields[index];
    size_t consumed = 0;
    long value = 0;
    try {
        value = std::stol(field, &consumed);
    } catch (const std::exception &) {
        return -1;
    }
    if (consumed != field.size() || value <= 0 || value > INT_MAX) {
        return -1;
    }
    return (int)value;
}

// Pick the record count out of an already split block header, preferring NDSEL
// (the number actually written) and falling back to NDMAX. Returns -1 when the
// deck omitted both.
static inline int GetBlockRecordCount(const std::vector<std::string> &fields) {
    int count = ParseOptionalCount(fields, 4); // NDSEL
    if (count < 0) {
        count = ParseOptionalCount(fields, 3); // NDMAX
    }
    return count;
}

//=============================================================================
// NBLOCK line classification
//=============================================================================
// Returns 1 for a node record, 0 for a clean end of the block, and -1 for a
// line that is neither (truncated or otherwise corrupt), which the caller
// reports as a corrupt archive.
//
// A node record starts with a right justified positive node number, so the
// first ``id_width`` characters are spaces and digits. The block terminator
// "N,R5.3,LOC,-1" starts with a letter, a short record list ends with "-1", and
// the next keyword in the file starts with a letter or "/".
static inline int ClassifyNBlockLine(const char *pos, const char *file_end, int id_width) {
    if (pos >= file_end) {
        return 0; // end of file ends the block
    }
    if (*pos == '\n' || *pos == '\r') {
        return 0; // a blank line ends the block
    }

    bool seen_digit = false;
    for (int c = 0; c < id_width && pos + c < file_end; ++c) {
        char ch = pos[c];
        if (ch == '\n' || ch == '\r') {
            break;
        } else if (ch == ' ') {
            continue;
        } else if (ch >= '0' && ch <= '9') {
            seen_digit = true;
        } else if (
            ch == '-' || ch == '/' || (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')) {
            return 0; // "-1", "N,R5.3,LOC,-1", "EBLOCK", "/EOF", ...
        } else {
            return -1; // NUL padding or other junk: the archive is corrupt
        }
    }

    return seen_digit ? 1 : 0;
}

// Reads various ansys float formats in the form of
// "3.7826539829200E+00"
// "1.0000000000000E-001"
// "        -6.01203 "
// "29.184036609179"
//
// The number of digits before the decimal point is not fixed: scientific
// notation always writes exactly one, but plain "F" style output (for example
// the "(3i8,6e16.9)" that some pre-processors emit) writes as many as the value
// needs. Every loop below advances ``raw`` unconditionally so that an
// unexpected character ends the field rather than spinning forever.
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

    // A field containing nothing but whitespace is zero, not garbage.
    if (raw >= end) {
        *arr = 0;
        return 0;
    }

    // either a number or a sign
    if (*raw == '-') {
        sign = -1;
        ++raw;
    } else if (*raw == '+') {
        ++raw;
    }

    // Integer part. Use integer arithmetic and then convert to a float.
    uint64_t val_int = 0;
    while (raw < end && *raw >= '0' && *raw <= '9') {
        val_int = val_int * 10 + (uint64_t)(*raw - '0');
        raw++;
    }

    // Fractional part, accumulated into the same integer and scaled below.
    int decimal_digits = 0;
    if (raw < end && *raw == '.') {
        raw++;
        while (raw < end && *raw >= '0' && *raw <= '9') {
            val_int = val_int * 10 + (uint64_t)(*raw - '0');
            decimal_digits++;
            raw++;
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
    if (raw < end && (*raw == 'e' || *raw == 'E')) { // incredibly, can be lowercase
        raw++;                                       // skip "E"
        // usually a sign of some sort, but it is not guaranteed
        if (raw < end && (*raw == '-' || *raw == '+')) {
            if (*raw == '-') {
                esign = -1;
            }
            raw++;
        }

        // Only digits belong to the exponent. Guarding on them keeps a trailing
        // "\r" or a short field from being folded in as if it were one.
        while (raw < end && *raw >= '0' && *raw <= '9') {
            evalue = evalue * 10 + (*raw - '0');
            raw++;
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

class MemoryMappedFile {
  private:
    size_t size;
    char *start;
#ifdef _WIN32
    HANDLE fileHandle;
    HANDLE mapHandle;
#else
    int fd;
#endif

  public:
    std::string line;
    char *current;

    MemoryMappedFile(const char *filename)
        : start(nullptr), current(nullptr), size(0)
#ifdef _WIN32
          ,
          fileHandle(INVALID_HANDLE_VALUE), mapHandle(nullptr)
#else
          ,
          fd(-1)
#endif
    {
#ifdef _WIN32
        fileHandle = CreateFile(
            filename,
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (fileHandle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Error opening file");
        }

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(fileHandle, &fileSize)) {
            CloseHandle(fileHandle);
            throw std::runtime_error("Error getting file size");
        }

        size = static_cast<size_t>(fileSize.QuadPart);
        mapHandle = CreateFileMapping(fileHandle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapHandle == nullptr) {
            CloseHandle(fileHandle);
            throw std::runtime_error("Error creating file mapping");
        }

        start = static_cast<char *>(MapViewOfFile(mapHandle, FILE_MAP_READ, 0, 0, size));
        if (start == nullptr) {
            CloseHandle(mapHandle);
            CloseHandle(fileHandle);
            throw std::runtime_error("Error mapping file");
        }
#else
        fd = open(filename, O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Error opening file");
        }

        struct stat st;
        if (fstat(fd, &st) == -1) {
            close(fd);
            throw std::runtime_error("Error getting file size");
        }

        size = st.st_size;
        start = static_cast<char *>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (start == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Error mapping file");
        }
#endif
        current = start;
    }

    ~MemoryMappedFile() {
        close_file();
#ifdef _WIN32
        if (fileHandle != INVALID_HANDLE_VALUE) {
            CloseHandle(fileHandle);
        }
        if (mapHandle != nullptr) {
            CloseHandle(mapHandle);
        }
#else
        if (fd != -1) {
            close(fd);
        }
#endif
    }

    void close_file() {
        if (start) {
#ifdef _WIN32
            UnmapViewOfFile(start);
#else
            munmap(start, size);
#endif
            start = nullptr;
            current = nullptr;
        }
    }

    char &operator[](size_t index) {
        // implement bounds checking?
        // if (index >= size) {
        //     throw std::out_of_range("Index out of bounds");
        // }
        return current[index];
    }

    void operator+=(size_t offset) { current += offset; }

    // Seek to the end of the line
    void seek_eol() {
        // check if at end of file
        if (current >= start + size) {
            // std::cout << "end" << std::endl;
            return;
        }

        while (current < start + size && *current != '\n') {
            current++;
        }

        if (current < start + size && *current == '\n') {
            current++;
        }
    }

    bool eof() { return current >= start + size; }

    // One past the last mapped byte. Needed to classify a line without running
    // off the end of the mapping.
    const char *file_end() const { return start + size; }

    // Rewind/forward to an absolute offset, as returned by tellg().
    void seekg(off_t pos) { current = start + pos; }

    bool read_line() {
        line.clear();
        if (current >= start + size) {
            return false;
        }

        char *line_start = current;
        while (current < start + size && *current != '\n') {
            line += *current++;
        }

        if (current < start + size && *current == '\n') {
            current++;
        }

        return line_start != current;
    }

    size_t current_line_length() const {
        char *temp = current;
        size_t length = 0;
        while (temp < start + size && *temp != '\n') {
            length++;
            temp++;
        }
        return length;
    }

    off_t tellg() const { return current - start; }

    void sanitize_line_ascii() {
        line.erase(
            std::remove_if(
                line.begin(), line.end(), [](unsigned char c) { return c < 32 || c > 126; }),
            line.end());
    }
};

// Count the node records of an NBLOCK whose header omitted NDMAX and NDSEL.
// Expects to be positioned on the first record and leaves the position
// unchanged. A corrupt record is included in the count so that the read pass
// still reaches it and reports it.
int CountNBlockRecords(MemoryMappedFile &memmap, int id_width) {
    off_t saved = memmap.tellg();
    int count = 0;

    while (!memmap.eof()) {
        int kind = ClassifyNBlockLine(memmap.current, memmap.file_end(), id_width);
        if (kind == 1) {
            count++;
            memmap.seek_eol();
            continue;
        }
        if (kind < 0) {
            count++;
        }
        break;
    }

    memmap.seekg(saved);
    return count;
}

// Upper bound on the number of element records of an EBLOCK whose header
// omitted its count. Expects to be positioned on the format line and leaves the
// position unchanged. An element occupies at least one line, so counting lines
// up to the "-1" terminator can only over-estimate, and ReadEBlockMemMap
// reports how many records it actually read.
int CountEBlockRecordLines(MemoryMappedFile &memmap, int isz) {
    off_t saved = memmap.tellg();
    memmap.seek_eol(); // skip the format line

    int count = 0;
    while (!memmap.eof()) {
        size_t line_length = memmap.current_line_length();
        if (line_length == 0) {
            break; // a blank line ends the block
        }
        // Never look past the end of the line: the last one of the file need
        // not be as wide as the integer field.
        if (checkneg(memmap.current, (int)std::min(line_length, (size_t)isz))) {
            break; // the "-1" terminator
        }
        count++;
        memmap.seek_eol();
    }

    memmap.seekg(saved);
    return count;
}

// Read up to ``nelem`` element records. ``isz`` is the width of the integer
// field, taken from the format line the memory map is positioned on. Reports
// the number of records actually read through ``n_elem_read``, which is less
// than ``nelem`` whenever the block ends early.
int ReadEBlockMemMap(
    MemoryMappedFile &memmap,
    int *elem_off,
    int *elem,
    const int nelem,
    const int isz,
    int *n_elem_read) {
    int i, j, n_node;

    int n_values;

    // Loop through elements, starting past the format line
    memmap.seek_eol();
    int c = 0;
    for (i = 0; i < nelem; ++i) {
        // A block that runs to the end of the file has no "-1" terminator to
        // stop on, and ``nelem`` is only an upper bound whenever the header did
        // not carry a count.
        if (memmap.eof()) {
            break;
        }

        // store start of each element
        elem_off[i] = c;

        // Read the line and determine the number of values. MAPDL does not write all the
        // values on a single line
        n_values = memmap.current_line_length() / isz;
        // std::cout << n_values << std::endl;

        // It's possible that less nodes are written to the record than
        // indicated.  In this case the line starts with a -1

        // Check if at end of the block
        if (checkneg(memmap.current, isz)) {
            memmap += isz;
            break;
        }

        // ANSYS archive format:
        // Field 1: material reference number
        elem[c++] = fast_atoi(memmap.current, isz);
        memmap += isz;

        // Field 2: element type number
        elem[c++] = fast_atoi(memmap.current, isz);
        memmap += isz;

        // Field 3: real constant reference number
        elem[c++] = fast_atoi(memmap.current, isz);
        memmap += isz;

        // Field 4: section number
        elem[c++] = fast_atoi(memmap.current, isz);
        memmap += isz;

        // Field 5: element coordinate system
        elem[c++] = fast_atoi(memmap.current, isz);
        memmap += isz;

        // Field 6: Birth/death flag
        elem[c++] = fast_atoi(memmap.current, isz);
        memmap += isz;

        // Field 7: Solid model reference
        elem[c++] = fast_atoi(memmap.current, isz);
        memmap += isz;

        // Field 8: Coded shape key
        elem[c++] = fast_atoi(memmap.current, isz);
        memmap += isz;

        // Field 9: Number of nodes, don't store this
        n_node = fast_atoi(memmap.current, isz);
        memmap += isz;

        //   /* // sanity check */
        //   /* if (n_node > 20){ */
        //   /*   printf("Element %d\n", i); */
        //   /*   perror("Greater than 20 nodes\n"); */
        //   /*   exit(1); */
        //   /* } */

        // Field 10: Not Used
        memmap += isz;

        // Field 11: Element number
        elem[c++] = fast_atoi(memmap.current, isz);
        memmap += isz;
        /* printf("reading element %d\n", elem[c - 1]); */

        // Need an additional value for consistency with other formats
        elem[c++] = 0;

        // Read the node indices in this line
        int n_read = n_values - 11;
        for (j = 0; j < n_read; j++) {
            elem[c++] = fast_atoi(memmap.current, isz);
            memmap += isz;
        }

        // There's a second line if we've not read all the nodes
        memmap.seek_eol();
        if (n_read < n_node) {
            n_read = memmap.current_line_length() / isz;
            for (j = 0; j < n_read; j++) {
                elem[c++] = fast_atoi(memmap.current, isz);
                memmap += isz;
            }
            memmap.seek_eol();
        }

        // Edge case where missing midside nodes are not written (because
        // MAPDL refuses to write zeros at the end of a line)
        if (n_node < 20 && n_node > 10) {
            for (j = n_node; j < 20; j++) {
                elem[c++] = 0;
            }
        }
    }

    // Return total data read. Write the closing offset at the number of records
    // actually read: when the block ends early (a "-1" before the declared
    // count, or a count that was only an upper bound) that is not ``nelem``.
    elem_off[i] = c;
    if (n_elem_read != NULL) {
        *n_elem_read = i;
    }
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

int ReadNBlockMemMap(
    MemoryMappedFile &memmap,
    int *nnum,
    double *nodes,
    double *node_angles,
    const int nnodes,
    std::array<int, 3> d_size,
    const int f_size) {

    int i, j, i_val, eol;
    for (i = 0; i < nnodes; i++) {
        // Read a line from the file
        int count = memmap.current_line_length();

        // It's possible that fewer nodes are written to the record than
        // indicated, and when the count came from NDMAX rather than NDSEL the
        // estimate is always high. Stop on anything that is not a node record:
        // a "-1", the "N,R5.3,LOC,-1" terminator, the next keyword, or EOF. A
        // line that is none of those falls through to the node number check
        // below, which reports the archive as corrupt.
        if (ClassifyNBlockLine(memmap.current, memmap.file_end(), d_size[0]) == 0) {
            break;
        }

        i_val = fast_atoi(memmap.current, d_size[0]);

        // Check if the node number is valid
        if (i_val < 1) {
            std::string error_message = "Failed to read NBLOCK node number.";
            if (i > 0) {
                error_message +=
                    " Last node number read was " + std::to_string(nnum[i - 1]) + ".";
            }
            error_message += " CDB is likely corrupt.";
            throw std::runtime_error(error_message);
        }

#ifdef DEBUG
        std::cout << "Node number " << i_val << std::endl;
#endif
        nnum[i] = i_val;

        memmap += d_size[0];
        memmap += d_size[1];
        memmap += d_size[2];

        // read nodes
        int n_read = (count - d_size[0] * 3) / f_size;
        // Check if the node number is valid
        if (n_read > 6) {
            std::string error_message = "Failed to read NBLOCK coordinates.";
            if (i > 0) {
                error_message +=
                    " Last node number read was " + std::to_string(nnum[i - 1]) + ".";
            }
            error_message += " CDB is likely corrupt.";
            throw std::runtime_error(error_message);
        }

#ifdef DEBUG
        std::cout << "n_read: " << n_read << std::endl;
#endif
        int n_read_nodes = (n_read < 3) ? n_read : 3;
        for (j = 0; j < n_read_nodes; j++) {
            ans_strtod(memmap.current, f_size, &nodes[3 * i + j]);
#ifdef DEBUG
            std::cout << " " << nodes[3 * i + j] << " ";
#endif
            memmap.current += f_size;
        }
#ifdef DEBUG
        std::cout << std::endl;
#endif

        // fill in unread nodes with zeros
        for (; j < 3; j++) {
            nodes[3 * i + j] = 0;
        }

        // read in node angles if applicable
        for (; j < n_read; j++) {
            eol = ans_strtod(memmap.current, f_size, &node_angles[3 * i + (j - 3)]);
            if (eol)
                break;
            memmap += f_size;
        }

        // seek to the end of the line
        memmap.seek_eol();
    }

    return i;
}

// Width of the integer field of an EBLOCK format line such as "(19i8)".
// Returns -1 when the line is not a format line.
int GetEBlockIntSize(MemoryMappedFile &memmap) {
    // Take a copy of the line: the memory map is not guaranteed to be NUL
    // terminated, so strchr and sscanf must not be pointed straight at it.
    std::string fmt(memmap.current, memmap.current_line_length());
    size_t i_pos = fmt.find('i');
    size_t close_paren_pos = fmt.find(')');
    if (i_pos == std::string::npos || close_paren_pos == std::string::npos ||
        i_pos > close_paren_pos) {
        return -1;
    }

    int isz = 0;
    if (sscanf(fmt.c_str() + i_pos + 1, "%d", &isz) != 1 || isz <= 0) {
        return -1;
    }
    return isz;
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

  private:
    // std::string line;
    bool debug;
    std::string filename;
    MemoryMappedFile memmap;

  public:
    bool read_parameters;

    bool read_eblock;
    bool eblock_is_read = false;
    bool nblock_is_read = false;

    std::vector<std::vector<int>> elem_type;
    std::unordered_map<int, std::vector<std::vector<int>>> keyopt;

    // Element block. Elements from every SOLID EBLOCK in the file accumulate
    // here in file order and are published through elem_arr/elem_off_arr.
    int n_elem = 0;
    std::vector<int> elem_data_all;
    std::vector<int> elem_off_all;
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
          memmap(fname.c_str()) {

        // Likely bogus leak warnings. See:
        // https://nanobind.readthedocs.io/en/latest/faq.html#why-am-i-getting-errors-about-leaked-functions-and-types
        nb::set_leak_warnings(false);

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

        // Assumes that the memory map is already at the ET line
        std::istringstream iss(memmap.line);
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

    // Read ETBLOCK
    void ReadETBlock() {
        std::string token;
        std::vector<int> values;

        // Assumes current line is an ETBLOCK
        std::istringstream iss(memmap.line);

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
        memmap.seek_eol();

        // Read each item
        int elem_id, elem_type_id;
        for (int i = 0; i < n_items; i++) {
            // std::getline(cfile, line);
            memmap.read_line();
            iss.clear();
            iss.str(memmap.line);

            // We only care about the first two integers in the block, the rest are keyopts
            if (!(iss >> elem_id >> elem_type_id)) {
                // allow a soft return
                std::cerr << "Failed to read ETBLOCK entry at line: " << memmap.line
                          << std::endl;
                return;
            }
            std::vector<int> entry;
            entry.push_back(elem_id);
            entry.push_back(elem_type_id);
            elem_type.push_back(entry);
        }
    }

    // Read EBLOCK
    //
    // A file may hold any number of SOLID element blocks; real decks split one
    // model across a block per component. Every SOLID block is read and the
    // elements are concatenated in file order. A model with a single block is
    // unaffected: the accumulated arrays are then exactly the one block.
    void ReadEBlock() {
        // Assumes already start of EBLOCK
        // Only read in SOLID eblocks
        std::transform(
            memmap.line.begin(), memmap.line.end(), memmap.line.begin(), [](unsigned char c) {
                return std::tolower(c);
            });
        if (memmap.line.find("solid") == std::string::npos) {
            return;
        }

        // Header: "EBLOCK,NUMNODES,Solkey,NDMAX,NDSEL", where NDMAX and NDSEL
        // are both optional. Examples seen in the wild: "EBLOCK,19,SOLID,4644,4",
        // "EBLOCK,19,SOLID      ,80" and "EBLOCK,19,SOLID,".
        std::vector<std::string> fields = SplitHeaderFields(memmap.line);
        int n_elem_declared = GetBlockRecordCount(fields);

        // The line after the header is the format, e.g. "(19i8)".
        int isz = GetEBlockIntSize(memmap);
        if (isz < 0) {
            throw std::runtime_error(
                "Unable to read the element block format line of \"" + memmap.line + "\".");
        }

        // When the deck did not write a count, count the records instead of
        // guessing. This is an upper bound; ReadEBlockMemMap reports the exact
        // number it read.
        int max_elem =
            (n_elem_declared > 0) ? n_elem_declared : CountEBlockRecordLines(memmap, isz);
        if (max_elem <= 0) {
            return; // an empty block contributes nothing
        }

        // We have to allocate for the maximum size since we don't know it a priori
        std::vector<int> block_elem((size_t)max_elem * 30);
        std::vector<int> block_off((size_t)max_elem + 1);

        int n_block_elem = 0;
        int elem_sz = ReadEBlockMemMap(
            memmap, block_off.data(), block_elem.data(), max_elem, isz, &n_block_elem);

        // Append this block to whatever has already been read, shifting its
        // offsets past the elements accumulated so far.
        int base = (int)elem_data_all.size();
        elem_off_all.reserve(elem_off_all.size() + (size_t)n_block_elem);
        for (int i = 0; i < n_block_elem; i++) {
            elem_off_all.push_back(block_off[i] + base);
        }
        elem_data_all.insert(
            elem_data_all.end(), block_elem.begin(), block_elem.begin() + elem_sz);

        eblock_is_read = true;
        BuildElementArrays();
    }

    // Publish the accumulated element blocks as the arrays exposed to Python.
    // Called after each block so that the members stay consistent whether the
    // file is read in one pass or read_eblock() is driven by hand.
    void BuildElementArrays() {
        n_elem = (int)elem_off_all.size();

        int *elem_data =
            AllocateArray<int>(elem_data_all.size() > 0 ? elem_data_all.size() : 1);
        std::copy(elem_data_all.begin(), elem_data_all.end(), elem_data);
        elem_arr = WrapNDarray<int, 1>(elem_data, {(int)elem_data_all.size()});

        // The offset array carries a closing entry with the total size.
        elem_off_arr = MakeNDArray<int, 1>({n_elem + 1});
        int *off_data = elem_off_arr.data();
        std::copy(elem_off_all.begin(), elem_off_all.end(), off_data);
        off_data[n_elem] = (int)elem_data_all.size();
    }

    // Read:
    // KEYOPT, ITYPE, KNUM, VALUE
    void ReadKEYOPTLine() {
        if (debug) {
            std::cout << "reading KEYOPT" << std::endl;
        }

        // Assumes at KEYOPT line
        std::istringstream iss(memmap.line);
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
            std::cerr << "Invalid format in KEYOPT line for elem_num:" << memmap.line
                      << std::endl;
            return;
        }

        // Read the rest of the values (Element Type, Key Option, Value)
        std::vector<int> keyopt_vals;
        while (std::getline(iss, token, ',')) {
            try {
                keyopt_vals.push_back(std::stoi(token));
            } catch (const std::invalid_argument &) {
                // soft error
                std::cerr << "Invalid format in KEYOPT line:" << memmap.line << std::endl;
                return;
            }
        }

        if (keyopt_vals.size() == 2) {
            keyopt[etype].push_back(keyopt_vals);
        }
    }

    // Read RLBLOCK
    void ReadRLBLOCK() {
        if (debug) {
            std::cout << "reading RLBLOCK" << std::endl;
        }

        std::vector<int> set_dat;

        // Assumes line at RLBLOCK
        std::istringstream iss(memmap.line);
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
            std::cerr << "Error RLBLOCK line when reading nset:" << memmap.line << std::endl;
            return;
        }
        try {
            std::getline(iss, token, ',');
            maxset = std::stoi(token);
        } catch (...) {
            std::cerr << "Error RLBLOCK line when reading maxset:" << memmap.line
                      << std::endl;
            return;
        }
        try {
            std::getline(iss, token, ',');
            maxitems = std::stoi(token);
        } catch (...) {
            std::cerr << "Error RLBLOCK line when reading maxitems:" << memmap.line
                      << std::endl;
            return;
        }
        try {
            std::getline(iss, token, ',');
            nperline = std::stoi(token);
        } catch (...) {
            std::cerr << "Error RLBLOCK line when reading nperline:" << memmap.line
                      << std::endl;
            return;
        }

        // Skip next two lines
        memmap.seek_eol(); // (2i8,6g16.9)
        memmap.seek_eol(); // (7g16.9)

        for (int i = 0; i < nset; i++) {
            // read real constant data in the form of:
            //        2       6  1.00000000     0.566900000E-07  0.00000000      0.000...
            memmap.read_line();
            std::vector<double> rcon;

            // real constant number
            try {
                rnum.push_back(std::stoi(memmap.line.substr(0, 8)));
            } catch (...) {
                std::cerr << "Failed to read real constant ID when reading:" << memmap.line
                          << std::endl;
                return;
            }

            int ncon = std::stoi(memmap.line.substr(8, 8));
            if (ncon > 6) {
                for (int j = 0; j < 6; j++) {
                    rcon.push_back(std::stod(memmap.line.substr(16 + 16 * j, 16)));
                    ncon--;
                }

                memmap.read_line();

                while (ncon > 0) {
                    for (int j = 0; j < 7 && ncon > 0; j++) {
                        rcon.push_back(std::stod(memmap.line.substr(16 * j, 16)));
                        ncon--;
                    }
                    if (ncon > 0) {
                        memmap.read_line();
                    }
                }
            } else {
                for (int j = 0; j < ncon; j++) {
                    rcon.push_back(std::stod(memmap.line.substr(16 + 16 * j, 16)));
                }
            }

            rdat.push_back(rcon);
        }
    }

    void ReadNBlock(const int pos) {

        // Sometimes, DAT files contains multiple node blocks. Read only the first block.
        if (nblock_is_read) {
            return;
        }
        nblock_start = pos;

        // Header: "NBLOCK,NUMFIELD,Solkey,NDMAX,NDSEL", where NDMAX and NDSEL
        // are both optional. Decks write all of "NBLOCK,6,SOLID,70631,10504",
        // "NBLOCK,6,SOLID,105" and "NBLOCK,6,SOLID", so the count has to be
        // taken by field position rather than from the last comma.
        // Assumes line is at NBLOCK
        std::vector<std::string> fields = SplitHeaderFields(memmap.line);
        int n_nodes_declared = GetBlockRecordCount(fields);

        // Get format of nblock, e.g. "(3i8,6e16.9)". This has to happen before
        // the allocation because counting the records needs the field widths.
        memmap.read_line();
        NodeBlockFormat nfmt = GetNodeBlockFormat(memmap.line);
        if (nfmt.d_size[0] <= 0 || nfmt.f_size <= 0) {
            throw std::runtime_error(
                "Unable to read the node block format line of \"" + memmap.line + "\".");
        }

        // When the deck did not write a count, count the records rather than
        // guessing at one and silently dropping the block.
        n_nodes = (n_nodes_declared > 0) ? n_nodes_declared
                                         : CountNBlockRecords(memmap, nfmt.d_size[0]);
        if (debug) {
            std::cout << "Reading " << n_nodes << " nodes" << std::endl;
        }

        int *nnum_data = AllocateArray<int>(n_nodes > 0 ? n_nodes : 1);
        double *nodes_data = AllocateArray<double>(n_nodes > 0 ? n_nodes * 3 : 1);

        // often angles aren't written, so it makes sense to initialize this to
        // zero
        double *node_angles_data = AllocateArray<double>(n_nodes > 0 ? n_nodes * 3 : 1, true);

        // Return actual number of nodes read and wrap the raw data
        // std::cout << memmap.tellg() << std::endl;
        n_nodes = ReadNBlockMemMap(
            memmap,
            nnum_data,
            nodes_data,
            node_angles_data,
            n_nodes,
            nfmt.d_size,
            nfmt.f_size);
        nnum_arr = WrapNDarray<int, 1>(nnum_data, {n_nodes});
        nodes_arr = WrapNDarray<double, 2>(nodes_data, {n_nodes, 3});
        node_angles_arr = WrapNDarray<double, 2>(node_angles_data, {n_nodes, 3});
        // std::cout << memmap.tellg() << std::endl;

        if (debug) {
            std::cout << "Finished reading " << n_nodes << " nodes" << std::endl;
        }

        // Read final line, this is always "N,R5.3,LOC, -1," and store file
        // position. This is used for later access (or rewrite) of the node
        // block.
        memmap.seek_eol();
        nblock_end = memmap.tellg();

        // Must mark nblock is read since we can only read one nblock per archive file.
        nblock_is_read = true;
        if (debug) {
            // std::cout << "Last line is: " << memmap.line << std::endl;
            std::cout << "NBLOCK complete, read " << n_nodes << " nodes." << std::endl;
        }
    }

    // Read CMBLOCK
    void ReadCMBlock() {

        // edge case with non ASCII characters
        memmap.sanitize_line_ascii();

        if (debug) {
            std::cout << "reading CMBLOCK" << std::endl;
            std::cout << "    " << memmap.line << std::endl;
        }

        // Assumes line at CMBLOCK
        std::istringstream iss(memmap.line);
        std::string token;
        std::vector<std::string> split_line;

        while (std::getline(iss, token, ',')) {
            split_line.push_back(token);
        }

        if (split_line.size() < 4) {
            std::cerr << "Poor formatting of CMBLOCK: " << memmap.line << std::endl;
            return;
        }

        int ncomp;
        try {
            ncomp = std::stoi(split_line[3].substr(0, split_line[3].find('!')).c_str());
        } catch (...) {
            std::cerr << "Poor formatting of CMBLOCK: " << memmap.line << std::endl;
            return;
        }

        memmap.read_line();
        int isz;
        try {
            isz = std::stoi(memmap.line.substr(
                memmap.line.find('i') + 1,
                memmap.line.find(')') - memmap.line.find('i') - 1));
        } catch (...) {
            std::cerr << "Failed to read integer size in CMBLOCK" << std::endl;
            return;
        }

        int nblock;
        try {
            nblock = std::stoi(memmap.line.substr(
                memmap.line.find('(') + 1,
                memmap.line.find('i') - memmap.line.find('(') - 1));
        } catch (...) {
            std::cerr << "Failed to read number of integers per line in CMBLOCK" << std::endl;
            return;
        }

        std::vector<int> component(ncomp);
        std::string tempstr(isz, '\0');

        for (int i = 0; i < ncomp; ++i) {
            if (i % nblock == 0) {
                memmap.read_line();
            }
            try {
                component[i] = std::stoi(memmap.line.substr(isz * (i % nblock), isz));
            } catch (...) {
                std::cerr << "Failed to parse integer from CMBLOCK line: " << memmap.line
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

        if (debug) {
            std::cout << "Done reading CMBLOCK" << std::endl;
        }
    }

    void Read() {
        int first_char, next_char;

        int position_start, position_end;
        while (true) {
            // Parse based on the first character rather than reading the entire
            // line. It's faster and the parsing logic is always based on the first
            // character

            if (memmap.eof()) {
#ifdef DEBUG
                std::cout << "Reached EOF" << std::endl;
#endif
                break;
            }

            first_char = memmap[0];
#ifdef DEBUG
            std::cout << "Read character: " << static_cast<char>(first_char) << std::endl;
#endif
            if (first_char == 'E' || first_char == 'e') {
                memmap.read_line();

                // E commands (ET or ETBLOCK)

                if (debug) {
                    std::cout << "Read E" << std::endl;
                }

                // Record element type
                if (memmap.line.compare(0, 3, "ET,") == 0 ||
                    memmap.line.compare(0, 3, "et,") == 0) {
                    ReadETLine();
                } else if (
                    memmap.line.compare(0, 6, "ETBLOC") == 0 ||
                    memmap.line.compare(0, 6, "etbloc") == 0) {
                    ReadETBlock();
                } else if (
                    (memmap.line.compare(0, 6, "EBLOCK") == 0 ||
                     memmap.line.compare(0, 6, "eblock") == 0) &&
                    read_eblock) {
                    // "&&" binds tighter than "||", so without these parentheses
                    // an upper case EBLOCK was read even when read_eblock was off.
                    ReadEBlock();
                }
            } else if (first_char == 'K' || first_char == 'k') {
                memmap.read_line();
                if (debug) {
                    std::cout << "Read K" << std::endl;
                }

                // Record keyopt
                if (memmap.line.compare(0, 5, "KEYOP") == 0 ||
                    memmap.line.compare(0, 5, "keyop") == 0) {
                    ReadKEYOPTLine();
                }

            } else if (first_char == 'R' || first_char == 'r') {
                memmap.read_line();
                // test for RLBLOCK
                if (debug) {
                    std::cout << "Read R" << std::endl;
                }

                // Record keyopt
                if (memmap.line.compare(0, 5, "RLBLO") == 0 ||
                    memmap.line.compare(0, 5, "rlblo") == 0) {
                    ReadRLBLOCK();
                }

            } else if (first_char == 'N' || first_char == 'n') {
                // store current position if we read in the node block
                const int pos = memmap.tellg();
                memmap.read_line();
                // test for NBLOCK

                if (debug) {
                    std::cout << "Read N" << std::endl;
                }

                // Record node block
                if (memmap.line.compare(0, 5, "NBLOC") == 0 ||
                    memmap.line.compare(0, 5, "nbloc") == 0) {
                    ReadNBlock(pos);
                }

            } else if (first_char == 'C' || first_char == 'c') {
                memmap.read_line();

                if (debug) {
                    std::cout << "Read C" << std::endl;
                }

                // Record component block
                if (memmap.line.compare(0, 5, "CMBLO") == 0 ||
                    memmap.line.compare(0, 5, "cmblo") == 0) {
                    // std::cout << "Reading CMBLOCK" << std::endl;
                    ReadCMBlock();
                }
            } else {
                if (debug) {
                    std::cout << "No match, continuing..." << std::endl;
                }
                // Skip remainder of the line
                memmap.seek_eol();
            }
        }
    }

    // Read line and return the position prior to reading the line
    int ReadLine() { return memmap.read_line(); }

    // Convert ansys style connectivity to VTK connectivity
    // type_ref is a mapping between ansys element types and VTK element types
    nb::tuple ToVTK(NDArray<int, 1> type_ref) {
        NDArray<int, 1> offset_arr = MakeNDArray<int, 1>({n_elem + 1});
        NDArray<uint8_t, 1> celltypes_arr = MakeNDArray<uint8_t, 1>({n_elem});

        // Allocate connectivity: max cell size 20 for VTK_HEXAHEDRAL
        int *cell_data = new int[n_elem * 20];

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

        NDArray<int, 1> cells_arr = WrapNDarray<int, 1>(cell_data, {loc});
        return nb::make_tuple(offset_arr, celltypes_arr, cells_arr);
    }

    ~Archive() { memmap.close_file(); }

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

    NDArray<int, 1> offset_arr = MakeNDArray<int, 1>({n_elem + 1});
    NDArray<uint8_t, 1> celltypes_arr = MakeNDArray<uint8_t, 1>({n_elem});

    // Allocate connectivity: max cell size 20 for VTK_HEXAHEDRAL
    int *cell_data = new int[n_elem * 20];

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

    NDArray<int, 1> cells_arr = WrapNDarray<int, 1>(cell_data, {loc});
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
        .def("to_vtk", &Archive::ToVTK)
        .def("read", &Archive::Read)
        .def("read_line", &Archive::ReadLine)
        .def("read_nblock", &Archive::ReadNBlock)
        .def("read_rlblock", &Archive::ReadRLBLOCK)
        .def("read_keyopt_line", &Archive::ReadKEYOPTLine)
        .def("read_et_line", &Archive::ReadETLine)
        .def("read_etblock", &Archive::ReadETBlock)
        .def("read_cmblock", &Archive::ReadCMBlock)
        .def("read_eblock", &Archive::ReadEBlock);
}
