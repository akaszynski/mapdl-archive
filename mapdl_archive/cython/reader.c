#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// consider using fgets_unlocked when compiling with GNU

#if defined(_WIN32) || defined(_WIN64)
/* We are on Windows */
# define strtok_r strtok_s
#endif

// #define DEBUG

  static const double DIV_OF_TEN[] = {
    1.0e-0,
    1.0e-1,
    1.0e-2,
    1.0e-3,
    1.0e-4,
    1.0e-5,
    1.0e-6,
    1.0e-7,
    1.0e-8,
    1.0e-9,
    1.0e-10,
    1.0e-11,
    1.0e-12,
    1.0e-13,
    1.0e-14,
    1.0e-15,
    1.0e-16,
    1.0e-17,
    1.0e-18,
    1.0e-19,
    1.0e-20,
    1.0e-21,
    1.0e-22,
    1.0e-23,
    1.0e-24,
  };


__inline double power_of_ten(int exponent) {
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
__inline int fast_atoi(const char *raw, const int intsz) {
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
__inline int checkneg(char *raw, int intsz) {
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
__inline int ans_strtod(char *raw, int fltsz, double *arr) {
  char *end = raw + fltsz;
  double sign = 1;

  while (raw < end) {
    if (*raw == '\r' || *raw == '\n') {
      // line empty, value is zero
      *arr = 0;
      /* printf("EOL"); */
      return 1;
    } else if (*raw != ' ') { // always skip whitespace
      break;
    }
    raw++;
  }

  // either a number of a sign
  if (*raw == '-') {
    sign = -1;
    ++raw;
  }

  // next value is always a number
  // Use integer arithmetric and then convert to a float
  uint64_t val_int = *raw++ - '0';
  raw++;  // next value is always a "."

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
  if (decimal_digits < 24){
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
      if (*raw == ' ' || *raw == '\r' || *raw == '\n') {
        break;
      }
      evalue = evalue * 10 + (*raw++ - '0');
    }
    // printf("%d\n", evalue);
    // val *= pow10(evalue);
    if (esign == 1) {
      val *= power_of_ten(evalue);

      // while (evalue > 0) {  // raises value to the power of the exponent
      //   val *= 10;
      //   evalue--;
      // }
    } else {
      val /= power_of_ten(evalue);
      // while (evalue > 0) {
      //   val *= 0.1;
      //   evalue--;
      // }
    }
  }

  // seek through end of float value
  if (sign == -1) {
    *arr = -val;
  } else {
    *arr = val;
  }
  /* printf(", %f", val); */

  return 0; // Return 0 when a number has a been read
}



static inline double ans_strtod2(char *raw, int fltsz) {
  int i;
  double sign = 1;

  for (i = 0; i < fltsz; i++) {
    if (*raw == '\r' || *raw == '\n') {
      // value is zero then
      return 0;
    } else if (*raw != ' ') { // always skip whitespace
      break;
    }
    raw++;
  }

  // either a number of a sign
  if (*raw == '-') {
    sign = -1;
    ++raw;
    ++i;
  }

  // next value is always a number
  double val = *raw++ - '0';
  i++;
  double k = 10;
  for (; i < fltsz; i++) {
    if (*raw == '.') {
      raw++;
      break;
    } else {
      val *= k;
      val += (*raw++ - '0');
      k *= 10;
    }
  }

  // Read through the rest of the number
  k = 0.1;
  for (; i < fltsz; i++) {
    if (*raw == 'e' || *raw == 'E') {
      break;
    } else if (*raw >= '0' && *raw <= '9') {
      val += (*raw++ - '0') * k;
      k *= 0.1;
    }
  }

  // Might have scientific notation left, for example:
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
    i++;
    i++; // skip E and sign
    for (; i < fltsz; i++) {
      // read to whitespace or end of the line
      if (*raw == ' ' || *raw == '\r' || *raw == '\n') {
        break;
      }
      evalue = evalue * 10 + (*raw++ - '0');
    }
    val *= pow(10, esign * evalue);
  }

  // seek through end of float value
  if (sign == -1) {
    return -val;
  }
  return val;
}

//=============================================================================
// reads nblock from ANSYS.  Raw string is from Python reader and file is
// positioned at the start of the data of NBLOCK
//
// Returns
// -------
// nnodes_read : int
//     Number of nodes read.
//=============================================================================
int read_nblock(char *raw, int *nnum, double *nodes, int nnodes, int *intsz,
                int fltsz, int64_t *n) {

  // set to start of the NBLOCK
  raw += *n;
  int64_t len_orig = strlen(raw);
  int i, j, i_val, eol;

  for (i = 0; i < nnodes; i++) {

    // It's possible that less nodes are written to the record than
    // indicated.  In this case the line starts with a -1
    if (raw[0] == '-') {
      break;
    }

    i_val = fast_atoi(raw, intsz[0]);
    /* printf("%d", i_val); */
    nnum[i] = i_val;
    raw += intsz[0];
    raw += intsz[1];
    raw += intsz[2];

    for (j = 0; j < 6; j++) {
      eol = ans_strtod(raw, fltsz, &nodes[6 * i + j]);
      if (eol) {
        break;
      } else {
        raw += fltsz;
      }
    }

    // remaining are zeros
    for (; j < 6; j++) {
      nodes[6 * i + j] = 0;
    }

    // possible whitespace (occurs in hypermesh generated files)
    while (*raw == ' ') {
      ++raw;
    }

    while (*raw == '\r' || *raw == '\n') {
      ++raw;
    }
    /* printf("\n"); */
  }

  // return file position
  *n += len_orig - strlen(raw);
  return i;
}

/* Read just the node coordinates from the output from the MAPDL
 *  NWRITE command
 * (I8, 6G20.13) to write out NODE,X,Y,Z,THXY,THYZ,THZX
 */
int read_nblock_from_nwrite(const char *filename, int *nnum, double *nodes,
                            int nnodes) {
  FILE *stream = fopen(filename, "r");

  if (stream == NULL) {
    printf("Error opening file");
    exit(1);
  }

  // set to start of the NBLOCK
  const int bufsize = 74; // One int, 3 floats, two end char max (/r/n)
  char buffer[74];
  int i;

  for (i = 0; i < nnodes; i++) {
    fgets(buffer, bufsize, stream);
    nnum[i] = fast_atoi(&buffer[0], 9);

    // X
    if (buffer[9] == '\r' || buffer[9] == '\n') {
      nodes[i * 3 + 0] = 0;
      nodes[i * 3 + 1] = 0;
      nodes[i * 3 + 2] = 0;
      continue;
    }
    nodes[i * 3 + 0] = ans_strtod2(&buffer[9], 21);

    // Y
    if (buffer[30] == '\r' || buffer[30] == '\n') {
      nodes[i * 3 + 1] = 0;
      nodes[i * 3 + 2] = 0;
      continue;
    }
    nodes[i * 3 + 1] = ans_strtod2(&buffer[30], 21);

    // Z
    if (buffer[51] == '\r' || buffer[51] == '\n') {
      nodes[i * 3 + 2] = 0;
      continue;
    }
    nodes[i * 3 + 2] = ans_strtod2(&buffer[51], 21);
  }

  fclose(stream);
  return 0;
}

/* ============================================================================
 * Function:  read_eblock
 *
 * Reads EBLOCK from ANSYS archive file.
 * raw : Raw string is from Python reader
 *
 * elem_off : Indices of the start of each element in ``elem``
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
 * nelem : Number of elements.
 *
 * pos : Position of the start of the EBLOCK.
 * ==========================================================================*/
int read_eblock(char *raw, int *elem_off, int *elem, int nelem, int intsz,
                int64_t *pos) {
  int i, j, nnode;

  // set to start of the EBLOCK
  raw += pos[0];
  int64_t len_orig = strlen(raw);
  int c = 0; // position in elem array

  // Loop through elements
  for (i = 0; i < nelem; ++i) {
    // store start of each element
    elem_off[i] = c;

    // Check if end of line
    while (raw[0] == '\r' || raw[0] == '\n') {
      ++raw;
    }

    // Check if at end of the block
    if (checkneg(raw, intsz)) {
      raw += intsz;
      break;
    }

    // ANSYS archive format:
    // Field 1: material reference number
    elem[c++] = fast_atoi(raw, intsz);
    raw += intsz;

    // Field 2: element type number
    elem[c++] = fast_atoi(raw, intsz);
    raw += intsz;

    // Field 3: real constant reference number
    elem[c++] = fast_atoi(raw, intsz);
    raw += intsz;

    // Field 4: section number
    elem[c++] = fast_atoi(raw, intsz);
    raw += intsz;

    // Field 5: element coordinate system
    elem[c++] = fast_atoi(raw, intsz);
    raw += intsz;

    // Field 6: Birth/death flag
    elem[c++] = fast_atoi(raw, intsz);
    raw += intsz;

    // Field 7: Solid model reference
    elem[c++] = fast_atoi(raw, intsz);
    raw += intsz;

    // Field 8: Coded shape key
    elem[c++] = fast_atoi(raw, intsz);
    raw += intsz;

    // Field 9: Number of nodes
    nnode = fast_atoi(raw, intsz);
    raw += intsz;

    /* // sanity check */
    /* if (nnode > 20){ */
    /*   printf("Element %d\n", i); */
    /*   perror("Greater than 20 nodes\n"); */
    /*   exit(1); */
    /* } */

    // Field 10: Not Used
    raw += intsz;

    // Field 11: Element number
    elem[c++] = fast_atoi(raw, intsz);
    raw += intsz;
    /* printf("reading element %d\n", elem[c - 1]); */

    // Need an additional value for consistency with other formats
    elem[c++] = 0;

    // Read nodes in element
    for (j = 0; j < nnode; j++) {
      /* printf("reading node %d\n", j); */
      // skip through EOL
      while (raw[0] == '\r' || raw[0] == '\n')
        ++raw;
      elem[c++] = fast_atoi(raw, intsz);
      raw += intsz;
    }

    // Edge case where missing midside nodes are not written (because
    // MAPDL refuse to write zeros at the end of a line)
    if (nnode < 20 && nnode > 10) {
      for (j = nnode; j < 20; j++) {
        elem[c++] = 0;
      }
    }
    /* else if (nnode < 8 && nnode > 4){ */
    /*   for (j=nnode; j<8; j++){ */
    /*     elem[c++] = 0; */
    /*   } */
    /* } */
  }

  // update file position
  *(pos) = len_orig - strlen(raw) + pos[0];

  // Return total data read
  elem_off[nelem] = c;
  return c;
}

int read_eblock_cfile(FILE *cfile, int *elem_off, int *elem, int nelem) {
  int i, j, nnode;

  // set to start of the NBLOCK
  char line[400];
  char *cursor = line;

  // Get size of integer
    if (fgets(line, sizeof(line), cfile) == NULL) {
      return 0;
    }

    char* i_pos = strchr(line, 'i');
    char* close_paren_pos = strchr(line, ')');
    if (i_pos == NULL || close_paren_pos == NULL || i_pos > close_paren_pos) {
      fprintf(stderr, "Invalid line format\n");
      return 0;
    }

    int isz;
    sscanf(i_pos + 1, "%d", &isz);

  // Loop through elements
  int c= 0;
  for (i = 0; i < nelem; ++i) {
    // store start of each element
    elem_off[i] = c;

    fgets(line, sizeof(line), cfile); cursor = line;

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

    // Field 9: Number of nodes
    nnode = fast_atoi(cursor, isz);
    cursor += isz;

  //   /* // sanity check */
  //   /* if (nnode > 20){ */
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

    // Read nodes in element
    for (j = 0; j < nnode; j++) {
      /* printf("reading node %d\n", j); */
      // skip through EOL
      if (*cursor == '\r' || *cursor == '\n'){
        fgets(line, sizeof(line), cfile); cursor = line;
      }
      elem[c++] = fast_atoi(cursor, isz);
      cursor += isz;
    }

    // Edge case where missing midside nodes are not written (because
    // MAPDL refuse to write zeros at the end of a line)
    if (nnode < 20 && nnode > 10) {
      for (j = nnode; j < 20; j++) {
        elem[c++] = 0;
      }
    }
  }

  // Return total data read
  elem_off[nelem] = c;
  return c;
}


// Simply write an array to disk as ASCII
int write_array_ascii(const char *filename, const double *arr,
                      const int nvalues) {
  FILE *stream = fopen(filename, "w");
  int i;

  for (i = 0; i < nvalues; i++) {
    fprintf(stream, "%20.12E\n", arr[i]);
  }

  fclose(stream);

  return 0;
}


int read_nblock_cfile(FILE *cfile, int *nnum, double *nodes, int nnodes, int* d_size, int f_size) {

  int i, j, i_val, eol;
  char line[256];

  for (i = 0; i < nnodes; i++) {
    // Read a line from the file
    if (fgets(line, sizeof(line), cfile) == NULL) {
      break;
    }

    // It's possible that less nodes are written to the record than
    // indicated.  In this case the line starts with a -1
    if (line[0] == '-') {
      break;
    }

    char *cursor = line;

    i_val = fast_atoi(cursor, d_size[0]);
    /* printf("%8d    \n", i_val); */
    nnum[i] = i_val;

    cursor += d_size[0];
    cursor += d_size[1];
    cursor += d_size[2];

    for (j = 0; j < 6; j++) {
      eol = ans_strtod(cursor, f_size, &nodes[6 * i + j]);
      if (eol) {
        break;
      } else {
        cursor += f_size;
      }
    }

    // remaining are zeros
    for (; j < 6; j++) {
      nodes[6 * i + j] = 0;
    }

    /* debug output */
// #ifdef DEBUG
//     for (j = 0; j < 6; j++) {
//         printf("  %lf", nodes[6 * i + j]);
//     }
//     printf("\n");
// #endif

    // possible whitespace (occurs in hypermesh generated files)
    while (*cursor == ' ') {
      ++cursor;
    }

    // handle newline characters
    while (*cursor == '\r' || *cursor == '\n') {
      ++cursor;
    }
  }

  return i;
}
