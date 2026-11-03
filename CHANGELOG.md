# Unreleased

- Read NBLOCK coordinates and optional node angles with one locale-independent native decimal conversion, preserving binary64 round trips from 17-digit CDB fields. Support Fortran D exponents, bounded fields, signed zero and long/subnormal literals. Report malformed, trailing, nonfinite and overflow fields instead of accepting a rounded integer/power-of-ten approximation or numeric prefix. Release temporary node arrays when parsing refuses.
