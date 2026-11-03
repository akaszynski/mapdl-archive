# Independent NBLOCK decimal fixtures

These are synthetic scalar interchange fixtures, not mesh coordinates. There is no runtime `float(text)` oracle. Tests send literal strings through the public Archive API/native C++ NBLOCK reader and inspect returned uint64 words. NumPy only views storage; it does not derive expected floating values.

For every dyadic `N/2^k`, the exact finite decimal is `N*5^k/10^k`. The committed literal strings were authored from this integer identity. Expected binary64 words follow directly from the sign, exponent and integer significand, with nearest/ties-to-even at midpoints. The derivation is recorded beside every literal in the JSON file.

At one, adjacent binary64 values are spaced by `2^-52`. Thus exact `1+2^-53` ties to one, while `1+3*2^-53` ties to the second successor. The shorter 17-digit literal `1.00000000000000011` lies strictly below the first midpoint because `11*2^53 <10^17`; the `...012` literal lies above it. This independently exposes multiple-rounding errors without overflowing uint64.

At `2^64`, the spacing is `2^12`. Adding `2^11` ties to the even lower significand; adding `3*2^11` ties to the even upper significand. The exact `2^64` case independently detects accumulation overflow.

The subnormal quantum is `2^-1074`; exact half/quarter quantum and three half-quanta exercise zero, negative zero and even-subnormal rounding. Exact `2^-1022` is the minimum normal. Exact `(2-2^-52)*2^1023 =2^1024-2^971` is the maximum finite value. Exact `2^-100` tests a long fractional token beyond the old hardcoded decimal cutoff.

Finally `1/10 *2^56 =7205759403792793.6`, so its nearest integer significand is7205759403792794, giving `0x3fb999999999999a`. Decimal/D/E/leading-plus/whitespace variants check lexical transport, not a new numeric engine. Field-adjacency, malformed/trailing/embedded-NUL/nonfinite and overflow cases exercise complete bounded conversion and explicit refusal. A leading NUL optional-angle field retains the previous end-of-line signal.
