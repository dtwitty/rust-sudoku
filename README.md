A Sudoku solver meant for exploring high-performance coding styles (and learning Rust).

Incorporates the following solving techniques:
  * Max-Conflicts Backtracking
  * Constraint propagation
  * Naked Singles
  * Hidden Singles

This solver was written with the following principles:
  * Profile ruthelessly
  * Use SIMD where possible
    * Encourage the compiler to auto-vectorize
  * Use branchless algorithms 
  * Minimize heap allocations
  * Prefer single-pass algorithms
  * Encourage caching
  * Propagate constraints rather than compute them

The core functions that speed up this solver are this:
  * Check if any cell in a u16 array is zero (to find conflicts)
  * Find the first u16 in an array with a single bit set (to find hidden/naked singles)

This solver attempts to perform these actions as fast as possible, using every trick I can think of to cheese the compiler into outputting faster code.

The optimal algorithm and constants will change depending on the processor (x86 vs arm) and available features (avx2, avx512)
    * Where possible, this solver uses code that is friendly to auto-vectorization, so the compiler can do this for you
