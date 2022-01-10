A Sudoku solver meant for exploring high-performance coding styles (and learning Rust).

Incorporates the following solving techniques:
  * Max-Conflicts Backtracking
  * Constraint propagation
  * Naked Singles
  * Hidden Singles

This solver was written with the following principles:
  * Profile ruthelessly
  * Use SIMD where possible
  * Use branchless algorithms
  * Minimize heap allocations
  * Prefer single-pass algorithms
  * Make the cache your friend
  * Propagate constraints rather than compute them

The core functions that speed up this solver are this:
  * Find the first zero u16 in an array (to find conflicts)
  * Find the first u16 in an array with a single bit set (to find hidden/naked singles)

This solver attempts to perform these actions as fast as possible, using every trick I can think of to cheese the compiler into outputting faster code.
