# Sudoku
A Sudoku solver meant for exploring high-performance coding styles (and learning Rust).

## Building and Benchmarking

To build:
```
cargo rustc --profile=release -- -C target-cpu=native -C opt-level=3
```

To benchmark with [hyperfine](https://crates.io/crates/hyperfine):
```
hyperfine -m 10 -w 2 "target/release/sudoku --boards-file puzzles/puzzles5_forum_hardest_1905_11+ --verify"
```

On my machine (Macbook Pro 2021, M1 Pro, 32GB memory):
```
hyperfine -m 10 -w 2 "target/release/sudoku --boards-file puzzles/puzzles5_forum_hardest_1905_11+ --verify"
Benchmark 1: target/release/sudoku --boards-file puzzles/puzzles5_forum_hardest_1905_11+ --verify
  Time (mean ± σ):      2.632 s ±  0.047 s    [User: 2.614 s, System: 0.006 s]
  Range (min … max):    2.602 s …  2.749 s    10 runs
```

To get your free beer 🍻:

Create a PR with an improvement to this solver, and show that it is significantly better.
The benchmarking script gives you $\mu$, $\sigma$, and $n$. Run the benchmark for both the main branch and your PR.
Input those results into a [2-sample t-test](https://www.wolframalpha.com/input?i=two+sample+t+test), with significance level 0.05.
If you show improvement, send the PR for review and we'll coordinate your free beer 🍻.


## About

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
