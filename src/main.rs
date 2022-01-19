#![feature(core_intrinsics)]
#![feature(const_assume)]
#![feature(stdsimd)]

extern crate aligned;
extern crate argmm;
extern crate structopt;
use aligned::{Aligned, A64};
use argmm::ArgMinMax;
use core::arch::x86_64::*;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use structopt::StructOpt;

// These type aliases help differentiate different kinds of ints.
type CellIdx = usize;
type GroupNum = usize;
type GroupIdx = usize;
type GroupCells = [CellIdx; 9];

// Groups are collections of cells that must contain 1-9.
// They are implemented such that the same code works for a Row, Col, or Box.
trait Group {
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx;
    fn group_idx(idx: CellIdx) -> GroupIdx;
    fn for_cell(idx: CellIdx) -> GroupNum;

    fn cells(g: GroupNum) -> GroupCells {
        unsafe {
            core::intrinsics::assume(g < 9);
        }
        [
            Self::cell_at(g, 0),
            Self::cell_at(g, 1),
            Self::cell_at(g, 2),
            Self::cell_at(g, 3),
            Self::cell_at(g, 4),
            Self::cell_at(g, 5),
            Self::cell_at(g, 6),
            Self::cell_at(g, 7),
            Self::cell_at(g, 8),
        ]
    }

    fn neighbors(idx: CellIdx) -> GroupCells {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        Self::cells(Self::for_cell(idx))
    }
}

struct Row;
impl Group for Row {
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx {
        unsafe {
            core::intrinsics::assume(g < 9);
            core::intrinsics::assume(idx < 9);
        }
        9 * g + idx
    }

    fn group_idx(idx: CellIdx) -> GroupIdx {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        idx % 9
    }

    fn for_cell(idx: CellIdx) -> GroupNum {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        idx / 9
    }
}

struct Col;
impl Group for Col {
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx {
        unsafe {
            core::intrinsics::assume(g < 9);
            core::intrinsics::assume(idx < 9);
        }
        idx * 9 + g
    }

    fn group_idx(idx: CellIdx) -> GroupIdx {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        idx / 9
    }

    fn for_cell(idx: CellIdx) -> GroupNum {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        idx % 9
    }
}

// To encourage SIMD, box positions are precomputed.
struct Box;
const STARTS: [CellIdx; 9] = [0, 3, 6, 27, 30, 33, 54, 57, 60];
const STEPS: [CellIdx; 9] = [0, 1, 2, 9, 10, 11, 18, 19, 20];
impl Group for Box {
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx {
        unsafe {
            core::intrinsics::assume(g < 9);
            core::intrinsics::assume(idx < 9);
        }
        let start = STARTS[g];
        let step = STEPS[idx];
        start + step
    }

    fn group_idx(idx: CellIdx) -> GroupIdx {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        let row = idx / 9;
        let col = idx % 9;
        (row % 3) * 3 + col % 3
    }

    fn for_cell(idx: CellIdx) -> GroupNum {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        let row = idx / 9;
        let col = idx % 9;
        let box_row = row / 3;
        let box_col = col / 3;
        box_row * 3 + box_col
    }

    fn cells(g: GroupNum) -> GroupCells {
        unsafe {
            core::intrinsics::assume(g < 9);
        }
        let s = STARTS[g];
        [
            s + 0,
            s + 1,
            s + 2,
            s + 9,
            s + 10,
            s + 11,
            s + 18,
            s + 19,
            s + 20,
        ]
    }
}

fn all_neighbors(idx: CellIdx) -> [CellIdx; 27] {
    unsafe {
        core::intrinsics::assume(idx < 81);
    }
    let mut arr: [CellIdx; 27] = [0; 27];
    arr[..9].clone_from_slice(&Row::neighbors(idx));
    arr[9..18].clone_from_slice(&Col::neighbors(idx));
    arr[18..].clone_from_slice(&Box::neighbors(idx));
    arr
}

pub type Value = u8;
const NO_VALUE: Value = 255;

pub trait ValueMethods {
    fn is_set(&self) -> bool;
}

impl ValueMethods for Value {
    fn is_set(&self) -> bool {
        *self != NO_VALUE
    }
}

pub type CandidateSet = u16;
const ALL_CANDS: CandidateSet = 511;
const SET_CANDS: CandidateSet = 0x600;

pub trait CandidateSetMethods {
    fn has_candidate(&self, v: Value) -> bool;
    fn num_candidates(&self) -> u32;
    fn remove_candidate(&mut self, v: Value);
    fn is_set(&self) -> bool;
}

impl CandidateSetMethods for CandidateSet {
    fn has_candidate(&self, v: Value) -> bool {
        (*self & (1 << v)) != 0
    }
    fn num_candidates(&self) -> u32 {
        self.count_ones()
    }
    fn remove_candidate(&mut self, v: Value) {
        unsafe {
            core::intrinsics::assume(v < 9);
        }
        *self &= !(1 << v);
    }
    fn is_set(&self) -> bool {
        *self == SET_CANDS
    }
}

struct MutGroupsForCandidate<'a> {
    groups: &'a mut [CandidateSet],
}
impl<'a> MutGroupsForCandidate<'a> {
    fn mut_row_candidates(&mut self, r: usize) -> &mut CandidateSet {
        unsafe {
            core::intrinsics::assume(r < 9);
            core::intrinsics::assume(r < self.groups.len());
        }
        &mut self.groups[r]
    }

    fn mut_col_candidates(&mut self, c: usize) -> &mut CandidateSet {
        unsafe {
            core::intrinsics::assume(c < 9);
            core::intrinsics::assume(9 + c < self.groups.len());
        }
        &mut self.groups[9 + c]
    }

    fn mut_box_candidates(&mut self, b: usize) -> &mut CandidateSet {
        unsafe {
            core::intrinsics::assume(b < 9);
            core::intrinsics::assume(18 + b < self.groups.len());
        }
        &mut self.groups[18 + b]
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C, align(64))]
struct CandidateToGroups {
    candidates: [CandidateSet; 9 * 3 * 9],
}

impl CandidateToGroups {
    fn mut_groups_for_candidate(&mut self, v: Value) -> MutGroupsForCandidate {
        unsafe {
            core::intrinsics::assume(v < 9);
        }
        let start = (v * 9 * 3) as usize;
        let end = ((v + 1) * 9 * 3) as usize;
        MutGroupsForCandidate {
            groups: &mut self.candidates[start..end],
        }
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C, align(64))]
pub struct Board {
    candidates: [CandidateSet; 81],
    candidate_to_groups: CandidateToGroups,
    values: [Value; 81],
    num_remaining_cells: usize,
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const LANE_SEP: &str = "------+-------+------";
        const CELL_SEP: &str = " ";
        const BLOCK_SEP: &str = " | ";

        for r in 0..9 {
            for c in 0..9 {
                let sep = if c == 2 || c == 5 {
                    BLOCK_SEP
                } else {
                    CELL_SEP
                };
                let v = self.values[r * 9 + c];
                if v.is_set() {
                    write!(f, "{}", v + 1).unwrap();
                } else {
                    write!(f, ".").unwrap();
                }
                write!(f, "{}", sep).unwrap();
            }
            writeln!(f).unwrap();
            if r == 2 || r == 5 {
                writeln!(f, "{}", LANE_SEP).unwrap();
            }
        }
        Ok(())
    }
}

// This function finds the first CandidateSet with a single set bit.
// It is the meat-and-potatoes of the solver and should be
// AS FAST AS POSSIBLE!!!
fn single_candidate_position(data: &[CandidateSet]) -> Option<usize> {
    const N: usize = 128;
    // Chunk the input array so that many positions are considered at once.
    data.chunks(N).enumerate().find_map(|(chunk, s)| {
        // Create a 64-byte aligned array of u8s where each position encodes 'Is Single Bit'.
        // It is faster to compute "Is Single Bit" all at once, then find the first in a second pass.
        let mut a: Aligned<A64, _> = Aligned([0u8; N]);
        let arr = a.as_mut_slice();
        arr.iter_mut()
            .zip(s.iter())
            // Calculate whether this position has a single bit.
            // We can ignore the case where c == 0 because that would imply a conflict.
            .for_each(|(a, &c)| *a = (((c & (c - 1)) == 0) as u8) * 0xFF);

        // Now that we have an array of "has_single_bit" flags, we need to find the first (if any).
        // The idea is to avoid branching at all costs! We populate a u128 where each bit position
        // encodes 'has_single_bit' in the input array.
        unsafe {
            let ptr = arr.as_ptr();

            let mut bitmask: u128 = 0;
            for i in 0..N / 32 {
                // Load 32 bytes of 0 or 0xFF.
                let packed_bits = _mm256_load_si256(ptr.offset(i as isize * 32) as *const __m256i);
                // Pack the first bit of each into a u32, then add that to a u128.
                let bits = _mm256_movemask_epi8(packed_bits) as u128;
                bitmask |= bits << (i * 32);
            }

            // This is the only branch needed to check a whole chunk of positions.
            // We simply check if our special bit array is non-zero, and find the first bit set.
            if bitmask != 0 {
                Some(chunk * N + (bitmask.trailing_zeros() as usize))
            } else {
                None
            }
        }
    })
}

// These are the possible outcomes of constraint propagation.
#[derive(PartialEq)]
enum ConstraintPropagationResult {
    NoConstraintsPropagated,
    PropagatedConstraint,
    FoundConflict,
    Solved,
}

impl Board {
    fn new() -> Board {
        Board {
            // Which values are candidates for this cell?
            candidates: [ALL_CANDS; 81],

            // What is the solved value at this cell?
            values: [NO_VALUE; 81],

            // How many cells are left to solve?
            num_remaining_cells: 81,

            // Where is each value available in each group?
            // For example, we can ask 'In Row 2, which cells can take a 5?'
            candidate_to_groups: CandidateToGroups {
                candidates: [ALL_CANDS; 9 * 3 * 9],
            },
        }
    }

    // Is this board solved?
    fn is_solved(&self) -> bool {
        self.num_remaining_cells == 0
    }

    // Is every cell set, and do its neighbors obey the Sudoku rules?
    fn is_complete(&self) -> bool {
        (0..81).all(|idx| {
            self.values[idx].is_set()
                && all_neighbors(idx).iter().all(|&other_idx| {
                    idx == other_idx || self.values[other_idx] != self.values[idx]
                })
        })
    }

    // Get mutable candidates at a cell, assuming a valid index.
    fn unsafe_mut_candidates_at(&mut self, idx: CellIdx) -> &mut CandidateSet {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        &mut self.candidates[idx]
    }

    // Get candidates at a cell, assuming a valid index.
    fn unsafe_candidates_at(&self, idx: CellIdx) -> &CandidateSet {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        &self.candidates[idx]
    }

    // Set a value at a cell.
    // This function handles updating the constraint propagation data structures.
    #[inline(never)]
    fn set_value_at(&mut self, idx: CellIdx, v: Value) {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }

        // Basic bookkeeping.
        self.values[idx] = v;
        self.candidates[idx] = SET_CANDS;
        self.num_remaining_cells -= 1;

        // Erase this value from the cell's neighbors.
        all_neighbors(idx).iter().for_each(|&other_idx| {
            self.unsafe_mut_candidates_at(other_idx).remove_candidate(v);
        });

        // Get the row, col, and box for this cell.
        let r = Row::for_cell(idx);
        let c = Col::for_cell(idx);
        let b = Box::for_cell(idx);
        unsafe {
            core::intrinsics::assume(v < 9);
            core::intrinsics::assume(r < 9);
            core::intrinsics::assume(c < 9);
            core::intrinsics::assume(b < 9);
        }

        // No candidate is available...
        for i in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(i)
                // In this row...
                .mut_row_candidates(r)
                // At this position.
                .remove_candidate(Row::group_idx(idx) as Value);
        }

        // No candidate is available...
        for i in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(i)
                // In this column...
                .mut_col_candidates(c)
                // At this position.
                .remove_candidate(Col::group_idx(idx) as Value);
        }

        // No candidate is available...
        for i in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(i)
                // In this box...
                .mut_box_candidates(b)
                // At this position.
                .remove_candidate(Box::group_idx(idx) as Value);
        }

        *self
            .candidate_to_groups
            // This value...
            .mut_groups_for_candidate(v)
            // Is no longer available in this row.
            .mut_row_candidates(r) = SET_CANDS;

        *self
            .candidate_to_groups
            // This value...
            .mut_groups_for_candidate(v)
            // Is no longer available in this column.
            .mut_col_candidates(c) = SET_CANDS;

        *self
            .candidate_to_groups
            // This value...
            .mut_groups_for_candidate(v)
            // Is no longer available in this box.
            .mut_box_candidates(b) = SET_CANDS;

        // In every row...
        for r in 0..9 {
            self.candidate_to_groups
                // This value...
                .mut_groups_for_candidate(v)
                .mut_row_candidates(r)
                // Is no longer available in this column.
                .remove_candidate(c as Value);
        }

        // For each position in this box...
        for br in 0..3 {
            for bc in 0..3 {
                let r = (b / 3) * 3 + br;
                let c = (b % 3) * 3 + bc;
                self.candidate_to_groups
                    // This value...
                    .mut_groups_for_candidate(v)
                    // Isn't available in the position's row...
                    .mut_row_candidates(r)
                    // At the position's column.
                    .remove_candidate(c as Value);
            }
        }

        // In every column...
        for c in 0..9 {
            self.candidate_to_groups
                // This value...
                .mut_groups_for_candidate(v)
                .mut_col_candidates(c)
                // Is no longer available in this row.
                .remove_candidate(r as Value);
        }

        // For each position in this box...
        for bc in 0..3 {
            for br in 0..3 {
                let r = (b / 3) * 3 + br;
                let c = (b % 3) * 3 + bc;
                self.candidate_to_groups
                    // This value...
                    .mut_groups_for_candidate(v)
                    // Isn't available in the position's column...
                    .mut_col_candidates(c)
                    // At the position's row.
                    .remove_candidate(r as Value);
            }
        }

        // The candidate isn't available at position in boxes that
        // overlap the current row.
        for c in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(v)
                .mut_box_candidates((r / 3) * 3 + (c / 3))
                .remove_candidate(((r % 3) * 3 + (c % 3)) as Value);
        }

        // The candidate isn't available at position in boxes that
        // overlap the current column.
        for r in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(v)
                .mut_box_candidates((r / 3) * 3 + (c / 3))
                .remove_candidate(((r % 3) * 3 + (c % 3)) as Value);
        }
    }

    // This function identifies the cell with the most constraints.
    // This is useful because this is the easiest cell to exhaustively check with backtracking.
    #[inline(never)]
    fn most_constrained_cell(&self) -> CellIdx {
        // An array of [i % 3 for i in range(81)]
        // This is needed because integer modulus is not provided by SIMD instructions.
        const MODS: [u8; 81] = [
            0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
            2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
            1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
        ];

        // Counts bits set in a u16.
        // This is needed because popcnt is not a vectorized instruction.
        fn cbs(mut v: u16) -> u16 {
            v = v - ((v >> 1) & 0x5555);
            v = (v & 0x3333) + ((v >> 2) & 0x3333);
            ((v + (v >> 4) & 0xF0F) * 0x101) >> 8
        }

        // This array holds values of the following format for each cell
        // <has_zero_candidates> (1 bit) - to toss out finsihed cells
        // <num_candidates> (4 bits, with one extra) - to prefer cells with fewer candidates
        // <cell_id % 3> (2 bits)
        // <cell_id % 3> is a poor-man's randomization. Without it, the solver will prefer cells
        // near the end of the array, leading to bunched backtracking that clears fewer candidates.
        // With it, earlier cells may trump later ones, leading to more even candidate clearing.
        let mut a: Aligned<A64, _> = Aligned([0u8; 81]);
        let arr = a.as_mut_slice();
        arr.iter_mut()
            .zip(self.candidates.iter())
            .zip(MODS.iter())
            .for_each(|((a, &c), &m)| {
                let is_set = c.is_set() as u8;
                let n = cbs(c as u16) as u8;
                *a = is_set << 7 | n << 2 | m;
            });

        // Select the cell_id with candidates the lowest number of candidates,
        // following constraints above.
        arr.argmin().unwrap()
    }

    // This function repeatedly propagates constraints until it can no longer make any
    // more inferences about the board.
    fn propagate_constraints(&mut self) -> ConstraintPropagationResult {
        let mut result = ConstraintPropagationResult::PropagatedConstraint;
        while result == ConstraintPropagationResult::PropagatedConstraint {
            result = self.try_propagate_constraints();
        }
        result
    }

    // This function attempts to propagate a constraint, or assert that the board is solved or unsolveable.
    fn try_propagate_constraints(&mut self) -> ConstraintPropagationResult {
        if self.is_solved() {
            return ConstraintPropagationResult::Solved;
        }
        if self.has_conflict() {
            return ConstraintPropagationResult::FoundConflict;
        }
        if self.set_naked_single() {
            return ConstraintPropagationResult::PropagatedConstraint;
        }
        if self.set_hidden_singles() {
            return ConstraintPropagationResult::PropagatedConstraint;
        }
        ConstraintPropagationResult::NoConstraintsPropagated
    }

    // This function detects whether the board has any conflicts that prove it is unsolveable.
    #[inline(never)]
    fn has_conflict(&self) -> bool {
        const N: usize = 128;

        let cand_chunks = self.candidates.chunks(N);
        let group_chunks = self.candidate_to_groups.candidates.chunks(N);

        // Check cells...
        cand_chunks
            // And then (value, group) pairs..
            .chain(group_chunks)
            // To see if any have no candidates.
            .any(|c_chunk| c_chunk.iter().map(|&cands| (cands == 0) as u8).sum::<u8>() != 0)
    }

    // This function finds and sets a 'naked single' if one exists.
    // A naked single is a cell that only has one candidate value.
    #[inline(never)]
    fn set_naked_single(&mut self) -> bool {
        // The candidates array contains a bitset of candidates for every position.
        // If any of these bitsets contains a single bit, it must be a naked single!
        let i = single_candidate_position(&self.candidates);

        if let Some(idx) = i {
            let cands = self.candidates[idx];
            let v = cands.trailing_zeros();
            self.set_value_at(idx, v as Value);
            return true;
        }

        false
    }

    // This function finds and sets a 'hidden single' if one exists.
    // A hidden single is a value that is only viable at one cell in a group.
    #[inline(never)]
    fn set_hidden_singles(&mut self) -> bool {
        // The candidate_to_groups_mapping tells where each value is viable in each group.
        // If there are any single-bit values in that array, that is a hidden single!
        let o = single_candidate_position(&self.candidate_to_groups.candidates);

        if let Some(i) = o {
            // We found a hidden single! We just need to extract its position information so
            // we can set the value at the right cell.
            let group_idx = i % 9;
            let group_type = (i / 9) % 3;
            let v = i / 9 / 3;

            let cell_in_group = self.candidate_to_groups.candidates[i].trailing_zeros() as usize;
            let row_cell = Row::cell_at(group_idx, cell_in_group);
            let col_cell = Col::cell_at(group_idx, cell_in_group);
            let box_cell = Box::cell_at(group_idx, cell_in_group);
            let idx = if group_type == 0 {
                row_cell
            } else if group_type == 1 {
                col_cell
            } else {
                box_cell
            };

            self.set_value_at(idx, v as Value);
            return true;
        }

        // No hidden singles found :(
        false
    }
}

pub fn solve(board: &mut Board) -> Option<Board> {
    let res = board.propagate_constraints();
    match res {
        // If the board is solved, return it!
        ConstraintPropagationResult::Solved => Some(*board),

        // If we found a conflict, we reject this board.
        ConstraintPropagationResult::FoundConflict => None,

        // We don't know if this board is on the right path, so we have to backtrack.
        _ => {
            // Pick the most constrained cell.
            let idx = board.most_constrained_cell();

            // Check every possible candidate for that cell.
            for v in 0..9 {
                if board.unsafe_candidates_at(idx).has_candidate(v) {
                    // Copy the current board, and set the value in the new board.
                    let mut new_board: Board = *board;
                    new_board.set_value_at(idx, v);

                    // Check whether the new board is now solveable.
                    let solution = solve(&mut new_board);
                    if solution.is_some() {
                        return solution;
                    }
                }
            }

            // We have checked every candidate for a given cell and found no suitable value.
            // This board must be unsolveable.
            None
        }
    }
}

fn parse_board(line: &str) -> Option<Board> {
    let mut board = Board::new();

    if line.starts_with('#') {
        return None;
    }

    line.chars().enumerate().for_each(|(idx, c)| {
        if let Some(d) = c.to_digit(10) {
            board.set_value_at(idx, (d - 1) as Value);
        }
    });
    Some(board)
}

/// Sudoku solver
#[derive(StructOpt, Debug)]
#[structopt(name = "sudoku")]
struct Args {
    /// Print the unsolved and solved boards
    #[structopt(short, long)]
    print: bool,

    /// Check solved boards for correctness
    #[structopt(short, long)]
    verify: bool,

    /// Number of benchmark rounds to run
    #[structopt(short, long, default_value = "1")]
    rounds: u32,

    /// File argument with one board per line
    #[structopt(short, long, default_value = "")]
    boards_file: String,

    /// Sudoku board argument
    #[structopt(short, long, default_value = "")]
    board: String,
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn do_board(args: &Args, line: &str) {
    let o = parse_board(&line);
    if let Some(mut board) = o {
        if args.print {
            println!("\nUnsolved:\n");
            print!("{}", board);
            println!("\n---------------------\n");
        }

        let solved = solve(&mut board);

        if let Some(solved_board) = solved {
            if args.print {
                print!("{}", solved_board);
            }
            if args.verify && !solved_board.is_complete() {
                panic!("Invalid solved board!");
            }
        } else {
            println!("Not Solved!");
            if args.verify {
                panic!("Unsolved board found!")
            }
        }
    }
}

fn main() {
    let args = Args::from_args();

    for _ in 0..args.rounds {
        if !args.board.is_empty() {
            do_board(&args, &args.board);
        } else if let Ok(lines) = read_lines(&args.boards_file) {
            lines.for_each(|r| do_board(&args, &r.unwrap()));
        } else {
            panic!("Must provide either --boards_file or --board !")
        }
    }
}
