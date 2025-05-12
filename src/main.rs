#![feature(core_intrinsics)]

use std::fmt;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use rayon::prelude::*;
use structopt::StructOpt;

macro_rules! assume {
    ($condition:expr) => {
        debug_assert!($condition);
        unsafe {
            core::intrinsics::assume($condition);
        }
    };
}

// These type aliases help differentiate different kinds of ints.
type CellIdx = usize;
type GroupNum = usize;
type GroupIdx = usize;
type GroupCells = [CellIdx; 9];

// Groups are collections of cells that must contain 1-9.
// They are implemented such that the same code works for a Row, Col, or Box.
trait Group {
    // What is the `idx`th cell in the `g`th group of this type?
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx;

    // What is the index of this cell within groups of this type?
    fn group_idx(idx: CellIdx) -> GroupIdx;

    // Which group of this type is the cell in?
    fn for_cell(idx: CellIdx) -> GroupNum;

    // Which cells does the `gth` group of this type contain?
    fn cells(g: GroupNum) -> GroupCells {
        assume!(g < 9);

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

    // What cells see cell `idx` within this group?
    fn neighbors(idx: CellIdx) -> GroupCells {
        assume!(idx < 81);

        Self::cells(Self::for_cell(idx))
    }
}

struct Row;
impl Group for Row {
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx {
        assume!(g < 9);
        assume!(idx < 9);

        9 * g + idx
    }

    fn group_idx(idx: CellIdx) -> GroupIdx {
        assume!(idx < 81);

        idx % 9
    }

    fn for_cell(idx: CellIdx) -> GroupNum {
        assume!(idx < 81);

        idx / 9
    }
}

struct Col;
impl Group for Col {
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx {
        assume!(g < 9);
        assume!(idx < 9);

        idx * 9 + g
    }

    fn group_idx(idx: CellIdx) -> GroupIdx {
        assume!(idx < 81);

        idx / 9
    }

    fn for_cell(idx: CellIdx) -> GroupNum {
        assume!(idx < 81);

        idx % 9
    }
}

// Boxes require some tricky math to compute indices.
// While it is possible to compute these on the fly, these computations don't play well with SIMD.
// To encourage SIMD, we precompute box positions where possible.
const STARTS: [CellIdx; 9] = [0, 3, 6, 27, 30, 33, 54, 57, 60];
const STEPS: [CellIdx; 9] = [0, 1, 2, 9, 10, 11, 18, 19, 20];
const GROUP_IDXS: [GroupIdx; 81] = [
    0, 1, 2, 0, 1, 2, 0, 1, 2,
    3, 4, 5, 3, 4, 5, 3, 4, 5,
    6, 7, 8, 6, 7, 8, 6, 7, 8,
    0, 1, 2, 0, 1, 2, 0, 1, 2,
    3, 4, 5, 3, 4, 5, 3, 4, 5,
    6, 7, 8, 6, 7, 8, 6, 7, 8,
    0, 1, 2, 0, 1, 2, 0, 1, 2,
    3, 4, 5, 3, 4, 5, 3, 4, 5,
    6, 7, 8, 6, 7, 8, 6, 7, 8
];
const BOXES: [GroupNum; 81] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2,
    0, 0, 0, 1, 1, 1, 2, 2, 2,
    0, 0, 0, 1, 1, 1, 2, 2, 2,
    3, 3, 3, 4, 4, 4, 5, 5, 5,
    3, 3, 3, 4, 4, 4, 5, 5, 5,
    3, 3, 3, 4, 4, 4, 5, 5, 5,
    6, 6, 6, 7, 7, 7, 8, 8, 8,
    6, 6, 6, 7, 7, 7, 8, 8, 8,
    6, 6, 6, 7, 7, 7, 8, 8, 8
];

struct Box;
impl Group for Box {
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx {
        assume!(g < 9);
        assume!(idx < 9);

        let start = STARTS[g];
        let step = STEPS[idx];
        start + step
    }

    fn group_idx(idx: CellIdx) -> GroupIdx {
        assume!(idx < 81);

        GROUP_IDXS[idx]
    }

    fn for_cell(idx: CellIdx) -> GroupNum {
        assume!(idx < 81);

        BOXES[idx]
    }
}

// Get a list (with possible repeats!) of all cells that see this cell.
fn all_neighbors(idx: CellIdx) -> [CellIdx; 27] {
    assume!(idx < 81);

    let mut arr: [CellIdx; 27] = [0; 27];
    arr[..9].clone_from_slice(&Row::neighbors(idx));
    arr[9..18].clone_from_slice(&Col::neighbors(idx));
    arr[18..].clone_from_slice(&Box::neighbors(idx));
    arr
}

// The type of values assigned to cells.
// Can also encode an unknown value.
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

// Represents a bit field of candidates from [0, 9).
// The lower bits are used to encode whether a candidate is available.
// The upper bits are used to encode flags, like whether a selection has been made.
// A cell with a selected candidate will be equal to SET_CANDS, and have no lower bits set.
// A CandidateSet with value 0 generally encodes a conflict, implying this board can't be solved.
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
        assume!(v < 9);

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
        assume!(r < 9);
        assume!(r < self.groups.len());

        &mut self.groups[r]
    }

    fn mut_col_candidates(&mut self, c: usize) -> &mut CandidateSet {
        assume!(c < 9);
        assume!(9 + c < self.groups.len());

        &mut self.groups[9 + c]
    }

    fn mut_box_candidates(&mut self, b: usize) -> &mut CandidateSet {
        assume!(b < 9);
        assume!(18 + b < self.groups.len());

        &mut self.groups[18 + b]
    }
}

// Encodes the following information:
//   for value `i`, in some group (row, col, or box), which cells of that group can hold `i`?
#[derive(Debug, Copy, Clone)]
#[repr(C, align(64))]
struct CandidateToGroups {
    candidates: [CandidateSet; 9 * 3 * 9],
}

impl CandidateToGroups {
    fn mut_groups_for_candidate(&mut self, v: Value) -> MutGroupsForCandidate {
        assume!(v < 9);

        let start = (v * 9 * 3) as usize;
        MutGroupsForCandidate {
            groups: &mut self.candidates[start..],
        }
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C, align(64))]
pub struct Board {
    candidates: [CandidateSet; 81],
    values: [Value; 81],
    num_remaining_cells: usize,
    candidate_to_groups: CandidateToGroups,
}

// This function finds the first CandidateSet with a single set bit.
// This is the heart of constraint propagation, so it should be
// AS FAST AS POSSIBLE!!!
fn single_candidate_position(data: &[CandidateSet]) -> Option<usize> {
    const N: usize = 81;
    // Process the data in chunks (to encourage SIMD)...
    data.chunks(N)
        .enumerate()
        // Find the first chunk with a single-bit value...
        .filter_map(|(i, c)| {
            // `k` is the index in this chunk of a single-bit value,
            // or chunk length if there is none.
            let k = c
                .iter()
                // We assume no zero values, so this checks for single-bits.
                .map(|&e| (e & (e - 1)) == 0)
                // This associates an index (or chunk length) with each result.
                .enumerate()
                .map(|(a, b)| if b { a } else { c.len() } as u8)
                // Get the min index.
                .min()
                .unwrap() as usize;
            (k != c.len()).then_some(k + i * N)
        })
        .next()
}

// This function checks if an array contains any zero elements.
// This is the heart of conflict detection, so it should be
// AS FAST AS POSSIBLE!!!
fn has_any_zeros(arr: &[CandidateSet]) -> bool {
    // Process the data in chunks (to encourage SIMD)...
    arr.chunks(81)
        // Process the data in reverse.
        // We tend to set values early in the puzzle first, so conflicts appear later.
        .rev()
        // Business logic here. This will compile to a simd min reduction.
        .any(|c| c.iter().map(|&x| x).min().unwrap() == 0)
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
                assume!(other_idx < 81);

                let is_same_idx = idx == other_idx;
                    let is_diff_value = self.values[other_idx] != self.values[idx];
                    is_same_idx | is_diff_value
                })
        })
    }

    // Get mutable candidates at a cell, assuming a valid index.
    fn unsafe_mut_candidates_at(&mut self, idx: CellIdx) -> &mut CandidateSet {
        assume!(idx < 81);

        &mut self.candidates[idx]
    }

    // Get candidates at a cell, assuming a valid index.
    fn unsafe_candidates_at(&self, idx: CellIdx) -> &CandidateSet {
        assume!(idx < 81);

        &self.candidates[idx]
    }

    // Set a value at a cell.
    // This function handles updating the constraint propagation data structures.
    fn set_value_at(&mut self, idx: CellIdx, v: Value) {
        assume!(idx < 81);

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

        assume!(v < 9);
        assume!(r < 9);
        assume!(c < 9);
        assume!(b < 9);


        for i in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(i)
                // In this row...
                .mut_row_candidates(r)
                // At this position.
                .remove_candidate(Row::group_idx(idx) as Value);
            self.candidate_to_groups
                .mut_groups_for_candidate(i)
                // In this column...
                .mut_col_candidates(c)
                // At this position.
                .remove_candidate(Col::group_idx(idx) as Value);
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

    // This function identifies the cell with the most constraints (ie the fewest candidates).
    // This is useful because this is the easiest cell to exhaustively check with backtracking.
    fn most_constrained_cell(&self) -> CellIdx {
        // Each cell is assigned a 16-bit code like the following:
        // <has_zero_candidates> (1 bit) - to prefer unfinished cells
        // <num_candidates> (4 bits, with one extra) - to prefer cells with fewer candidates
        // <cell_id % 6> (3 bits) - heuristic
        //   <cell_id % 6> is a poor-man's randomization. Without it, the solver will prefer cells
        //   near the end of the array, leading to bunched backtracking that clears fewer candidates.
        //   With it, earlier cells may trump later ones, leading to more even candidate clearing.
        // <idx> (7 bits) - holds the argmin index that will be returned.
        (self
            .candidates
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let is_set = c.is_set() as u16;
                let n = c.count_ones() as u16;
                let m = (i as u16) % 6;
                is_set << 15 | n << 10 | m << 7 | (i as u16)
            })
            .min()
            .unwrap()
            & 0x7F)
            .into()
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
        if has_any_zeros(&self.candidates) {
            return ConstraintPropagationResult::FoundConflict;
        }
        if self.set_naked_single() {
            return ConstraintPropagationResult::PropagatedConstraint;
        }
        if has_any_zeros(&self.candidate_to_groups.candidates) {
            return ConstraintPropagationResult::FoundConflict;
        }
        if self.set_hidden_singles() {
            return ConstraintPropagationResult::PropagatedConstraint;
        }
        ConstraintPropagationResult::NoConstraintsPropagated
    }

    // This function finds and sets a 'naked single' if one exists.
    // A naked single is a cell that only has one candidate value.
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
    fn set_hidden_singles(&mut self) -> bool {
        // The candidate_to_groups_mapping tells where each value is viable in each group.
        // If there are any single-bit values in that array, that is a hidden single!
        let o = single_candidate_position(&self.candidate_to_groups.candidates);

        if let Some(i) = o {
            // We found a hidden single! We just need to extract its position information so that
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
                write!(f, "{sep}").unwrap();
            }
            writeln!(f).unwrap();
            if r == 2 || r == 5 {
                writeln!(f, "{LANE_SEP}").unwrap();
            }
        }
        Ok(())
    }
}

fn parse_board(line: &str) -> Option<Board> {
    let mut board = Board::new();

    if line.starts_with('#') {
        return None;
    }

    line.bytes().enumerate().for_each(|(idx, c)| {
        if c >= b'1' && c <= b'9' {
            let d = c - b'0';
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
    #[structopt(long, default_value = "")]
    boards_file: String,

    /// Sudoku board argument
    #[structopt(long, default_value = "")]
    board: String,

    /// Whether to solve boards in parallel
    #[structopt(long)]
    parallel: bool,
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn do_board(args: &Args, line: &str) {
    let o = parse_board(line);
    if let Some(mut board) = o {
        if args.print {
            println!("\nUnsolved:\n");
            print!("{board}");
            println!("\n---------------------\n");
        }

        let solved = solve(&mut board);

        if let Some(solved_board) = solved {
            if args.print {
                print!("{solved_board}");
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
            if args.parallel {
                lines
                    .par_bridge()
                    .for_each(|r| do_board(&args, &r.unwrap()));
            } else {
                lines.for_each(|r| do_board(&args, &r.unwrap()));
            }
        } else {
            panic!("Must provide either --boards_file or --board !")
        }
    }
}
