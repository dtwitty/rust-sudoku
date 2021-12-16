#![feature(core_intrinsics)]
#![feature(const_assume)]
#![feature(stdsimd)]

extern crate aligned;
extern crate structopt;
use aligned::{Aligned, A32};
use core::arch::x86_64::*;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use structopt::StructOpt;

type CellIdx = usize;
type GroupNum = usize;
type GroupIdx = usize;
type GroupCells = [CellIdx; 9];

trait Group {
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx;
    fn group_idx(idx: CellIdx) -> GroupIdx;
    fn for_cell(idx: CellIdx) -> GroupNum;

    fn cells(g: GroupNum) -> GroupCells {
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
        Self::cells(Self::for_cell(idx))
    }
}

struct Row;
impl Group for Row {
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx {
        9 * g + idx
    }

    fn group_idx(idx: CellIdx) -> GroupIdx {
        idx % 9
    }

    fn for_cell(idx: CellIdx) -> GroupNum {
        idx / 9
    }
}

struct Col;
impl Group for Col {
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx {
        idx * 9 + g
    }

    fn group_idx(idx: CellIdx) -> GroupIdx {
        idx / 9
    }

    fn for_cell(idx: CellIdx) -> GroupNum {
        idx % 9
    }
}

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
        let row = idx / 9;
        let col = idx % 9;
        (row % 3) * 3 + col % 3
    }

    fn for_cell(idx: CellIdx) -> GroupNum {
        let row = idx / 9;
        let col = idx % 9;
        let box_row = row / 3;
        let box_col = col / 3;
        box_row * 3 + box_col
    }
}

fn all_neighbors(idx: CellIdx) -> [CellIdx; 27] {
    let mut arr: [CellIdx; 27] = [0; 27];
    arr[..9].clone_from_slice(&Row::neighbors(idx));
    arr[9..18].clone_from_slice(&Col::neighbors(idx));
    arr[18..].clone_from_slice(&Box::neighbors(idx));
    arr
}

pub type Value = u16;
const NO_VALUE: Value = 10;

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

pub trait CandidateSetMethods {
    fn has_candidate(&self, v: Value) -> bool;
    fn num_candidates(&self) -> u32;
    fn remove_candidate(&mut self, v: Value);
}

impl CandidateSetMethods for CandidateSet {
    fn has_candidate(&self, v: Value) -> bool {
        (*self & (1 << v)) != 0
    }
    fn num_candidates(&self) -> u32 {
        self.count_ones()
    }
    fn remove_candidate(&mut self, v: Value) {
        *self &= !(1 << v);
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
pub struct Board {
    candidates: [CandidateSet; 81],
    values: [Value; 81],
    num_remaining_cells: usize,
    candidate_to_groups: CandidateToGroups,
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

fn single_candidate_position(data: &[CandidateSet]) -> Option<usize> {
    const N: usize = 128;
    data.chunks(N).enumerate().find_map(|(chunk, s)| {
        let mut a: Aligned<A32, _> = Aligned([0u8; N]);
        let arr = a.as_mut_slice();
        arr.iter_mut()
            .zip(s.iter())
            .for_each(|(a, &c)| *a = (((c != 0) & ((c & (c - 1)) == 0)) as u8) * 0xFF);
        unsafe {
            let ptr = arr.as_ptr();

            let mut bitmask: u128 = 0;
            for i in 0..N / 32 {
                let packed_bits = _mm256_load_si256(ptr.offset(i as isize * 32) as *const __m256i);
                let bits = _mm256_movemask_epi8(packed_bits) as u128;
                bitmask |= bits << (i * 32);
            }

            if bitmask != 0 {
                Some(chunk * N + (bitmask.trailing_zeros() as usize))
            } else {
                None
            }
        }
    })
}

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
            candidates: [ALL_CANDS; 81],
            values: [NO_VALUE; 81],
            num_remaining_cells: 81,
            candidate_to_groups: CandidateToGroups {
                candidates: [ALL_CANDS; 9 * 3 * 9],
            },
        }
    }

    fn is_solved(&self) -> bool {
        self.num_remaining_cells == 0
    }

    fn is_complete(&self) -> bool {
        (0..81).all(|idx| {
            self.values[idx].is_set()
                && all_neighbors(idx).iter().all(|&other_idx| {
                    idx == other_idx || self.values[other_idx] != self.values[idx]
                })
        })
    }

    fn unsafe_mut_candidates_at(&mut self, idx: CellIdx) -> &mut CandidateSet {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        &mut self.candidates[idx]
    }

    fn unsafe_candidates_at(&self, idx: CellIdx) -> &CandidateSet {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        &self.candidates[idx]
    }

    #[inline(never)]
    fn set_value_at(&mut self, idx: CellIdx, v: Value) {
        unsafe {
            core::intrinsics::assume(idx < 81);
        }
        self.values[idx] = v;
        self.candidates[idx] = 0;
        self.num_remaining_cells -= 1;

        all_neighbors(idx).iter().for_each(|&other_idx| {
            self.unsafe_mut_candidates_at(other_idx).remove_candidate(v);
        });

        let r = Row::for_cell(idx);
        let c = Col::for_cell(idx);
        let b = Box::for_cell(idx);
        unsafe {
            core::intrinsics::assume(v < 9);
            core::intrinsics::assume(r < 9);
            core::intrinsics::assume(c < 9);
            core::intrinsics::assume(b < 9);
        }

        for i in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(i)
                .mut_row_candidates(r)
                .remove_candidate(Row::group_idx(idx) as Value);
        }
        for i in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(i)
                .mut_col_candidates(c)
                .remove_candidate(Col::group_idx(idx) as Value);
        }
        for i in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(i)
                .mut_box_candidates(b)
                .remove_candidate(Box::group_idx(idx) as Value);
        }

        *self
            .candidate_to_groups
            .mut_groups_for_candidate(v)
            .mut_row_candidates(r) = 0;
        *self
            .candidate_to_groups
            .mut_groups_for_candidate(v)
            .mut_col_candidates(c) = 0;
        *self
            .candidate_to_groups
            .mut_groups_for_candidate(v)
            .mut_box_candidates(b) = 0;

        for r in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(v)
                .mut_row_candidates(r)
                .remove_candidate(c as Value);
        }

        for br in 0..3 {
            for bc in 0..3 {
                let r = (b / 3) * 3 + br;
                let c = (b % 3) * 3 + bc;
                self.candidate_to_groups
                    .mut_groups_for_candidate(v)
                    .mut_row_candidates(r)
                    .remove_candidate(c as Value);
            }
        }

        for c in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(v)
                .mut_col_candidates(c)
                .remove_candidate(r as Value);
        }

        for bc in 0..3 {
            for br in 0..3 {
                let r = (b / 3) * 3 + br;
                let c = (b % 3) * 3 + bc;
                self.candidate_to_groups
                    .mut_groups_for_candidate(v)
                    .mut_col_candidates(c)
                    .remove_candidate(r as Value);
            }
        }

        for c in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(v)
                .mut_box_candidates((r / 3) * 3 + (c / 3))
                .remove_candidate(((r % 3) * 3 + (c % 3)) as Value);
        }

        for r in 0..9 {
            self.candidate_to_groups
                .mut_groups_for_candidate(v)
                .mut_box_candidates((r / 3) * 3 + (c / 3))
                .remove_candidate(((r % 3) * 3 + (c % 3)) as Value);
        }
    }

    #[inline(never)]
    fn most_constrained_cell(&self) -> CellIdx {
        const MODS: [u8; 81] = [
            0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
            2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
            1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
        ];

        fn cbs(mut v: u16) -> u16 {
            v = v - ((v >> 1) & 0x5555);
            v = (v & 0x3333) + ((v >> 2) & 0x3333);
            ((v + (v >> 4) & 0xF0F) * 0x101) >> 8
        }

        let mut arr = [0u8; 81];
        arr.iter_mut()
            .zip(self.candidates.iter())
            .zip(MODS.iter())
            .enumerate()
            .for_each(|(_i, ((a, &c), &m))| {
                let is_zero = (c == 0) as u8;
                let n = cbs(c as u16) as u8;
                *a = is_zero << 7 | n << 2 | m;
            });
        (arr.iter()
            .enumerate()
            .map(|(i, &a)| (i as u8, a))
            .reduce(|(i, a), (j, b)| if b <= a { (j, b) } else { (i, a) })
            .unwrap()
            .0) as usize
    }

    fn propagate_constraints(&mut self) -> ConstraintPropagationResult {
        let mut result = ConstraintPropagationResult::PropagatedConstraint;
        while result == ConstraintPropagationResult::PropagatedConstraint {
            result = self.try_propagate_constraints();
        }
        result
    }

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

    #[inline(never)]
    fn has_conflict(&self) -> bool {
        const N: usize = 64;
        return self
            .candidates
            .chunks(N)
            .zip(self.values.chunks(N))
            .any(|(c_chunk, v_chunk)| {
                c_chunk
                    .iter()
                    .zip(v_chunk.iter())
                    .map(|(&cands, &v)| (!v.is_set() & (cands == 0)) as u8)
                    .sum::<u8>()
                    != 0
            });
    }

    #[inline(never)]
    fn set_naked_single(&mut self) -> bool {
        let i = single_candidate_position(&self.candidates);

        if let Some(idx) = i {
            let cands = self.candidates[idx];
            let v = cands.trailing_zeros();
            self.set_value_at(idx, v as Value);
            return true;
        }

        false
    }

    #[inline(never)]
    fn set_hidden_singles(&mut self) -> bool {
        let o = single_candidate_position(&self.candidate_to_groups.candidates);
        if let Some(i) = o {
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
        false
    }
}

pub fn solve(board: &mut Board) -> Option<Board> {
    let res = board.propagate_constraints();
    match res {
        ConstraintPropagationResult::Solved => Some(*board),
        ConstraintPropagationResult::FoundConflict => None,
        _ => {
            let idx = board.most_constrained_cell();
            for v in 0..9 {
                if board.unsafe_candidates_at(idx).has_candidate(v) {
                    let mut new_board: Board = *board;
                    new_board.set_value_at(idx, v);

                    let solution = solve(&mut new_board);
                    if solution.is_some() {
                        return solution;
                    }
                }
            }
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
