use crate::assume::*;
use crate::board::ConstraintPropagationResult::NoConstraintsPropagated;
use crate::low_level::*;
use crate::neighbors::*;
use crate::types::*;
use std::fmt;

struct MutGroupsForCandidate<'a> {
    groups: &'a mut [CandidateSet],
}

impl<'a> MutGroupsForCandidate<'a> {
    fn mut_row_candidates(&mut self, r: GroupNum) -> &mut CandidateSet {
        assume!(r < 9);
        assume!(r < self.groups.len());

        &mut self.groups[r]
    }

    fn mut_col_candidates(&mut self, c: GroupNum) -> &mut CandidateSet {
        assume!(c < 9);
        assume!(9 + c < self.groups.len());

        &mut self.groups[9 + c]
    }

    fn mut_box_candidates(&mut self, b: GroupNum) -> &mut CandidateSet {
        assume!(b < 9);
        assume!(18 + b < self.groups.len());

        &mut self.groups[18 + b]
    }
}

// Encodes the following information:
//   for value `i`, in some group (row, col, or box), which cells of that group can hold `i`?
// Each value's 27 group entries are padded to 32 (2 AVX2 ymm vectors = 64 bytes) for SIMD alignment.
const GROUP_STRIDE: usize = 32;
const NUM_GROUP_CANDS: usize = 9 * GROUP_STRIDE;
const INIT_GROUP_CANDS: [CandidateSet; NUM_GROUP_CANDS] = {
    let mut arr = [SET_CANDS; NUM_GROUP_CANDS];
    let mut v = 0;
    while v < 9 {
        let mut g = 0;
        while g < 27 {
            arr[v * GROUP_STRIDE + g] = ALL_CANDS;
            g += 1;
        }
        v += 1;
    }
    arr
};

#[derive(Debug, Copy, Clone)]
#[repr(C, align(64))]
struct CandidateToGroups {
    candidates: [CandidateSet; NUM_GROUP_CANDS],
}

impl CandidateToGroups {
    fn mut_groups_for_candidate(&mut self, v: Value) -> MutGroupsForCandidate<'_> {
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

// These are the possible outcomes of constraint propagation.
#[derive(PartialEq)]
enum ConstraintPropagationResult {
    NoConstraintsPropagated,
    PropagatedConstraint,
    FoundConflict,
    Solved,
}

impl Board {
    pub fn new() -> Board {
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
                candidates: INIT_GROUP_CANDS,
            },
        }
    }

    // Is this board solved?
    pub fn is_solved(&self) -> bool {
        self.num_remaining_cells == 0
    }

    // Is every cell set, and do its neighbors obey the Sudoku rules?
    pub fn is_complete(&self) -> bool {
        (0..81).all(|idx| {
            self.values[idx].is_set()
                && unique_neighbors(idx).iter().all(|&other_idx| {
                    assume!(other_idx < 81);
                    self.values[other_idx] != self.values[idx]
                })
        })
    }

    // Get mutable candidates at a cell, assuming a valid index.
    fn mut_candidates_at(&mut self, idx: CellIdx) -> &mut CandidateSet {
        assume!(idx < 81);

        &mut self.candidates[idx]
    }

    // Get candidates at a cell, assuming a valid index.
    fn candidates_at(&self, idx: CellIdx) -> &CandidateSet {
        assume!(idx < 81);

        &self.candidates[idx]
    }

    // Set a value at a cell.
    // This function handles updating the constraint propagation data structures.
    #[inline(always)]
    fn set_value_at(&mut self, idx: CellIdx, v: Value) {
        assume!(idx < 81);
        assume!(v < 9);

        // Basic bookkeeping.
        self.values[idx] = v;
        self.num_remaining_cells -= 1;

        // Erase this value from the cell's unique neighbors.
        let v_bit = 1u16 << v;
        let nmask = neighbor_mask(idx);
        for j in 0..81 {
            self.candidates[j] &= !(nmask[j] & v_bit);
        }
        self.candidates[idx] = SET_CANDS;

        // This cell is taken, so for each value, remove this cell's position
        // from the row, column, and box group candidate sets.
        let cg_mask = cell_group_mask(idx);
        for i in 0..9 {
            let g_base = i * GROUP_STRIDE;
            for j in 0..GROUP_STRIDE {
                self.candidate_to_groups.candidates[g_base + j] &= cg_mask[j];
            }
        }

        // This value can no longer appear in any group that sees this cell,
        // and is now SET_CANDS in this cell's own row, column, and box.
        let mask = value_block_mask(idx);
        let smask = cell_group_set_mask(idx);
        let base = (v as usize) * GROUP_STRIDE;
        assume!(base + GROUP_STRIDE <= self.candidate_to_groups.candidates.len());
        for j in 0..GROUP_STRIDE {
            self.candidate_to_groups.candidates[base + j] &= mask[j];
        }
        for j in 0..GROUP_STRIDE {
            self.candidate_to_groups.candidates[base + j] |= smask[j];
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
                let n = c.num_candidates() as u16;
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
        use ConstraintPropagationResult::*;
        if self.is_solved() {
            return Solved;
        }

        let naked_single = self.set_naked_single();
        if naked_single != NoConstraintsPropagated {
            return naked_single;
        }

        let hidden_single = self.set_hidden_singles();
        if hidden_single != NoConstraintsPropagated {
            return hidden_single;
        }

        NoConstraintsPropagated
    }

    // This function finds and sets a 'naked single' if one exists.
    // A naked single is a cell that only has one candidate value.
    fn set_naked_single(&mut self) -> ConstraintPropagationResult {
        // The candidates array contains a bitset of candidates for every position.
        // If any of these bitsets contains a single bit, it must be a naked single!
        let scan_result = scan(&self.candidates);

        match scan_result {
            // We found a conflict, so we can't propagate any constraints.
            ScanResult::Conflict => ConstraintPropagationResult::FoundConflict,

            // We found a single candidate at the given position.
            ScanResult::Single(idx) => {
                let v = self.candidates[idx].trailing_zeros() as Value;
                self.set_value_at(idx, v);
                ConstraintPropagationResult::PropagatedConstraint
            }

            // We found nothing interesting.
            ScanResult::Nothing => ConstraintPropagationResult::NoConstraintsPropagated,
        }
    }

    // This function finds and sets a 'hidden single' if one exists.
    // A hidden single is a value that is only viable at one cell in a group.
    fn set_hidden_singles(&mut self) -> ConstraintPropagationResult {
        // The candidate_to_groups_mapping tells where each value is viable in each group.
        // If there are any single-bit values in that array, that is a hidden single!
        let scan_result = scan(&self.candidate_to_groups.candidates);
        match scan_result {
            // We found nothing interesting.
            ScanResult::Nothing => NoConstraintsPropagated,

            // We found a conflict, so we can't propagate any constraints.
            ScanResult::Conflict => ConstraintPropagationResult::FoundConflict,

            // We found a single candidate at the given position.
            ScanResult::Single(i) => {
                // We found a hidden single! We just need to extract its position information so that
                // we can set the value at the right cell.
                let g = i % GROUP_STRIDE;
                if g >= 27 {
                    return NoConstraintsPropagated;
                }
                let group_idx = g % 9;
                let group_type = g / 9;
                let v = (i / GROUP_STRIDE) as Value;

                let cell_in_group =
                    self.candidate_to_groups.candidates[i].trailing_zeros() as usize;
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
                ConstraintPropagationResult::PropagatedConstraint
            }
        }
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
                if board.candidates_at(idx).has_candidate(v) {
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

pub fn parse_board(line: &str) -> Option<Board> {
    if line.starts_with('#') {
        return None;
    }

    let mut board = Board::new();

    line.bytes().enumerate().for_each(|(idx, c)| {
        if (b'1'..=b'9').contains(&c) {
            let d = c - b'0';
            board.set_value_at(idx, (d - 1) as Value);
        }
    });
    Some(board)
}
