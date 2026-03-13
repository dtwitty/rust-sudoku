use crate::assume::assume;
use crate::{CellIdx, GroupIdx, GroupNum};

// Groups are collections of cells that must contain 1-9.
// They are implemented such that the same code works for a Row, Col, or Box.
pub trait Group {
    // What is the `idx`th cell in the `g`th group of this type?
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx;

    // What is the index of this cell within groups of this type?
    fn group_idx(idx: CellIdx) -> GroupIdx;

    // Which group of this type is the cell in?
    fn for_cell(idx: CellIdx) -> GroupNum;
}

pub struct Row;
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

pub struct Col;
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

pub struct Box;
impl Group for Box {
    fn cell_at(g: GroupNum, idx: GroupIdx) -> CellIdx {
        assume!(g < 9);
        assume!(idx < 9);

        let box_row = g / 3;
        let box_col = g % 3;
        let cell_row = idx / 3;
        let cell_col = idx % 3;
        (box_row * 27) + (box_col * 3) + (cell_row * 9) + cell_col
    }

    fn group_idx(idx: CellIdx) -> GroupIdx {
        assume!(idx < 81);

        let row = idx / 9;
        let col = idx % 9;
        let box_row = row % 3;
        let box_col = col % 3;
        box_row * 3 + box_col
    }

    fn for_cell(idx: CellIdx) -> GroupNum {
        assume!(idx < 81);

        let row = idx / 9;
        let col = idx % 9;
        let box_row = row / 3;
        let box_col = col / 3;
        box_row * 3 + box_col
    }
}

// Get a list of all cells that see this cell.
pub fn neighbors(idx: CellIdx) -> [CellIdx; 20] {
    assume!(idx < 81);
    UNIQUE_NEIGHBORS[idx]
}

// Precomputed table of unique neighbors (peers) for each cell, excluding self.
// Each cell has exactly 20 unique peers: 8 row + 8 col (not in row) + 4 box (not in row or col).
const UNIQUE_NEIGHBORS: [[CellIdx; 20]; 81] = {
    let mut table = [[0usize; 20]; 81];
    let mut idx = 0;
    while idx < 81 {
        let r = idx / 9;
        let c = idx % 9;
        let br = (r / 3) * 3;
        let bc = (c / 3) * 3;
        let mut count = 0;
        // Row peers (excluding self)
        let mut j = 0;
        while j < 9 {
            let peer = r * 9 + j;
            if peer != idx {
                table[idx][count] = peer;
                count += 1;
            }
            j += 1;
        }
        // Col peers (excluding self and row duplicates)
        j = 0;
        while j < 9 {
            let peer = j * 9 + c;
            if peer != idx && j != r {
                table[idx][count] = peer;
                count += 1;
            }
            j += 1;
        }
        // Box peers not already in same row or col
        let mut bi = 0;
        while bi < 3 {
            let mut bj = 0;
            while bj < 3 {
                let peer = (br + bi) * 9 + (bc + bj);
                if peer != idx && (br + bi) != r && (bc + bj) != c {
                    table[idx][count] = peer;
                    count += 1;
                }
                bj += 1;
            }
            bi += 1;
        }
        idx += 1;
    }
    table
};

// Get the 20 unique peers for a cell (excludes self, no duplicates).
pub fn unique_neighbors(idx: CellIdx) -> &'static [CellIdx; 20] {
    &UNIQUE_NEIGHBORS[idx]
}
