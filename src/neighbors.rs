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

// Get the unique peers for a cell (excludes self, no duplicates).
pub fn unique_neighbors(idx: CellIdx) -> &'static [CellIdx; 20] {
    assume!(idx < 81);

    &UNIQUE_NEIGHBORS[idx]
}

const NEIGHBOR_MASK: [[u16; 81]; 81] = {
    let mut table = [[0u16; 81]; 81];
    let mut idx = 0;
    while idx < 81 {
        let peers = &UNIQUE_NEIGHBORS[idx];
        let mut k = 0;
        while k < 20 {
            table[idx][peers[k]] = 0xFFFF;
            k += 1;
        }
        idx += 1;
    }
    table
};

pub fn neighbor_mask(idx: CellIdx) -> &'static [u16; 81] {
    assume!(idx < 81);
    &NEIGHBOR_MASK[idx]
}

const CELL_GROUP_MASK: [[u16; 32]; 81] = {
    let mut table = [[0xFFFFu16; 32]; 81];
    let mut idx = 0;
    while idx < 81 {
        let r = idx / 9;
        let c = idx % 9;
        let b = (r / 3) * 3 + (c / 3);
        let box_idx = (r % 3) * 3 + (c % 3);
        table[idx][r] = !(1u16 << c);
        table[idx][9 + c] = !(1u16 << r);
        table[idx][18 + b] = !(1u16 << box_idx);
        idx += 1;
    }
    table
};

pub fn cell_group_mask(idx: CellIdx) -> &'static [u16; 32] {
    assume!(idx < 81);
    &CELL_GROUP_MASK[idx]
}

const CELL_GROUP_SET_MASK: [[u16; 32]; 81] = {
    let mut table = [[0u16; 32]; 81];
    let mut idx = 0;
    while idx < 81 {
        let r = idx / 9;
        let c = idx % 9;
        let b = (r / 3) * 3 + (c / 3);
        table[idx][r] = crate::types::SET_CANDS;
        table[idx][9 + c] = crate::types::SET_CANDS;
        table[idx][18 + b] = crate::types::SET_CANDS;
        idx += 1;
    }
    table
};

pub fn cell_group_set_mask(idx: CellIdx) -> &'static [u16; 32] {
    assume!(idx < 81);
    &CELL_GROUP_SET_MASK[idx]
}

pub fn value_block_mask(idx: CellIdx) -> &'static [u16; 32] {
    assume!(idx < 81);

    &VALUE_BLOCK_MASK[idx]
}

const VALUE_BLOCK_MASK: [[u16; 32]; 81] = {
    let mut table = [[0xFFFFu16; 32]; 81];
    let mut idx = 0usize;
    while idx < 81 {
        let r = idx / 9;
        let c = idx % 9;
        let br = r / 3;
        let bc = c / 3;
        let b = br * 3 + bc;

        // Row group entries (positions 0..8).
        // For each row: this value can no longer go in this cell's column.
        // For rows in the same box band: also can't go in any column of this box.
        let mut j = 0;
        while j < 9 {
            table[idx][j] &= !(1u16 << c);
            if j / 3 == br {
                table[idx][j] &= !(1u16 << (bc * 3));
                table[idx][j] &= !(1u16 << (bc * 3 + 1));
                table[idx][j] &= !(1u16 << (bc * 3 + 2));
            }
            j += 1;
        }

        // Col group entries (positions 9..17).
        // For each col: this value can no longer go in this cell's row.
        // For cols in the same box band: also can't go in any row of this box.
        let mut k = 0;
        while k < 9 {
            table[idx][9 + k] &= !(1u16 << r);
            if k / 3 == bc {
                table[idx][9 + k] &= !(1u16 << (br * 3));
                table[idx][9 + k] &= !(1u16 << (br * 3 + 1));
                table[idx][9 + k] &= !(1u16 << (br * 3 + 2));
            }
            k += 1;
        }

        // Box group entries (positions 18..26).
        // For boxes in the same row band: this value can't go in positions sharing this cell's row.
        // For boxes in the same col band: this value can't go in positions sharing this cell's col.
        let mut bx = 0;
        while bx < 9 {
            if bx / 3 == br {
                let m = (r % 3) * 3;
                table[idx][18 + bx] &= !(1u16 << m);
                table[idx][18 + bx] &= !(1u16 << (m + 1));
                table[idx][18 + bx] &= !(1u16 << (m + 2));
            }
            if bx % 3 == bc {
                let m = c % 3;
                table[idx][18 + bx] &= !(1u16 << m);
                table[idx][18 + bx] &= !(1u16 << (m + 3));
                table[idx][18 + bx] &= !(1u16 << (m + 6));
            }
            bx += 1;
        }

        table[idx][r] = 0;
        table[idx][9 + c] = 0;
        table[idx][18 + b] = 0;

        idx += 1;
    }
    table
};
