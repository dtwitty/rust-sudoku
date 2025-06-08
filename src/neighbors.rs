use crate::assume::assume;
use crate::{CellIdx, GroupCells, GroupIdx, GroupNum};

// Groups are collections of cells that must contain 1-9.
// They are implemented such that the same code works for a Row, Col, or Box.
pub trait Group {
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

// Get a list (with possible repeats!) of all cells that see this cell.
pub fn all_neighbors(idx: CellIdx) -> [CellIdx; 27] {
    assume!(idx < 81);

    let mut arr: [CellIdx; 27] = [0; 27];
    arr[..9].clone_from_slice(&Row::neighbors(idx));
    arr[9..18].clone_from_slice(&Col::neighbors(idx));
    arr[18..].clone_from_slice(&Box::neighbors(idx));
    arr
}
