use crate::{CellIdx, GroupCells, GroupIdx, GroupNum};
use crate::assume::assume;

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

pub struct Box;
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
pub fn all_neighbors(idx: CellIdx) -> [CellIdx; 27] {
    assume!(idx < 81);

    let mut arr: [CellIdx; 27] = [0; 27];
    arr[..9].clone_from_slice(&Row::neighbors(idx));
    arr[9..18].clone_from_slice(&Col::neighbors(idx));
    arr[18..].clone_from_slice(&Box::neighbors(idx));
    arr
}
