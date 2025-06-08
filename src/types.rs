use crate::assume::assume;

/// The index of a cell in the board.
pub type CellIdx = usize;

/// The number of a group in the board. For "the 5th row", this would be 4.
pub type GroupNum = usize;

/// The index of a cell in a group. For "the 5th cell in the 3rd row", this would be 4.
pub type GroupIdx = usize;

/// A group of cells in the board, like a row, column, or box.
pub type GroupCells = [CellIdx; 9];

// The type of values assigned to cells.
// Can also encode an unknown value.
pub type Value = u8;
pub const NO_VALUE: Value = 255;

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
pub const ALL_CANDS: CandidateSet = 511;
pub const SET_CANDS: CandidateSet = 0x600;

pub trait CandidateSetMethods {
    fn has_candidate(&self, v: Value) -> bool;
    fn num_candidates(&self) -> u32;
    fn remove_candidate(&mut self, v: Value);
    fn remove_candidates(&mut self, values: &[Value]);
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
    fn remove_candidates(&mut self, values: &[Value]) {
        let mut mask = 0 as CandidateSet;
        for &v in values {
            assume!(v < 9);
            mask |= 1 << v;
        }
        *self &= !mask;
    }
    fn is_set(&self) -> bool {
        *self == SET_CANDS
    }
}
