use crate::CandidateSet;


/// This function finds the first CandidateSet with a single set bit.
/// This is the heart of constraint propagation, so it should be
/// AS FAST AS POSSIBLE!!!
pub fn single_candidate_position(data: &[CandidateSet]) -> Option<usize> {
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
                .map(|&e| e.count_ones() == 1)
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

/// This function checks if an array contains any zero elements.
/// This is the heart of conflict detection, so it should be
/// AS FAST AS POSSIBLE!!!
pub fn has_any_zeros(arr: &[CandidateSet]) -> bool {
    // Process the data in chunks (to encourage SIMD)...
    arr.chunks(64)
        // Business logic here. This will compile to a simd min reduction.
        .any(|c| c.iter().map(|&x| x).min().unwrap() == 0)
}

pub enum ScanResult {
    /// We found nothing interesting.
    Nothing,

    /// We found a conflict (a candidate set with no candidates).
    Conflict,

    /// We found a single candidate at the given position.
    Single(usize),
}

/// This function scans the data for single candidate positions or conflicts.
pub fn scan(data: &[CandidateSet]) -> ScanResult {
    if has_any_zeros(data) {
        return ScanResult::Conflict;
    }

    if let Some(pos) = single_candidate_position(data) {
        return ScanResult::Single(pos);
    }

    ScanResult::Nothing
}
