/// In debug mode, check that the condition is true. In release mode, assume it is true, allowing
/// unsafe optimizations to be applied.
macro_rules! assume {
    ($condition:expr) => {
        debug_assert!($condition);
        unsafe {
            core::hint::assert_unchecked($condition);
        }
    };
}


pub(crate) use assume;
