//! Common utilities and types shared across backends.
//!
//! Provides shared functionality for code generation, type mapping,
//! and common patterns used across different compilation targets.

pub mod types;
pub mod utils;

/// Common data types and mappings
pub use types::*;

/// Utility functions
pub use utils::*;

