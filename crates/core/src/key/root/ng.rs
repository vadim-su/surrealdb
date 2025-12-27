//! Stores namespace ID generator global counter
//!
//! This key stores the global counter for namespace ID generation.
//! It tracks the next available batch start value to avoid range scans
//! during batch allocation, significantly reducing transaction conflicts
//! in concurrent scenarios.

use storekey::{BorrowDecode, Encode};

use crate::key::category::{Categorise, Category};
use crate::kvs::impl_kv_key_storekey;

/// Key structure for storing the namespace ID generator global counter.
///
/// This key stores a single i64 value representing the next available
/// batch start for namespace ID generation. Using a global counter
/// instead of range scanning all batch allocations reduces the read-set
/// in optimistic transactions and prevents conflicts.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Encode, BorrowDecode)]
pub(crate) struct NamespaceIdGeneratorGlobalKey {
	__: u8,
	_a: u8,
	_b: u8,
	_c: u8,
}

impl_kv_key_storekey!(NamespaceIdGeneratorGlobalKey => Vec<u8>);

impl Categorise for NamespaceIdGeneratorGlobalKey {
	fn categorise(&self) -> Category {
		Category::NamespaceIdentifierBatch
	}
}

impl NamespaceIdGeneratorGlobalKey {
	/// Creates a new namespace ID generator global counter key.
	pub fn new() -> Self {
		Self {
			__: b'/',
			_a: b'!',
			_b: b'n',
			_c: b'g',
		}
	}
}

impl Default for NamespaceIdGeneratorGlobalKey {
	fn default() -> Self {
		Self::new()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::kvs::KVKey;

	#[test]
	fn key() {
		let val = NamespaceIdGeneratorGlobalKey::new();
		let enc = NamespaceIdGeneratorGlobalKey::encode_key(&val).unwrap();
		assert_eq!(&enc, b"/!ng");
	}
}
