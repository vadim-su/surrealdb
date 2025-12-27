//! Stores database ID generator global counter
//!
//! This key stores the global counter for database ID generation within a namespace.
//! It tracks the next available batch start value to avoid range scans
//! during batch allocation, significantly reducing transaction conflicts
//! in concurrent scenarios.

use storekey::{BorrowDecode, Encode};

use crate::catalog::NamespaceId;
use crate::key::category::{Categorise, Category};
use crate::kvs::impl_kv_key_storekey;

/// Key structure for storing the database ID generator global counter.
///
/// This key stores a single i64 value representing the next available
/// batch start for database ID generation within a namespace. Using a global counter
/// instead of range scanning all batch allocations reduces the read-set
/// in optimistic transactions and prevents conflicts.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Encode, BorrowDecode)]
pub(crate) struct DatabaseIdGeneratorGlobalKey {
	__: u8,
	_a: u8,
	pub ns: NamespaceId,
	_b: u8,
	_c: u8,
	_d: u8,
}

impl_kv_key_storekey!(DatabaseIdGeneratorGlobalKey => Vec<u8>);

impl Categorise for DatabaseIdGeneratorGlobalKey {
	fn categorise(&self) -> Category {
		Category::DatabaseIdentifierBatch
	}
}

impl DatabaseIdGeneratorGlobalKey {
	/// Creates a new database ID generator global counter key.
	///
	/// # Arguments
	/// * `ns` - The namespace ID
	pub fn new(ns: NamespaceId) -> Self {
		Self {
			__: b'/',
			_a: b'*',
			ns,
			_b: b'!',
			_c: b'd',
			_d: b'g',
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::kvs::KVKey;

	#[test]
	fn key() {
		let val = DatabaseIdGeneratorGlobalKey::new(NamespaceId(1));
		let enc = DatabaseIdGeneratorGlobalKey::encode_key(&val).unwrap();
		assert_eq!(&enc, b"/*\x00\x00\x00\x01!dg");
	}
}
