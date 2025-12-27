//! Stores table ID generator global counter
//!
//! This key stores the global counter for table ID generation within a database.
//! It tracks the next available batch start value to avoid range scans
//! during batch allocation, significantly reducing transaction conflicts
//! in concurrent scenarios.

use storekey::{BorrowDecode, Encode};

use crate::catalog::{DatabaseId, NamespaceId};
use crate::key::category::{Categorise, Category};
use crate::kvs::impl_kv_key_storekey;

/// Key structure for storing the table ID generator global counter.
///
/// This key stores a single i64 value representing the next available
/// batch start for table ID generation within a database. Using a global counter
/// instead of range scanning all batch allocations reduces the read-set
/// in optimistic transactions and prevents conflicts.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Encode, BorrowDecode)]
pub(crate) struct TableIdGeneratorGlobalKey {
	__: u8,
	_a: u8,
	pub ns: NamespaceId,
	_b: u8,
	pub db: DatabaseId,
	_c: u8,
	_d: u8,
	_e: u8,
}

impl_kv_key_storekey!(TableIdGeneratorGlobalKey => Vec<u8>);

impl Categorise for TableIdGeneratorGlobalKey {
	fn categorise(&self) -> Category {
		Category::DatabaseTableIdentifierBatch
	}
}

impl TableIdGeneratorGlobalKey {
	/// Creates a new table ID generator global counter key.
	///
	/// # Arguments
	/// * `ns` - The namespace ID
	/// * `db` - The database ID
	pub fn new(ns: NamespaceId, db: DatabaseId) -> Self {
		Self {
			__: b'/',
			_a: b'*',
			ns,
			_b: b'*',
			db,
			_c: b'!',
			_d: b't',
			_e: b'g',
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::kvs::KVKey;

	#[test]
	fn key() {
		let val = TableIdGeneratorGlobalKey::new(NamespaceId(1), DatabaseId(2));
		let enc = TableIdGeneratorGlobalKey::encode_key(&val).unwrap();
		assert_eq!(&enc, b"/*\x00\x00\x00\x01*\x00\x00\x00\x02!tg");
	}
}
