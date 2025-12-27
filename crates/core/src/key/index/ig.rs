//! Stores FT index document ID generator global counter
//!
//! This key stores the global counter for full-text index document ID generation.
//! It tracks the next available batch start value to avoid range scans
//! during batch allocation, significantly reducing transaction conflicts
//! in concurrent scenarios.

use std::borrow::Cow;

use storekey::{BorrowDecode, Encode};

use crate::catalog::{DatabaseId, IndexId, NamespaceId};
use crate::key::category::{Categorise, Category};
use crate::kvs::impl_kv_key_storekey;
use crate::val::TableName;

/// Key structure for storing the FT index document ID generator global counter.
///
/// This key stores a single i64 value representing the next available
/// batch start for document ID generation in a full-text search index.
/// Using a global counter instead of range scanning all batch allocations
/// reduces the read-set in optimistic transactions and prevents conflicts.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Encode, BorrowDecode)]
#[storekey(format = "()")]
pub(crate) struct Ig<'a> {
	__: u8,
	_a: u8,
	pub ns: NamespaceId,
	_b: u8,
	pub db: DatabaseId,
	_c: u8,
	pub tb: Cow<'a, TableName>,
	_d: u8,
	pub ix: IndexId,
	_e: u8,
	_f: u8,
	_g: u8,
}

impl_kv_key_storekey!(Ig<'_> => Vec<u8>);

impl Categorise for Ig<'_> {
	fn categorise(&self) -> Category {
		Category::SequenceBatch
	}
}

impl<'a> Ig<'a> {
	/// Creates a new FT index document ID generator global counter key.
	///
	/// # Arguments
	/// * `ns` - The namespace ID
	/// * `db` - The database ID
	/// * `tb` - The table name
	/// * `ix` - The index ID
	pub fn new(ns: NamespaceId, db: DatabaseId, tb: &'a TableName, ix: IndexId) -> Self {
		Self {
			__: b'/',
			_a: b'*',
			ns,
			_b: b'*',
			db,
			_c: b'*',
			tb: Cow::Borrowed(tb),
			_d: b'+',
			ix,
			_e: b'!',
			_f: b'i',
			_g: b'g',
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::kvs::KVKey;

	#[test]
	fn key() {
		let tb = TableName::from("testtb");
		let val = Ig::new(NamespaceId(1), DatabaseId(2), &tb, IndexId(3));
		let enc = Ig::encode_key(&val).unwrap();
		assert_eq!(
			&enc,
			b"/*\x00\x00\x00\x01*\x00\x00\x00\x02*testtb\0+\0\0\0\x03!ig"
		);
	}
}
