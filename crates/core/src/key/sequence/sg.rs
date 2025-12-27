//! Stores user sequence global counter
//!
//! This key stores the global counter for user-defined sequence ID generation.
//! It tracks the next available batch start value to avoid range scans
//! during batch allocation, significantly reducing transaction conflicts
//! in concurrent scenarios.

use std::borrow::Cow;

use storekey::{BorrowDecode, Encode};

use crate::catalog::{DatabaseId, NamespaceId};
use crate::key::category::{Categorise, Category};
use crate::kvs::impl_kv_key_storekey;

/// Key structure for storing the user sequence global counter.
///
/// This key stores a single i64 value representing the next available
/// batch start for user-defined sequence generation. Using a global counter
/// instead of range scanning all batch allocations reduces the read-set
/// in optimistic transactions and prevents conflicts.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Encode, BorrowDecode)]
pub(crate) struct Sg<'a> {
	__: u8,
	_a: u8,
	pub ns: NamespaceId,
	_b: u8,
	pub db: DatabaseId,
	_c: u8,
	_d: u8,
	_e: u8,
	pub sq: Cow<'a, str>,
	_f: u8,
	_g: u8,
	_h: u8,
}

impl_kv_key_storekey!(Sg<'_> => Vec<u8>);

impl Categorise for Sg<'_> {
	fn categorise(&self) -> Category {
		Category::SequenceBatch
	}
}

impl<'a> Sg<'a> {
	/// Creates a new user sequence global counter key.
	///
	/// # Arguments
	/// * `ns` - The namespace ID
	/// * `db` - The database ID
	/// * `sq` - The sequence name
	pub fn new(ns: NamespaceId, db: DatabaseId, sq: &'a str) -> Self {
		Self {
			__: b'/',
			_a: b'*',
			ns,
			_b: b'*',
			db,
			_c: b'!',
			_d: b's',
			_e: b'q',
			sq: Cow::Borrowed(sq),
			_f: b'!',
			_g: b's',
			_h: b'g',
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::kvs::KVKey;

	#[test]
	fn key() {
		let val = Sg::new(NamespaceId(1), DatabaseId(2), "testsq");
		let enc = Sg::encode_key(&val).unwrap();
		assert_eq!(&enc, b"/*\x00\x00\x00\x01*\x00\x00\x00\x02!sqtestsq\0!sg");
	}
}
