//! Distributed sequence and ID generation management.
//!
//! This module provides a distributed ID generation system that uses a batch allocation
//! strategy to efficiently generate unique identifiers across multiple nodes. The system
//! maintains both state (per-node tracking) and batch allocations (reserved ID ranges)
//! to ensure uniqueness while minimizing coordination overhead.
//!
//! # Key Components
//!
//! - **Sequences**: Main coordinator for all sequence operations
//! - **SequenceDomain**: Defines different types of sequences (namespace IDs, database IDs, etc.)
//! - **BatchValue**: Represents a batch allocation of IDs owned by a specific node
//! - **SequenceState**: Tracks the next available ID for a node
//!
//! # ID Generation Strategy
//!
//! Each node maintains local state and coordinates with other nodes through batch allocations
//! stored in the key-value store. When a node needs IDs, it allocates a batch and uses those
//! IDs locally until the batch is exhausted, then allocates a new batch.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use rand::{Rng, thread_rng};
use revision::revisioned;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tokio::time::sleep;
use uuid::Uuid;

use crate::catalog::providers::DatabaseProvider;
use crate::catalog::{DatabaseId, IndexId, NamespaceId, TableId};
use crate::ctx::Context;
use crate::err::Error;
use crate::idx::IndexKeyBase;
use crate::idx::seqdocids::DocId;
use crate::key::database::tg::TableIdGeneratorGlobalKey;
use crate::key::database::th::TableIdGeneratorBatchKey;
use crate::key::database::ti::TableIdGeneratorStateKey;
use crate::key::namespace::dg::DatabaseIdGeneratorGlobalKey;
use crate::key::namespace::dh::DatabaseIdGeneratorBatchKey;
use crate::key::namespace::di::DatabaseIdGeneratorStateKey;
use crate::key::root::ng::NamespaceIdGeneratorGlobalKey;
use crate::key::root::nh::NamespaceIdGeneratorBatchKey;
use crate::key::root::ni::NamespaceIdGeneratorStateKey;
use crate::key::sequence::Prefix;
use crate::key::sequence::ba::Ba;
use crate::key::sequence::sg::Sg;
use crate::key::sequence::st::St;
use crate::key::table::ig::IndexIdGeneratorGlobalKey;
use crate::key::table::ih::IndexIdGeneratorBatchKey;
use crate::key::table::is::IndexIdGeneratorStateKey;
use crate::kvs::ds::TransactionFactory;
use crate::kvs::{KVKey, LockType, Transaction, TransactionType, impl_kv_value_revisioned};
use crate::val::TableName;

type SequencesMap = Arc<RwLock<HashMap<Arc<SequenceDomain>, Arc<Mutex<Sequence>>>>>;

/// Manager for all sequence operations in the system.
///
/// The Sequences struct coordinates ID generation across different domains
/// (namespaces, databases, tables, indexes, and user sequences) and manages
/// the lifecycle of sequence allocations.
#[derive(Clone)]
pub struct Sequences {
	tf: TransactionFactory,
	nid: Uuid,
	sequences: SequencesMap,
}

/// Defines the different types of sequences supported by the system.
///
/// Each variant represents a distinct ID generation domain with its own
/// namespace and allocation strategy.
#[derive(Hash, PartialEq, Eq)]
enum SequenceDomain {
	/// A user-defined sequence in a database
	UserName(NamespaceId, DatabaseId, String),
	/// A sequence generating DocIds for a FullText search index
	FullTextDocIds(IndexKeyBase),
	/// A sequence generating IDs for namespaces
	NameSpacesIds,
	/// A sequence generating IDs for databases
	DatabasesIds(NamespaceId),
	/// A sequence generating IDs for tables
	TablesIds(NamespaceId, DatabaseId),
	/// A sequence generating IDs for indexes
	IndexIds(NamespaceId, DatabaseId, TableName),
}

impl SequenceDomain {
	fn new_user(ns: NamespaceId, db: DatabaseId, sq: &str) -> Self {
		Self::UserName(ns, db, sq.to_string())
	}

	pub(crate) fn new_ft_doc_ids(ikb: IndexKeyBase) -> Self {
		Self::FullTextDocIds(ikb)
	}

	pub(crate) fn new_namespace_ids() -> Self {
		Self::NameSpacesIds
	}

	pub(crate) fn new_database_ids(ns: NamespaceId) -> Self {
		Self::DatabasesIds(ns)
	}

	pub(crate) fn new_table_ids(ns: NamespaceId, db: DatabaseId) -> Self {
		Self::TablesIds(ns, db)
	}

	pub(crate) fn new_index_ids(ns: NamespaceId, db: DatabaseId, tb: TableName) -> Self {
		Self::IndexIds(ns, db, tb)
	}

	/// Returns the range of batch keys for this sequence domain.
	///
	/// This method was used in the old range-scan approach but is kept for
	/// debugging and potential recovery scenarios.
	#[allow(dead_code)]
	fn new_batch_range_keys(&self) -> Result<Range<Vec<u8>>> {
		match self {
			Self::UserName(ns, db, sq) => Prefix::new_ba_range(*ns, *db, sq),
			Self::FullTextDocIds(ibk) => ibk.new_ib_range(),
			Self::NameSpacesIds => NamespaceIdGeneratorBatchKey::range(),
			Self::DatabasesIds(ns) => DatabaseIdGeneratorBatchKey::range(*ns),
			Self::TablesIds(ns, db) => TableIdGeneratorBatchKey::range(*ns, *db),
			Self::IndexIds(ns, db, tb) => IndexIdGeneratorBatchKey::range(*ns, *db, tb),
		}
	}

	fn new_batch_key(&self, start: i64) -> Result<Vec<u8>> {
		match &self {
			Self::UserName(ns, db, sq) => Ba::new(*ns, *db, sq, start).encode_key(),
			Self::FullTextDocIds(ikb) => ikb.new_ib_key(start).encode_key(),
			Self::NameSpacesIds => NamespaceIdGeneratorBatchKey::new(start).encode_key(),
			Self::DatabasesIds(ns) => DatabaseIdGeneratorBatchKey::new(*ns, start).encode_key(),
			Self::TablesIds(ns, db) => TableIdGeneratorBatchKey::new(*ns, *db, start).encode_key(),
			Self::IndexIds(ns, db, tb) => {
				IndexIdGeneratorBatchKey::new(*ns, *db, tb, start).encode_key()
			}
		}
	}

	fn new_state_key(&self, nid: Uuid) -> Result<Vec<u8>> {
		match &self {
			Self::UserName(ns, db, sq) => St::new(*ns, *db, sq, nid).encode_key(),
			Self::FullTextDocIds(ikb) => ikb.new_is_key(nid).encode_key(),
			Self::NameSpacesIds => NamespaceIdGeneratorStateKey::new(nid).encode_key(),
			Self::DatabasesIds(ns) => DatabaseIdGeneratorStateKey::new(*ns, nid).encode_key(),
			Self::TablesIds(ns, db) => TableIdGeneratorStateKey::new(*ns, *db, nid).encode_key(),
			Self::IndexIds(ns, db, tb) => {
				IndexIdGeneratorStateKey::new(*ns, *db, tb, nid).encode_key()
			}
		}
	}

	/// Returns the global counter key for this sequence domain.
	///
	/// The global counter stores the next available batch start value,
	/// eliminating the need for range scans during batch allocation.
	/// This significantly reduces transaction conflicts in concurrent scenarios.
	fn new_global_counter_key(&self) -> Result<Vec<u8>> {
		match &self {
			Self::UserName(ns, db, sq) => Sg::new(*ns, *db, sq).encode_key(),
			Self::FullTextDocIds(ikb) => ikb.new_ig_key().encode_key(),
			Self::NameSpacesIds => NamespaceIdGeneratorGlobalKey::new().encode_key(),
			Self::DatabasesIds(ns) => DatabaseIdGeneratorGlobalKey::new(*ns).encode_key(),
			Self::TablesIds(ns, db) => TableIdGeneratorGlobalKey::new(*ns, *db).encode_key(),
			Self::IndexIds(ns, db, tb) => IndexIdGeneratorGlobalKey::new(*ns, *db, tb).encode_key(),
		}
	}
}

/// Represents a batch allocation of IDs in the key-value store.
///
/// A batch allocation reserves a range of IDs for a specific node (identified by `owner`).
/// The range is from some starting value (stored in the key) up to (but not including) `to`.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Serialize, Deserialize, Hash)]
pub(crate) struct BatchValue {
	/// The exclusive upper bound of the batch allocation
	to: i64,
	/// The UUID of the node that owns this batch allocation
	owner: Uuid,
}
impl_kv_value_revisioned!(BatchValue);

/// Tracks the next available ID for a specific node in a sequence.
///
/// Each node maintains its own `SequenceState` which tracks the next ID it will
/// allocate from its current batch. This state is persisted to coordinate with
/// batch allocations and ensure no ID is used twice.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Serialize, Deserialize, Hash)]
pub(crate) struct SequenceState {
	/// The next ID to be allocated by this node
	next: i64,
}
impl_kv_value_revisioned!(SequenceState);

impl Sequences {
	pub(super) fn new(tf: TransactionFactory, nid: Uuid) -> Self {
		Self {
			tf,
			sequences: Arc::new(Default::default()),
			nid,
		}
	}
	/// Cleans up all sequences associated with a removed namespace.
	///
	/// This method is called when a namespace is deleted to remove all cached
	/// sequence state for databases within that namespace.
	pub(crate) async fn namespace_removed(&self, tx: &Transaction, ns: NamespaceId) -> Result<()> {
		for db in tx.all_db(ns).await?.iter() {
			self.database_removed(tx, ns, db.database_id).await?;
		}
		Ok(())
	}

	/// Cleans up all sequences associated with a removed database.
	///
	/// This method is called when a database is deleted to remove all cached
	/// sequence state for user-defined sequences within that database.
	pub(crate) async fn database_removed(
		&self,
		tx: &Transaction,
		ns: NamespaceId,
		db: DatabaseId,
	) -> Result<()> {
		for sqs in tx.all_db_sequences(ns, db).await?.iter() {
			self.sequence_removed(ns, db, &sqs.name).await;
		}
		Ok(())
	}

	/// Removes a specific user-defined sequence from the cache.
	///
	/// This method is called when a sequence is deleted to clean up its cached state.
	pub(crate) async fn sequence_removed(&self, ns: NamespaceId, db: DatabaseId, sq: &str) {
		let key = SequenceDomain::new_user(ns, db, sq);
		self.sequences.write().await.remove(&key);
	}

	/// Core internal method for retrieving the next value from a sequence.
	///
	/// This method coordinates sequence loading, caching, and value generation.
	/// It ensures that only one Sequence instance exists per domain by checking
	/// the cache first, then loading if needed.
	///
	/// # Arguments
	/// * `ctx` - Optional mutable context for timeout checking
	/// * `seq` - The sequence domain to generate values from
	/// * `start` - The starting value if the sequence hasn't been initialized
	/// * `batch` - The batch size for ID allocations
	/// * `timeout` - Optional timeout for batch allocation operations
	///
	/// # Returns
	/// The next sequential value
	async fn next_val(
		&self,
		ctx: Option<&Context>,
		seq: Arc<SequenceDomain>,
		start: i64,
		batch: u32,
		timeout: Option<Duration>,
	) -> Result<i64> {
		let sequence = self.sequences.read().await.get(&seq).cloned();
		if let Some(s) = sequence {
			return s.lock().await.next(self, ctx, &seq, batch).await;
		}
		let s = match self.sequences.write().await.entry(seq.clone()) {
			Entry::Occupied(e) => e.get().clone(),
			Entry::Vacant(e) => {
				let s = Arc::new(Mutex::new(
					Sequence::load(ctx, self, &seq, start, batch, timeout).await?,
				));
				e.insert(s).clone()
			}
		};
		s.lock().await.next(self, ctx, &seq, batch).await
	}

	/// Generates the next namespace ID.
	///
	/// # Arguments
	/// * `ctx` - Optional mutable context for transaction operations
	///
	/// # Returns
	/// A new unique namespace ID
	pub(crate) async fn next_namespace_id(&self, ctx: Option<&Context>) -> Result<NamespaceId> {
		let domain = Arc::new(SequenceDomain::new_namespace_ids());
		let id = self.next_val(ctx, domain, 0, 100, None).await?;
		Ok(NamespaceId(id as u32))
	}

	/// Generates the next database ID within a namespace.
	///
	/// # Arguments
	/// * `ctx` - Optional mutable context for transaction operations
	/// * `ns` - The namespace ID to generate the database ID within
	///
	/// # Returns
	/// A new unique database ID for the given namespace
	pub(crate) async fn next_database_id(
		&self,
		ctx: Option<&Context>,
		ns: NamespaceId,
	) -> Result<DatabaseId> {
		let domain = Arc::new(SequenceDomain::new_database_ids(ns));
		let id = self.next_val(ctx, domain, 0, 100, None).await?;
		Ok(DatabaseId(id as u32))
	}

	/// Generates the next table ID within a database.
	///
	/// # Arguments
	/// * `ctx` - Optional mutable context for transaction operations
	/// * `ns` - The namespace ID
	/// * `db` - The database ID to generate the table ID within
	///
	/// # Returns
	/// A new unique table ID for the given database
	pub(crate) async fn next_table_id(
		&self,
		ctx: Option<&Context>,
		ns: NamespaceId,
		db: DatabaseId,
	) -> Result<TableId> {
		let domain = Arc::new(SequenceDomain::new_table_ids(ns, db));
		let id = self.next_val(ctx, domain, 0, 100, None).await?;
		Ok(TableId(id as u32))
	}

	/// Generates the next index ID within a table.
	///
	/// # Arguments
	/// * `ctx` - Optional mutable context for transaction operations
	/// * `ns` - The namespace ID
	/// * `db` - The database ID
	/// * `tb` - The table name to generate the index ID within
	///
	/// # Returns
	/// A new unique index ID for the given table
	pub(crate) async fn next_index_id(
		&self,
		ctx: Option<&Context>,
		ns: NamespaceId,
		db: DatabaseId,
		tb: TableName,
	) -> Result<IndexId> {
		let domain = Arc::new(SequenceDomain::new_index_ids(ns, db, tb));
		let id = self.next_val(ctx, domain, 0, 100, None).await?;
		Ok(IndexId(id as u32))
	}

	/// Generates the next value for a user-defined sequence.
	///
	/// # Arguments
	/// * `ctx` - Optional mutable context for transaction operations
	/// * `tx` - The transaction to use for accessing sequence configuration
	/// * `ns` - The namespace ID
	/// * `db` - The database ID
	/// * `sq` - The sequence name
	///
	/// # Returns
	/// The next value in the user-defined sequence
	pub(crate) async fn next_user_sequence_id(
		&self,
		ctx: Option<&Context>,
		tx: &Transaction,
		ns: NamespaceId,
		db: DatabaseId,
		sq: &str,
	) -> Result<i64> {
		let seq = tx.get_db_sequence(ns, db, sq).await?;
		let domain = Arc::new(SequenceDomain::new_user(ns, db, sq));
		self.next_val(ctx, domain, seq.start, seq.batch, seq.timeout).await
	}

	/// Generates the next document ID for a full-text search index.
	///
	/// # Arguments
	/// * `ctx` - Optional mutable context for transaction operations
	/// * `ikb` - The index key base identifying the full-text index
	/// * `batch` - The batch size for ID allocation
	///
	/// # Returns
	/// A new unique document ID for the full-text search index
	pub(crate) async fn next_fts_doc_id(
		&self,
		ctx: Option<&Context>,
		ikb: IndexKeyBase,
		batch: u32,
	) -> Result<DocId> {
		let domain = Arc::new(SequenceDomain::new_ft_doc_ids(ikb));
		let id = self.next_val(ctx, domain, 0, batch, None).await?;
		Ok(id as DocId)
	}
}

/// Internal per-node sequence state manager.
///
/// This struct manages the local state for a specific sequence on a specific node.
/// It tracks the current position within an allocated batch and coordinates with
/// the distributed batch allocation system when the current batch is exhausted.
struct Sequence {
	/// Transaction factory for creating transactions to persist state
	tf: TransactionFactory,
	/// The current state tracking the next ID to allocate
	st: SequenceState,
	/// Optional timeout for batch allocation operations
	timeout: Option<Duration>,
	/// The exclusive upper bound of the current batch allocation
	to: i64,
	/// The key used to persist this sequence's state
	state_key: Vec<u8>,
}

impl Sequence {
	/// Loads or initializes a sequence instance for the current node.
	///
	/// This method reads the persisted state for this sequence (if it exists) and
	/// allocates an initial batch of IDs. If no state exists, it starts from the
	/// provided `start` value.
	///
	/// # Arguments
	/// * `ctx` - Optional mutable context for timeout checking
	/// * `sqs` - The sequences manager
	/// * `seq` - The sequence domain identifying which sequence to load
	/// * `start` - The starting value if no state exists
	/// * `batch` - The batch size for ID allocations
	/// * `timeout` - Optional timeout for batch allocation operations
	async fn load(
		ctx: Option<&Context>,
		sqs: &Sequences,
		seq: &SequenceDomain,
		start: i64,
		batch: u32,
		timeout: Option<Duration>,
	) -> Result<Self> {
		let state_key = seq.new_state_key(sqs.nid)?;
		// Create a separate transaction for reading sequence state to avoid conflicts
		// with the parent transaction in strict serialization mode (e.g., FDB)
		let tx =
			sqs.tf.transaction(TransactionType::Read, LockType::Optimistic, sqs.clone()).await?;
		let mut st: SequenceState = if let Some(v) = tx.get(&state_key, None).await? {
			revision::from_slice(&v)?
		} else {
			SequenceState {
				next: start,
			}
		};
		tx.cancel().await?;
		let (from, to) =
			Self::find_batch_allocation(sqs, ctx, seq, st.next, batch, timeout).await?;
		st.next = from;
		Ok(Self {
			tf: sqs.tf.clone(),
			state_key,
			to,
			st,
			timeout,
		})
	}

	/// Gets the next ID from this sequence.
	///
	/// If the current batch is exhausted, this method will allocate a new batch
	/// before returning the next ID. The state is persisted to the key-value store
	/// after each allocation.
	///
	/// # Arguments
	/// * `sqs` - The sequences manager
	/// * `ctx` - Optional mutable context for timeout checking
	/// * `seq` - The sequence domain
	/// * `batch` - The batch size for new allocations if needed
	async fn next(
		&mut self,
		sqs: &Sequences,
		ctx: Option<&Context>,
		seq: &SequenceDomain,
		batch: u32,
	) -> Result<i64> {
		if self.st.next >= self.to {
			(self.st.next, self.to) =
				Self::find_batch_allocation(sqs, ctx, seq, self.st.next, batch, self.timeout)
					.await?;
		}
		let v = self.st.next;
		self.st.next += 1;
		// write the state on the KV store
		let tx =
			self.tf.transaction(TransactionType::Write, LockType::Optimistic, sqs.clone()).await?;

		// Execute operations and ensure transaction is cancelled on error
		match tx.set(&self.state_key, &revision::to_vec(&self.st)?, None).await {
			Ok(_) => {
				tx.commit().await?;
				Ok(v)
			}
			Err(e) => {
				tx.cancel().await?;
				Err(e)
			}
		}
	}

	/// Finds and allocates a batch of IDs with retry logic and exponential backoff.
	///
	/// This method repeatedly attempts to allocate a batch until successful or until
	/// a timeout is reached. It uses exponential backoff with jitter to reduce
	/// contention when multiple nodes are competing for batch allocations.
	///
	/// # Arguments
	/// * `sqs` - The sequences manager
	/// * `ctx` - Optional mutable context for timeout checking
	/// * `seq` - The sequence domain
	/// * `next` - The next ID that needs to be allocated
	/// * `batch` - The batch size to allocate
	/// * `to` - Optional timeout duration for the entire operation
	///
	/// # Returns
	/// A tuple of (start, end) representing the allocated batch range [start, end)
	async fn find_batch_allocation(
		sqs: &Sequences,
		ctx: Option<&Context>,
		seq: &SequenceDomain,
		next: i64,
		batch: u32,
		to: Option<Duration>,
	) -> Result<(i64, i64)> {
		// Use for exponential backoff
		let mut tempo = 4;
		const MAX_BACKOFF: u64 = 32_768;
		let start = if to.is_some() {
			Some(Instant::now())
		} else {
			None
		};
		// Loop until we have a successful allocation.
		// We check the timeout inherited from the context
		loop {
			if let Some(ctx) = ctx {
				ctx.expect_not_timedout().await?;
			} else {
				yield_now!();
			}
			if let (Some(ref start), Some(ref to)) = (start, to) {
				// We check the time associated with the sequence
				if start.elapsed().ge(to) {
					let timeout = (*to).into();
					return Err(anyhow::Error::new(Error::QueryTimedout(timeout)));
				}
			}
			match Self::check_batch_allocation(sqs, seq, next, batch).await {
				Ok(r) => return Ok(r),
				Err(_) => {
					// Increment conflict counter for test verification
					#[cfg(test)]
					{
						TEST_CONFLICT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
					}
				}
			}
			// exponential backoff with full jitter
			let sleep_ms = thread_rng().gen_range(1..=tempo);
			sleep(Duration::from_millis(sleep_ms)).await;
			if tempo < MAX_BACKOFF {
				tempo *= 2;
			}
		}
	}

	/// Attempts to allocate a batch of IDs in a single transaction.
	///
	/// This method uses a global counter to determine the next available batch start,
	/// avoiding range scans that caused transaction conflicts in concurrent scenarios.
	/// The global counter approach significantly reduces the read-set size and
	/// prevents conflicts when multiple nodes allocate batches simultaneously.
	///
	/// # Arguments
	/// * `sqs` - The sequences manager
	/// * `seq` - The sequence domain
	/// * `next` - The next ID that needs to be allocated
	/// * `batch` - The batch size to allocate
	///
	/// # Returns
	/// A tuple of (start, end) representing the allocated batch range [start, end)
	async fn check_batch_allocation(
		sqs: &Sequences,
		seq: &SequenceDomain,
		next: i64,
		batch: u32,
	) -> Result<(i64, i64)> {
		let tx =
			sqs.tf.transaction(TransactionType::Write, LockType::Optimistic, sqs.clone()).await?;

		// Execute operations and ensure transaction is cancelled on error
		let result = async {
			// Read the global counter to find the next available batch start.
			// This replaces the range scan of all batch keys, dramatically reducing
			// the transaction's read-set and preventing conflicts.
			let global_key = seq.new_global_counter_key()?;
			let global_counter: i64 = match tx.get(&global_key, None).await? {
				Some(bytes) if bytes.len() == 8 => {
					i64::from_be_bytes(bytes.try_into().expect("length checked"))
				}
				_ => 0,
			};

			// Compute the new batch start as the maximum of the global counter
			// and the requested next value
			let next_start = next.max(global_counter);
			let next_to = next_start + batch as i64;

			// Update the global counter to reflect the new allocation
			let next_to_bytes = next_to.to_be_bytes().to_vec();
			tx.set(&global_key, &next_to_bytes, None).await?;

			// Store the batch allocation for this node (for visibility and recovery)
			let bv = revision::to_vec(&BatchValue {
				to: next_to,
				owner: sqs.nid,
			})?;
			let batch_key = seq.new_batch_key(next_start)?;
			tx.set(&batch_key, &bv, None).await?;

			Ok::<(i64, i64), anyhow::Error>((next_start, next_to))
		}
		.await;

		match result {
			Ok(res) => {
				tx.commit().await?;
				Ok(res)
			}
			Err(e) => {
				tx.cancel().await?;
				Err(e)
			}
		}
	}
}

/// Test-only counter for tracking allocation conflicts.
/// This is used to verify that the global counter approach reduces conflicts.
#[cfg(test)]
pub(crate) static TEST_CONFLICT_COUNT: std::sync::atomic::AtomicU64 =
	std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
mod tests {
	use std::collections::HashSet;
	use std::sync::Arc;
	use std::sync::atomic::Ordering;

	use tokio::sync::Barrier;
	use uuid::Uuid;

	use super::{Sequences, TEST_CONFLICT_COUNT};
	use crate::dbs::node::Timestamp;
	use crate::idx::IndexKeyBase;
	use crate::catalog::{DatabaseId, IndexId, NamespaceId};
	use crate::kvs::clock::{FakeClock, SizedClock};
	use crate::kvs::ds::{TransactionFactory, DatastoreFlavor};
	use crate::kvs::Datastore;
	use crate::val::TableName;

	/// Performance test: measures how allocation time scales with existing keys.
	///
	/// With range-scan: O(n) - time grows as batch keys accumulate
	/// With global counter: O(1) - time stays constant
	///
	/// This test compares allocation time with FEW keys vs MANY keys.
	/// The ratio should be close to 1.0 with global counter, but high with range-scan.
	#[cfg(feature = "kv-rocksdb")]
	#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
	async fn test_batch_allocation_performance_rocksdb() {
		use std::time::Instant;
		use temp_dir::TempDir;

		const WARMUP_KEYS: usize = 10;
		const MEASURE_ALLOCATIONS: usize = 50;
		const KEYS_TO_ADD: usize = 1000;
		const BATCH_SIZE: u32 = 1;

		// Create RocksDB-backed storage
		let temp_dir = TempDir::new().unwrap();
		let path = format!("rocksdb:{}", temp_dir.path().to_string_lossy());
		let clock = Arc::new(SizedClock::Fake(FakeClock::new(Timestamp::default())));
		let flavor = crate::kvs::rocksdb::Datastore::new(&path)
			.await
			.map(DatastoreFlavor::RocksDB)
			.unwrap();
		let tf = TransactionFactory::new(clock, Box::new(flavor));

		let ikb = IndexKeyBase::new(
			NamespaceId(1),
			DatabaseId(1),
			TableName::from("perf_test_table"),
			IndexId(1),
		);

		let sequences = Sequences::new(tf.clone(), Uuid::new_v4());

		println!("\n=== Performance Scaling Test ===");

		// Phase 1: Warmup with few keys
		for _ in 0..WARMUP_KEYS {
			sequences
				.next_fts_doc_id(None, ikb.clone(), BATCH_SIZE)
				.await
				.expect("warmup should succeed");
		}

		// Phase 2: Measure with FEW existing keys
		let start_few = Instant::now();
		for _ in 0..MEASURE_ALLOCATIONS {
			sequences
				.next_fts_doc_id(None, ikb.clone(), BATCH_SIZE)
				.await
				.expect("allocation should succeed");
		}
		let time_few = start_few.elapsed();
		let avg_few_us = time_few.as_micros() as f64 / MEASURE_ALLOCATIONS as f64;

		println!("With ~{} existing keys: {:.1} µs/allocation", WARMUP_KEYS + MEASURE_ALLOCATIONS, avg_few_us);

		// Phase 3: Add many more keys
		for _ in 0..KEYS_TO_ADD {
			sequences
				.next_fts_doc_id(None, ikb.clone(), BATCH_SIZE)
				.await
				.expect("populate should succeed");
		}

		// Phase 4: Measure with MANY existing keys
		let total_keys = WARMUP_KEYS + MEASURE_ALLOCATIONS + KEYS_TO_ADD;
		let start_many = Instant::now();
		for _ in 0..MEASURE_ALLOCATIONS {
			sequences
				.next_fts_doc_id(None, ikb.clone(), BATCH_SIZE)
				.await
				.expect("allocation should succeed");
		}
		let time_many = start_many.elapsed();
		let avg_many_us = time_many.as_micros() as f64 / MEASURE_ALLOCATIONS as f64;

		println!("With ~{} existing keys: {:.1} µs/allocation", total_keys, avg_many_us);

		let slowdown_ratio = avg_many_us / avg_few_us;
		println!("Slowdown ratio: {:.2}x", slowdown_ratio);
		println!("================================\n");

		// With global counter (O(1)), ratio should be close to 1.0 (< 3.0)
		// With range-scan (O(n)), ratio would be much higher (> 5.0)
		// because scanning 1000+ keys takes much longer than scanning 10 keys
		assert!(
			slowdown_ratio < 3.0,
			"Performance degradation too high: {:.2}x slowdown. \
			 With global counter approach (O(1)), we expect < 3.0x slowdown. \
			 With range-scan (O(n)), you would see > 5x slowdown as keys accumulate.",
			slowdown_ratio
		);
	}

	/// Test with memory backend - verifies uniqueness of allocated IDs.
	/// Note: Memory backend may not exhibit the same conflict behavior as RocksDB.
	#[tokio::test]
	async fn test_multi_node_parallel_batch_allocation_memory() {
		const NUM_CONCURRENT_TASKS: usize = 50;
		const BATCH_SIZE: u32 = 1;

		TEST_CONFLICT_COUNT.store(0, Ordering::SeqCst);

		let clock = Arc::new(SizedClock::Fake(FakeClock::new(Timestamp::default())));
		let flavor = crate::kvs::mem::Datastore::new().await.map(DatastoreFlavor::Mem).unwrap();
		let tf = TransactionFactory::new(clock, Box::new(flavor));

		let ikb = IndexKeyBase::new(
			NamespaceId(1),
			DatabaseId(1),
			TableName::from("shared_table"),
			IndexId(1),
		);

		let barrier = Arc::new(Barrier::new(NUM_CONCURRENT_TASKS));

		let handles: Vec<_> = (0..NUM_CONCURRENT_TASKS)
			.map(|task_id| {
				let tf = tf.clone();
				let barrier = barrier.clone();
				let ikb = ikb.clone();

				tokio::spawn(async move {
					let sequences = Sequences::new(tf, Uuid::new_v4());
					barrier.wait().await;

					let id = sequences
						.next_fts_doc_id(None, ikb, BATCH_SIZE)
						.await
						.expect("allocation should succeed");

					(task_id, id)
				})
			})
			.collect();

		let mut all_ids = HashSet::new();
		for handle in handles {
			let (task_id, id) = handle.await.unwrap();
			assert!(all_ids.insert(id), "Task {task_id} got duplicate ID {id}!");
		}

		assert_eq!(all_ids.len(), NUM_CONCURRENT_TASKS);

		let conflicts = TEST_CONFLICT_COUNT.load(Ordering::SeqCst);
		println!(
			"\n=== Memory Multi-node test ===\n\
			 Concurrent tasks: {NUM_CONCURRENT_TASKS}\n\
			 Conflicts: {conflicts}\n\
			 ==============================="
		);
	}

	/// Test that verifies conflict detection is actually working.
	/// If this test passes, it means our test infrastructure can detect conflicts.
	#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
	async fn test_verify_conflict_detection_works() {
		use crate::kvs::{LockType, TransactionType};

		let ds = Datastore::new("memory").await.unwrap();
		const KEY: &[u8] = b"/test/conflict/key";

		// Initialize key
		{
			let tx = ds.transaction(TransactionType::Write, LockType::Optimistic).await.unwrap();
			tx.set(&KEY.to_vec(), &b"initial".to_vec(), None).await.unwrap();
			tx.commit().await.unwrap();
		}

		// Create 5 concurrent transactions writing to the same key
		let tx1 = ds.transaction(TransactionType::Write, LockType::Optimistic).await.unwrap();
		let tx2 = ds.transaction(TransactionType::Write, LockType::Optimistic).await.unwrap();
		let tx3 = ds.transaction(TransactionType::Write, LockType::Optimistic).await.unwrap();
		let tx4 = ds.transaction(TransactionType::Write, LockType::Optimistic).await.unwrap();
		let tx5 = ds.transaction(TransactionType::Write, LockType::Optimistic).await.unwrap();

		// All read the key
		let _ = tx1.get(&KEY.to_vec(), None).await.unwrap();
		let _ = tx2.get(&KEY.to_vec(), None).await.unwrap();
		let _ = tx3.get(&KEY.to_vec(), None).await.unwrap();
		let _ = tx4.get(&KEY.to_vec(), None).await.unwrap();
		let _ = tx5.get(&KEY.to_vec(), None).await.unwrap();

		// All write to the key
		tx1.set(&KEY.to_vec(), &b"tx1".to_vec(), None).await.unwrap();
		tx2.set(&KEY.to_vec(), &b"tx2".to_vec(), None).await.unwrap();
		tx3.set(&KEY.to_vec(), &b"tx3".to_vec(), None).await.unwrap();
		tx4.set(&KEY.to_vec(), &b"tx4".to_vec(), None).await.unwrap();
		tx5.set(&KEY.to_vec(), &b"tx5".to_vec(), None).await.unwrap();

		// First commits successfully
		tx1.commit().await.expect("tx1 should succeed");

		// Others should conflict
		let mut conflicts = 0;
		if tx2.commit().await.is_err() { conflicts += 1; }
		if tx3.commit().await.is_err() { conflicts += 1; }
		if tx4.commit().await.is_err() { conflicts += 1; }
		if tx5.commit().await.is_err() { conflicts += 1; }

		println!("Conflict detection test: {conflicts}/4 transactions conflicted");
		assert!(conflicts >= 3, "Conflict detection should work - got only {conflicts} conflicts");
	}
}

