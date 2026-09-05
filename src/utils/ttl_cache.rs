//! Bounded time-to-live cache shared by the Telegraph/Twitter extractors,
//! the Telegram file-URL lookup and web search.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

struct Entry<V> {
    value: V,
    stored_at: Instant,
    /// Monotonic insertion counter; eviction drops the lowest sequence
    /// first so capacity pruning is deterministic even when two inserts
    /// land on the same `Instant`.
    sequence: u64,
}

/// In-memory cache whose entries expire `ttl` after insertion and which
/// never holds more than `max_entries` live entries (oldest insertions are
/// evicted first). A zero `ttl` disables caching entirely: inserts are
/// dropped and lookups always miss.
///
/// Expired entries are pruned lazily on every access, so the cache needs no
/// background task. Callers wrap it in a `Mutex`; all operations are cheap
/// and never await.
pub struct TtlCache<K, V> {
    ttl: Duration,
    max_entries: usize,
    next_sequence: u64,
    entries: HashMap<K, Entry<V>>,
}

impl<K, V> TtlCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            ttl,
            max_entries,
            next_sequence: 0,
            entries: HashMap::new(),
        }
    }

    /// `false` when the TTL is zero and the cache stores nothing.
    pub fn is_enabled(&self) -> bool {
        !self.ttl.is_zero()
    }

    /// Return a clone of the fresh value stored under `key`, if any.
    pub fn get<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.prune_expired();
        self.entries.get(key).map(|entry| entry.value.clone())
    }

    /// Store `value` under `key`, replacing any previous entry and making
    /// this the newest entry for eviction purposes.
    pub fn insert(&mut self, key: K, value: V) {
        if !self.is_enabled() {
            return;
        }
        self.prune_expired();
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        self.entries.insert(
            key,
            Entry {
                value,
                stored_at: Instant::now(),
                sequence,
            },
        );
        self.evict_oldest_beyond_capacity();
    }

    fn prune_expired(&mut self) {
        let ttl = self.ttl;
        self.entries
            .retain(|_, entry| entry.stored_at.elapsed() < ttl);
    }

    fn evict_oldest_beyond_capacity(&mut self) {
        if self.entries.len() <= self.max_entries {
            return;
        }
        let mut ordered = self
            .entries
            .iter()
            .map(|(key, entry)| (key.clone(), entry.sequence))
            .collect::<Vec<_>>();
        ordered.sort_by_key(|(_, sequence)| *sequence);
        let remove_count = self.entries.len() - self.max_entries;
        for (key, _) in ordered.into_iter().take(remove_count) {
            self.entries.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_fresh_values_and_drops_expired_ones() {
        let mut fresh = TtlCache::new(Duration::from_secs(3600), 8);
        fresh.insert("a".to_string(), 1);
        assert_eq!(fresh.get("a"), Some(1));

        let mut short = TtlCache::new(Duration::from_millis(1), 8);
        short.insert("a".to_string(), 1);
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(short.get("a"), None);
    }

    #[test]
    fn evicts_the_oldest_entries_beyond_capacity() {
        let mut cache = TtlCache::new(Duration::from_secs(3600), 2);
        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        assert_eq!(cache.get("a"), None);
        assert_eq!(cache.get("b"), Some(2));
        assert_eq!(cache.get("c"), Some(3));
    }

    #[test]
    fn re_inserting_a_key_makes_it_the_newest_entry() {
        let mut cache = TtlCache::new(Duration::from_secs(3600), 2);
        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("a".to_string(), 10);
        cache.insert("c".to_string(), 3);

        assert_eq!(cache.get("b"), None);
        assert_eq!(cache.get("a"), Some(10));
        assert_eq!(cache.get("c"), Some(3));
    }

    #[test]
    fn zero_ttl_disables_caching() {
        let mut cache = TtlCache::new(Duration::ZERO, 8);
        assert!(!cache.is_enabled());
        cache.insert("a".to_string(), 1);
        assert_eq!(cache.get("a"), None);
    }
}
