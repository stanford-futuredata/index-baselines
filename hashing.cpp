#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <random>
#include <vector>

using namespace std;

const uint32_t INVALID_KEY = 0xffffffff;     // An unusable key we'll treat as a sentinel value
const uint32_t BUCKET_SIZE = 8;              // Bucket size for cuckoo hash
const double LOAD_FACTOR = 0.99;             // Load factor used for hash tables
const uint32_t NUM_KEYS = 20 * 1000 * 1000;  // How many keys to generate if not loading a file
const uint32_t BUF_SIZE = 8192;              // For reading data

// Finalization step of Murmur3 hash
uint32_t hash32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x85ebca6b;
  value ^= value >> 13;
  value *= 0xc2b2ae35;
  value ^= value >> 16;
  return value;
}

// Fast alternative to modulo from Daniel Lemire
uint32_t alt_mod(uint32_t x, uint32_t n) {
  return ((uint64_t) x * (uint64_t) n) >> 32 ;
}

// A bucketed cuckoo hash map with keys of type uint32_t and values of type V
template <typename V>
class CuckooHashMap {
public:
  struct SearchResult {
    bool found;
    V value;
  };

private:
  struct Bucket {
    uint32_t keys[BUCKET_SIZE] __attribute__((aligned(32)));
    V values[BUCKET_SIZE];
  };

  Bucket *buckets_;
  uint32_t num_buckets_;  // Total number of buckets
  uint32_t size_;         // Number of entries filled
  mt19937 rand_;          // RNG for moving items around
  V uninitialized_value_;

public:
  CuckooHashMap(uint32_t capacity): size_(0) {
    num_buckets_ = (capacity + BUCKET_SIZE - 1) / BUCKET_SIZE;
    int r = posix_memalign((void **) &buckets_, 32, num_buckets_ * sizeof(Bucket));
    assert(r == 0);
    for (uint32_t i = 0; i < num_buckets_; i++) {
      for (size_t j = 0; j < BUCKET_SIZE; j++) {
        buckets_[i].keys[j] = INVALID_KEY;
      }
    }
  }

  ~CuckooHashMap() {
    free(buckets_);
  }

  SearchResult get(uint32_t key) {
    uint32_t hash = hash32(key);
    uint32_t i1 = alt_mod(hash, num_buckets_);
    Bucket *b1 = &buckets_[i1];

    __m256i vkey = _mm256_set1_epi32(key);
    __m256i vbucket = _mm256_load_si256((const __m256i *) &b1->keys);
    __m256i cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    int mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      return { true, b1->values[index] };
    }

    uint32_t i2 = alt_mod(hash32(key ^ hash), num_buckets_);
    if (i2 == i1) {
      i2 = (i1 == num_buckets_ - 1) ? 0 : i1 + 1;
    }
    Bucket *b2 = &buckets_[i2];

    vbucket = _mm256_load_si256((const __m256i *) &b2->keys);
    cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      return { true, b2->values[index] };
    }

    return { false, uninitialized_value_ };
  }

  void insert(uint32_t key, V value) {
    insert(key, value, false);
  }

  uint32_t size() {
    return size_;
  }

private:
  // Insert a key into the table if it's not already inside it;
  // if this is a re-insert, we won't increase the size_ field.
  void insert(uint32_t key, V value, bool is_reinsert) {
    uint32_t hash = hash32(key);
    uint32_t i1 = alt_mod(hash, num_buckets_);
    uint32_t i2 = alt_mod(hash32(key ^ hash), num_buckets_);
    if (i2 == i1) {
      i2 = (i1 == num_buckets_ - 1) ? 0 : i1 + 1;
    }

    Bucket *b1 = &buckets_[i1];
    Bucket *b2 = &buckets_[i2];

    // Update old value if the key is already in the table
    __m256i vkey = _mm256_set1_epi32(key);
    __m256i vbucket = _mm256_load_si256((const __m256i *) &b1->keys);
    __m256i cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    int mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      b1->values[index] = value;
      return;
    }

    vbucket = _mm256_load_si256((const __m256i *) &b2->keys);
    cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      b2->values[index] = value;
      return;
    }

    if (!is_reinsert) {
      size_++;
    }

    size_t count1 = 0;
    for (size_t i = 0; i < BUCKET_SIZE; i++) {
      count1 += (b1->keys[i] != INVALID_KEY ? 1 : 0);
    }
    size_t count2 = 0;
    for (size_t i = 0; i < BUCKET_SIZE; i++) {
      count2 += (b2->keys[i] != INVALID_KEY ? 1 : 0);
    }

    if (count1 <= count2 && count1 < BUCKET_SIZE) {
      // Add it into bucket 1
      b1->keys[count1] = key;
      b1->values[count1] = value;
    } else if (count2 < BUCKET_SIZE) {
      // Add it into bucket 2
      b2->keys[count2] = key;
      b2->values[count2] = value;
    } else {
      // Both buckets are full; evict a random item from one of them
      assert(count1 == BUCKET_SIZE);
      assert(count2 == BUCKET_SIZE);

      Bucket *victim_bucket = b1;
      if (rand_() % 2 == 0) {
        victim_bucket = b2;
      }
      uint32_t victim_index = rand_() % BUCKET_SIZE;
      uint32_t old_key = victim_bucket->keys[victim_index];
      V old_value = victim_bucket->values[victim_index];
      victim_bucket->keys[victim_index] = key;
      victim_bucket->values[victim_index] = value;
      insert(old_key, old_value, true);
    }
  }
};

// A linked hash map with keys of type uint32_t and values of type V
template <typename V>
class LinkedHashMap {
public:
  struct SearchResult {
    bool found;
    V value;
  };

private:
  struct Cell {
    uint32_t key;
    V value;
    Cell *next;
  };

  uint32_t num_cells_;
  Cell *cells_;
  V uninitialized_value_;

public:
  LinkedHashMap(uint32_t num_cells): num_cells_(num_cells) {
    cells_ = new Cell[num_cells];
    for (size_t i = 0; i < num_cells; i++) {
      cells_[i].key = INVALID_KEY;
      cells_[i].next = 0;
    }
  }

  ~LinkedHashMap() {
    delete[] cells_;
  }

  SearchResult get(uint32_t key) {
    uint32_t i = hash32(key) % num_cells_;
    //uint32_t i = alt_mod(hash32(key), num_cells_);
    Cell *cell = &cells_[i];
    while (true) {
      if (cell->key == key) {
        return { true, cell->value };
      } else if (!cell->next) {
        return { false, uninitialized_value_ };
      } else {
        cell = cell->next;
      }
    }
  }

  void insert(uint32_t key, V value) {
    uint32_t i = hash32(key) % num_cells_;
    //uint32_t i = alt_mod(hash32(key), num_cells_);
    Cell *cell = &cells_[i];
    if (cell->key == INVALID_KEY) {
      cell->key = key;
      cell->value = value;
      return;
    } else {
      while (cell->next) {
        cell = cell->next;
      }
      Cell* new_cell = new Cell;
      cell->next = new_cell;
      new_cell->key = key;
      new_cell->value = value;
      new_cell->next = 0;
    }
  }

  uint32_t filled() {
    uint32_t res = 0;
    for (size_t i = 0; i < num_cells_; i++) {
      res += (cells_[i].key != INVALID_KEY ? 1 : 0);
    }
    return res;
  }
};

vector<uint32_t> read_keys(const char *path) {
  vector<uint32_t> vec;
  FILE *fin = fopen(path, "rb");
  uint32_t buf[BUF_SIZE];
  while (true) {
    size_t num_read = fread(buf, sizeof(uint32_t), BUF_SIZE, fin);
    for (size_t i = 0; i < num_read; i++) {
      vec.push_back(buf[i]);
    }
    if (num_read < BUF_SIZE) break;
  }
  fclose(fin);
  printf("Loaded %lu keys\n", vec.size());
  return vec;
}

vector<uint32_t> generate_keys() {
  printf("Generating %u keys\n", NUM_KEYS);
  vector<uint32_t> keys(NUM_KEYS);
  for (uint32_t i = 1; i <= NUM_KEYS; i++) {
    keys[i] = i;
  }
  return keys;
}

struct NoValue {};

void benchmark_cuckoo_no_value(vector<uint32_t>& keys) {
  printf("Benchmarking CuckooHashMap without values\n");

  CuckooHashMap<NoValue> set(uint32_t(keys.size() / LOAD_FACTOR));
  for (uint32_t i = 0; i < keys.size(); i++) {
    set.insert(keys[i], {});
  }
  printf("Done inserting keys, set size = %u\n", set.size());

  // Reshuffle the keys so we don't query them in insert order
  random_shuffle(keys.begin(), keys.end());

  auto start = clock();
  uint32_t hits = 0;
  for (uint32_t i = 0; i < keys.size(); i++) {
    hits += set.get(keys[i]).found ? 1 : 0;
  }
  auto end = clock();
  printf("CuckooHashMap average time taken: %lf ns, hits: %u\n",
    double(end - start) / CLOCKS_PER_SEC / keys.size() * 1e9, hits);
}

void benchmark_cuckoo_32(vector<uint32_t>& keys) {
  printf("Benchmarking CuckooHashMap with 32-bit values\n");

  CuckooHashMap<uint32_t> set(uint32_t(keys.size() / LOAD_FACTOR));
  for (uint32_t i = 0; i < keys.size(); i++) {
    set.insert(keys[i], keys[i]);
  }
  printf("Done inserting keys, set size = %u\n", set.size());

  // Reshuffle the keys so we don't query them in insert order
  random_shuffle(keys.begin(), keys.end());

  auto start = clock();
  uint32_t hits = 0;
  for (uint32_t i = 0; i < keys.size(); i++) {
    hits += (set.get(keys[i]).value == keys[i]) ? 1 : 0;
  }
  auto end = clock();
  printf("CuckooHashMap average time taken: %lf ns, hits: %u\n",
    double(end - start) / CLOCKS_PER_SEC / keys.size() * 1e9, hits);
}

void benchmark_linked_32(vector<uint32_t>& keys) {
  printf("Benchmarking LinkedHashMap with 32-bit values\n");

  LinkedHashMap<uint32_t> set(uint32_t(keys.size()));
  for (uint32_t i = 0; i < keys.size(); i++) {
    set.insert(keys[i], keys[i]);
  }
  printf("Done inserting keys, filled = %u\n", set.filled());

  // Reshuffle the keys so we don't query them in insert order
  random_shuffle(keys.begin(), keys.end());

  auto start = clock();
  uint32_t hits = 0;
  for (uint32_t i = 0; i < keys.size(); i++) {
    hits += (set.get(keys[i]).value == keys[i]) ? 1 : 0;
  }
  auto end = clock();
  printf("LinkedHashMap average time taken: %lf ns, hits: %u\n",
    double(end - start) / CLOCKS_PER_SEC / keys.size() * 1e9, hits);
}

int main(int argc, char **argv) {
  vector<uint32_t> keys = (argc == 2 ? read_keys(argv[1]) : generate_keys());
  benchmark_linked_32(keys);
  benchmark_cuckoo_32(keys);
  benchmark_cuckoo_no_value(keys);
}
