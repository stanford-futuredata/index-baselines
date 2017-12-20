//
// Linear and binary search code derived from:
// https://dirtyhandscoding.wordpress.com/2017/08/25/performance-comparison-linear-search-vs-binary-search/
// https://www.pvk.ca/Blog/2012/07/03/binary-search-star-eliminates-star-branch-mispredictions/
// https://github.com/stgatilov/linear-vs-binary-search/blob/master/search.cpp
//

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <numeric>
#include <random>
#include <immintrin.h>
#include <stx/btree_set.h>

#define BUF_SIZE 2048
#define QUERIES_PER_TRIAL (50 * 1000 * 1000)

#define SHUF(i0, i1, i2, i3) (i0 + i1*4 + i2*16 + i3*64)
#define FORCEINLINE __attribute__((always_inline)) inline

// btree configuration
class btree_traits : public stx::btree_default_set_traits<int> {
 public:
  static const int leafslots = 64;
  static const int innerslots = 64;
  static const size_t binsearch_threshold = 256;
};

// power of 2 at most x, undefined for x == 0
FORCEINLINE uint32_t bsr(uint32_t x) {
  return 31 - __builtin_clz(x);
}

static int binary_search_std (const int *arr, int n, int key) {
    return (int) (std::lower_bound(arr, arr + n, key) - arr);
}

static int binary_search_simple(const int *arr, int n, int key) {
  intptr_t left = -1;
  intptr_t right = n;
  while (right - left > 1) {
    intptr_t middle = (left + right) >> 1;
    if (arr[middle] < key)
      left = middle;
    else
      right = middle;
  }
  return (int) right;
}

static int binary_search_branchless(const int *arr, int n, int key) {
  intptr_t pos = -1;
  intptr_t logstep = bsr(n - 1);
  intptr_t step = intptr_t(1) << logstep;

  pos = (arr[pos + n - step] < key ? pos + n - step : pos);
  step >>= 1;

  while (step > 0) {
    pos = (arr[pos + step] < key ? pos + step : pos);
    step >>= 1;
  }
  pos += 1;

  return (int) (arr[pos] >= key ? pos : n);
}

static uint32_t interpolation_search( const int32_t* data, uint32_t n, int32_t key ) {
  uint32_t low = 0;
  uint32_t high = n-1;
  uint32_t mid;

  if ( key <= data[low] ) return low; /* in the first page */

  uint32_t iters = 0;
  while ( data[high] > data[low] and
          data[high] > key and
          data[low] < key ) {
    iters += 1;
    if ( iters > 1 ) return binary_search_branchless( data + low, high-low, key );
    
    mid = low + (((long)key - (long)data[low]) * (double)(high - low) / ((long)data[high] - (long)data[low]));

    if ( data[mid] < key ) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  if ( key <= data[low] ) return low;
  if ( key <= data[high] ) return high;
  return high + 1;
}

static int linear_search(const int *arr, int n, int key) {
  intptr_t i = 0;
  while (i < n) {
    if (arr[i] >= key)
      break;
    ++i;
  }
  return i;
}

static int linear_search_avx (const int *arr, int n, int key) {
  __m256i vkey = _mm256_set1_epi32(key);
  __m256i cnt = _mm256_setzero_si256();
  for (int i = 0; i < n; i += 16) {
    __m256i mask0 = _mm256_cmpgt_epi32(vkey, _mm256_load_si256((__m256i *)&arr[i+0]));
    __m256i mask1 = _mm256_cmpgt_epi32(vkey, _mm256_load_si256((__m256i *)&arr[i+8]));
    __m256i sum = _mm256_add_epi32(mask0, mask1);
    cnt = _mm256_sub_epi32(cnt, sum);
  }
  __m128i xcnt = _mm_add_epi32(_mm256_extracti128_si256(cnt, 1), _mm256_castsi256_si128(cnt));
  xcnt = _mm_add_epi32(xcnt, _mm_shuffle_epi32(xcnt, SHUF(2, 3, 0, 1)));
  xcnt = _mm_add_epi32(xcnt, _mm_shuffle_epi32(xcnt, SHUF(1, 0, 3, 2)));
  return _mm_cvtsi128_si32(xcnt);
}

class OneLevelIndex {
  const int* data_;
  int k_, stride_;
  int* table_;

 public:
  explicit OneLevelIndex(const std::vector<int>& data, int k)
      : data_(&data[0]), k_(k) {
    table_ = (int*) calloc((size_t) k_, sizeof(int));
    stride_ = (int) data.size() / k;

    // populate index
    for (int i = 0; i < k_; i++) {
      table_[i] = data[(i + 1) * stride_  - 1];
    }
  }

  ~OneLevelIndex() {
    free(table_);
  }

  int find(int key) {
    int i = binary_search_branchless(table_, k_, key);
    int pos = i * stride_;
    int offset = binary_search_branchless(data_ + pos, stride_, key);
    return pos + offset;
  }
};


class TwoLevelIndex {
  const int* data_;
  int k1_, k2_, k2stride_, page_size_;
  int* tables_;

 public:
  explicit TwoLevelIndex(std::vector<int>& data, int k2, int page_size)
      : k2_(k2), page_size_(page_size) {

    int hang = data.size() % page_size;
    if (hang > 0) {
      for (int i = 0; i < page_size - hang; i++) {
        data.push_back(INT_MAX);
      }
    }

    data_ = &data[0];

    k1_ = (int) std::ceil(data.size() / double(k2_ * page_size_));
    std::cerr << k1_ << std::endl;
    tables_ = (int*) calloc((size_t) k1_ * (k2 + 1), sizeof(int));
    k2stride_ = k2 * page_size_;

    // populate second level index
    for (int i = 0; i < k1_ * k2_; i++) {
      tables_[k1_ + i] = ((i + 1) * page_size_ > data.size()) ? INT_MAX : data[(i + 1) * page_size_ - 1];
    }

    // populate top level index
    for (int i = 0; i < k1_; i++) {
      tables_[i] = tables_[k1_ + (i + 1) * k2 - 1];
    }
  }

  ~TwoLevelIndex() {
    free(tables_);
  }

  // assumes that key is <= max(data)
  int find(int key) {
    int i = binary_search_branchless(tables_, k1_, key);
    //int i = interpolation_search(tables_, k1_, key);
    //int i = linear_search_avx(tables_, k1_, key);
    //int j = binary_search_branchless(tables_ + k1_ + i * k2_, k2_, key);
    int j = linear_search_avx(tables_ + k1_ + i * k2_, k2_, key);
    int pos = i * k2stride_ + j * page_size_;
    int offset = linear_search_avx(data_ + pos, page_size_, key);
    return pos + offset;
  }
};


class ThreeLevelIndex {
  const int* data_;
  int k1_, k2_, k3_, stride1_, stride2_, page_size_;
  int* tables_;
  int* table2_;
  int* table3_;

 public:
  explicit ThreeLevelIndex(std::vector<int>& data, int k2, int k3, int page_size)
      : k2_(k2), k3_(k3), page_size_(page_size) {

    int hang = data.size() % page_size;
    if (hang > 0) {
      for (int i = 0; i < page_size - hang; i++) {
        data.push_back(INT_MAX);
      }
    }

    data_ = &data[0];

    k1_ = (int) std::ceil(data.size() / double(k2_ * k3_ * page_size_));
    //k1_ = 2 << bsr((int) std::ceil(data.size() / double(k2_ * k3_ * page_size_)) - 1);
    std::cerr << k1_ << std::endl;
    tables_ = (int*) calloc((size_t) k1_ * (1 + k2_ * (1 + k3_)), sizeof(int));
    table2_ = tables_ + k1_;
    table3_ = table2_ + k1_ * k2_;

    stride1_ = k2_ * k3_ * page_size_;
    stride2_ = k3_ * page_size_;

    // populate third level index
    for (int i = 0; i < k1_ * k2_ * k3_; i++) {
      table3_[i] = ((i + 1) * page_size_ > data.size()) ? INT_MAX : data[(i + 1) * page_size_ - 1];
    }

    // populate second level index
    for (int i = 0; i < k1_ * k2_; i++) {
      table2_[i] = table3_[(i + 1) * k3_ - 1];
    }

    // populate top level index
    for (int i = 0; i < k1_; i++) {
      tables_[i] = table2_[(i + 1) * k2_ - 1];
    }
  }

  ~ThreeLevelIndex() {
    free(tables_);
  }

  // assumes that key is <= max(data)
  int find(int key) {
    int i = binary_search_branchless(tables_, k1_, key);
    //int i = interpolation_search(tables_, k1_, key);
    //int i = linear_search_avx(tables_, k1_, key);
    int j = linear_search_avx(table2_ + i * k2_, k2_, key);
    int k = linear_search_avx(table3_ + i * k2_ * k3_ + j * k3_, k3_, key);
    int pos = i * stride1_ + j * stride2_ + k * page_size_;
    int offset = linear_search_avx(data_ + pos, page_size_, key);
    return pos + offset;
  }
};


std::vector<int> read_data(const char *path) {
  std::vector<int> vec;
  FILE *fin = fopen(path, "rb");
  int buf[BUF_SIZE];
  while (true) {
    size_t num_read = fread(buf, sizeof(int), BUF_SIZE, fin);
    for (int i = 0; i < num_read; i++) {
      vec.push_back(buf[i]);
    }
    if (num_read < BUF_SIZE) break;
  }
  fclose(fin);
  return vec;
}


int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0] << " DATA_PATH TRIALS K1 K2 K3" << std::endl;
    exit(1);
  }

  int num_trials = std::atoi(argv[2]);
  int k1 = std::atoi(argv[3]);
  int k2 = std::atoi(argv[4]);
  int k3 = std::atoi(argv[5]);
  printf("k1: %d, k2: %d, k3: %d\n", k1, k2, k3);

  std::vector<int> vec = read_data(argv[1]);
  vec.push_back(INT_MAX);
  const int* data = &vec[0];
  int n = (int) vec.size();
  printf("num elements: %d\n", n);
  
  // Clone vec so we don't bring pages from it into cache when selecting random keys
  std::vector<int> vec_clone(vec.begin(), vec.end());

  // construct indices
  stx::btree_set<int, std::less<int>, btree_traits> btree(vec.begin(), vec.begin() + n);
  //OneLevelIndex index1(vec, k1);
  TwoLevelIndex index2(vec, k1, k2);
  ThreeLevelIndex index3(vec, k1, k2, k3);

  uint32_t seed = std::random_device()();
  std::mt19937 rng;
  std::uniform_int_distribution<> dist(0, n - 1);
  std::vector<int> queries(QUERIES_PER_TRIAL);

  std::vector<long> checksums;
  std::vector<double> times_bs;
  std::vector<double> times_bt;
  std::vector<double> times_h2;
  std::vector<double> times_h3;

  // binary search baseline
  /*
  rng.seed(seed);
  for (int t = 0; t < num_trials; t++) {
    long check = 0;
    auto start = clock();
    for (const int& key : queries) {
      volatile int res = binary_search_branchless(data, n, key);
      check += res;
    }
    double elapsed = double(clock() - start) / CLOCKS_PER_SEC;

    times_bs.push_back(elapsed);
    checksums.push_back(check);
  }
  */
  
  // stx::btree baseline
  /*
  rng.seed(seed);
  for (int t = 0; t < num_trials; t++) {
    auto start = clock();
    for (const int& key : queries) {
      volatile auto res = btree.find(key);
    }
    double elapsed = double(clock() - start) / CLOCKS_PER_SEC;
    times_bt.push_back(elapsed);
  }
  */
  
  // benchmark 2-level index
  rng.seed(seed);
  for (int t = 0; t < num_trials; t++) {
    for (int &query : queries) {
      query = vec_clone[dist(rng)];
    }

    long check = 0;
    auto start = clock();
    for (const int& key : queries) {
      int res = index2.find(key);
      check += res;
    }
    double elapsed = double(clock() - start) / CLOCKS_PER_SEC;
    if (check % 13337 == 0) printf("%ld\n", check);

    times_h2.push_back(elapsed);
//    if (check != checksums[t]) {
//      std::cerr << "ERROR: checksum mismatch " << checksums[t] << " vs " << check << std::endl;
//      exit(1);
//    }
  }

  // benchmark 3-level index
  rng.seed(seed);
  for (int t = 0; t < num_trials; t++) {
    for (int &query : queries) {
      query = vec_clone[dist(rng)];
    }

    long check = 0;
    auto start = clock();
    for (const int& key : queries) {
      int res = index3.find(key);
      check += res;
    }
    double elapsed = double(clock() - start) / CLOCKS_PER_SEC;
    if (check % 13337 == 0) printf("%ld\n", check);

    times_h3.push_back(elapsed);
//    if (check != checksums[t]) {
//      std::cerr << "ERROR: checksum mismatch " << checksums[t] << " vs " << check << std::endl;
//      exit(1);
//    }
  }

  // double time_bs = 1e+9 * std::accumulate(times_bs.begin(), times_bs.end(), 0.) / (num_trials * QUERIES_PER_TRIAL);
  // double time_bt = 1e+9 * std::accumulate(times_bt.begin(), times_bt.end(), 0.) / (num_trials * QUERIES_PER_TRIAL);
  double time_h2 = 1e+9 * std::accumulate(times_h2.begin(), times_h2.end(), 0.) / (num_trials * QUERIES_PER_TRIAL);
  double time_h3 = 1e+9 * std::accumulate(times_h3.begin(), times_h3.end(), 0.) / (num_trials * QUERIES_PER_TRIAL);

  // printf("%8.1lf ns : %.40s\n", time_bs, "binary search");
  // printf("%8.1lf ns : %.40s\n", time_bt, "btree");
  printf("%8.1lf ns : %.40s\n", time_h2, "2-level index");
  printf("%8.1lf ns : %.40s\n", time_h3, "3-level index");

  return 0;
}
