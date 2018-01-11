//
// Generates 190M values drawn from a
//

#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <climits>
#include <set>
#include <vector>

int main()
{
  double scale = 1e+9;
  double max = double(INT_MAX) / scale;
  int nelements = 190000000;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::lognormal_distribution<double> dist(0.0, 2.0);

  std::set<int> samples;

  while (samples.size() < nelements) {
    double r = dist(rng);
    if (r > max) continue;
    samples.insert(int(r * scale));
    if (samples.size() % 1000000 == 0) {
      std::cerr << "Generated " << samples.size() << std::endl;
    }
  }

  std::vector<int> vec(samples.begin(), samples.end());
  std::sort(vec.begin(), vec.end());
  std::cerr << "min = " << vec[0] << std::endl;
  std::cerr << "max = " << vec[vec.size() - 1] << std::endl;

  FILE *fout = fopen("lognormal.sorted.190M", "wb");
  for (int x : vec) {
    fwrite(&x, sizeof(x), 1, fout);
  }
  fclose(fout);

  return 0;
}
