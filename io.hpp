#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <cassert>
#include <string>
#include <cmath>

using namespace std;
typedef uint8_t byte;
typedef long long ll;
typedef double ld;
typedef vector<vector<ld>> mat;
typedef vector<ld> vec;

std::vector<byte> read_file(string filename) {
    std::basic_ifstream<char> file(filename, std::ios::in | std::ios::binary);
    istreambuf_iterator<char> eof;
    vector<byte> ans;
    for(auto it = istreambuf_iterator<char>(file); it!=eof; ++it) {
      ans.push_back(byte(*it));
    }
    return ans;
}

ll read32(vector<byte>& A, ll i) {
  return A[i]*256*256*256 + A[i+1]*256*256 + A[i+2]*256 + A[i+3];
}

vec normalize(const vec& A) {
  ld mx = A[0];
  for(ll i=0; i<A.size(); i++) {
    mx = max(A[i], mx);
  }
  if(mx < 1e-9) {
    return A;
  } else {
    vector<ld> B(A.size(), 0.0);
    for(ll i=0; i<A.size(); i++) {
      B[i] = A[i]/mx;
    }
    return B;
  }
}

void write_ppm(const vector<ld>& R, const vector<ld>& G, const vector<ld>& B, ll rows, ll cols, string filename) {
  ofstream out(filename);
  out << "P3" << endl;
  out << cols << " " << rows << " " << 255 << endl;
  assert(R.size() == rows*cols);
  /*
  for(ld i=0; i<rows; i++) {
    for(ll j=0; j<cols; j++) {
      cout << R[i*cols+j] << " ";
    }
    cout << endl;
  }
  cout << endl << endl << endl;
  */
  vector<ld> R2 = normalize(R);
  vector<ld> G2 = normalize(G);
  vector<ld> B2 = normalize(B);
  auto pixel = [](ld x) { return static_cast<ll>(255*(1.0-x)); };
  for(ll i=0; i<rows*cols; i++) {
    out << pixel(R2[i]) << " " << pixel(G2[i]) << " " << pixel(B2[i]) << endl;
  }
  out.close();
}

void write_ppm_bw(const vector<ld>& G, ll R, ll C, string filename) {
  write_ppm(G, G, G, R, C, filename);
}

mat read_data(ll n, ll R, ll C, const string& filename) {
  vector<byte> data = read_file(filename);
  assert(read32(data, 0) == 2051);
  assert(read32(data, 4) == n);
  assert(read32(data, 8) == R);
  assert(read32(data, 12) == C);

  ll ix = 16;
  mat X(n, vector<ld>());
  for(ll i=0; i<n; i++) {
    for(ll r=0; r<R*C; r++) {
      X[i].push_back(data[ix]);
      ix++;
    }
  }
  assert(ix == data.size());
  return X;
}

vector<ll> read_labels(ll n, const string& filename) {
  vector<byte> data = read_file(filename);
  assert(data.size() == n+8);
  assert(read32(data, 0) == 2049);
  assert(read32(data, 4) == n);
  vector<ll> Y(n, 0);
  for(ll i=0; i<n; i++) {
    Y[i] = data[i+8];
  }
  return Y;
}

