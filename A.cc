#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <cassert>
#include <string>
#include <cmath>

#include "io.hpp"
#include "math.hpp"

using namespace std;
typedef uint8_t byte;
typedef long long ll;
typedef double ld;
typedef vector<vector<ld>> mat;
typedef vector<ld> vec;

ld score(const mat& B, const mat& X, const vector<ll>& Y) {
  ll n = X.size();
  assert(Y.size() == n);

  mat Y2;
  for(ll d=0; d<10; d++) {
    Y2.push_back(predict(X, B[d]));
  }

  vector<pair<ll, ld>> Y3(Y.size(), make_pair(0, 0.0));
  for(ll i=0; i<Y.size(); i++) {
    ld sum = 0.0;
    ll best = 0;
    for(ll d=0; d<10; d++) {
      sum += Y2[d][i];
      if(Y2[d][i] > Y2[best][i]) {
        best = d;
      }
    }
    Y3[i] = make_pair(best, Y2[best][i]/sum);
  }

  vector<pair<ld, ll>> HARD;
  ll score = 0;
  for(ll i=0; i<Y.size(); i++) {
    if(Y3[i].first==Y[i]) {
      score++;
    } else {
      HARD.push_back(make_pair(Y3[i].second, i));
    }
  }

  sort(HARD.begin(), HARD.end());
  for(ll i=0; i<10; i++) {
    ll j = HARD[i].second;
    cout << j << ": correct=" << Y[j] << " guess=" << Y3[j].first << " p=" << Y3[j].second << endl;
    write_ppm_bw(X[j], 28, 28, to_string(j)+".ppm");
  }
  return (score+0.0)/(n+0.0);
}

int main() {
  ll n1 = 60000;
  ll n2 = 10000;
  ll R = 28;
  ll C = 28;
  mat X1 = read_data(n1, R, C, "train_data.dat");
  vector<ll> Y1 = read_labels(n1, "train_label.dat");

  //X1.resize(1000);
  //Y1.resize(1000);

  mat B;
  for(ll d=0; d<=9; d++) {
    cout << "d=" << d << endl;
    vector<ld> Yd;
    for(ll i=0; i<Y1.size(); i++) {
      Yd.push_back(Y1[i]==d);
    }
    vec Bd = regress_wrap(X1, Yd);
    assert(Bd.size() == X1[0].size()+1);
    B.push_back(Bd);
  }

  vector<mat> VIS(10, mat(3, vec(R*C, 0.0)));
  for(ll d=0; d<10; d++) {
    assert(B[d].size() == 1+R*C);
    for(ll i=0; i<R*C; i++) {
      if(B[d][i+1] > 0.0) {
        VIS[d][0][i] = B[d][i+1];
      } else {
        VIS[d][1][i] = -B[d][i+1];
      }
    }
  }
  for(ll d=0; d<10; d++) {
    for(ll i=0; i<VIS[d][2].size(); i++) {
      assert(eq(VIS[d][2][i], 0.0));
    }
    write_ppm(VIS[d][0], VIS[d][1], VIS[d][2], R, C, "C"+to_string(d)+".ppm");
  }

  mat X2 = read_data(n2, R, C, "test_data.dat");
  vector<ll> Y2 = read_labels(n2, "test_label.dat");

//  ld s1 = score(B, X1, Y1);
  ld s2 = score(B, X2, Y2);
  cout << "Train: " << 0 << endl << "Test: " << s2 << endl;
}
