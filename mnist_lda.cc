#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <cassert>
#include <string>
#include <cmath>

#include "io.hpp"
#include "math.hpp"
#include "lda.hpp"

using namespace std;

int main() {
  ll n1 = 60000;
  ll n2 = 10000;
  ll R = 28;
  ll C = 28;
  mat X1 = read_data(n1, R, C, "train_data.dat");
  vector<ll> Y1 = read_labels(n1, "train_label.dat");

  cout << "READ TRAIN" << endl;

  model m = make_model(X1, Y1);

  cout << "MODEL" << endl;

  mat X2 = read_data(n2, R, C, "test_data.dat");
  vector<ll> Y2 = read_labels(n2, "test_label.dat");

  cout << "READ TEST" << endl;

  ll score = 0;
  for(ll i=0; i<n2; i++) {
    ll p = classify(X2[i], m);
    if(p == Y2[i]) {
      score++;
    }
  }
  cout << score << endl;
}
