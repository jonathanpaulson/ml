#include <vector>

#include "math.hpp"

typedef long long ll;
typedef double ld;

int main() {
    vector<vector<ld>> X;
    vector<ld> Y;
    for(ll i=0; i<10; i++) {
        ld x0 = static_cast<ld>(i);
        ld x1 = static_cast<ld>(i*i);
        ld y = 1 + 2*x0 + 3*x1;
        X.push_back({x0, x1});
        Y.push_back(y);
    }
    vector<ld> B = regress_wrap(X, Y);
    cout << "Y = " << B[0] << " + " << B[1] << "*X1 + " << B[2] << "*X2" << endl;
}

