#include <algorithm>
 
#include "math.hpp"
#include "rng.hpp"
#include "lda.hpp"
 
using namespace std;
 
typedef long long ll;
typedef double ld;
typedef vector<ld> vec;
typedef vector<vec> mat;
 
ll K = 3; // K classes
ll P = 3; // P features
 
ll N1 = 1000; // train
ll N2 = 1000; // test
 
int main() {
    vector<vec> means(K, vec(P, 0.0));
    for(ll i=0; i<K; i++) {
        for(ll j=0; j<P; j++) {
            means[i][j] = rld(-1, 1);
        }
    }
    mat cov =
      {{1.0, 0.5, 0.5},
       {0.5, 1.0, 0.5},
       {0.5, 0.5, 1.0}};
 
    vector<vec> X;
    vector<ll> Y;
    for(ll i=0; i<N1; i++) {
        ll y = rll(0, K-1);
        vec x = multivariate_gaussian(means[y], cov);
        X.push_back(x);
        Y.push_back(y);
    }
 
    model m = make_model(X, Y);
 
    model perfect;
    perfect.cov_inv = mat_inv(cov);
    for(ll i=0; i<K; i++) {
        perfect.means.push_back(means[i]);
        perfect.log_priors.push_back(0.0);
    }
 
    ll s1 = 0;
    ll s2 = 0;
    for(ll i=0; i<N2; i++) {
        ll y = rll(0, K-1);
        vec x = multivariate_gaussian(means[y], cov);
        ll p1 = classify(x, perfect);
        ll p2 = classify(x, m);
        if(p1 == y) {
            s1++;
        }
        if(p2 == y) {
            s2++;
        }
    }
    cout << "perfect=" << s1 << " (" << 100.0*s1/N2 << "%) model=" << s2 << " (" << 100.0*s2/N2 << "%)" << endl;
}
