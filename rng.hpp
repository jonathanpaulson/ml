#include <random>

#include "math.hpp"
 
using namespace std;
 
typedef long long ll;
typedef double ld;
 
default_random_engine& RNG() {
    static default_random_engine RNG(random_device{}());
    return RNG;
}
 
ld standard_normal() {
    static normal_distribution<ld> N(0, 1.0);
    return N(RNG());
}
ld rld(ld lo, ld hi) {
    uniform_real_distribution<ld> D(lo, hi);
    return D(RNG());
}
ll rll(ll lo, ll hi) {
    uniform_int_distribution<ll> D(lo, hi);
    return D(RNG());
}
 
vec multivariate_gaussian(vec mu, mat cov) {
    vector<ld> zs;
    for(unsigned i=0; i<mu.size(); i++) {
        zs.push_back(standard_normal());
    }
    mat A = cholesky(cov);
    return vec_add(mu, mat_vec_mul(A, zs));
}
