#include <random>
#include <algorithm>
 
#include "math.hpp"
 
using namespace std;
 
typedef long long ll;
typedef double ld;
typedef vector<ld> vec;
typedef vector<vec> mat;
 
ll K = 3; // K classes
ll P = 3; // P features
 
ll N1 = 1000; // train
ll N2 = 1000; // test
 
struct model {
    vector<vec> means;
    mat cov_inv;
    vector<ld> log_priors;
};
 
 
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
    for(unsigned i=0; i<P; i++) {
        zs.push_back(standard_normal());
    }
    mat A = cholesky(cov);
    return vec_add(mu, mat_vec_mul(A, zs));
}
 
ll classify(vec xv, model model) {
    mat x = mat_of_vec(xv);
    mat SI = model.cov_inv;
 
    vector<ld> scores;
    for(ll i=0; i<K; i++) {
        mat mu = mat_of_vec(model.means[i]);
        // x : Px1
        // SI : PxP
        // mu : Px1
        mat s1 = mat_mul(transpose(x), mat_mul(SI, mu));
        assert(s1.size() == 1 && s1[0].size() == 1);
        mat s2 = mat_mul(transpose(mu), mat_mul(SI, mu));
 
        ld score = mat_to_ld(s1) - 0.5*mat_to_ld(s2) + model.log_priors[i];
        scores.push_back(score);
    }
    return distance(scores.begin(), max_element(scores.begin(), scores.end()));
}
 
model make_model(vector<vec> X, vector<ll> Y) {
    ll N = X.size();
    assert(Y.size() == N);
    for(ll i=0; i<X.size(); i++) {
        assert(X[i].size() == P);
    }
    for(ll i=0; i<Y.size(); i++) {
        assert(0 <= Y[i] && Y[i] < K);
    }
 
    vector<ll> C(K, 0);
    for(ll i=0; i<Y.size(); i++) {
        C[Y[i]]++;
    }
 
    vector<vec> means(K, vec(P, 0.0));
    for(ll i=0; i<X.size(); i++) {
        ll k = Y[i];
        means[Y[i]] = vec_add(means[k], vec_scale(X[i], 1.0/C[k]));
    }
    mat cov = mat(P, vec(P, 0.0));
    for(ll i=0; i<X.size(); i++) {
        mat diff = mat_of_vec(vec_sub(X[i], means[Y[i]]));
        cov = mat_add(cov, mat_scale(mat_mul(diff, transpose(diff)), 1.0/(N-K)));
    }
 
    model m;
    m.cov_inv = mat_inv(cov);
    for(ll i=0; i<K; i++) {
        m.means.push_back(means[i]);
        m.log_priors.push_back(log(static_cast<ld>(C[i])/N));
    }
    return m;
}
 
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
