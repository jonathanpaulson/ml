#include <algorithm>
 
#include "math.hpp"
 
using namespace std;
 
struct model {
    vector<vec> means;
    mat cov_inv;
    vector<ld> log_priors;
};
 
ll classify(vec xv, model model) {
    mat x = mat_of_vec(xv);
    mat SI = model.cov_inv;
 
    vector<ld> scores;
    for(ll i=0; i<model.means.size(); i++) {
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
    ll P = X[0].size();
    assert(Y.size() == N);
    for(ll i=0; i<X.size(); i++) {
        assert(X[i].size() == P);
    }
    ll K = Y[0]+1;
    for(ll i=0; i<Y.size(); i++) {
      assert(Y[i] >= 0);
      K = max(K, Y[i]+1);
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
