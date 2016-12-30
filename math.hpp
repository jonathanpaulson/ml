#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <tuple>
#include <random>
#include <chrono>
 
using namespace std;
typedef long long ll;
typedef double ld;
typedef vector<vector<ld>> mat;
typedef vector<ld> vec;
 
ld eps = 1e-6;
ld eq(ld x, ld y) {return fabs(x-y)<eps; }
 
ostream& operator<<(ostream& out, const vec& v) {
    out << "[";
    for(ld x : v) {
        out << " " << x;
    }
    out << " ]";
    return out;
}
ostream& operator<<(ostream& out, const mat& m) {
    for(vec v : m) {
        out << v << endl;
    }
    return out;
}
 
vec vec_add(const vec& A, const vec& B) {
    vec C(A.size());
    assert(B.size() == A.size());
    for(ll i=0; i<A.size(); i++) {
        C[i] = A[i]+B[i];
    }
    return C;
}
vec vec_sub(const vec& A, const vec& B) {
    vec C(A.size());
    assert(B.size() == A.size());
    for(ll i=0; i<A.size(); i++) {
        C[i] = A[i]-B[i];
    }
    return C;
}
mat mat_add(const mat& A, const mat& B) {
    mat C(A.size(), vec(A[0].size(), 0.0));
    assert(B.size()==A.size() && B[0].size()==A[0].size());
    for(ll i=0; i<A.size(); i++) {
        for(ll j=0; j<A[0].size(); j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}
 
vec vec_scale(const vec& A, ld by) {
    vec B(A.size(), 0.0);
    for(ll i=0; i<A.size(); i++) {
        B[i] = A[i]*by;
    }
    return B;
}
mat mat_scale(const mat& A, ld by) {
    mat B(A.size(), vec(A[0].size(), 0.0));
    for(ll i=0; i<A.size(); i++) {
        for(ll j=0; j<A[0].size(); j++) {
            B[i][j] = A[i][j]*by;
        }
    }
    return B;
}
 
mat transpose(const mat& X) {
    mat A(X[0].size(), vec(X.size()));
    for(ll r=0; r<X.size(); r++) {
        for(ll c=0; c<X[0].size(); c++) {
            A[c][r] = X[r][c];
        }
    }
    return A;
}
 
mat mat_mul(const mat& A, const mat& B) {
    assert(A[0].size() == B.size());
    mat C(A.size(), vec(B[0].size(), 0.0));
    for(ll i=0; i<A.size(); i++) {
        for(ll k=0; k<B.size(); k++) {
            for(ll j=0; j<B[0].size(); j++) {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    return C;
}
 
mat mat_of_vec(const vec& A) {
    mat B(A.size(), vec(1, 0.0));
    for(ll i=0; i<A.size(); i++) {
        B[i][0] = A[i];
    }
    return B;
}
vec vec_of_mat(const mat& A) {
    assert(A[0].size() == 1);
    vec B(A.size(), 0.0);
    for(ll i=0; i<A.size(); i++) {
        B[i] = A[i][0];
    }
    return B;
}
ld mat_to_ld(const mat& A) {
    assert(A.size()==1 && A[0].size()==1);
    return A[0][0];
}
vec mat_vec_mul(const mat& A, const vec& B) {
    return vec_of_mat(mat_mul(A, mat_of_vec(B)));
}
 
bool mat_eq(const mat& A, const mat& B) {
    if(A.size() != B.size()) {
        return false;
    }
    for(ll r=0; r<A.size(); r++) {
        if(A[r].size() != B[r].size()) {
            return false;
        }
        for(ll c=0; c<A[r].size(); c++) {
            if(!eq(A[r][c], B[r][c])) {
                return false;
            }
        }
    }
    return true;
}
 
ld dot(const vec& A, const vec& B) {
    assert(A.size() == B.size());
    ld ans = 0.0;
    for(ll i=0; i<A.size(); i++) {
        ans += A[i]*B[i];
    }
    return ans;
}
 
mat ZERO(ll n) {
    return mat(n, vec(n, 0.0));
}
mat I(ll n) {
    mat A(n, vec(n, 0.0));
    for(ll i=0; i<n; i++) {
        A[i][i] = 1.0;
    }
    return A;
}
 
bool is_upper_triangular(const mat& R) {
    for(ll r=0; r<R.size(); r++) {
        for(ll c=0; c<r; c++) {
            if(!eq(R[r][c], 0.0)) {
                return false;
            }
        }
    }
    return true;
}
bool is_inverse(const mat& A, const mat& AI) {
    ll n = A.size();
    // A is square
    if(A[0].size() != n) {
        return false;
    }
    // AI is square
    if(AI.size()!=n || AI[0].size()!=n) {
        return false;
    }
    // Right inverse
    if(!mat_eq(mat_mul(A,AI), I(n))) {
        return false;
    }
    // Left inverse
    if(!mat_eq(mat_mul(AI, A), I(n))) {
        return false;
    }
    return true;
}
 
// X is NxP
// Let R = rank(X)
// Q is NxR
// R is RxP
pair<mat, mat> QR(const mat& X) {
    ll N = X.size();
    ll P = X[0].size();
 
    mat XT = transpose(X); // PxN
    mat Z;
    mat R;
 
    for(ll i=0; i<P; i++) {
        vector<ld> zi = XT[i];
        vector<ld> ri;
        for(ll j=0; j<Z.size(); j++) {
            ld bij = dot(zi, Z[j])/dot(Z[j], Z[j]);
            for(ll k=0; k<N; k++) {
                zi[k] = zi[k] - bij*Z[j][k];
            }
            ri.push_back(bij);
        }
        if(!eq(dot(zi, zi), 0.0)) {
            ri.push_back(1.0);
            Z.push_back(zi);
            R.push_back(ri);
        } else {
            R.push_back(ri);
        }
    }
 
    for(int i=0; i<R.size(); i++) {
        while(R[i].size() < R[R.size()-1].size()) {
            R[i].push_back(0.0);
        }
    }
 
    R = transpose(R);
 
    mat DI(R.size(), vec(R.size(), 0.0));
    mat D(R.size(), vec(R.size(), 0.0));
    for(ll i=0; i<R.size(); i++) {
        D[i][i] = sqrt(dot(Z[i], Z[i]));
        DI[i][i] = 1.0/D[i][i];
    }
 
    mat Q = mat_mul(transpose(Z), DI);
    // Q is orthagonal
    assert(mat_eq(mat_mul(transpose(Q), Q), I(Q[0].size())));
 
    R = mat_mul(D, R);
    assert(is_upper_triangular(R));
 
    // X = QR
    assert(mat_eq(X, mat_mul(Q, R)));
    return make_pair(Q, R);
}
 
// Compute L s.t. LL^T = A and L is lower triangular
mat cholesky(const mat& A) {
    assert(A.size() == A[0].size());
    mat L(A.size(), vec(A[0].size(), 0.0));
    for(ll i=0; i<A.size(); i++) {
        L[i][i] = A[i][i];
        for(ll k=0; k<i; k++) {
            L[i][i] -= L[i][k]*L[i][k];
        }
        L[i][i] = sqrt(L[i][i]);
        for(ll j=i+1; j<A.size(); j++) {
            L[j][i] = A[j][i];
            for(ll k=0; k<i; k++) {
                L[j][i] -= L[j][k]*L[i][k];
            }
            L[j][i] = L[j][i]/L[i][i];
        }
    }
    assert(mat_eq(mat_mul(L, transpose(L)), A));
    return L;
}
 
mat mat_tri_inv(const mat& R) {
    assert(is_upper_triangular(R));
    ll n = R.size();
    mat D(n, vec(n, 0.0));
    mat DI(n, vec(n, 0.0));
    mat RU = R;
    for(ll i=0; i<n; i++) {
        assert(R[i][i] > 0.0);
        D[i][i] = R[i][i];
        DI[i][i] = 1.0/D[i][i];
        RU[i][i] = 0.0;
    }
    assert(is_inverse(D, DI));
 
    RU = mat_mul(DI, RU);
    mat R2 = mat_add(I(n), RU);
    assert(mat_eq(R, mat_mul(D, R2)));
 
    mat R2I = ZERO(n);
    mat RUp = I(n);
    while(true) {
        R2I = mat_add(R2I, RUp);
        RUp = mat_mul(RUp, mat_scale(RU, -1));
        if(mat_eq(RUp, ZERO(n))) {
            break;
        }
    }
    assert(is_inverse(R2, R2I));
    mat RI = mat_mul(R2I, DI);
    assert(is_inverse(R, RI));
    return RI;
}
 
// inv(X) = inv(QR) = inv(R)inv(Q) = inv(R)Q^T
mat mat_inv(const mat& X) {
    // X is square
    assert(X.size() == X[0].size());
    mat Q;
    mat R;
    std::tie(Q, R) = QR(X);
    // X is not singular
    assert(R.size() == X.size() && R[0].size() == X.size());
 
    mat RI = mat_tri_inv(R);
 
    mat XI = mat_mul(RI, transpose(Q));
    assert(is_inverse(X, XI));
    return XI;
}

// return B s.t. \sum_i (Y[i][0] - mat_mul(X,B)[i][0])^2 is minimized
// X = [ x0 x1 x2] - NxP
// Y is Nx1
// B (return value) is Px1
mat regress(mat X, mat Y) {
    ll N = X.size();
    ll P = X[0].size();
    assert(Y.size() == N && Y[0].size()==1);
 
    mat Q; // Nxrank(X)
    mat R; // rank(X)xP
    std::tie(Q, R) = QR(X);

    cout << "Rank(X): " << Q[0].size() << endl;
 
    // Px1
    mat C = mat_mul(transpose(Q), Y);

    vector<ll> CI;
    // Solve RB = C. R is upper triangular
    mat B(P, vec(1, 0.0));
    for(ll i=R.size()-1; i>=0; i--) {
      ll c = 0;
      while(eq(R[i][c], 0.0)) { c++; }

      B[c][0] = C[i][0];
      for(ll j=0; j<CI.size(); j++) {
        B[c][0] = B[c][0] - R[i][CI[j]]*B[CI[j]][0];
      }
      B[c][0] /= R[i][c];
      CI.push_back(c);
    }
    assert(mat_eq(mat_mul(R, B), C));
    return B;
}

mat add_ones(const mat& X) {
  mat X2(X.size(), vec(X[0].size()+1, 0.0));
  for(ll i=0; i<X.size(); i++) {
    X2[i][0] = 1.0;
    for(ll j=0; j<X[0].size(); j++) {
      X2[i][j+1] = X[i][j];
    }
  }
  return X2;
}
 
vec regress_wrap(const mat& X, const vec& Y) {
    ll N = X.size();
    ll P = X[0].size();
    assert(Y.size() == N);
 
    mat X2 = add_ones(X);
    mat Y2(N, vec(1, 0.0));
    for(ll i=0; i<N; i++) {
        Y2[i][0] = Y[i];
    }
    mat B = regress(X2, Y2);
    assert(B.size() == P+1 && B[0].size()==1);
    return vec_of_mat(B);
}

// X=NxP 
// B=(P+1)x1
// Return vec(N)
vector<ld> predict(const mat& X, const vec& B) {
  assert(B.size() == X[0].size()+1);
  return vec_of_mat(mat_mul(add_ones(X), mat_of_vec(B)));
}


