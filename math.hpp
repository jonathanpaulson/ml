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
    out << " ]" << endl;
    return out;
}
ostream& operator<<(ostream& out, const mat& m) {
    for(vec v : m) {
        out << v;
    }
    return out;
}

mat column_vector_of_vec(const vec& V) {
  mat A(V.size(), vec(1, 0.0));
  for(ll i=0; i<V.size(); i++) {
    A[i][0] = V[i];
  }
  return A;
}
vec vec_of_column_vector(const mat& X) {
  assert(X[0].size() == 1);
  vec A(X.size(), 0.0);
  for(ll i=0; i<X.size(); i++) {
    A[i] = X[i][0];
  }
  return A;
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
            for(ll j=0; j<B[k].size(); j++) {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    return C;
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
 
mat I(ll n) {
    mat A(n, vec(n, 0.0));
    for(ll i=0; i<n; i++) {
        A[i][i] = 1.0;
    }
    return A;
}

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
    // R is upper-triangular
    for(ll r=0; r<R.size(); r++) {
        for(ll c=0; c<r; c++) {
            assert(eq(R[r][c], 0.0));
        }
    }
 
    // X = QR
    assert(mat_eq(X, mat_mul(Q, R)));
    return make_pair(Q, R);
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
    return vec_of_column_vector(B);
}

// X=NxP 
// B=(P+1)x1
// Return vec(N)
vector<ld> predict(const mat& X, const vec& B) {
  assert(B.size() == X[0].size()+1);
  return vec_of_column_vector(mat_mul(add_ones(X), column_vector_of_vec(B)));
}
