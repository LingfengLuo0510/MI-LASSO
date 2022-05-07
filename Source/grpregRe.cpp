//#include <Rcpp.h>
#include <RcppArmadillo.h>
#include <math.h>
#include <iostream>
#include <RcppArmadilloExtensions/sample.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;
using namespace std;


double norm(const vec &x, const int &p) {
  double x_norm = 0;
  for (int j=0; j<p; j++) {
    x_norm = x_norm + pow(x(j),2);
  }
  x_norm = sqrt(x_norm);
  return(x_norm);
}

// Cross product of the jth column of x with y
double crossprod(const mat &x, const vec &y, const int &n, const int &j) {
  double val = 0;
  for (int i=0; i<n; i++) val += x(i,j) * y(i);
  return(val);
}

// [[Rcpp::export]]
double  maxgrad(const mat &X, const vec &y, const vec &K, const vec &m){
  
  //intialize
  int n = X.n_rows;
  int J = K.n_elem - 1;
  double zmax = 0;

  for (int g = 0; g < J; ++g)
  {
    int Kg = K(g+1) - K(g);
    vec Z = zeros<vec>(Kg);
    for (int j = K(g); j < K(g+1); ++j)
    { 
      Z(j - K(g)) = crossprod(X, y, n, j);
    }
    double z = norm(Z, Kg) / m(g);
    if ( z > zmax) zmax = z;
  }
  
  return zmax;
}

// Soft-thresholding operator
double S(double z, double l) {
  if (z > l) return(z-l);
  if (z < -l) return(z+l);
  return(0);
}


// standardize function
// [[Rcpp::export]]
List standardize(const mat &x) {
	int n = x.n_rows;
	int p = x.n_cols;
	mat XX = zeros<mat>(n,p) ;
	vec c = zeros<vec>(p), s = zeros<vec>(p);

	for (int j = 0; j < p; ++j)
	{
		//center:
		c(j) = 0;
		for (int i = 0; i < n; ++i)
		{
			c(j) = c(j) + x(i,j);
		}
		c(j) = c(j)/n;
		for (int i = 0; i < n; ++i)
		{
			XX(i,j) = x(i,j) - c(j);
		}

		//Scale:
		s(j) = 0;
		for (int i = 0; i < n; ++i)
		{
			s(j) += pow(XX(i,j),2);
		}
		s(j) = sqrt(s(j)/n);
		for (int i = 0; i < n; ++i)
		{
			XX(i,j) = XX(i,j)/s(j);
		}
	}

  return List::create(_["XX"]=XX,
                      _["c"]=c,
                      _["s"]=s);

}




// Group descent update -- cox
// call in main loop: gd_cox(beta, X, r, eta, v, g, K1, n, l, p, penalty, l1, l2, gamma, df, a, maxChange);
// r: first order derivative (n*1)
// eta: X*beta
// v: =1, hessian matrix approximation
// g: index of groups
// l: lambda
//	 lam1 = lambda(l) * group_multiplier(g) * alpha;
//	 lam2 = 0, (lambda(l) * group_multiplier(g) * (1-alpha);)
void gd_cox(mat &b, const mat &x, vec &r, vec &eta, double &v, const int &g,
            const uvec &K1, const int &n, int &l, int &p, const string &penalty,  double &lam1,
            double &lam2, const double &gamma, vec &df, vec &a, double &maxChange) {

  // Calculate z
  int K = K1(g+1) - K1(g);        //In group g, there are K covaraites
  vec z = zeros<vec>(K);          //z_g = 1/n * (X_g^T * r) + beta_g; a K-dimensional vector.
  for (int j=K1(g); j<K1(g+1); j++) {z(j-K1(g)) = crossprod(x, r, n, j)/n + a(j);}  
  double z_norm = norm(z,K);      //||z_g||_2,  L2 norm.

  // Update b
  double len;
  len = S(v * z_norm, lam1) / (v * (1 + lam2));   //||z_g||_2 - lambda(l) * group_multiplier(g)
  if (len != 0 || a(K1(g)) != 0) {
    // If necessary, update the K-dim beta, r, eta;
    for (int j=K1(g); j<K1(g+1); j++) { 
      b(l,j) = len * z(j-K1(g)) / z_norm;
      double shift = b(l,j)-a(j);
      if (fabs(shift) > maxChange) maxChange = fabs(shift);
      for (int i=0; i<n; i++) {
        double si = shift*x(i,j);
        r(i)   	 -= si;
        eta(i) 	 += si;
      }
    }
  }

  // Update df
  if (len > 0) df(l) += K * len / z_norm;
}



// [[Rcpp::export]]
List gdfit_cox(const mat &X, const vec &d, const string &penalty, 
                const uvec &K1, const int &K0, const vec &lambda, const double &alpha, const double &eps,
                const int &max_iter, const double &gamma, const vec &group_multiplier, const int &dfmax, const int &gmax,
                const bool &warn, const bool &user){
  
  // cout<<"done 1"<<endl;
  // Lengths/dimensions
  int n = d.n_elem;                     //number of samples
  int L = lambda.n_elem;                //number of penalized coefficients (for grid search)
  int J = K1.n_elem - 1;                //number of groups
  int p = X.n_cols;                     //number of predictors

  int tot_iter = 0;                     // iteration index
  
  // outcomes:
  List res;
  mat beta = zeros<mat>(L,p);           // store L*p beta
  vec Loss = zeros<vec>(L);             // loss function for each lambda
  uvec iter = zeros<uvec>(L);             // iterations taken for each lambda
  vec df   = zeros<vec>(L);
  mat Eta  = zeros<mat>(L,n);

  // interrmediate quantities  add notes:
  vec a    = zeros<vec>(p);      
  vec r    = zeros<vec>(n), h = zeros<vec>(n), haz = zeros<vec>(n), rsk = zeros<vec>(n), eta = zeros<vec>(n);
  vec e    = zeros<vec>(J);

  int lstart, ng, nv, violations;
  double shift, l1, l2, nullDev, v, s, maxChange;

  // Initialization
  // If lam[0]=lam_max, skip lam[0] -- closed form sol'n available
  rsk(n-1) = 1;                            //rsk: (e.g.  n = 5,  rsk = (5,4,3,2,1)  )
  for (int i = n-2; i >= 0; i--) {
    rsk(i) = rsk(i+1) + 1;
  }
  nullDev = 0;  
  for (int i = 0; i < n; i++){
    nullDev -= d(i)*log(rsk(i));           //nullDev : - sum(delta_i * log(rsk(i)))
  }
  if (user) { 
    lstart = 0;
  } else {
    lstart = 1;                                           
    Loss(0) = nullDev;                                     //Loss initialize as nullDev
  }

  // cout<<"done 2"<<endl;
  // Path through different  lambda:
  for (int l = lstart; l < L; ++l){
      Rcpp::checkUserInterrupt();
      if(l != 0) {
        // Assign previous beta to a (b: store all the beta)
        for (int j = 0; j < p; ++j) a(j) = beta(l-1,j);
      }

      // Check dfmax, gmax
      ng = 0;
      nv = 0;             // sotre all the number of covariates in each group
      for (int g = 0; g < J; g++) {
        if (a(K1(g)) != 0) {
          ng++;
          nv = nv + (K1(g+1)-K1(g));
        }
      }
      if (ng > gmax || nv > dfmax || tot_iter == max_iter) {
        for (int ll=l; ll<L; ll++) iter(ll) = NA_INTEGER;
        break;
      }

      //begin iteration:
      while(tot_iter < max_iter){
        while(tot_iter < max_iter){
		      iter(l)++;
		      tot_iter++;
		      Loss(l) = 0;
		      df(l)   = 0;

		      // Calculate haz, risk
		      haz = exp(eta);                                        // eta (to be updated below), for (int i=0; i<n; i++) haz(i) = exp(eta(i));
		      rsk(n-1) = haz(n-1);                                    
		      for (int i=n-2; i>=0; i--) {
		        rsk(i) = rsk(i+1) + haz(i);                          // rsk : reverse culmulative sum of haz
		      }  
		      for (int i=0; i<n; i++) {
		        Loss(l) += d(i)*eta(i) - d(i)*log(rsk(i));           // update Loss
		      }

		      //Approximate L:
		      h(0) = d(0)/rsk(0);                                    // delta_0 / rsk0
		      v = 1;                                                 // identitical matrix to approximate 
		      for (int i=1; i<n; i++) {
		        h(i) = h(i-1) + d(i)/rsk(i);                         // hi: sum(delta_l / rsk_l) (l: at risk set)   this is cumulative baseline hazard
		      }
		      for (int i=0; i<n; i++) {
		        h(i) = h(i)*haz(i);                                  
		        s    = d(i) - h(i);                  // corresponds to our first order derivative wrt pseudo-response
		        if (h(i)==0) r(i)=0;                 // corresponds to our l1/eta derivative
		        else r(i) = s/v;                     // use identical matrix to approximate the hessian matrix
		      }

		      // Check for saturation
		      if (Loss(l)/nullDev < .01) {
		        if (warn) warning("Model saturated; exiting...");
		        for (int ll=l; ll<L; ll++) iter(ll) = NA_INTEGER;
		        tot_iter = max_iter;
		        break;
		      }

		      // Update unpenalized covariates
		      maxChange = 0;
		      for (int j=0; j<K0; j++) {                      
		        shift = crossprod(X, r, n, j)/n;
		        if (fabs(shift) > maxChange) maxChange = fabs(shift);
		        beta(l,j) = shift + a(j);
		        for (int i=0; i<n; i++) {
		          double si = shift * X(i,j);
		          r(i)   -= si;              // minus si ()
		          eta(i) += si;
		        }
		        df(l)++;
		      }
		      // Update penalized groups
		      for (int g=0; g<J; g++) {
		        l1 = lambda(l) * group_multiplier(g) * alpha;
		        l2 = lambda(l) * group_multiplier(g) * (1-alpha);
		        if (e(g)!=0) gd_cox(beta, X, r, eta, v, g, K1, n, l, p, penalty, l1, l2,
		                         gamma, df, a, maxChange);
		      }
		      // Check convergence
		      for (int j=0; j<p; j++) a(j) = beta(l,j);
		      if (maxChange < eps) break;
        }

        // Scan for violations
        violations = 0;
        for (int g=0; g<J; g++) {
          if (e(g)==0) {
            l1 = lambda(l) * group_multiplier(g) * alpha;
            l2 = lambda(l) * group_multiplier(g) * (1-alpha);
            gd_cox(beta, X, r, eta, v, g, K1, n, l, p, penalty, l1, l2, gamma, df, a, maxChange);
            if (beta(l,K1(g)) != 0) {
              e(g) = 1;
              violations++;
            }
          }
        }
        if (violations==0) {
          for (int i=0; i<n; i++) Eta(l,i) = eta(i);  // can use .row
          break;
        }

        for (int j=0; j<p; j++) a(j) = beta(l,j);     // can use .row
      }
  }



  res["beta"]  = beta;
  res["iter"]  = iter;
  res["df"]    = df;
  res["Loss"]  = Loss;
  res["Eta"]   = Eta;
  
  return res; 
  
}