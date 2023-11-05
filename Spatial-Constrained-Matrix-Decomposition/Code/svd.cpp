// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <iostream>
using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List mySVD(const arma::sp_mat& X, int k) {
  arma::mat U;
  arma::vec s;
  arma::mat V;
  
  // Calculate the first k singular values using svds
  arma::svds(U, s, V, X, k);
  
  return Rcpp::List::create(Rcpp::Named("U") = U, 
                            Rcpp::Named("s") = s, 
                            Rcpp::Named("V") = V);
}



// Calculate R(XV)
// [[Rcpp::export]]
double calculate_regularization(const arma::mat& X, const arma::mat& V, const arma::mat& D) {
  arma::mat XV = X * V;
  arma::mat reg_term = D * XV; 
  return arma::accu(reg_term % reg_term);
}


// [[Rcpp::export]]
Rcpp::List gradient_descent_svd(const arma::mat& X, arma::mat U, arma::mat Sigma, arma::mat V, 
                                const arma::mat& D, double lambda, double alpha, double tol, int max_iter) {
  // Pre-compute constant terms
  arma::mat DX = D * X;  // X-transpose times D-transpose times D times X
  arma::mat Sigma_inv = arma::diagmat(1 / Sigma.diag());
  
  // Calculate initial objective function value
  double f = arma::norm(X - U * Sigma * V.t(), "fro") + lambda * calculate_regularization(X, V, D);
  
  int print_interval = max_iter / 100; // Print progress information 100 times
  for (int iter = 0; iter < max_iter; ++iter) {
    
    // Compute the gradient for V
    arma::mat grad_V = -2 * Sigma.t() * U.t() * X + 2 * Sigma.t() * U.t() * U * Sigma * V.t() + 2 * lambda * V.t() * DX.t()*DX;
    grad_V = grad_V.t(); // Transpose gradient to match dimensions of V
    
    // Update the matrix V
    V -= alpha * grad_V;
    
    // Update the matrix U using the new V
    U = X * V * Sigma_inv; // Reusing pre-computed Sigma_inv
    
    // Compute the new objective function value
    double f_new = arma::norm(X - U * Sigma * V.t(), "fro") + lambda * calculate_regularization(X, V, D);
    
    // Check for NaN in the objective function value
    if (!arma::is_finite(f_new)) {
      Rcpp::Rcout << "Non-finite value encountered at iteration " << iter << std::endl;
      break; // Exit the loop if a non-finite value is encountered
    }
    
    // Check if new objective function value is greater than the previous one
    if (f_new >= f) {
      Rcpp::Rcout << "Objective function increased at iteration " << iter << std::endl;
      break; // Exit the loop if the new value is not smaller than the old one
    }
    
    // Check for convergence
    if (std::abs(f_new - f) < tol) {
      Rcpp::Rcout << "Convergence reached at iteration " << iter << std::endl;
      break;
    }
    
    // Optionally print the progress
    if (iter % print_interval == 0) {
      Rcpp::Rcout << "Iteration " << iter << " Objective Function: " << f_new << std::endl;
    }
    
    // Update the objective function value only if the new one is smaller
    f = f_new;
  }
  
  return Rcpp::List::create(Rcpp::Named("U") = U,
                            Rcpp::Named("Sigma") = Sigma,
                            Rcpp::Named("V") = V);
}


