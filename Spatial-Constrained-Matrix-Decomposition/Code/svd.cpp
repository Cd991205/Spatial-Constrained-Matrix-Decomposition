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


// Gradient Descent Algorithm for Matrix Factorization with Regularization
// [[Rcpp::export]]
Rcpp::List gradient_descent_svd(const arma::mat& X, arma::mat U, arma::mat Sigma, arma::mat V, 
                                const arma::mat& D, double lambda, double alpha, double tol, int max_iter) {
  arma::mat I_U = arma::eye<arma::mat>(U.n_rows, U.n_rows);
  arma::mat I_V = arma::eye<arma::mat>(V.n_cols, V.n_cols);
  double f = arma::norm(X - U * Sigma * V.t(), "fro") + lambda * calculate_regularization(X, V, D);
  
  for (int iter = 0; iter < max_iter; ++iter) {
    // Compute the gradients based on provided derivatives
    arma::mat grad_U = -2 * Sigma * V.t() * X.t() + 2 * Sigma * V.t() * V * Sigma.t() * U.t();
    arma::mat grad_V = -2 * Sigma.t() * U.t() * X + 2 * Sigma.t() * U.t() * U * Sigma * V.t() + 2 * lambda * V.t() * X.t() * D.t() * D * X;
    
    // Transpose gradients to match dimensions of U and V
    grad_U = grad_U.t();
    grad_V = grad_V.t();
    
    // Update the matrices U and V
    U -= alpha * grad_U;
    V -= alpha * grad_V;
    
    // Compute the new objective function value
    double f_new = arma::norm(X - U * Sigma * V.t(), "fro") + lambda * calculate_regularization(X, V, D);
    
    // Check for convergence
    if (std::abs(f_new - f) < tol) {
      Rcpp::Rcout << "Convergence reached at iteration " << iter << std::endl;
      break;
    }
    
    f = f_new; // Update the objective function value
  }
  
  return Rcpp::List::create(Rcpp::Named("U") = U,
                            Rcpp::Named("Sigma") = Sigma,
                            Rcpp::Named("V") = V);
}


