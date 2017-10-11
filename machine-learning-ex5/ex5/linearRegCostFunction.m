function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
errors = X * theta - y;
theta_no_theta0 = theta(2:end, :);
J = 1 / (2 * m) * (errors' * errors) ...
     + lambda / (2 * m) * (theta_no_theta0' * theta_no_theta0);

grad_regular = lambda / m * theta;
grad_regular(1) = 0;
grad = 1 / m * X' * errors + grad_regular;










% =========================================================================

grad = grad(:);

end
