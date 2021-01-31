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


% Size:
% X = 12x2 (already padded)
% y = 12x1
% theta = 2x1
% lambda = scalar

% Padding column for X is added outside of function

% Calculate Cost function
J = (1/(2*m))*sum((X*theta-y).**2)+(lambda/(2*m))*(sum(theta(2:end,:).**2));

% temp_theta will be used in the regularization term with temp_theta(1) set to zero
% theta will be used to calculate the errors between prediction and actual y values
temp_theta = theta;
temp_theta(1) = 0;

grad = (1/m)*(X'*(X*theta-y))+(lambda/m)*temp_theta;

% alternative way to compute gradient with regularization:
% grad(1) = (1/m)*(X(:,1)'*(X*theta-y));
% grad(2:end) = (1/m)*(X(:,2:end)'*(X*theta-y))+(lambda/m)*temp_theta(2:end);

% =========================================================================

grad = grad(:);

end
