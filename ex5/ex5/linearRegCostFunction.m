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

hx =  X * theta;

errorTerm = sum((hx - y).^2);

%regularizationTheta = regularator = (sum(sum(theta(:,2:end).^2)) * (lambda/(2*m));
regularizationThetaForNonZero = [0 ; theta(2:end, :)];

costRegularizer = lambda * (regularizationThetaForNonZero' * regularizationThetaForNonZero);

% don't regularize the zero bias unit
J = 1./ (2 * m) *(errorTerm + costRegularizer )
  
 delta = X*theta - y;
%% each X' is Xj i.e. foreach J foreach M training example sume these together and divide by M. X[j] * errorTerm return vector 
grad = 1/m *(X' * delta + (lambda * regularizationThetaForNonZero));


% =========================================================================

grad = grad(:);

end
