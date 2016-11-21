function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
hx = sigmoid( X * theta);


% J(theta) = 1/m*sum((-y_i)*log(h(x_i)-(1-y_i)*log(1-h(x_i))))+(lambda/2*m)*sum(theta_j)

%eliminate theta0 that doesn' t factor into the regularization
regularizationTheta = [];

% for i = 2:length(theta)
 %   regularizationTheta(jj) = theta(i)^2;
% end

%or  this is faster, note we square regulartion term in cost function, leaving out the first term 
regularizationTheta = theta(2:end).^2;

thetaRegularizer = (lambda/(2*m) ) *sum(regularizationTheta);

J = 1./m * ( -y' * log(hx) - ( 1 - y' ) * log ( 1 - hx) ) + thetaRegularizer;


% vectorizing 
% this is the equation hnaded to us in the class
% grad = 1./m * X' * (hx - y);    // note  aT b = bT a
% there is no regularation term for zero so we zero that dimension out. 
% also it isn't squared can either do ((hx - y)' * X  or X' * ((hx - y)' 
% core = ((X' * (hx - y) * / m)' ;
core = ((hx - y)' * X / m)' ;
grad = core + lambda .* theta .* [0; ones(length(theta)-1, 1)] ./ m ;


% =============================================================

grad = grad(:);

end
