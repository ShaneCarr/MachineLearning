function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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
hx = sigmoid( X * theta);


% J(theta) = 1/m*sum((-y_i)*log(h(x_i)-(1-y_i)*log(1-h(x_i))))+(lambda/2*m)*sum(theta_j)

%eliminate theta0 that doesn' t factor into the regularization
regularizationTheta = [];

% for i = 2:length(theta)
 %   regularizationTheta(jj) = theta(i)^2;
% end

%or  this is faster
regularizationTheta = theta(2:end).^2;

thetaRegularizer = (lambda/(2*m) ) *sum(regularizationTheta);

J = 1./m * ( -y' * log(hx) - ( 1 - y' ) * log ( 1 - hx) ) + thetaRegularizer;


% grad = 1./m * X' * (hx - y);
% there is no regularation term for zero so we zero that dimension out. 
core = ((hx - y)' * X / m)' ;
grad = core + lambda .* theta .* [0; ones(length(theta)-1, 1)] ./ m ;

% =============================================================

end
