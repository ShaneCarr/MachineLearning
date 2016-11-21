function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Fill out J_vals
%for i = 1:length(theta0_vals)
%    for j = 1:length(theta1_vals)
	  %t = [theta0_vals(i); theta1_vals(j)];    
	  %J_vals(i,j) = computeCost(X, y, t);
    %end
%end

%Generate Predictoin
% each row of X is a sample
% we use constants theta which is generated for us
% we multiply those constants by X 
%building the first part of teh equations
predictionY = X * theta; 

% Get the error 1,ActualInstance * theta0(i) theta1(i)
predictiveError = (predictionY - y);

%square each element; 
predictiveErrorSquared = predictiveError.^2; 

%equation provided
costOfPrediction = 1/(2*m) * sum(predictiveErrorSquared);
J = costOfPrediction;

end
