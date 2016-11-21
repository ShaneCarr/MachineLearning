function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
% num labels is the number of of classifiers (the numbers 1-10 we are trying to match)
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 500);

for c = 1:num_labels
  %t is a handle of the objective function http://stackoverflow.com/questions/30280903/one-vs-all-regression
  % y == c sets the classifier (0-9) == C makes Y look like 00100 etc if c=2
  % each column of theta is going to be a trained set of theta for that letter or character.
  % initial_theta is initialized to some random values. I am not exactly sure how they are picked but it looked like
  % zero, bigger and smaller. 
  %lambda is a constant.
%  options= optimset('GradObj', 'on', 'MaxIter', '100'); % define the options data structure
%initial_theta= zeros(2,1); # set the initial dimensions for theta % initialize the theta values 
% this is initialized to N + 1, 1 note sigmoid(X*Theta) we distribute those across x.
%[optTheta, funtionVal, exitFlag]= fminunc(@costFunction, initialTheta, options); % run the algorithm
%need a theta for all the x plus the constant x0 term.
%explanation of these functions http://www.holehouse.org/mlclass/06_Logistic_Regression.html

%to get the gradient we take the derative and walk down until it is minimized by some algorithm.
% (y == c): set the 'classifier' we care about
  [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);

  all_theta(c, :) = theta';

end






% =========================================================================


end
