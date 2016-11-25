function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

%these guys accumulate the errors the big delta.
BigDeltaTheta1Gradient = zeros(size(Theta1));
BigDeltaTheta2Gradient= zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         BigDeltaTheta1Gradient and BigDeltaTheta2Gradient. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in BigDeltaTheta1Gradient and
%         BigDeltaTheta2Gradient, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to BigDeltaTheta1Gradient
%               and BigDeltaTheta2Gradientfrom Part 2.
%


%% Part 1 implementation forward propigation.

% add bias unit
% X is 5k X 400 ones(m,1) is 5kX1
% so a1 is 5K x 401
a1 = [ones(m, 1) X];

% intermediate value
% Theta is already set up with bias unit. 
% a1 -> 5000 X 401
% this is the input

% z2 -> 5000 X 25 
% this is input when theta' is applied
% theta1 -> 25 X 401 we take the tranpose though!)
z2 = a1 * Theta1';

% get the a2 our 
% 5000 X 25
a2 = sigmoid(z2);

% 5000 X 26
%add bias unit the one columns
a2 = [ones(size(a2,1), 1) a2];

% 5000 X 10
z3 = a2 * Theta2';

% 5000 x 10
a3 = sigmoid(z3);
% hx -> 5000 X 10
hx = a3;
% 5000 X 10
yRowVec = zeros(m,num_labels);

%create a vectory for each y
% set of row vectors each column i szero except 1 where yval indexes into column set taht to 1.
for i = 1:m
    yRowVec(i,y(i)) = 1;
end
% j 1x 1
J = 1/m * sum(sum(-1 * yRowVec .* log(hx)-(1-yRowVec) .* log(1-hx)));
regularator = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));

J = J + regularator;

%% Part 2 implementation
for t = 1:m

  %create row vector and make it a bitmask with this y.
  % 1 : 10 == 5 0000010000
	TrainingSetExampleExpectedResultY = ([1:num_labels]==y(t))';
  
	% For the input layer, where l=1:
	% For the hidden layers
  %a1 -> 401 x1
  a1 = [1; X(t,:)'];
	
  % z2: -> 25 x 1 /theta 1: 25 x 401 
  z2 = Theta1 * a1;
  
  % a2 : 26 x 1
	a2 = [1; sigmoid(z2)];
	
  % 10 x 1
  z3 = Theta2 * a2;
	
  % 10 X 1
  a3 = sigmoid(z3);

  % d3: 10 X 1
   %a3 -> 10X1, a2 -> 26x1, a1 -> 401 X 1; TrainingSetExampleExpectedResultY -> 10 X1; note the transpose.  
	% athree is the prediction for this example t. subtract the expected result TrainingSetExampleExpectedResultY.
  % to get our error. that is intuitive. row vector since a3 is a row vector.
	d3 = a3 - TrainingSetExampleExpectedResultY;
  %d2: 25 X 1
  % THE .* Is becaues we are applying the derative to the cost but not transforming anything.
  % we are taking the porportion of the error and applying it here.
	d2 = (Theta2' * d3) .* [1; sigmoidGradient(z2)];
	d2 = d2(2:end); % Taking off the bias column

	% d1 is not calculated because there is no relationship between the error 
  % and the original input value. X is just x. 
	% accumulates the errors for each training example.
	BigDeltaTheta1Gradient = BigDeltaTheta1Gradient + d2 * a1';
	BigDeltaTheta2Gradient= BigDeltaTheta2Gradient+ d3 * a2';
end
    % The error for theta.
    % BigDeltaTheta2Gradient -> 10 X 26; BigDeltaTheta1Gradient -> 25 x 401                                         % add zeros to make it compatible for number of rows remove teh bias unit
BigDeltaTheta1Gradient = (1/m) * BigDeltaTheta1Gradient + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
BigDeltaTheta2Gradient = (1/m) * BigDeltaTheta2Gradient + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients 10285 x 1
grad = [BigDeltaTheta1Gradient(:) ; BigDeltaTheta2Gradient(:)];


end
