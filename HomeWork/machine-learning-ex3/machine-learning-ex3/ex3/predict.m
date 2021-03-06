function p = predict(Theta1, Theta2, X)

%PREDICT Predict the label of an input given a trained neural network

%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the

%   trained weights of a neural network (Theta1, Theta2)



% Useful values

m = size(X, 1);

num_labels = size(Theta2, 1);



% You need to return the following variables correctly 

p = zeros(size(X, 1), 1);


%this is a little confusing because they are already "Trained" we are just building the network.
% take the first level and plug it into the second level.

% G(Z(x)) = sigmoid Z(x) X*Theta'  (reversed with transpose to make column vector.
a1 = sigmoid([ones(m, 1) X] * Theta1');


%each a2 has a prediction for each letter or whatever so you 'd see 1x10 if you run ex3_nn
% We pass in A one (as we did with X for the second layer.
a2 = sigmoid([ones(m, 1) a1] * Theta2');

% returns the value that is max and the dimension. ( in one case for example if 4 is the biggest then thta index would be returned).
[dummy, p] = max(a2, [], 2);


% =========================================================================





end