function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% =========================================================================

% create vector of all the places where y is 1
trainingSetRowOrdinalPCategory = find(y==1);

% create vectory of where y is 0
trainingSetRowOrdinalNCategory = find(y == 0);

% Plot Examples 

%plot the xy (the two features really in the X VECTOR) cord position is the "postiive ordinals, and the 1,2 is the x or y. This si a little confusing because the vector is X
xFeature1 = X(trainingSetRowOrdinalPCategory, 1);
xFeature2 = X(trainingSetRowOrdinalPCategory, 2);
plot(xFeature1, xFeature2, 'k+','LineWidth', 2, 'MarkerSize', 7);


%plot the xy cord position is the "n ordinals, and the 1,2 is the x or y. This si a little confusing because the vector is X
xnfeature1 = X(trainingSetRowOrdinalNCategory, 1);
xnFeature2 = X(trainingSetRowOrdinalNCategory, 2);

plot(xnfeature1, xnFeature2, 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

hold off;

end
