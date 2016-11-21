function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================
   %for each row in theta there are two in this case but could be more.
   
   thetaTemp = theta;
   hypothesis = X * theta;
   currentRowError = (hypothesis - y);
 
   % there was some proof of this.
   currentRowErrorT = currentRowError';
   
  % for each theta  
  % i think this can be paralleized more.
  for t = 1:size(theta)(:,1)
      % printf("\%s\n", "Size Theta");
      % size(theta)(:,1);

      #take the right element
      xColumnCurrent = X(:,t); 
      yDiffsX = currentRowErrorT * xColumnCurrent;
      
      %new theta note this should approach zero if we make small steps. 
      %make a note to study later derative again. But rate of change slows to zero at a minimum.
      thetaTemp(t) = theta(t) - (alpha * (1/m) * sum(yDiffsX));
      
    end
    
    theta = thetaTemp
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    J_history(iter)
end

end
