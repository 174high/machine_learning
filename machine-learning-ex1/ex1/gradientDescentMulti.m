function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
        %       of the cost function (computeCostMulti) and gradient here.
        %

        % we store theta afterwards because these operations are acted upon concurrently := to update the vector

        % store_theta = [];
        % for feature = 1:size(X,2)
        %     theta(feature) = theta(feature) - (alpha/m) * sum( ((X*theta)-y).*X(:,feature) );
        % end

        % 1/m*alpha*sum( ((X*theta)-y)*X);
        % theta = store_theta;

        % ============================================================

        % Save the cost J in every iteration    
        % http://stackoverflow.com/questions/32274474/machine-learning-linear-regression-using-batch-gradient-descent
        theta = theta - alpha * 1/m * (X')*(X*theta-y);
        J_history(iter) = computeCostMulti(X, y, theta);
    end

end


% Theta computed from gradient descent: 
%  340412.659574 
%  110631.050279 
%  -6649.474271 