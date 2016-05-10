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

% X = [ones(m, 1) X];

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Activation layer calculations.  Also known as h(x)

% Activation layer 1 = X input layer
a_1 = [ones(m, 1) X];

% Activation layer 2
a_2 = sigmoid(a_1*Theta1');
a_2_size = size(a_2, 1);
a_2 = [ones(a_2_size, 1) a_2];

% Activation layer 3
a_3 = sigmoid(a_2*Theta2');
h = a_3;

total_sum = 0;

y_matrix = eye(num_labels)(y,:); % 5000x10 matrix

% Failed solutions:

% for i = 1:num_labels
% 	% total_sum += sum( (log(h) * (-1 * y_value==)) - (log(1-h)*(1-y_value)) );
	% total_sum += sum( (( -(y==i)')*log(h)) - ((1-(y==i)')*log(1-h)) ) ;
	% total_sum += sum( (( -(y_matrix(:,i))')*log(h)) - ((1-(y_matrix(:,i))')*log(1-h)) ) ;

	% total_sum += sum( (( -(y_matrix(:,i))).*log(h)) - ((1-(y_matrix(:,i))).*log(1-h)) ) ;
% endfor
% J = 1/m*sum(total_sum)
% J = 1/m * sum(sum(-y_matrix'*log(h) - (1-y_matrix')*(log(1-h))))
	
% -y_matrix.*log(h) - (1-y_matrix).*log(1-h)

% Important, we are performing element wise multiplication 
% Y-matrix contains vectorized matrix of single values
% When doing element wise multiplication, all other columns but colum set to 1s
% will be set to 0 afer multiplying

%
%  Non-regularized cost function:
%
% J = 1/m * sum(sum(-y_matrix.*log(h) - (1-y_matrix).*log(1-h)));


% Part 1
% Weight regularization parameter
% Theta1(:,2:end) removes first column because this is the bias vector
weight_reg_param = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
% Main cost function
J = 1/m * sum(sum(-y_matrix.*log(h) - (1-y_matrix).*log(1-h))) + weight_reg_param;


% J = 1/m * sum(-y' * log(h) - (1-y')*log(1-h)) + lambda/(2*m) * sum( (theta(2:end).^2) );

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
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
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
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));

for t = 1:m
%
% Let:
% m = the number of training examples
% n = the number of training features, including the initial bias unit.
% h = the number of units in the hidden layer - NOT including the bias unit
% r = the number of output classifications
%
	%
	% Step 1
	% activation layer 1 = input X
	a_1 = X(t,:);
	a_1_size = size(a_1, 1);
	a_1 = [ones(a_1_size,1) a_1];

	% Calculate activation layer 2
	z_2 = a_1*Theta1';
	a_2 = sigmoid(z_2);
	a_2_size = size(a_2, 1);
	a_2 = [ones(a_2_size,1) a_2];

	z_3 = a_2*Theta2';
	a_3 = sigmoid(z_3); % h(x)

	% Step 2, output layer -- revisit?
	d_3 = a_3-y_matrix(t,:);

	% Step 3, hidden layer 
	% (m x r) * (r x h) = m x h    
	% (1 x 10) * (10 x 25) = (1 x 25)
	d_2 = (d_3*Theta2(:,2:end)) .* sigmoidGradient(z_2);

	% Step 4, accumulating gradient
	% (h x m) * (m x n) = (h x n)
	% Delta_1 = a_1' * d_2(:,2:end);  
	% Delta_2 = a_2' * d_3(:,2:end);
	Delta_1 += (a_1' * d_2)'; % (401 x 1 * 1 x 25) = (401 x 25)
	Delta_2 += (a_2' * d_3)'; % (26 x 1 * 1 x 10) = (26 x 10) 
endfor

% Step 5
% (h x n)

% Regularization parameter is lambda/m * ....
% Remove first column of Theta and replace with 0s, so that when multiplied, it will not affect
%  the first parameter
Theta1_grad = 1/m * Delta_1 + lambda/m*[zeros(size(Delta_1,1),1), Theta1(:,2:end)];
Theta2_grad = 1/m * Delta_2 + lambda/m*[zeros(size(Delta_2,1),1), Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
