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
a_2_size = size(a_2, 1)
a_2 = [ones(a_2_size, 1) a_2];
% a_2 = [ones(m, 1) a_2];

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
J = 1/m * sum(sum(-y_matrix.*log(h) - (1-y_matrix).*log(1-h)));



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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
