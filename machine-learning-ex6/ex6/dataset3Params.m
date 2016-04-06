function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

steps = [.01, .03, .1, .3, 1, 3, 10, 30];

all_time_low = 1000;
for c = 1:columns(steps)         
    for s = 1:columns(steps)   
        % debug:
        % printf('%d, %d = %f, %f\n', i, j, steps(i), steps(j) )

        % attempt to remove inverse later
        x_1 = X(:,1)';
        x_2 = X(:,2)';
        model = svmTrain(X, y, steps(c), @(x_1, x_2) gaussianKernel(x_1, x_2, steps(s))); 
        prediction = svmPredict(model, Xval);

        current_low = mean(double(prediction ~= yval));
        if (current_low < all_time_low)
            all_time_low = current_low;
            C = steps(c);
            sigma = steps(s);
        endif
    endfor
endfor








% =========================================================================

end
