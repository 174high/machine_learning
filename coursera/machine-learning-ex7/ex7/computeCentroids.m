function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
for k = 1:K
	% cent_assign_mat = idx == k; % centroid assignment matrix.  Get all elements assigned to 1,2,3
	% C_k = sum(cent_assign_mat); % count set of examples assigned to centroid K
	% data_assigned_matrice = X .* (repmat(cent_assign_mat),1,n);
	% centroids(k) = sum(data_assigned_matrice) / C_k;


	% simpler:
	% i_m = idx==k contains a vector of 1s and 0s.  1s where the value equals the cur cluster
	% 1/C_k = 1/(sum(i_m))
	% repmat(i_m,1,n) repeats the first column matrix n times
	% Zero out other clusters not assigned to cur cluster with X .* ( repmat(i_m,1,n) )
	% Finally, sum the non zero entries, and divide by C_k
	i_m = idx==k;
	centroids(k,:) = sum(X .* ( repmat(i_m,1,n) ))/(sum(i_m));
endfor







% =============================================================


end

