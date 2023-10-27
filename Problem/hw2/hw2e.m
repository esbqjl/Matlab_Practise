% EC 503 Learning from Data
% Homework 2
% by (fill in name)
%
% Nearest Neighbor Classifier
%
% Problem 2.5e

clc, clear;
rng('default');
defaultseed = rng;

fprintf("==== Loading data_mnist_train.mat\n");
load("data_mnist_train.mat");
fprintf("==== Loading data_mnist_test.mat\n");
load("data_mnist_test.mat");

% show test image
imshow(reshape(X_train(200,:), 28,28)')

% determine size of dataset
[Ntrain, dims] = size(X_train);
[Ntest, ~] = size(X_test);

% precompute components

% Note: To improve performance, we split our calculations into
% batches. A batch is defined as a set of operations to be computed
% at once. We split our data into batches to compute so that the 
% computer is not overloaded with a large matrix.
batch_size = 500;  % fit 4 GB of memory
num_batches = Ntest / batch_size;
ypred = zeros(Ntest,1);

% Using (x - y) * (x - y)' = x * x' + y * y' - 2 x * y'
for bn = 1:num_batches
  batch_start = 1 + (bn - 1) * batch_size;
  batch_stop = batch_start + batch_size - 1;
  % calculate cross term
  Xtest_batch = X_test(batch_start:batch_stop,:);

  % compute euclidean distance
  
  fprintf("==== Doing 1-NN classification for batch %d\n", bn);
  Xsq=sum(X_train.^2,2)';
  Ysq=sum(Xtest_batch.^2,2);
  XY=Xtest_batch*X_train';
  distances = Xsq+Ysq+2*XY;
  % find minimum distance for k = 1
  [min_value,min_indexes] = min(distances,[],2);
  ypred(batch_start:batch_stop) = Y_train(min_indexes);
end

% compute confusion matrix
conf_mat = confusionmat(Y_test,ypred)
% compute CCR from confusion matrix
ccr = sum(diag(conf_mat)) / Ntest;