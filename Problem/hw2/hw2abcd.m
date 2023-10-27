% EC 503 Learning from Data
% Homework 2
% by Wenjun Zhang
%
% Nearest Neighbor Classifier
%
% Problem 2.5 a, b, c, d


clc, clear;
rng('default');
defaultseed = rng;

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

%% a) Plotting
% include a scatter plot
% MATLAB function: gscatter()
figure;
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain,'rgb','o',5);
grid on;
% label axis and include title
xlabel('Feature1');
ylabel('Feature2');
title('scatter graph of class 1 2 3');

%% b)Plotting Probabilities on a 2D map
K = 10;
% specify grid
[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
Xtest = [Xgrid(:),Ygrid(:)];
[Ntest,dim]=size(Xtest);

% compute probabilities of being in class 2 for each point on grid
probabilities = zeros(Ntest,1);
%for class 2
for i =1:Ntest
    test_point = Xtest(i,:)
    %find the Euclidean distance from this test point to all training point
    distances = sqrt(sum((Xtrain-test_point).^2,2));
    [values,indexes]=sort(distances);
    Knearest = indexes(1:K);
    probabilities(i) = sum(ytrain(Knearest) == 2)/K;
    
end
% Figure for class 2
figure
class2ProbonGrid = reshape(probabilities,size(Xgrid));
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('Feature1')
ylabel('Feature2')
title('Probability of being in class 2')


% repeat steps above for class 3 below

% compute probabilities of being in class 3 for each point on grid
probabilities = zeros(Ntest, 1);
%for class 3
for i =1:Ntest
    %find the Euclidean distance from this test point to all training point
    test_point = Xtest(i,:)
    distances = sqrt(sum((Xtrain-test_point).^2,2));
    
    [values,indexes]=sort(distances);
    Knearest = indexes(1:K);
    probabilities(i) = sum(ytrain(Knearest) == 3)/K;
end
% Figure for class 2
figure
class3ProbonGrid = reshape(probabilities,size(Xgrid));
contourf(Xgrid,Ygrid,class3ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('Feature1')
ylabel('Feature2')
title('Probability of being in class 3')
%% c) Class label predictions
K = 1 ; % K = 1 case

% compute predictions 
ypred = zeros(Ntest,1)
for i=1:Ntest 
    test_point=Xtest(i,:);
    distances = sqrt(sum((Xtrain-test_point).^2,2))
    [values,indexes]=min(distances);
    ypred(i)=ytrain(indexes);
end
figure
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
grid on;
xlim([-3.5,6]);
ylim([-3,6.5]);
% remember to include title and labels!
xlabel('Feature 1')
ylabel('Feature 2')
title('k=1 kNN classification')

% repeat steps above for the K=5 case. Include code for this below.

K = 5 ; % K = 5 case

% compute predictions 
ypred = zeros(Ntest,1)
for i=1:Ntest 
    test_point=Xtest(i,:);
    distances = sqrt(sum((Xtrain-test_point).^2,2))
    [values,indexes]=sort(distances);
    Knearest = indexes(1:K);
    % using majority vote to predict the label
    ypred(i)=mode(ytrain(Knearest));
end
figure
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
grid on;
xlim([-3.5,6]);
ylim([-3,6.5]);
% remember to include title and labels!
xlabel('Feature 1')
ylabel('Feature 2')
title('k=5 kNN classification')
%% d) LOOCV CCR computations

for k = 1:2:11
    % determine leave-one-out predictions for k
    ypred = zeros(Ntrain,1);
    for i=1:Ntrain
        X_LOOCV_train=Xtrain;
        X_LOOCV_train(i,:) = [];
        y_LOOCV_train=ytrain;
        y_LOOCV_train(i)=[];
    
        test_point = Xtrain(i,:);
    
        distances = sqrt(sum((X_LOOCV_train-test_point).^2,2));
        [values,indexes] = sort(distances);
        Knearest = y_LOOCV_train(indexes(1:k));
        ypred(i)=mode(Knearest);
    end
    % compute confusion matrix
    conf_mat = confusionmat(ytrain, ypred);
    % from confusion matrix, compute CCR
    CCR = sum(diag(conf_mat)) / Ntrain;
    
    % below is logic for collecting CCRs into one vector
    if k == 1
        CCR_values = CCR;
    else
        CCR_values = [CCR_values, CCR];
    end
end

% plot CCR values for k = 1,3,5,7,9,11
% label x/y axes and include title
plot(1:2:11, CCR_values, '-o');
xlabel('k');
ylabel('CCR');
title('Correct Classification Rate (CCR) as a function of k');
grid on;