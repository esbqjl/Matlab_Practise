%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503 (Ishwar) Fall 2023
% HW 4.4
% <Your full name and BU email> Wenjun wjz@bu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You are always welcome to vectorize your loop!

clear, clc, close all,

%% 4.4 a) Normalization of data

% load data
load prostateStnd.mat
disp('Original Mean of Features:');
disp(mean(Xtrain));
disp('Original Variance of Features:');
disp(var(Xtrain));
disp('Original Mean of Labels:');
disp(mean(ytrain));
disp('Original Variance of Labels:');
disp(var(ytrain));
% obtain mean and stds
mean_vec = mean(Xtrain);
std_vec  = std(Xtrain);
mean_ytrain = mean(ytrain);
std_ytrain = std(ytrain)
% Normalize data:
ytrain_normalized = ((ytrain-mean_ytrain)/std_ytrain);
ytest_normalized  = (ytest-mean_ytrain)/std_ytrain;

Xtrain_normalized = (Xtrain - mean_vec) ./ std_vec;
Xtest_normalized = (Xtest - mean_vec) ./ std_vec;

%% 4.4 b) Train ridge regression model for various lambda (regularization parameter) values

xx = [-5:10];
lambda_vec = exp(xx);
coefficient_mat = zeros(size(Xtrain,2),length(lambda_vec));

disp('Iterating through different lambdas...')
for i = 1:length(lambda_vec)
    fprintf('Progress: %d/%d...\n',i,length(lambda_vec))
    lambda = lambda_vec(i);
    B = ridge(ytrain_normalized,Xtrain_normalized,lambda,0);
    coefficient_mat(:,i) = B(2:end)

end
disp('Done.')


%% 4.4 c) Plotting ridge regression coefficients

figure, grid on; hold on; xlabel('ln(lambda)'), ylabel('Feature coefficient values'); %title('Feature coefficent values for various regularization amounts')
colors = lines(size(coefficient_mat,1));
for i = 1:size(coefficient_mat,1)
    plot(xx,coefficient_mat(i,:),'color',colors(i,:),'DisplayName',names{i});

end
legend('Location','Best');

%% 4.4 d) Plotting MSE values as function of ln(lambda)

train_MSE = zeros(1,length(lambda_vec));
test_MSE  = zeros(1,length(lambda_vec));

for i =1:length(lambda_vec)
    predicted_train = [ones(size(Xtrain_normalized,1),1) Xtrain_normalized] * [0;coefficient_mat(:,i)];
    predicted_test = [ones(size(Xtest_normalized,1),1) Xtest_normalized] * [0;coefficient_mat(:,i)];
    
    train_MSE(i) = mean((predicted_train - ytrain_normalized).^2);
    test_MSE(i) = mean((predicted_test-ytest_normalized).^2);
end

figure, grid on; hold on; xlabel('ln(lambda)'), ylabel('MSE');
plot(xx, train_MSE, '-o', 'DisplayName', 'Training MSE');
plot(xx, test_MSE, '-x', 'DisplayName', 'Testing MSE');
legend('Location', 'Best');






