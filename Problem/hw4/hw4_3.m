%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503 (Ishwar) Fall 2023
% HW 4.3
% <Your full name and BU email> Wenjun Zhang, wjz@bu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;
rng('default')  % For reproducibility of data and results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(a)
% Generate and plot the data points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1 = 50;
n2 = 100;
mu1 = [1; 2];
mu2 = [3; 2];

% Generate dataset (i) 

lambda1 = 1;
lambda2 = 0.25;
theta = 0*pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% See below for function two_2D_Gaussians which you need to complete.

% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure;
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(0),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code with suitable modifications here to create and plot 
% datasets (ii), (iii), and (iv)
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (ii)
lambda1 = 1;
lambda2 = 0.25;
theta = 1*pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% See below for function two_2D_Gaussians which you need to complete.

% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure;
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (iii)
lambda1 = 1;
lambda2 = 0.25;
theta = 2*pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% See below for function two_2D_Gaussians which you need to complete.

% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure;
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(2),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (iv)
lambda1 = 0.25;
lambda2 = 1;
theta = 1*pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% See below for function two_2D_Gaussians which you need to complete.

% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure;
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(b)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each phi = 0 to pi in steps of pi/48 compute the signal power, noise 
% power, and snr along direction phi and plot them against phi 

phi_array = 0:pi/48:pi;
signal_power_array = zeros(1,length(phi_array));
noise_power_array = zeros(1,length(phi_array));
snr_array = zeros(1,length(phi_array));
for i=1:1:length(phi_array)
    [signal, noise, snr] = signal_noise_snr(X, Y, phi_array(i), false);
    % See below for function signal_noise_snr which you need to complete.
    signal_power_array(i) = signal;
    noise_power_array(i) = noise;
    snr_array(i) = snr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to create plots of signal power versus phi, noise
% power versus phi, and snr versus phi and to locate the values of phi
% where the signal power is maximized, the noise power is minimized, and
% the snr is maximized
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
plot(phi_array,signal_power_array,'-b','LineWidth',2);
hold on;
plot(phi_array,noise_power_array,'-r','LineWidth',2);
plot(phi_array,snr_array,'-g','LineWidth',2);
legend('Signal Power', 'Noise Power', 'SNR');
xlabel('Direction \phi (radians)');
ylabel('Value');
title('Signal Power, Noise Power, and SNR vs. \phi');
hold off;

% Determine values of phi for the required conditions
[~, idx_max_signal] = max(signal_power_array);
[~, idx_min_noise] = min(noise_power_array);
[~, idx_max_snr] = max(snr_array);

phi_max_signal = phi_array(idx_max_signal);
phi_min_noise = phi_array(idx_min_noise);
phi_max_snr = phi_array(idx_max_snr);

fprintf('Phi which maximizes squared distance: %.4f radians\n', phi_max_signal);
fprintf('Phi which minimizes average within-class variance: %.4f radians\n', phi_min_noise);
fprintf('Phi which maximizes the SNR: %.4f radians\n', phi_max_snr);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For phi = 0, pi/6, and pi/3, generate plots of estimated class 1 and 
% class 2 densities of the projections of the feature vectors along 
% direction phi. To do this, set phi to the desired value, set 
% want_class_density_plots = true; 
% and then invoke the function: 
% signal_noise_snr(X, Y, phi, want_class_density_plots);
% Insert your script here 
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
phi_values = [0, pi/6, pi/3];
for i=1:length(phi_values)
    signal_noise_snr(X, Y, phi_values(i), true);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(c)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the LDA solution by writing and invoking a function named LDA 

w_LDA = LDA(X,Y);

% See below for the LDA function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to create a scatter plot and overlay the LDA vector and the 
% difference between the class means. Use can use Matlab's quiver function 
% to do this.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
scatter(X(1,Y==1), X(2,Y==1), 'o', 'fill', 'b');
hold on;
scatter(X(1,Y==2), X(2,Y==2), '^', 'fill', 'r');
quiver(mean(X(1,:)), mean(X(2,:)), w_LDA(1), w_LDA(2), 'k', 'LineWidth', 2, 'MaxHeadSize', 8);
quiver(mean(X(1,:)), mean(X(2,:)), mu1(1) - mu2(1), mu1(2) - mu2(2), 'g', 'LineWidth', 2, 'MaxHeadSize', 2);
legend('Class 1', 'Class 2', 'LDA Direction', 'Difference of Means');
xlabel('x_1');
ylabel('x_2');
title('Scatter plot with LDA direction and Mean Difference');
axis equal;
grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(d)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create CCR vs b plot
n = length(X);
X_project = w_LDA' * X;
X_project_sorted = sort(X_project);
b_array = X_project_sorted * (diag(ones(1,n))+ diag(ones(1,n-1),-1)) / 2;
b_array = b_array(1:(n-1));
ccr_array = zeros(1,n-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise: decode what the last 6 lines of code are doing and why
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:(n-1)
    ccr_array(i) = compute_ccr(X, Y, w_LDA, b_array(i));
end

% See below for the compute_ccr function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to plote CCR as a function of b and determine the value of b
% which maximizes the CCR.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
plot(b_array, ccr_array, 'LineWidth', 2);
xlabel('b value');
ylabel('CCR');
title('CCR vs. b');
[~, index_max_ccr] = max(ccr_array);
b_optimal = b_array(index_max_ccr);
hold on;
plot(b_optimal, ccr_array(index_max_ccr), 'ro'); % Highlight the max CCR
legend('CCR', 'Max CCR');
fprintf('Value of b that maximizes CCR: %.4f\n', b_optimal);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complete the following 4 functions defined below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [X, Y] = two_2D_Gaussians(n1,n2,mu1,mu2,lambda1,lambda2,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function should generate a labeled dataset of 2D data points drawn 
% independently from 2 Gaussian distributions with the same covariance 
% matrix but different mean vectors
%
% Inputs:
%
% n1 = number of class 1 examples
% n2 = number of class 2 examples
% mu1 = 2 by 1 class 1 mean vector
% mu2 = 2 by 1 class 2 mean vector
% theta = orientation of eigenvectors of common 2 by 2 covariance matrix shared by both classes
% lambda1 = first eigenvalue of common 2 by 2 covariance matrix shared by both classes
% lambda2 = second eigenvalue of common 2 by 2 covariance matrix shared by both classes
% 
% Outputs:
%
% X = a 2 by (n1 + n2) matrix with first n1 columns containing class 1
% feature vectors and the last n2 columns containing class 2 feature
% vectors
%
% Y = a 1 by (n1 + n2) matrix with the first n1 values equal to 1 and the 
% last n2 values equal to 2


%%%%%%%%%%%%%%%%%%%%%%
%Insert your code here
%%%%%%%%%%%%%%%%%%%%%%
    U =[cos(theta),sin(theta);sin(theta),-cos(theta)];
    Lambda = [lambda1,0;0,lambda2];
    Sigma = U * Lambda * U';
    % generate random data points
    D1 = mvnrnd(mu1,Sigma,n1)';
    D2 = mvnrnd(mu2,Sigma,n2)';
    % construct X and Y matrixs
    X = [D1,D2];
    Y = [ones(1,n1),2*ones(1,n2)];
end

function [signal, noise, snr] = signal_noise_snr(X, Y, phi, want_class_density_plots)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to project data along direction phi and then comput the
% resulting signal power, noise power, and snr 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To generate plots of estimated class 1 and class 2 densities of the 
% projections of the feature vectors along direction phi, set:
% want_class_density_plots = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    X1 = X(:,Y==1);
    X2 = X(:,Y==2);
    % Project data on that direction phi
    
    direction = [cos(phi);sin(phi)];
    X_proj_phi_1 = X1' * direction;
    X_proj_phi_2 = X2' * direction;
    % get the means for both classes on that direction
    mean_phi_1 = mean(X_proj_phi_1);
    mean_phi_2 = mean(X_proj_phi_2);
    % get the variances for both classes
    var_phi_1 = var(X_proj_phi_1);
    var_phi_2 = var(X_proj_phi_2);
    % compute signal and snr
    signal = (mean_phi_1 - mean_phi_2)^2;
    n_samples = length(Y);
    noise = (length(Y(Y==1))/n_samples)* var_phi_1+(length(Y(Y==2))/n_samples)* var_phi_2;
    snr = signal/noise;
if want_class_density_plots == true
    % Plot density estimates for both classes along chosen direction phi
    figure();
    [pdf1,z1] = ksdensity(X_proj_phi_1);
    plot(z1,pdf1)
    hold on;
    [pdf2,z2] = ksdensity(X_proj_phi_2);
    plot(z2,pdf2)
    grid on;
    hold off;
    legend('Class 1', 'Class 2')
    xlabel('projected value')
    ylabel('density estimate')
    title("Estimated class density estimates of data projected along \phi ="+ phi + "\times \pi/6. Ground-truth \phi = \pi/6")
end

end

function w_LDA = LDA(X, Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to compute and return the LDA solution
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    X1 = X(:,Y==1);
    X2 = X(:,Y==2);
    mu1 = mean(X1,2);
    mu2 = mean(X2,2);

    Sw = cov(X1')* (size(X1,2)-1) + cov(X2')*(size(X2,2)-1)

    % compare w_LDA
    w_LDA = Sw \ (mu1-mu2);
end

function ccr = compute_ccr(X, Y, w_LDA, b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to compute the CCR for the given labeled dataset
% (X,Y) when you classify the feature vectors in X using w_LDA and b
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    predictions = ((w_LDA' * X + b)<=0) +1
    
    classified = sum(predictions ==Y);
    ccr = classified/length(Y);
end