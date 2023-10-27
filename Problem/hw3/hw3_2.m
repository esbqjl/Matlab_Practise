% EC 503 - HW 3 - Fall 2023
% K-Means starter code

clear, clc, close all;
rng('default');
defaultseed = rng;

%% Generate Gaussian data:
% Add code below:
mu1 = [2,2]; mu2=[-2,2]; mu3 = [0,-3.25];
sigma1 = 0.2 * eye(2); sigma2 = 0.05* eye(2); sigma3 = 0.07 * eye(2);
cluster1 = mvnrnd(mu1,sigma1,50);
cluster2 = mvnrnd(mu2,sigma2,50);
cluster3 = mvnrnd(mu3,sigma3,50);
Gaussian_DATA = [cluster1;cluster2;cluster3];
% Scatter plot
figure;

hold on;
scatter(cluster1(:,1), cluster1(:,2), 'r');
scatter(cluster2(:,1), cluster2(:,2), 'g');
scatter(cluster3(:,1), cluster3(:,2), 'b');
hold off;
title('Generated Gaussian Data');
xlabel('x1');
ylabel('x2');

%% K-Means implementation
% Add code below
DATA = Gaussian_DATA;
K = 3
%% 3.2.a
MU_init = [3,3;-4,-1;2,-4];
%% 3.2.b
MU_init = [-0.14,2.61;3.15,-0.84;-3.28,-1.58];
MU_previous = MU_init;
MU_current = MU_init;

% initializations
labels = ones(length(DATA),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration);

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % While calculating the distance, use the same trick as the HW 2.5(e)
    % to avoid innner for loop.
    % Write code below here:
    distance = zeros(length(DATA),K);
    for k=1:K
        distance(:,k)=sum((DATA-MU_current(k,:)).^2,2);
    end
    [value,labels] = min(distance,[],2);
    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    for k= 1:K
        MU_current(k,:) = mean(DATA(labels==k,:),1);
    end
    %% CODE 4 - Check for convergence 
    % Write code below here:
    if (max(abs(MU_previous-MU_current))) < convergence_threshold

        converged=1;
    end
    MU_previous = MU_current;
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n');
        figure;
        
        scatter(cluster1(:,1), cluster1(:,2), 'r');
        hold on;
        scatter(cluster2(:,1), cluster2(:,2), 'g');
        scatter(cluster3(:,1), cluster3(:,2), 'b');
        scatter(MU_current(:,1), MU_current(:,2), 100, 'k*', 'LineWidth', 1.5);
        hold off;
        title('K-Means Clustering Results for Gaussian Data');
        xlabel('x1');
        ylabel('x2');
       
        
        %% If converged, get WCSS metric
        % Add code below
        WCSS = 0;  % Initialize the WCSS
        for i = 1:size(DATA, 1)  % Loop through each data point
            for j = 1:size(DATA, 2)  % Loop through each dimension
                WCSS = WCSS + (DATA(i, j) - MU_current(labels(i), j))^2;
            end
        end
        fprintf('WCSS: %d\n',WCSS);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end
%% 3.2(c)
num_inits = 10; % Number of random initializations
K = 3; % Number of clusters


best_WCSS = Inf; % Initialize the best WCSS to a large value
WCSS_values = zeros(num_inits, 1); % Store WCSS values for each initialization

for init = 1:num_inits
    % Random the initialization 
    min_data = min(DATA, [], 1);
    max_data = max(DATA, [], 1);
    MU_init= repmat(min_data,K,1) + rand(K,2) .* (repmat(max_data - min_data, K, 1));

    MU_current = MU_init;
    disp(MU_current)
    MU_previous = MU_init;
    labels = ones(length(DATA),1);
    converged = 0;

    while (converged==0)

        distance = zeros(length(DATA),K);
        for k=1:K
            distance(:,k)=sum((DATA-MU_current(k,:)).^2,2);
        end
        [value,labels] = min(distance,[],2);
        for k= 1:K
            MU_current(k,:) = mean(DATA(labels==k,:),1);
        end

       if max(abs(MU_previous-MU_current)) < convergence_threshold

            converged=1;
       end
       MU_previous = MU_current;
    end

    % Calculate WCSS for this initialization

    WCSS = 0;  % Initialize the WCSS
    for i = 1:size(DATA, 1)  % Loop through each data point
        for j = 1:size(DATA, 2)  % Loop through each dimension
            WCSS = WCSS + (DATA(i, j) - MU_current(labels(i), j))^2;
        end
    end

    WCSS_values(init) = WCSS;
    % Check if this is the best initialization so far
    if WCSS < best_WCSS
        best_WCSS = WCSS;
        best_MU = MU_current;
        best_labels = labels;
    end
end

% Plot the best clustering results
figure;
hold on;
scatter(cluster1(:,1), cluster1(:,2), 'r');
scatter(cluster2(:,1), cluster2(:,2), 'g');
scatter(cluster3(:,1), cluster3(:,2), 'b');
scatter(best_MU(:,1), best_MU(:,2), 100, 'k*', 'LineWidth', 1.5);
hold off;
title(sprintf('Best K-Means Clustering Result (WCSS = %.2f)', best_WCSS));
xlabel('x1');
ylabel('x2');

% Display all WCSS values
disp('WCSS values for each initialization:');
disp(WCSS_values);
%% 3.2(d)
rng('shuffle');
DATA = Gaussian_DATA;
K_range = [2, 3, 4, 5, 6, 7, 8, 9, 10];
WCSS_values = zeros(length(K_range), 1) % Store WCSS values for each initialization
for K_index=1:length(K_range)
    num_inits = 10; % Number of random initializations
    K = K_range(K_index); % Number of clusters
    best_WCSS = Inf; % Initialize the best WCSS to a large value
    for init = 1:num_inits
        % Random the initialization 
        min_data = min(DATA, [], 1);
        max_data = max(DATA, [], 1);
        MU_init= repmat(min_data,K,1) + rand(K,2) .* (repmat(max_data - min_data, K, 1));
    
        MU_current = MU_init;
        MU_previous = MU_init;
        labels = ones(length(Gaussian_DATA),1);
        converged = 0;
    
        while (converged==0)
    
            distance = zeros(length(DATA),K);
            for k=1:K
                distance(:,k)=sum((DATA-MU_current(k,:)).^2,2);
            end
            [value,labels] = min(distance,[],2);
            for k= 1:K
                MU_current(k,:) = mean(DATA(labels==k,:),1);
            end
    
           if max(abs(MU_previous-MU_current)) < convergence_threshold
    
                converged=1;
           end
           MU_previous = MU_current;
        end
    
        % Calculate WCSS for this initialization
    
        WCSS = 0;  % Initialize the WCSS
        for i = 1:size(DATA, 1)  % Loop through each data point
            for j = 1:size(DATA, 2)  % Loop through each dimension
                WCSS = WCSS + (DATA(i, j) - MU_current(labels(i), j))^2;
            end
        end
        % Check if this is the best initialization so far
        if WCSS < best_WCSS
            best_WCSS = WCSS;
            best_MU = MU_current;
            best_labels = labels;
        end
    end
    
    WCSS_values(K_index)=best_WCSS;
    % Display all WCSS values
    fprintf('WCSS values is %d when k = %d\n',best_WCSS,K);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
figure;
plot(K_range, WCSS_values, '-o', 'LineWidth', 2);
title('WCSS values for different k');
xlabel('Number of clusters (k)');
ylabel('WCSS');
grid on;



%% 3.2(e)
%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here
NBA_data_raw = readmatrix('NBA_stats_2018_2019.xlsx');
NBA_data = NBA_data_raw(:, 2:end);  
NBA_data = (NBA_data - mean(NBA_data))./std(NBA_data); 
PPG = NBA_data(:, 7); 
MPG = NBA_data(:, 5); 
DATA = [PPG MPG];

num_inits = 10; % Number of random initializations
K = 10; % Number of clusters


best_WCSS = Inf; % Initialize the best WCSS to a large value
WCSS_values = zeros(num_inits, 1); % Store WCSS values for each initialization

for init = 1:num_inits
    % Random the initialization 
    min_data = min(DATA, [], 1);
    max_data = max(DATA, [], 1);
    MU_init= repmat(min_data,K,1) + rand(K,2) .* (repmat(max_data - min_data, K, 1));

    MU_current = MU_init;
    disp(MU_current)
    MU_previous = MU_init;
    labels = ones(length(DATA),1);
    converged = 0;

    while (converged==0)

        distance = zeros(length(DATA),K);
        for k=1:K
            distance(:,k)=sum((DATA-MU_current(k,:)).^2,2);
        end
        [value,labels] = min(distance,[],2);
        for k= 1:K
            MU_current(k,:) = mean(DATA(labels==k,:),1);
        end

       if max(abs(MU_previous-MU_current)) < convergence_threshold

            converged=1;
       end
       MU_previous = MU_current;
    end

    % Calculate WCSS for this initialization

    WCSS = 0;  % Initialize the WCSS
    for i = 1:size(DATA, 1)  % Loop through each data point
        for j = 1:size(DATA, 2)  % Loop through each dimension
            WCSS = WCSS + (DATA(i, j) - MU_current(labels(i), j))^2;
        end
    end

    WCSS_values(init) = WCSS;
    % Check if this is the best initialization so far
    if WCSS < best_WCSS
        best_WCSS = WCSS;
        best_MU = MU_current;
        best_labels = labels;
    end
end

% Plot the best clustering results
figure;
hold on;
scatter(DATA(:,1), DATA(:,2), 25, 'b');
scatter(best_MU(:,1), best_MU(:,2), 100, 'k*', 'LineWidth', 1.5);
hold off;
title(sprintf('Best K-Means Clustering Result (WCSS = %.2f)', best_WCSS));
xlabel('x1');
ylabel('x2');

% Display all WCSS values
disp('WCSS values for each initialization:');
disp(WCSS_values);
a=sample_circle(3);
function [data ,label] = sample_circle( num_cluster, points_per_cluster )
% Function to sample 2-D circle-shaped clusters
% Input:
% num_cluster: the number of clusters 
% points_per_cluster: a vector of [num_cluster] numbers, each specify the
% number of points in each cluster 
% Output:
% data: sampled data points. Each row is a data point;
% label: ground truth label for each data points.
%
% EC 503: Learning from Data
% Fall 2022
% Instructor: Prakash Ishwar
% HW 3, Problem 3.2(f) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin == 0
  num_cluster = 2;
  points_per_cluster = 500*ones(num_cluster,1);
end
if nargin == 1
   points_per_cluster = 500*ones(num_cluster,1);
end
points_per_cluster=points_per_cluster(:);

data = zeros([sum(points_per_cluster), 2]);
label = zeros(sum(points_per_cluster),1);
idx = 1;
bandwidth = 0.1;

for k = 1 : num_cluster
    theta = 2 * pi * rand(points_per_cluster(k), 1);
    rho = k + randn(points_per_cluster(k), 1) * bandwidth;
    [x, y] = pol2cart(theta, rho);
    data(idx:idx+points_per_cluster(k)-1,:) = [x, y];
    label(idx:idx+points_per_cluster(k)-1)=k;
    idx = idx + points_per_cluster(k);
end
DATA = data;

num_inits = 10; % Number of random initializations
K = 10; % Number of clusters
labels = ones(length(DATA),1);
converged = 0;

convergence_threshold = 0.025;

best_WCSS = Inf; % Initialize the best WCSS to a large value
WCSS_values = zeros(num_inits, 1); % Store WCSS values for each initialization

for init = 1:num_inits
    % Random the initialization 
    min_data = min(DATA, [], 1);
    max_data = max(DATA, [], 1);
    MU_init= repmat(min_data,K,1) + rand(K,2) .* (repmat(max_data - min_data, K, 1));

    MU_current = MU_init;
    disp(MU_current)
    MU_previous = MU_init;
    labels = ones(length(DATA),1);
    converged = 0;

    while (converged==0)

        distance = zeros(length(DATA),K);
        for k=1:K
            distance(:,k)=sum((DATA-MU_current(k,:)).^2,2);
        end
        [value,labels] = min(distance,[],2);
        for k= 1:K
            MU_current(k,:) = mean(DATA(labels==k,:),1);
        end

       if max(abs(MU_previous-MU_current)) < convergence_threshold

            converged=1;
       end
       MU_previous = MU_current;
    end

    % Calculate WCSS for this initialization

    WCSS = 0;  % Initialize the WCSS
    for i = 1:size(DATA, 1)  % Loop through each data point
        for j = 1:size(DATA, 2)  % Loop through each dimension
            WCSS = WCSS + (DATA(i, j) - MU_current(labels(i), j))^2;
        end
    end

    WCSS_values(init) = WCSS;
    % Check if this is the best initialization so far
    if WCSS < best_WCSS
        best_WCSS = WCSS;
        best_MU = MU_current;
        best_labels = labels;
    end
end

% Plot the best clustering results
figure;
hold on;
scatter(DATA(:,1), DATA(:,2), 25, 'b');
scatter(best_MU(:,1), best_MU(:,2), 100, 'k*', 'LineWidth', 1.5);
hold off;
title(sprintf('Best K-Means Clustering Result (WCSS = %.2f)', best_WCSS));
xlabel('x1');
ylabel('x2');

% Display all WCSS values
disp('WCSS values for each initialization:');
disp(WCSS_values);
end
