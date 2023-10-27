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