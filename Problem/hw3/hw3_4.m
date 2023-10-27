% EC 503 - HW 3 - Fall 2023
% DP-Means starter code

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
DATA = Gaussian_DATA;
%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here
NBA_data_raw = readmatrix('NBA_stats_2018_2019.xlsx');
NBA_data = NBA_data_raw(:, 2:end); 
PPG = NBA_data(:, 7); 
MPG = NBA_data(:, 5); 
DATA = [PPG MPG];
%% DP Means method:

% Parameter Initializations
LAMBDA_VALUES = [0.15, 0.4, 3, 20];  % For Gaussian data
LAMBDA_VALUES = [44,100,450]
convergence_threshold = 1;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DP Means - Initializations for algorithm %%%
% cluster count
for lambda_index = 1: length(LAMBDA_VALUES)
    LAMDA = LAMBDA_VALUES(lambda_index);
    num_points = length(DATA);
    total_indices = [1:num_points];
    K = 1;
    
    % sets of points that make up clusters
    L = {};
    L = [L total_indices];
    
    % Class indicators/labels
    Z = ones(1,num_points);
    
    % means
    MU = [];
    MU = [MU; mean(DATA,1)];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Initializations for algorithm:
    converged = 0;
    t = 0;
    while (converged == 0)
        t = t + 1;
        fprintf('Current iteration: %d...\n',t)
        
        %% Per Data Point:
        for i = 1:num_points
            
            %% CODE 1 - Calculate distance from current point to all currently existing clusters
            % Write code below here:
            distances = [];
            for j=1:K
                distances = [distances,norm(DATA(i,:)-MU(j,:))^2];
            end
            %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
            % Write code below here:
            [min_distance, index] = min(distances);
    
            if min_distance > LAMDA
                K = K + 1;
                MU = [MU; DATA(i,:)]; % set new cluster mean at this data point
                Z(i) = K;
                L{K} = [i]; % create new cluster
            else
                Z(i) = index;
                L{index} = [L{index}, i];
            end
        end
        
        %% CODE 3 - Form new sets of points (clusters)
        % Write code below here:
        for j=1:K
            L{j}= find(Z==j);
        end
        %% CODE 4 - Recompute means per cluster
        % Write code below here:
        new_MU = [];
        for j=1:K
            cluster_data = DATA(L{j},:);
            new_MU = [new_MU;mean(cluster_data,1)]
        end
        %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same %%%
        % Write code below here:
        if size(new_MU,1)==size(MU,1)&&max(max(abs(new_MU-MU)))<convergence_threshold
            converged =1;
        end
        %% CODE 6 - Plot final clusters after convergence 
        % Write code below here:
        
        if (converged)
            figure;
            hold on;
            colors = 'rgbmkcy'; % add more colors if needed
            for j = 1:K
                scatter(DATA(Z == j, 1), DATA(Z == j, 2), 50, colors(mod(j, length(colors)) + 1), 'filled');
            end
            hold off;
            title(['DP-means Clustering with Lambda = ', num2str(LAMDA)]);
            xlabel('x1');
            ylabel('x2');
        end
        MU = new_MU;
    end
end



