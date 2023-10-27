%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503 (Ishwar) Fall 2023
% HW 5.2
% Wenjun Zhang wjz@bu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% You can directly run the following function without calling it to get results

function [lambda_top5, k_] = skeleton_hw5_2()
%% Q5.2
%% Load AT&T Face dataset, unzip the zip file in the running folder
    img_size = [112,92];   % image size (rows,columns)
    % Load the AT&T Face data set using load_faces()
    %%%%% TODO
    faces = load_faces();
    %% Compute mean face and the covariance matrix of faces
    % compute X_tilde1
    %%%%% TODO
    mean_face = mean(faces,1);
    X_tilde = faces- mean_face;
    % Compute covariance matrix using X_tilde
    %%%%% TODO
    S_x = cov(X_tilde);
    %% Compute the eigenvalue decomposition of the covariance matrix
    %%%%% TODO
    [U, Lamda] = eig(S_x)
    %% Sort the eigenvalues and their corresponding eigenvectors construct the U and Lambda matrices
    %%%%% TODO
    [eigenvalues, order]=sort(diag(Lamda),'descend');
    U = U(:,order);
    %% Compute the principal components: Y
    %%%%% TODO
    Y = X_tilde * U;
%% Q5.2 a) Visualize the loaded images and the mean face image
    figure(1)
    sgtitle('Data Visualization')
    
    % Visualize image number 120 in the dataset
    % practice using subplots for later parts
    subplot(1,2,1)
    %%%%% TODO
    imshow(uint8(reshape(faces(120,:),img_size)));
    title("Faces");
    % Visualize the mean face image
    subplot(1,2,2)
    %%%%% TODO
    imshow(uint8(reshape(mean_face,img_size)));
    title("Mean Face");
%% Q5.2 b) Analysing computed eigenvalues
    warning('off')
    d = 450;
    % Report the top 5 eigenvalues
    % lambda_top5 = ?; %%%%% TODO
    lamda_top5 = eigenvalues(1:5)
    
    % Plot the eigenvalues in from largest to smallest
    k = 1:d;
    figure(2)
    sgtitle('Eigenvalues from largest to smallest')

    % Plot the eigenvalue number k against k
    subplot(1,2,1)
    %%%%% TODO
    plot(k,eigenvalues(k));
    title('Eigenvalues');
    xlabel('Order');
    ylabel('Value');
    % Plot the sum of top k eigenvalues, expressed as a fraction of the sum of all eigenvalues, against k
    %%%%% TODO: Compute eigen fractions
    eigen_fraction = cumsum(eigenvalues) /sum(eigenvalues);
    subplot(1,2,2)
    plot(k,eigen_fraction(k));
    title('Eigenvalues Fraction');
    xlabel('Order');
    ylabel('Cumulative Fraction');
    %%%%% TODO
    
    % find & report k for which the eigen fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
    ef = [0.51, 0.75, 0.9, 0.95, 0.99];
    
    %%%%% TODO (Hint: ismember())
    % k_ = ?; %%%%% TODO
    k_ = find(ismember(round(eigen_fraction,2),ef))
%% Q5.2 c) Approximating an image using eigen faces
    test_img_idx = 43;
    test_img = faces(test_img_idx,:);    
    % Compute eigenface coefficients
    %%%% TODO
    img_minus_mean = test_img - mean_face;

    Y_PCA = U' * img_minus_mean';
    
    % add eigen faces weighted by eigen face coefficients to the mean face
    % for each K value
    % 0 corresponds to adding nothing to the mean face

    % visulize and plot in a single figure using subplots the resulating image approximations obtained by adding eigen faces to the mean face.

    %%%% TODO 
    % 2,6,29,105,179,300 are the samllest k value from each k_ eigen fraction
    K = [0,1,2,6,29,105,179,300,400,d];
    recon_imgs = zeros(length(K),size(test_img,2));
    for i=1:length(K)
        k=K(i);
        approx = mean_face'+U(:,1:k)*Y_PCA(1:k);
        recon_imgs(i,:) = approx';
    end
    figure(3)
    sgtitle('Approximating original image by adding eigen faces')
    for i=1:length(K)
        subplot(1,length(K)+1,i)
        imshow(uint8(reshape(recon_imgs(i,:),img_size)));
        title(['k=',num2str(K(i))]);
    end
    subplot(i,length(K)+1,length(K)+1);
    imshow(uint8(reshape(test_img,img_size)));
    title('Original Image');

%% Q5.2 d) Principal components capture different image characteristics
%% Loading and pre-processing MNIST Data-set
     % Data Prameters
    q = 5;                  % number of percentile points
    noi = 3;                % Number of interest
    img_size = [16, 16];
    
    % load mnist into workspace
    mnist = load('mnist256.mat').mnist;
    label = mnist(:,1);
    X = mnist(:,(2:end));
    num_idx = (label == noi);
    X = X(num_idx,:);
    [n,~] = size(X);
    
    %% Compute the mean face and the covariance matrix
    % compute X_tilde
    %%%%% TODO
    mean_img = mean(X,1);
    X_tilde = X - mean_img;
    % Compute covariance using X_tilde
    %%%%% TODO
    S_x = cov(X_tilde);
    %% Compute the eigenvalue decomposition
    %%%%% TODO
    [U, Lamda] = eig(S_x)
    %% Sort the eigenvalues and their corresponding eigenvectors in the order of decreasing eigenvalues.
    %%%%% TODO
    [eigenvalues, order]=sort(diag(Lamda),'descend');
    U = U(:,order);
    %% Compute principal components
    %%%%% TODO
    Y = X_tilde * U;
    %% Computing the first 2 pricipal components
    %%%%% TODO
    pc1 = Y(:,1);
    pc2 = Y(:,2);
    % finding percentile points
    percentile_vals = [5, 25, 50, 75, 95];
    %%%%% TODO (Hint: Use the provided fucntion - percentile_points())
    pc1_percentile_vals = percentile_values(pc1,percentile_vals);
    pc2_percentile_vals = percentile_values(pc2,percentile_vals);
    % Finding the cartesian product of percentile points to find grid corners
    %%%%% TODO
    [pc1_mesh, pc2_mesh] = meshgrid(pc1_percentile_vals, pc2_percentile_vals);
    grid_corners = [pc1_mesh(:) pc2_mesh(:)];

    
    %% Find images whose PCA coordinates are closest to the grid coordinates 
    
    %%%%% TODO
    Closest_points =zeros(size(grid_corners,1),2);
    closest_images_indices = zeros(size(grid_corners,1),1);
    for i = 1:size(grid_corners,1)
        [val,closest_index] = min(vecnorm(Y(:,1:2)-grid_corners(i,:),2,2));
        
        closest_points(i,:) = Y(closest_index,1:2);
        closest_images_indices(i)=closest_index;
    end
    
    %% Visualize loaded images
    % random image in dataset
    figure(4)
    sgtitle('Data Visualization')

    % Visualize the 120th image
    subplot(1,2,1)
    %%%%% TODO
    imshow(reshape(X(120,:),img_size));
    title("Image #120");
    % Mean face image
    subplot(1,2,2)
    %%%%% TODO
    imshow(reshape(mean_img,img_size));
    title("Mean Image")
    %% Image projections onto principal components and their corresponding features
    
    figure(5)    
    hold on
    grid on
    
    % Plotting the principal component 1 vs principal component 2. Draw the
    % grid formed by the percentile points and highlight the image points that are closest to the 
    % percentile grid corners
    
    %%%%% TODO (hint: Use xticks and yticks)
    scatter(Y(:,1),Y(:,2),'bo');
    scatter(Closest_points(:,1),Closest_points(:,2));
    xticks(pc1_percentile_vals);
    yticks(pc2_percentile_vals);
    xlabel('Principal component 1')
    ylabel('Principal component 2')
    title('Image points closest to percentile grid corners')
    hold off
    
    figure(6)
    sgtitle('Images closest to percentile grid corners')
    hold on
    % Plot the images whose PCA coordinates are closest to the percentile grid 
    % corners. Use subplot to put all images in a single figure in a grid.
    
    %%%%% TODO
  
    for i=1:length(closest_images_indices)
        subplot(size(pc1_percentile_vals,1),size(pc2_percentile_vals,1),i);
        imshow(reshape(X(closest_images_indices(i),:),img_size));
    end
    hold off    
end

function [faces] = load_faces()    
    % Data Parameters
    num_people = 40;       % # People in dataset
    num_img_pp = 10;       % # images per person in each subdirectory
    img_fmt = '.pgm';      % image format (portable grayscale map)
    img_size = [112,92];   % image size (rows,columns)
  
    % Load data from directory into workspace
    faces = zeros(num_people*num_img_pp,prod(img_size));
    for person = 1:num_people
        for img = 1:num_img_pp
            img_path = strcat('./att-database-of-faces/s',num2str(person), ...
                        '/', num2str(img), img_fmt);
            person_img = imread(img_path);
            faces((person-1)*num_img_pp+img,:) = person_img(:)'; 
        end
    end
end
function [percentile_values] = percentile_values(v, percentiles)
    % assumes v as a column vector
    % percentiles is an array of percentiles 
    % percentile_values is an array of percentile-values corresponding
    % to percentiles 
    [n, ~] = size(v);
    [sorted_v,~] = sort(v, 'ascend');
    percentile_indices = ceil(n*percentiles/100);
    percentile_values = sorted_v(percentile_indices,:);
end