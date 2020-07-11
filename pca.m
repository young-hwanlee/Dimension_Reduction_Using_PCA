% close all;	% clear all existing figures
clc;        % clear the command window
clear all;  % clear all objects in the workspace

%% ---------------------------------------------------------------------------

% read the face images (N=1924)
% imgs = dir(fullfile('/Users/young-hwanlee/OneDrive/Young-hwan Lee/2) 2015 Fall Semester/ENEE 620 Probability and Random Processes/Project1/faces','*.jpg'));
imgs = dir(fullfile('/Users/young-hwanlee/Desktop/Young-hwan Lee/2) 2015 Fall Semester/ENEE 620 Probability and Random Processes/Projects/Project 1/faces','*.jpg'));
num_imgs = size(imgs,1);
for i=1:num_imgs
    single_face = imread(imgs(i).name);     % read each single face (168x168)
%     imshow(single_face);
%     pause;
    single_face = double(single_face);

    % form 28224*1924 dimensional data matrix X by vectorizing each image
    % & concatenating them as columns
    faces(:,i) = reshape(single_face,[size(single_face,1)*size(single_face,2),1]);
    centralized_faces(:,i) = faces(:,i)-mean(faces(:,i));
end

% imshow('Aaron_Eckhart_0001.jpg');
% imshow(faces);
% imshow(imgs(1).name);
% imshow(reshape(faces(:,1),[size(single_face,1),size(single_face,2)]));

%% ---------------------------------------------------------------------------

% perform the principal component analysis (PCA) using singular value decomposition (SVD)
% [U0,S0,V0] = svd(double(faces),'econ');
[U,S,V] = svd(centralized_faces);
figure(1);
plot(diag(S),'LineWidth',2);
grid;
xlabel('n'); ylabel('Singular values');

% display a number of 168x168-dimensional eigenvectors as images
for i=1:num_imgs
    eigenvectors(:,:,i) = reshape(U(:,i),[sqrt(size(U,1)),sqrt(size(U,1))]);
%     eig_vec(:,:,i) = eigenvectors(:,:,i)+mean(mean(eigenvectors(:,:,i)));
end

% Min = min(min(min(eigenvectors)));
% Max = max(max(max(eigenvectors)));
% figure(2);
% for i=1:8
%     subplot(2,4,i); imagesc(eigenvectors(:,:,i),[Min, Max]);
% end

figure(2);
for i=1:8
    subplot(2,4,i); imagesc(eigenvectors(:,:,i)); axis square;
end
subplot(2,4,2); title('Eigenfaces corresponding to                ');
subplot(2,4,3); title('                8 eigenvalues with higher values');

% figure(3);
% for i=1:8
%     subplot(2,4,i); imagesc(eigenvectors(:,:,i+8));
% end

%% ---------------------------------------------------------------------------

% select a number of sample face images from the database
for i=1:num_imgs
    origin(:,:,i) = reshape(centralized_faces(:,i),[sqrt(size(U,1)),sqrt(size(U,1))]);
end

% show lower dimensional representations using K eigenimages with higher values (< rank N)
% K = 5
A_hat_K5 = zeros(size(faces));
for K=1:5
    A_hat_K5 = A_hat_K5 + S(K,K)*U(:,K)*V(:,K)';
end
for i=1:num_imgs
    A_K5(:,:,i) = reshape(A_hat_K5(:,i),[sqrt(size(U,1)),sqrt(size(U,1))]);
end

% K = 30
A_hat_K30 = zeros(size(faces));
for K=1:30
    A_hat_K30 = A_hat_K30 + S(K,K)*U(:,K)*V(:,K)';
end
for i=1:num_imgs
    A_K30(:,:,i) = reshape(A_hat_K30(:,i),[sqrt(size(U,1)),sqrt(size(U,1))]);
end

% K = 100
A_hat_K100 = zeros(size(faces));
for K=1:100
    A_hat_K100 = A_hat_K100 + S(K,K)*U(:,K)*V(:,K)';
end
for i=1:num_imgs
    A_K100(:,:,i) = reshape(A_hat_K100(:,i),[sqrt(size(U,1)),sqrt(size(U,1))]);
end

% K = 600
A_hat_K600 = zeros(size(faces));
for K=1:600
    A_hat_K600 = A_hat_K600 + S(K,K)*U(:,K)*V(:,K)';
end
for i=1:num_imgs
    A_K600(:,:,i) = reshape(A_hat_K600(:,i),[sqrt(size(U,1)),sqrt(size(U,1))]);
end

figure(3);
for i=1:3
    subplot(5,3,i); imagesc(origin(:,:,i)); axis square;
end
for i=1:3
    subplot(5,3,i+3); imagesc(A_K5(:,:,i)); axis square;
end
for i=1:3
    subplot(5,3,i+6); imagesc(A_K30(:,:,i)); axis square;
end
for i=1:3
    subplot(5,3,i+9); imagesc(A_K100(:,:,i)); axis square;
end
for i=1:3
    subplot(5,3,i+12); imagesc(A_K600(:,:,i)); axis square;
end
subplot(5,3,2); title('Original faces');
subplot(5,3,5); title('K = 5');
subplot(5,3,8); title('K = 30');
subplot(5,3,11); title('K = 100');
subplot(5,3,14); title('K = 600');

%% ---------------------------------------------------------------------------

figure(4);
for i=1:8
    subplot(2,4,i); imagesc(eigenvectors(:,:,i+1916)); axis square;
end
subplot(2,4,2); title('Eigenfaces corresponding to                ');
subplot(2,4,3); title('                8 eigenvalues with lower values');

% show lower dimensional representations using K eigenimages with lower values (< rank N)
% K = 600
B_hat_K600 = zeros(size(faces));
for K=1:600
    B_hat_K600 = B_hat_K600 + S(K+1324,K+1324)*U(:,K+1324)*V(:,K+1324)';
end
for i=1:num_imgs
    B_K600(:,:,i) = reshape(B_hat_K600(:,i),[sqrt(size(U,1)),sqrt(size(U,1))]);
end

% K = 1000
B_hat_K1000 = zeros(size(faces));
for K=1:1000
    B_hat_K1000 = B_hat_K1000 + S(K+924,K+924)*U(:,K+924)*V(:,K+924)';
end
for i=1:num_imgs
    B_K1000(:,:,i) = reshape(B_hat_K1000(:,i),[sqrt(size(U,1)),sqrt(size(U,1))]);
end

% K = 1500
B_hat_K1500 = zeros(size(faces));
for K=1:1500
    B_hat_K1500 = B_hat_K1500 + S(K+424,K+424)*U(:,K+424)*V(:,K+424)';
end
for i=1:num_imgs
    B_K1500(:,:,i) = reshape(B_hat_K1500(:,i),[sqrt(size(U,1)),sqrt(size(U,1))]);
end

figure(5);
for i=1:3
    subplot(4,3,i); imagesc(origin(:,:,i)); axis square;
end
for i=1:3
    subplot(4,3,i+3); imagesc(B_K600(:,:,i)); axis square;
end
for i=1:3
    subplot(4,3,i+6); imagesc(B_K1000(:,:,i)); axis square;
end
for i=1:3
    subplot(4,3,i+9); imagesc(B_K1500(:,:,i)); axis square;
end
subplot(4,3,2); title('Original faces');
subplot(4,3,5); title('K = 600');
subplot(4,3,8); title('K = 1000');
subplot(4,3,11); title('K = 1500');


