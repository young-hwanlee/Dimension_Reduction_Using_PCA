% close all;	% close all existing figures
clc;        % clear the command window
clear all;  % clear all objects in the workspace


% read the face images (N=1924)
imgs = dir(fullfile('path to data','*.jpg'));
num_imgs = size(imgs,1);
for i=1:num_imgs
    single_face = imread(imgs(i).name);     % read each single face (168x168)
    single_face = double(single_face);

    % form 28224*1924 dimensional data matrix X by vectorizing each image
    % & concatenating them as columns
    faces(:,i) = reshape(single_face,[size(single_face,1)*size(single_face,2),1]);
    centralized_faces(:,i) = faces(:,i)-mean(faces(:,i));
end

% perform the principal component analysis (PCA) using singular value decomposition (SVD)
[U,S,V] = svd(centralized_faces);

figure(1);
plot(diag(S),'LineWidth',2);
grid;
xlabel('K'); ylabel('Singular values');

% display a number of 168x168-dimensional eigenvectors as images
for i=1:num_imgs
    eigenvectors(:,:,i) = reshape(U(:,i),[sqrt(size(U,1)),sqrt(size(U,1))]);
end

figure(2);
for i=1:10
    subplot(2,5,i); imagesc(eigenvectors(:,:,i)); axis square;
end
subplot(2,5,2); title('Eigenfaces corresponding to');
subplot(2,5,3); title('                                                10 eigenvalues with higher values');

% select a number of sample face images from the database
for i=1:num_imgs
    origin(:,:,i) = reshape(centralized_faces(:,i),[sqrt(size(U,1)),sqrt(size(U,1))]);
end

% show lower dimensional representations using K eigenimages with higher values (< rank N)
figure(3);
for i=1:5
    subplot(4,5,i); imagesc(origin(:,:,i)); axis square;
end
for i=1:5
    subplot(4,5,i+5); imagesc(A_K50(:,:,i)); axis square;
end
for i=1:5
    subplot(4,5,i+10); imagesc(A_K100(:,:,i)); axis square;
end
for i=1:5
    subplot(4,5,i+15); imagesc(A_K500(:,:,i)); axis square;
end
subplot(4,5,3); title('Original faces');
subplot(4,5,8); title('K = 50');
subplot(4,5,13); title('K = 100');
subplot(4,5,18); title('K = 500');


