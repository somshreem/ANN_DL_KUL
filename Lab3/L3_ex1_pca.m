data_set = load('threes.mat','-ascii');

%compute mean
mean_value = mean(data_set);

figure
plot(mean_value)
title('Mean')

%compute covariance matrix for whole data set

cov_matrix = cov(data_set);

%compute eigen vector and eigen values
[right_eigen_vec, eigen_val, left_eigen_vec] = eig(cov_matrix);

eigen_values = diag(eigen_val)';

figure
plot(diag(eigen_val))
title('Eigen Values')
grid on;

%compress dataset by projecting to 1,2,3,4 principle components and
%reconstruct images

[principle_component_vector, projection_on_pc_space, eigen_cov_matrix] = pca(data_set);


Compress_one_pc = projection_on_pc_space(:,1:1) * principle_component_vector(:,1:1)';
Reconstruct_one_pc = mean_value + Compress_one_pc; % add the mean back in

Compress_two_pc = projection_on_pc_space(:,1:2) * principle_component_vector(:,1:2)';
Reconstruct_two_pc = mean_value + Compress_two_pc; % add the mean back in

Compress_three_pc = projection_on_pc_space(:,1:3) * principle_component_vector(:,1:3)';
Reconstruct_three_pc = mean_value + Compress_three_pc; % add the mean back in

Compress_four_pc = projection_on_pc_space(:,1:4) * principle_component_vector(:,1:4)';
Reconstruct_four_pc = mean_value + Compress_four_pc; % add the mean back in


%comparison of original image and reconstructed image
figure
subplot(4,2,[1,2,3,4])
colormap('gray')
imagesc(reshape(data_set(1,:),16,16),[0,1]);
title('Original image')
subplot(4, 2, 5);
imagesc(reshape(Reconstruct_one_pc(1,:),16,16),[0,1]);
title('1 component')
subplot(4, 2, 6);
imagesc(reshape(Reconstruct_two_pc(1,:),16,16),[0,1]);
title('2 compontents')
subplot(4, 2, 7);
imagesc(reshape(Reconstruct_three_pc(1,:),16,16),[0,1]);
title('3 components')
subplot(4, 2, 8);
imagesc(reshape(Reconstruct_four_pc(1,:),16,16),[0,1]);
title('4 components')

%compress to 1-50 principle components, reconstruct and measure
%reconstruction error
reconstruction_error = [];
for i = 1:50
    Compress_pc = projection_on_pc_space(:,1:i) * principle_component_vector(:,1:i)';
    Reconstruct_pc = mean_value + Compress_pc; % add the mean back in
    reconstruction_error = [reconstruction_error, sqrt(mean(mean((data_set-Reconstruct_pc).^2)))];
end

figure
plot(reconstruction_error);
title('Reconstruction Error')
xlabel('Principle Components'); ylabel('Reconstruction Error'); grid on;


%for 256 principle components

Compress_256_pc = projection_on_pc_space(:,1:256) * principle_component_vector(:,1:256)';
Reconstruct_256_pc = mean_value + Compress_256_pc; % add the mean back in
reconstruction_error_256 = sqrt(mean(mean((data_set-Reconstruct_256_pc).^2)))

%cumulative sum of eigen values

eigen_vector_new = zeros(size((eigen_values)));

for i=1:50
    eigen_vector_new(end-i+1) = eigen_values(i);
end

cumulative_eigen_value= cumsum(eigen_vector_new); 

figure
plot(cumulative_eigen_value)
title('Cumulative Eigenvalues')
xlabel('Eigenvalues')
ylabel('Amount of information')
grid on;




figure
plot(1:50, reconstruction_error/reconstruction_error(1),'b');
hold on
plot(1:50, cumulative_eigen_value(1:50)/20.3776,'r');
grid on
legend('Reconstruction error', 'Eigenvalues information');
title('Error vs Eigenvalues information');
xlabel('Number of components')
ylabel('Relative magnitude of errors & eigenvalues information');
saveas(gcf,strcat('ErrorVSEigenvalues.jpg'));



