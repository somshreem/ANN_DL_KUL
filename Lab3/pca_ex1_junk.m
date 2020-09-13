X = load('threes.mat','-ascii');
[w, pc, ev] = pca(X);

mu = mean(X);
xhat = X - mu; % subtract the mean
norm(pc * w' - xhat)
plot(mu)
title('mean')

ev;

Xapprox1 = pc(:,1:1) * w(:,1:1)';
Xapprox1 = mu + Xapprox1; % add the mean back in

Xapprox2 = pc(:,1:2) * w(:,1:2)';
Xapprox2 = mu + Xapprox2; % add the mean back in

Xapprox3 = pc(:,1:3) * w(:,1:3)';
Xapprox3 = mu + Xapprox3; % add the mean back in

Xapprox4 = pc(:,1:4) * w(:,1:4)';
Xapprox4 = mu + Xapprox4; % add the mean back in



figure
colormap('gray')
imagesc(reshape(X(1,:),16,16),[0,1]);
title('Original image')

figure
colormap('gray')
subplot(2, 2, 1);
imagesc(reshape(Xapprox1(1,:),16,16),[0,1]);
title('1 component')
subplot(2, 2, 2);
imagesc(reshape(Xapprox2(1,:),16,16),[0,1]);
title('2 compontents')
subplot(2, 2, 3);
imagesc(reshape(Xapprox3(1,:),16,16),[0,1]);
title('3 components')
subplot(2, 2, 4);
imagesc(reshape(Xapprox4(1,:),16,16),[0,1]);
title('4 components')

figure
plot(100*ev/sum(ev));
xlabel('Column'); ylabel('%'); grid on;

[v, d] = eig(cov(X));
figure
plot(diag(d));
title('Eigenvalues')

error = [];
for i = 1:50
    Xapprox = pc(:,1:i) * w(:,1:i)';
    Xapprox = mu + Xapprox; % add the mean back in
    error = [error, sqrt(mean(mean((X-Xapprox).^2)))];
end

figure
plot(error);
title('Error')
xlabel('Components'); ylabel('Error'); grid on;


Xapprox256 = pc(:,1:256) * w(:,1:256)';
Xapprox256 = mu + Xapprox256; % add the mean back in
sqrt(mean(mean((X-Xapprox).^2)))