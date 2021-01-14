clear; close all; clc
pat = 'C:\Users\ceskild\Documents\fromHdrive\Publications\000_In_Preparation\underfit_PLS\matlab';
cd(pat);

load data

% Calculate true reg. vec and loadings, NIPALS
[B,P1true,~,~,~,~] = nipals_pls1(X1,y1,K);
btrue = B(:,K);
[~,P2true,~,~,~,~] = nipals_pls1(X2,y2,K);
Proj = btrue*inv(btrue'*btrue)*btrue'; % projection matrix for Net Analyte Signal, NAS
clear B

% add noise to test data
Xtest = Xtest+Ex_test;
ytest = ytest+ey_test;

% %% Start iterations
% preallocation
MSE1 = nan(K,It);       % test set prediction error, model 1
MSE2 = nan(K,It);       % test set prediction error, model 2
MSEP1 = nan(K,It);      % error between estimated and true loadings, model 1
MSEP2 = nan(K,It);      % error between estimated and true loadings, model 2
SNm1 = nan(K,It);       % signal-to-noise, calibration set 1
SNm2 = nan(K,It);       % signal-to-noise, calibration set 2

for it = 1:It
    % Fit PLS models, NIPALS
    [B1,P1,~,T1,~,~] = nipals_pls1(X1+Ex(:,:,it),y1+ey(:,it),K);
    [B2,P2,~,T2,~,~] = nipals_pls1(X2+Ex(:,:,it),y2+ey(:,it),K);
    
    % Prediction error in test set
    Yhat1 = Xtest*B1;
    Yhat2 = Xtest*B2;
    E1 = repmat(ytest,1,K)-Yhat1;
    E2 = repmat(ytest,1,K)-Yhat2;
    MSE1(:,it) = diag(E1'*E1)./Itest;
    MSE2(:,it) = diag(E2'*E2)./Itest;
    
    % Error in loading vector
    EP1 = P1true-P1;
    EP2 = P2true-P2;
    
    MSEP1(:,it) = diag(EP1'*EP1)./J;
    MSEP2(:,it) = diag(EP2'*EP2)./J;
    
    % Calculate signal-to-noise
    X1temp = X1+Ex(:,:,it);
    X2temp = X2+Ex(:,:,it);
    E1temp = Ex(:,:,it);
    E2temp = Ex(:,:,it);
    for k = 1:K
        NAS1 = Proj*X1temp';    % NAS for calibration set 1
        NAS2 = Proj*X2temp';    % NAS for calibration set 2
        
        ProjE1 = Proj*E1temp';  % noise in NAS direction, calibration set 1
        ProjE2 = Proj*E2temp';  % noise in NAS direction, calibration set 2
        
        SN1 = nan(I,1);
        SN2 = nan(I,1);
        for i = 1:I
            SN1(i) = norm(NAS1(:,i))/norm(ProjE1(:,i)); % signal-to-noise
            SN2(i) = norm(NAS2(:,i))/norm(ProjE2(:,i)); % signal-to-noise
        end
        SNm1(k,it) = mean(SN1);
        SNm2(k,it) = mean(SN2);
        
        % Deflate X and E with component k
        X1temp = X1temp-T1(:,k)*P1(:,k)';
        X2temp = X2temp-T2(:,k)*P2(:,k)';
        E1temp = E1temp-E1temp*P1(:,k)*P1(:,k)';
        E2temp = E2temp-E2temp*P2(:,k)*P2(:,k)';
    end
end
clear B1 B2 E1 E1temp E2 E2temp EP1 EP2 i it k NAS1 NAS2 ProjE1 ProjE2 SN1 SN2 T1 T2 X1temp X2temp

% %% plot results
lim = round(0.125*It);

figure;
subplot(1,3,1);
plot(1:K, median(SNm1,2),'s--','markerfacecolor','b'); hold on;
plot(1:K, median(SNm2,2),'s--','markerfacecolor','r');
for k = 1:K
    [~,Idx1] = sort(SNm1(k,:));
    [~,Idx2] = sort(SNm2(k,:));
    errorbar(k,median(SNm1(k,:)),SNm1(k,Idx1(lim+1)), SNm1(k,Idx1(It-lim)),'color','b');
    errorbar(k,median(SNm2(k,:)),SNm2(k,Idx2(lim+1)), SNm2(k,Idx2(It-lim)),'color','r');
end
clear Idx1 Idx2 k
not_so_tight; axis square; grid on; hold off;
xlabel('#LV');
ylabel('L2(s)/L2(e)')
title('signal-to-noise');

subplot(1,3,2);
plot(1:K,median(MSEP1,2),'s--','markerfacecolor','b'); hold on;
plot(1:K, median(MSEP2,2),'s--','markerfacecolor','r');
for k = 1:K
    [~,Idx1] = sort(MSEP1(k,:));
    [~,Idx2] = sort(MSEP2(k,:));
    errorbar(k,median(MSEP1(k,:)),MSEP1(k,Idx1(lim+1)), MSEP1(k,Idx1(It-lim)),'color','b');
    errorbar(k,median(MSEP2(k,:)),MSEP2(k,Idx2(lim+1)), MSEP2(k,Idx2(It-lim)),'color','r');
end
clear Idx1 Idx2 k
not_so_tight; axis square; grid on; hold off;
xlabel('#LV');
ylabel('MSE');
title('loadings');

subplot(1,3,3);
plot(1:K, median(MSE1,2),'s--','markerfacecolor','b'); hold on;
plot(1:K, median(MSE2,2),'s--','markerfacecolor','r');
for k = 1:K
    [~,Idx1] = sort(MSE1(k,:));
    [~,Idx2] = sort(MSE2(k,:));
    errorbar(k,median(MSE1(k,:)),MSE1(k,Idx1(lim+1)), MSE1(k,Idx1(It-lim)),'color','b');
    errorbar(k,median(MSE2(k,:)),MSE2(k,Idx2(lim+1)), MSE2(k,Idx2(It-lim)),'color','r');
end
clear Idx1 Idx2 k
not_so_tight; axis square; grid on; hold off;
xlabel('#LV');
ylabel('MSE');
title('prediction error');
shg