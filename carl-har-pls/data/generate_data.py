#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 21:59:41 2021.

@author: peterp
"""

# %%%%%%%%%%%%%%%%%%%%% Anonymous functions %%%%%%%%%%%
# mncn = @(x) (x-mean(x)); % column-wise mean center
# auto = @(x) ((x-mean(x))./std(x)); % column-wise mean center and scale to unit variance
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np

def auto(x):
    """Compute column-wise mean center and scale to unit variance."""
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

# Constants
I = 100  # number of calibration samples
Itest = 500  # number of test samples
It = 50  # number of iterations
J = 50  # number of variables in X, explainatory variables
K = 3  # rank of system
r = np.array([[0.2, 0.95], [0.3, 0.9]])  # correlation coefficient

T1 = auto(np.random.rand(I, K))  # left factor matrix, calibration set 1

### kommet hertil
V_orth = (eye(I)-T1(:,1)*inv(T1(:,1)'*T1(:,1))*T1(:,1)')*T1(:,2:K);
          
          
T2 = nan(size(T1));
T2(:,1) = T1(:,1);
for k = 2:K
    T1(:,k) = T1(:,1)*r(k-1,1)+Vorth(:,k-1)*(1-r(k-1,1));
    T2(:,k) = T2(:,1)*r(k-1,2)+Vorth(:,k-1)*(1-r(k-1,2));
end
clear k

T1(:,K) = rand(I,1);
T2(:,K) = rand(I,1);

T1 = auto(T1);
T2 = auto(T2);
Ttest = auto(rand(Itest,K));  % left factor matrix, test set

% P = rand(J,K);          % right factor matrix
P = nan(J,K);   % Signals
sigma = 5;
mu = [15;20;30];
for k = 1:K
    %     S(:,i) = normpdf(1:J,mu(1),sigma)+i*normpdf(1:J,mu(2),sigma);
    P(:,k) = normpdf(1:J,mu(k),sigma);
    P(:,k) = P(:,k)/sqrt(P(:,k)'*P(:,k));
end
clear k mu sigma

X1 = T1*P';             % explainatory variables, calibration set 1
X2 = T2*P';             % explainatory variables, calibration set 2
Xtest = Ttest*P';       % explainatroy variables, test set

y1 = T1(:,1);              % response, calibration set 1
y2 = T2(:,1);              % response, calibration set 2
ytest = Ttest(:,1);        % response, test set

s = P(:,1);        % analyte signal at unit concentration
clear T1 T2 Ttest P

% %% Calculate true reg. vec and loadings, NIPALS
[B,P1true,~,~,~,~] = nipals_pls1(X1,y1,K);
btrue = B(:,K);
[~,P2true,~,~,~,~] = nipals_pls1(X2,y2,K);
Proj = btrue*inv(btrue'*btrue)*btrue'; % projection matrix for Net Analyte Signal, NAS
clear B

% %% Add noise to X and y
sigX = 0.6E-1;
sigy = 5E-1;

Ex_test = mncn(randn(Itest,J)*sigX);    % Error for test set
ey_test = mncn(randn(Itest,1)*sigy);    % Error for test set
Xtest = Xtest+Ex_test;
ytest = ytest+ey_test;