clear; close all; clc
pat = 'C:\Users\ceskild\Documents\fromHdrive\Publications\000_In_Preparation\underfit_PLS\matlab';
cd(pat);

%%%%%%%%%%%%%%%%%%%%% Anonymous functions %%%%%%%%%%%
mncn = @(x) (x-mean(x)); % column-wise mean center
auto = @(x) ((x-mean(x))./std(x)); % column-wise mean center and scale to unit variance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I = 100;            % number of calibration samples
Itest = 200;        % number of test samples
It = 75;            % number of iterations
J = 50;             % number of variables in X, explainatory variables
K = 3;              % rank of system
r = [0.5;0.9;0];  % correlation coefficient

C1 = mncn(rand(I,K));       % left factor matrix, calibration set 1
C2 = C1;                    % left factor matrix, calibration set 2
Ct = mncn(rand(Itest,K));   % left factor matrix, test set
Vcal = (eye(I)-C1(:,1)*inv(C1(:,1)'*C1(:,1))*C1(:,1)')*C1(:,2:K);
Vt = (eye(Itest)-Ct(:,1)*inv(Ct(:,1)'*Ct(:,1))*Ct(:,1)')*Ct(:,2:K);
for k = 2:K
    C1(:,k) = C1(:,1)*r(1)+Vcal(:,k-1)*(1-r(1));
    C2(:,k) = C2(:,1)*r(2)+Vcal(:,k-1)*(1-r(2));
    Ct(:,k) = Ct(:,1)*r(3)+Vt(:,k-1)*(1-r(3));
end
clear k Vcal Vt r

C1 = auto(C1);
C2 = auto(C2);
Ct = auto(Ct);

S = nan(J,K);       % right factor matrix
sigma = 5;
mu = [15;20;30];
for k = 1:K
    S(:,k) = normpdf(1:J,mu(k),sigma);
    S(:,k) = S(:,k)/sqrt(S(:,k)'*S(:,k));
end
clear k mu sigma

X1 = C1*S';         % explainatory variables, calibration set 1
X2 = C2*S';         % explainatory variables, calibration set 2
Xtest = Ct*S';      % explainatroy variables, test set

y1 = C1(:,1);       % response, calibration set 1
y2 = C2(:,1);       % response, calibration set 2
ytest = Ct(:,1);    % response, test set

% %% Add noise to X and y
sigX = 0.5E-1;
sigy = 1E-1;

% test data
Ex = mncn(randn(I,J,It)*sigX);     % Error for calibration sets
ey = mncn(randn(I,It)*sigy);     % Error for calibration sets

Ex_test = mncn(randn(Itest,J)*sigX);    % Error for test set
ey_test = mncn(randn(Itest,1)*sigy);    % Error for test set

save data Ex Ex_test ey ey_test K S X1 X2 Xtest y1 y2 ytest I It J Itest
