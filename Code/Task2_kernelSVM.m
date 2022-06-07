% Task 2 using Kernel SVM
clc; clear all; close all;
%% 
load('data.mat');
d = 24*21; % dimension of each sample image
c = 2; % No. of classes (Neutral or facial expression)
ni = 160; % training data per class
n = 320; % training data
nt = 80; %test data
D = zeros(d, n); % Training data set 504 * 320 
DT = zeros(d, nt); % Test data set 504 * 80 
L = zeros(n,1); % label for training data
LT = zeros(nt,1); % label for test data
%% 
for i=1:1:200
    face1(:, :, 2*i-1)=face(:, :, 3*i-2); % Reshaping to remove 3rd image of every subject
    face1(:, :, 2*i)=face(:, :, 3*i-1);
end
%%
% Training dataset
for i=1:1:160   
    D( :, 2*i-1)=reshape(face1(:, :, 2*i-1), [d, 1]);
    L(2*i-1)= +1;
    D( :, 2*i)=reshape(face1(:, :, 2*i), [d, 1]);
    L(2*i)= -1;
end
% Testing dataset
for i=1:1:40    
    DT( :, 2*i-1)=reshape(face1(:, :, 320+((2*i)-1)), [d, 1]);
    LT(2*i-1)= +1;
    DT( :, 2*i)=reshape(face1(:, :,320+(2*i)), [d, 1]);
    LT(2*i)= -1;
end
%%
Kernel_Type={'ploynomial';'RBF'};
X=transpose(D);
X1=transpose(DT);
Y=L;
global poly_deg 
poly_deg=2; % For Polynomial Kernel degree
kernel=char(Kernel_Type(1));% Choose the kernel
[a,Ker,b0]=SVM(X,Y, kernel);
Y_pred = SVM_pred(X1, X, Y,'polynomial',a,b0);
Y_pred1=sign(Y_pred);
accuracy=sum(Y_pred1 == LT) / length(LT);
disp(accuracy);
%%
function [a,Ker,b0]=SVM(X,Y, kernel)
precision=10^-5;Cost=1000;
switch kernel
    case 'ploynomial'
        Ker=Kernel_Polynomial(X,X);
    case 'RBF'
        Ker=Kernel_RBF(X,X);
end
N=size(X,1);
H=diag(Y)*Ker*diag(Y);
f= -ones(N,1);
Aeq=Y';
beq=0;
A=[];
b=[];
lb = zeros(N,1);
ub = repmat(Cost,N,1);
a=quadprog(H,f,A,b,Aeq,beq, lb, ub);

sr_num=(1:size(X,1))';
sr_sv=sr_num(a>precision&a<Cost);

temp_b0=0;
for i=1:size(sr_sv,1)
    temp_b0=temp_b0+Y(sr_sv(i));
    temp_b0=temp_b0-sum(a(sr_sv(i))*Y(sr_sv(i))*Ker(sr_sv,sr_sv(i)));
end
b0=temp_b0/size(sr_sv,1);
return
end
%%
function Y=Kernel_RBF(X1,X2) 
Y=zeros(size(X1,1),size(X2,1));% Gram Matrix
for i=1:size(X1,1)
    for j=1:size(X2,1)
        Y(i,j)=exp(-0.005*norm(X1(i,:)-X2(j,:))^2);
    end
end
return
end
%%
function Y=Kernel_Polynomial(X1,X2)
global poly_deg
Y=zeros(size(X1,1),size(X2,1));%Gram Matrix
for i=1:size(X1,1)
    for j=1:size(X2,1)
        Y(i,j)=(1+dot(X1(i,:),X2(j,:))).^poly_deg;
    end
end
return
end
%%
function Y_new = SVM_pred(X_new, X, Y,kernel,a,b0)
M = size(X_new,1);
switch kernel
    case 'polynomial'
        Ker=Kernel_Polynomial(X,X_new);
    case 'RBF'
        Ker=Kernel_RBF(X,X_new);
end
Y_new = sum(diag(a.*Y)*Ker,1)'+b0*ones(M,1);
return
end