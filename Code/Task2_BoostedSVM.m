% Task 2 using Boosted SVM
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

[ada_train, ada_test]= adaboost_func(D',L,DT');
accuracy=sum(ada_test == LT) / length(LT);
disp(accuracy);
%%
function [ada_train, ada_test]= adaboost_func(Xtrain,Ytrain, Xtest)

N=size(Xtrain,1);
a=[Xtrain Ytrain];
D=(1/N)*ones(N,1);
Dt=[]; h1=[];
C=1;
er=zeros(C,1);
for k=1:C
    p_min=min(D);
    p_max=max(D);
    
    for i=1:length(D)
        p = (p_max-p_min)*rand(1) + p_min;
        
        if D(i)>=p
            d(i,:)=a(i,:);
        end
        
        t=randi(size(d,1));
        Dt=[Dt ;d(t,:)];
    end
    X=Dt(:,1:end-1);
    Y=Dt(:,end);
    
    if k==1
        % svm 
        svm_in=fitcsvm(X,Y,'KernelFunction','linear');
        svm_out=predict(svm_in, X);
        h=svm_out;
        Dt=Dt(length(Dt)+1:end,:);
    end
    
    h1=[h1 h];   
    for i=1:length(Y)
        if (h1(i,k)~=Y(i))
            er(k)=er(k)+D(i,:); % Error
        end  
    end
    
    alpha(k)=0.5*log((1-er(k))/er(k)); % H weight
    D=D.*exp((-1).*Y.*alpha(k).*h);  % Updating weights
    D=D./sum(D);
end
% Training
H=predict(svm_in, Xtrain);
ada_train=sign(H*alpha');
% Testing
Htest=predict(svm_in, Xtest);
ada_test=sign(Htest*alpha');
end
