% Task 2 Bayes Classifier for Neutral vs Facial expression detection

clc; clear all; close all;
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
    L(2*i-1)= 1;
    D( :, 2*i)=reshape(face1(:, :, 2*i), [d, 1]);
    L(2*i)= 2;
end
% Testing dataset
for i=1:1:40    
    DT( :, 2*i-1)=reshape(face1(:, :, 320+((2*i)-1)), [d, 1]);
    LT(2*i-1)= 1;
    DT( :, 2*i)=reshape(face1(:, :,320+(2*i)), [d, 1]);
    LT(2*i)= 2;
end

%% 
mu = zeros(d, c); %mean of each class
for i=1:c
    for j=1:ni
        mu(:,i) = mu(:,i) + D(:, ni*(i-1)+j);
    end
    mu(:,i) = 1/ni * mu(:,i);
end
%% 
var = zeros(d, d, c); % Variance of each class
var_inv = zeros(d, d, c); % Inverse of variance
delta = 1; % To adjust singularity
for i=1:c
   for j=1:ni
      var(:,:,i) = var(:,:,i) +  ( D(:, ni*(i-1)+j) - mu(:,i) ) * ( D(:, ni*(i-1)+j) - mu(:,i) ).';
   end
   var(:,:,i) = 1/ni * var(:,:,i);
   var(:, :, i) = var(:, :, i ) + delta * eye(d);
   if det(var(:,:,i)) == 0
      disp('singular matrix');
      pause;
   end
   var_inv(:, :, i) = inv(var(:, : ,i));
end
%% 
% Calculating the discriminant function
for i = 1:c
    W(:,:,i) = -1/2 * var_inv(:, :, i);
    w(:,i) = var_inv(:, :, i) * mu(:,i);
    w0(i) = -1/2 * mu(:, i)' * var_inv(:, :, i) * mu(:, i) - 1/2 * log(det(var(:,:,i))); 
end
%%
%Testing
pred = zeros(nt,1);
for i = 1:nt % test data DT
    disp(i);
    max_J = DT(:,i)' * W(:,:,1) * DT(:,i) + w(:,1)' * DT(:,i) + w0(1);
    for j = 1:c % discriminant functions
        J = DT(:,i)' * W(:,:,j) * DT(:,i) + w(:,j)' * DT(:,i) + w0(j);
        if(J >= max_J)
           max_J = J; 
           pred(i) = j;
        end
    end
end
%%
accuracy = 0.0;
for i=1:nt
   if pred(i) == LT(i)
       accuracy = accuracy + 1; 
   end
end
accuracy = accuracy / nt; % Finding accuracy
display(accuracy);


