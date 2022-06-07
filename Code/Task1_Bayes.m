% Task 1 using Bayes Classifier

clc; clear all; close all; 
load('data.mat'); % 24 * 21 * 600 (200 subjects)
d = 24*21; % dimension of image
c = 200; % No. of class
ni = 2; %  Training data per subject
n = 400; % Training data size
nt = 200; % Test data
D = zeros(d, n); % Training data set 504 * 400 
DT = zeros(d, nt); % Test data set 504 * 200 
L = zeros(n,1); % label for training data
LT = zeros(nt,1); % label for test data
%%
% Assigning training and testing dataset
for i=0:c-1
    count = 1;
    for j=1:3
        if j==1 || j==2 % Training dataset
            D(:,2*i+count)=reshape(face(:,:,3*i+j), [d,1]);
            L(2*i+count) = i+1; 
            count = count + 1;
        else
            DT(:,i+1)=reshape(face(:,:,3*i+j), [d,1]); % Testing dataset
            LT(i+1) = i+1;
        end
    end
end
%%
mu = zeros(d, c); % Matrix to assign mean of each class
for i=1:c
    for j=1:ni
        mu(:,i) = mu(:,i) + D(:, ni*(i-1)+j);
    end
    mu(:,i) = 1/ni * mu(:,i);
end
%%
var = zeros(d, d, c); % Matrix to assign variance to each class
var_inv = zeros(d, d, c); % Matrix to assign variance inverse to each class
delta = 1; % To adjust singular matrix
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
%calculate discriminant function
W = zeros(d, d, c);
w = zeros(d, c);
w0 = zeros(c,1);

for i = 1:c
    W(:,:,i) = -1/2 * var_inv(:, :, i);
    w(:,i) = var_inv(:, :, i) * mu(:,i);
    w0(i) = -1/2 * mu(:, i)' * var_inv(:, :, i) * mu(:, i) - 1/2 * log(det(var(:,:,i))); 
end
%%
%Testing
pred = zeros(nt,1);
for i = 1:nt % test data DT
    display(i);
    max_J = DT(:,i)' * W(:,:,1) * DT(:,i) + w(:,1)' * DT(:,i) + w0(1);
    for j = 1:c    % discriminant functions
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
disp('Accuracy=');
disp(accuracy);

