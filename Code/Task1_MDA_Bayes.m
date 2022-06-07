% Task 1 LDA followed by Bayes Classifier

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
% mean
mu = zeros(d, c); 
for i=1:c
    mu(:, i) = (D(:,2*i-1) + D(:,2*i))/2;
end

mu_all = zeros(d, 1);
for i=1:n
    mu_all = mu_all + D(:,i);
end
mu_all = 1/n * mu_all;
%%
% Within scatter matrix
delta = 0.05;  %To adjust singularity
SW = zeros(d,d);
for i=1:c
    for j=1:2
        S = ( D(:,2*(i-1)+j) - mu(:,i) ) * ( D(:,2*(i-1)+j) - mu(:,i) ).';
    end
    S = S + delta * eye(d);
    SW = SW + S;
end

if(det(SW)==0)
    disp('singular');
    pause;
end
%%
% Between scatter matrix
SB = zeros(d,d);
for i=1:c
   SB = SB + 2 * ( mu(:,i) - mu_all ) * ( mu(:,i) - mu_all ).';  
end

[W,EV] = eigs(SB,SW, c-1); %Eigen vector decomposition

Y = zeros(c-1, n);
YT = zeros(c-1, nt);
for i = 1:n
   Y(:, i) = W.' * D(:,i);
end
for i = 1:nt
    YT(:, i) = W.' * DT(:,i);
end

%%
delta2 = 1; % To adjust singularity
pred = BAYESfunc(Y, YT, LT, c, delta2);

%%
function [pred] = BAYESfunc(D, DT, LT, c, delta)

d = size(D,1); %  dimension
nt = size(DT,2); %  test data size

mu = zeros(d, c); % mean of each class
for i=1:c
    mu(:, i) = (D(:,2*i-1) + D(:,2*i))/2;
end

var = zeros(d, d, c); % variance of each class
var_inv = zeros(d, d, c); % variance inverse

for i=1:c
   var(:, :, i) = 1/2 * ( D(:, 2*i-1) - mu(:,i) ) * ( D(:, 2*i-1) - mu(:,i) ).' ...
                + 1/2 * ( D(:, 2*i) - mu(:,i) ) * ( D(:, 2*i) - mu(:,i) ).'; 
   var(:, :, i) = var(:, :, i ) + delta * eye(d);
   if det(var(:,:,i)) == 0
      disp('singular'); % Display if matrix is singular, to change delta
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
    w0(i) = -1/2 * mu(:, i)' * var_inv(:, :, i) * mu(:, i) - 1/2 * log(det(var(:,:,i))); % ignore lnP(wi)
end
%%
%Testing
pred = zeros(nt,1);
for i = 1:nt % test data DT
    display(i);
    max_J = -100000; % Cost
    for j = 1:c % discriminant function
        J = DT(:,i)' * W(:,:,j) * DT(:,i) + w(:,j)' * DT(:,i) + w0(j);
        if(J > max_J)
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
disp('Accuracy=')
disp(accuracy);
end