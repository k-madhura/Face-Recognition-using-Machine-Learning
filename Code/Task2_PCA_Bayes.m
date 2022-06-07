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
[W,S,V] = svds(D,c-1); % To find singular values of W
Y = zeros(c-1, n);
YT = zeros(c-1, nt);
for i = 1:n
   Y(:, i) = W.' * D(:,i);
end
for i = 1:nt
    YT(:, i) = W.' * DT(:,i);
end

delta = 0.05; %To adjust singularity
pred = BAYESfunc(Y, YT, LT, c, delta);

%%
function [pred] = BAYESfunc(D, DT, LT, c, delta)

d = size(D,1); %  dimension
nt = size(DT,2); %  test data size

mu = zeros(d, c); % mean of each class
for i=1:c
    mu(:, i) = (D(:,2*i-1) + D(:,2*i))/2;
end

var = zeros(d, d, c);
var_inv = zeros(d, d, c);

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