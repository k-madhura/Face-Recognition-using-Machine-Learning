% Task 2 kNN
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
pred = zeros(nt,1); % Matrix for prediction
for i=1:nt
min_dist = (DT(:,i) - D(:,1))'*(DT(:,i) - D(:,1));
for j=1:n
  if (DT(:,i) - D(:,j))'*(DT(:,i) - D(:,j)) <= min_dist
     min_dist =  (DT(:,i) - D(:,j))'*(DT(:,i) - D(:,j));
     pred(i) = L(j);
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