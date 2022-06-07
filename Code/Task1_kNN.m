% Task 1 using kNN
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
accuracy = accuracy / nt;  % Finding accuracy
disp('Accuracy=');
disp(accuracy);