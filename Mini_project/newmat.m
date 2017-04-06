clear all
clc


% You can customize and fix initial directory paths
TrainDatabasePath = uigetdir('C:\Users\arpan\Desktop\face_recognition', 'Select training database path' );
TestDatabasePath = uigetdir('C:\Users\arpan\Desktop\face_recognition', 'Select test database path');

prompt = {'Enter test image number'};
dlg_title = 'Input of PCA-Based Face Recognition System';
num_lines= 1;
def = {'1'};

TestImage  = inputdlg(prompt,dlg_title,num_lines,def);
TestImage = strcat(TestDatabasePath,'\',char(TestImage),'.jpg');
im = imread(TestImage);
T = CreateDatabase(TrainDatabasePath);

[m, A, Eigenfaces] = EigenfaceCore(T);
OutputName = Recognition(TestImage, m, A, Eigenfaces);

SelectedImage = strcat(TrainDatabasePath,'\',OutputName);
SelectedImage = imread(SelectedImage);
imshow(im)
title('Input Image');
figure,imshow(SelectedImage);
title('Matching Image');

str = strcat('Matched image is :  ',OutputName);
disp(str)
ProjectedImages = [];
Train_Number = size(Eigenfaces,2);
for i = 1 : Train_Number
    temp = Eigenfaces'*A(:,i); % Projection of centered images into facespace
    ProjectedImages = [ProjectedImages temp]; 
end
% Extracting the PCA features from test image
InputImage = imread(TestImage);
G=imnoise(InputImage,'salt & pepper',0.06);
figure,imshow(G);

temp = InputImage(:,:,1);
[irow icol] = size(temp);
InImage = reshape(temp',irow*icol,1);
Difference = double(InImage)-m; % Centered test image
ProjectedTestImage = Eigenfaces'*Difference; % Test image feature vector
Euc_dist = [];
for i = 1 : Train_Number
    q = ProjectedImages(:,i);
    temp = ( norm( ProjectedTestImage - q ) )^2;
    Euc_dist = [Euc_dist temp];
end
[Euc_dist_min , Recognized_index] = min(Euc_dist);
OutputName = strcat(int2str(Recognized_index),'.jpg');
m = mean(T,2); % Computing the average face image m = (1/P)*sum(Tj's)    (j = 1 : P)
Train_Number = size(T,2);
A = [];  
for i = 1 : Train_Number
    temp = double(T(:,i)) - m; % Computing the difference image for each image in the training set Ai = Ti - m
    A = [A temp]; % Merging all centered images
end


 L = A'*A; % L is the surrogate of covariance matrix C=A*A'.
 [V D] = eig(L); 
 L_eig_vec = [];
 for i = 1 : size(V,2) 
     if( D(i,i)>1 )
         L_eig_vec = [L_eig_vec V(:,i)];
     end
 end
