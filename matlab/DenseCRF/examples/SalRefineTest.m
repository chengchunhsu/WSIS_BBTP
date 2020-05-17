clear all; close all;
addpath([pwd '\..\..\DenseCRF'])
addpath([pwd '\..\..\DenseCRF\include'])

image = imread('img_1001.jpg');
SalImage = im2single(imread('img_1001_stage2.png'));
SalImage = repmat(SalImage, [1 1 3]);
SalImage(:,:,1) = 1-SalImage(:,:,1);

unary = SalImage;
D = Densecrf();
D.SetImage(image);
D.SetUnary(unary);

% Some settings.
% D.gaussian_x_stddev = 3;
% D.gaussian_y_stddev = 3;
% D.gaussian_weight = 0.25; 
% 
% D.bilateral_x_stddev = 64;
% D.bilateral_y_stddev = 64;
% D.bilateral_r_stddev = 4;
% D.bilateral_g_stddev = 4;
% D.bilateral_b_stddev = 4;
% D.bilateral_weight = 0.25; 

D.mean_field;
D.optimization_time
Segmap = D.segmentation-1;
figure;imshow(Segmap)

D.mean_field_inferprob;
D.optimization_time
Prob = D.prob;
figure;imshow(Prob)