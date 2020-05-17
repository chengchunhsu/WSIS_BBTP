clear all; close all;
addpath('..');

image = imread('../include/densecrf/examples/im2.ppm');
unary = -single(image);
D = Densecrf();
D.SetImage(image);
D.SetUnary(unary);
D.iterations = 10;
% Some settings.
D.gaussian_x_stddev = 3;
D.gaussian_y_stddev = 3;
D.gaussian_weight = 2^-5; 

D.bilateral_x_stddev = 64;
D.bilateral_y_stddev = 64;
D.bilateral_r_stddev = 4;
D.bilateral_g_stddev = 4;
D.bilateral_b_stddev = 4;
D.bilateral_weight = 2^-5; 

D.mean_field;
D.optimization_time


%% Some settings.
D.gaussian_x_stddev = 3;
D.gaussian_y_stddev = 3;
D.gaussian_weight = 2^-5; 

D.bilateral_x_stddev = 60;
D.bilateral_y_stddev = 60;
D.bilateral_r_stddev = 10;
D.bilateral_g_stddev = 10;
D.bilateral_b_stddev = 10;
D.bilateral_weight = 2^-5; 
D.mean_field;
D.optimization_time
