clear all; close all;
addpath('..');

image = imread('../include/densecrf/examples/im2.ppm');
unary = -single(image);
D = Densecrf();
D.SetImage(image);
D.SetUnary(unary);
% Some settings.
D.gaussian_x_stddev = 3;
D.gaussian_y_stddev = 3;
D.gaussian_weight = 1; 

D.bilateral_x_stddev = 60;
D.bilateral_y_stddev = 60;
D.bilateral_r_stddev = 10;
D.bilateral_g_stddev = 10;
D.bilateral_b_stddev = 10;
D.bilateral_weight = 1; 


%% Threhold
% figure(1);
% D.threshold;
% D.display();

%% Meanfield
figure(2);
D.mean_field;
D.display();

% %% TRW-S (adding very few edges)
% figure()
% 
% % set to zero to solve the densecrf problem (requires alot of memory)
% D.min_pairwise_cost = 1; 
% D.trws;
% D.display()