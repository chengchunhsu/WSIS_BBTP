clear all; close all;
addpath('..');

%%
image_name= '2_14_s';
M = MSRC(image_name);

M.bilateral_weight =1;
M.gaussian_weight = 1;

%%
M.mean_field();
M.display()