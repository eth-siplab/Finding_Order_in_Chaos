clear all
clc
close all
%%
addpath('Competition_data\')
%%
data_12 = load('IEEE_Small.mat');
data_10 = load('IEEE_SPC10.mat');
data_10_whole = data_10.whole_dataset;
data_12_whole = data_12.whole_dataset;
whole_dataset = [data_10_whole;data_12_whole];
%%
save('IEEE_Big.mat','whole_dataset')

