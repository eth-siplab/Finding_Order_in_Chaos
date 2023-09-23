clear all
clc
close all
%%
data_file = 'Training_data\';
addpath(data_file)
%%
folders = dir(fullfile(data_file,'*.mat'));
window_duration = 8;
overlap_duration = 6;
downsample_flag = 1;
data = {};
counter = 1;
for i = 1:2:23
fs = 125;
current_data = load(folders(i).name).sig;
bpm_trace = load(folders(i+1).name).BPM0;
ecg = current_data(1,:);
ppg1 = current_data(2,:);
ppg2 = current_data(3,:);

ppg1_filtered = filter_butter(ppg1,fs);
ppg2_filtered = filter_butter(ppg2,fs);
ppg_avg = (ppg1_filtered + ppg2_filtered)/2;

if downsample_flag == 1
    ppg_avg_segments = downsample(normalize(buffer(ppg_avg,window_duration*fs,overlap_duration*fs),'zscore'),5).';
    fs = 25;
else
    ppg_avg_segments = normalize(buffer(ppg_avg,window_duration*fs,overlap_duration*fs),'zscore').';
end 
%%
if downsample_flag == 0
    L = 8192;
else
    L = 512;
end
f = fs*(0:(L/2))/L;
bpm_values = f*60;
[~, index_low] = min(abs(bpm_values-30));
[~, index_high] = min(abs(bpm_values-210));
length_of_segment = index_high - index_low+1;
%%
whole_dataset{counter,1} = ppg_avg_segments(4:end-1,:);
if length(whole_dataset{counter,1}(:,1)) ~= length(bpm_trace)
whole_dataset{counter,1} = ppg_avg_segments(4:end,:);
end
whole_dataset{counter,2} = round(bpm_trace)-30;
counter = counter + 1;
end
bpm_values = whole_dataset(:,2);
bpmmat = cell2mat(bpm_values);
min_HR = min(bpmmat);
max_HR = max(bpmmat);

save('IEEE_Small.mat','whole_dataset')

%% Filter Butterworth
function filtered = filter_butter(x,fs)
f1=0.5;
f2=4;
Wn=[f1 f2]*2/fs;
N = 4;
[b,a] = butter(N,Wn);
% [b,a] = ellip(6,5,50,20/(fs/2));
filtered = filtfilt(b,a,x);
filtered = normalize(filtered,'range');
end
