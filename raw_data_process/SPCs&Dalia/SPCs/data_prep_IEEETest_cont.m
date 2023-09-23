clear all
clc
close all
%%
data_file = 'IEEEPPG\Competition_data';
addpath('IEEEPPG\Training_data\')
addpath(data_file)
%%
folders = dir(fullfile(data_file,'*.mat'));
fs = 125;
window_duration = 8;
overlap_duration = 6;
downsample_flag = 1;
data = {};
counter = 1;
for i = 1:10
fs = 125;
current_data = load(folders(i).name).sig;
bpm_trace = load(folders(i+10).name).BPM0;
ppg1 = current_data(1,:);
ppg2 = current_data(2,:);

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
ppg_fft_pdf = zeros((length(ppg_avg_segments(:,1))),length_of_segment);
for k = 1:length(ppg_avg_segments(:,1))
    current_segment = ppg_avg_segments(k,:);
    Y = fft(current_segment,L);
    P1 = Y(1:L/2+1);
    P1 = (1/(fs*L)) * abs(P1).^2;
    P1(2:end-1) = 2*P1(2:end-1);
    P1 = P1/sum(P1);
    P1_bpm = P1(index_low:index_high);
    f_bpm = bpm_values(index_low:index_high);
    ppg_fft_pdf(k,:) = P1_bpm;
end
%%
whole_dataset{counter,1} = ppg_avg_segments(4:end-1,:);
data_ppg_avg_fft{counter,1} = ppg_fft_pdf(4:end-1,:);
if length(whole_dataset{counter,1}(:,1)) ~= length(bpm_trace)
whole_dataset{counter,1} = ppg_avg_segments(4:end,:);
data_ppg_avg_fft{counter,1} = ppg_fft_pdf(4:end,:);
end
whole_dataset{counter,2} = round(bpm_trace)-30;
counter = counter + 1;
end
if downsample_flag == 1
    save('IEEE_10.mat','whole_dataset')
else
    save('IEEETest_cont.mat','data_ppg_avg','data_bpm_values','data_ppg_avg_fft')
end
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