clc
close all
clear all
%%
addpath('USC-HAD')
whole_dataset = {};
files = dir("USC-HAD");
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags); % A structure with extra info.
% Get only the folder names into a cell array.
subFolderNames = {subFolders(3:end).name};
fs = 100;
for i=1:length(subFolderNames)
current_folder = dir(fullfile("USC-HAD\" + string(subFolderNames{i}),'*.mat'));
    subject_data = [];
    subject_label = [];
    for k=1:length(current_folder)
        gg = current_folder(k).name;
        folder = current_folder.folder;
        current_data = load(strcat(folder,'\',gg));
        current_data_cell = struct2cell(current_data);
        data = current_data_cell{13};
        if ischar(data)
            data = current_data_cell{14};
        end
        label = str2num(current_data_cell{9});
        data = normalize(data,'zscore');
        windowed_data = windowed_section(data);
        label_data = repelem(label, length(windowed_data(:,1,1))).';
        subject_data = cat(1,subject_data,windowed_data);
        subject_label = cat(1,subject_label,label_data);
    end
whole_dataset{i,1} = subject_data;
whole_dataset{i,2} = subject_label;
end
save('usc_data.mat','whole_dataset')

function section_windowed = windowed_section(data)
window_duration = 100;
overlap_duration = 50;
section_windowed=[];
for channels = 1:6
    temp = buffer(data(:,channels),window_duration,overlap_duration).';
    if isempty(section_windowed)
        section_windowed = temp;
    else
        section_windowed = cat(3,section_windowed,temp);
    end
end

end