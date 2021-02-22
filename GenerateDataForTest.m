%% Initialization
clear all;
clc;
addpath(genpath('./Functions/'))

%% Parameters setting

angRes = 5; % Angular Resolution, options, e.g., 3, 5, 7, 9. Default: 5
factor = 4; % SR factor, Default: 4
downRatio = 1/factor;
sourceDataPath = './Datasets/';
sourceDatasets = dir(sourceDataPath);
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);
idx = 0;

for DatasetIndex = 1 : datasetsNum
    DatasetName = sourceDatasets(DatasetIndex).name;
    SavePath = ['./Data/TestData_', num2str(factor), 'xSR_', num2str(angRes), 'x', num2str(angRes), '/', DatasetName];
    if exist(SavePath, 'dir')==0
        mkdir(SavePath);
    end
    
    sourceDataFolder = [sourceDataPath, sourceDatasets(DatasetIndex).name, '/test/'];
    folders = dir(sourceDataFolder); % list the scenes
    folders(1:2) = [];
    sceneNum = length(folders);
    
    for iScene = 1 : sceneNum
        sceneName = folders(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating test data of Scene_%s in Dataset %s......\n', sceneName, sourceDatasets(DatasetIndex).name);
        dataPath = [sourceDataFolder, folders(iScene).name];
        data = load(dataPath);
        LF = data.LF;
        [U, V, H, W, ~] = size(LF);
        while mod(H, 4) ~= 0
            H = H - 1;
        end
        while mod(W, 4) ~= 0
            W = W - 1;
        end
        
        LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), 1:H, 1:W, 1:3); % Extract central angRes*angRes views
        [U, V, H, W, ~] = size(LF);
        
        data = single(zeros(U*H*downRatio, V*W*downRatio));
        label = single(zeros(U*H, V*W));

        for u = 1 : U
            for v = 1 : V
                SAI_rgb = squeeze(LF(u, v, :, :, :));
                SAI_ycbcr = rgb2ycbcr(double(SAI_rgb));
                label((u-1)*H+1 : u*H, (v-1)*W+1 : v*W) = SAI_ycbcr(:, :, 1);                
                data((u-1)*H*downRatio+1 : u*H*downRatio, (v-1)*W*downRatio+1 : v*W*downRatio) = imresize(SAI_ycbcr(:, :, 1), downRatio);
            end
        end
        
        SavePath_H5 = [SavePath, '/', sceneName, '.h5'];
        h5create(SavePath_H5, '/data', size(data), 'Datatype', 'single');
        h5write(SavePath_H5, '/data', single(data), [1,1], size(data));
        h5create(SavePath_H5, '/label', size(label), 'Datatype', 'single');
        h5write(SavePath_H5, '/label', single(label), [1,1], size(label));
        idx = idx + 1;
    end
end


