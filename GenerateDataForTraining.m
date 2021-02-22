%% Initialization
clear all;
clc;
%close all;
addpath(genpath('./Functions/'))


%% Parameters setting

angRes = 5;
patchsize = 64;
stride = 32;
factor = 4;
downRatio = 1/factor;
sourceDataPath = '../Datasets/';
sourceDatasets = dir(sourceDataPath);
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);
idx = 0;
SavePath = ['../Data/TrainingData', '_', num2str(factor), 'xSR', '_', num2str(angRes), 'x', num2str(angRes), '/'];
if exist(SavePath, 'dir')==0
    mkdir(SavePath);
end

for DatasetIndex = 1 : 5
    sourceDataFolder = [sourceDataPath, sourceDatasets(DatasetIndex).name, '/training/'];
    folders = dir(sourceDataFolder); % list the scenes
    if isempty(folders)
        continue
    end
    folders(1:2) = [];
    sceneNum = length(folders);
    
    for iScene = 1 : sceneNum
        idx_s = 0;
        sceneName = folders(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating training data of Scene_%s in Dataset %s......\t\t', sceneName, sourceDatasets(DatasetIndex).name);
        dataPath = [sourceDataFolder, folders(iScene).name];
        data = load(dataPath);
        
        LF = data.LF;        
        LF = LF(:, :, :, :, 1:3);
        [U, V, H, W, ~] = size(LF);
        
        for h = 1 : stride : H-patchsize+1
            for w = 1 : stride : W-patchsize+1                
                lrSAI = single(zeros(U*patchsize*downRatio, V*patchsize*downRatio));
                HrSAI = single(zeros(U*patchsize, V*patchsize));
                for u = 1 : U
                    for v = 1 : V
                        k = (u-1)*V + v;
                        SAItemp = squeeze(LF(u, v, h:h+patchsize-1, w:w+patchsize-1, :));
                        SAItemp = rgb2ycbcr(double(SAItemp));
                        temp = squeeze(SAItemp(:,:,1));
                        HrSAI((u-1)*patchsize+1 : u*patchsize, (v-1)*patchsize+1 : v*patchsize) = temp;
                        lrSAI((u-1)*patchsize*downRatio+1 : u*patchsize*downRatio,...
                            (v-1)*patchsize*downRatio+1 : v*patchsize*downRatio) = imresize(temp, downRatio);                        
                    end
                end 
                
                %ku = floor((10-angRes)*rand());
                %kv = floor((10-angRes)*rand());
                ku = (9-angRes)/2;
                kv = (9-angRes)/2;
                idx = idx + 1;
                data = lrSAI(ku*patchsize*downRatio+1 : (ku+angRes)*patchsize*downRatio,...
                    kv*patchsize*downRatio+1 : (kv+angRes)*patchsize*downRatio);
                label = HrSAI(ku*patchsize+1 : (ku+angRes)*patchsize, kv*patchsize+1 : (kv+angRes)*patchsize);
                SavePath_H5 = [SavePath, num2str(idx,'%06d'),'.h5'];
                h5create(SavePath_H5, '/data', size(data), 'Datatype', 'single');
                h5write(SavePath_H5, '/data', single(data), [1,1], size(data));
                h5create(SavePath_H5, '/label', size(label), 'Datatype', 'single');
                h5write(SavePath_H5, '/label', single(label), [1,1], size(label));                
            end
        end
        fprintf([num2str(idx), ' training samples have been generated\n']);
    end
end

