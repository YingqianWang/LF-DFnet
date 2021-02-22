clear all;
clc;
close all;

%% The upscaling factor must match to the super-resolved LFs in './Results/'
factor = 4;

%%
sourceDataPath = './Datasets/';
sourceDatasets = dir(sourceDataPath);
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);

resultsFolder = './Results/';

for DatasetIndex = 1 : datasetsNum
    DatasetName = sourceDatasets(DatasetIndex).name;
    gtFolder = [sourceDataPath, sourceDatasets(DatasetIndex).name, '/test/'];
    scenefiles = dir(gtFolder);
    scenefiles(1:2) = [];
    sceneNum = length(scenefiles);
    
    resultsFolder = ['./Results/', DatasetName, '/'];
    
    for iScene = 1 : sceneNum
        sceneName = scenefiles(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating result images of Scene_%s in Dataset %s......\n', sceneName, sourceDatasets(DatasetIndex).name);
        
        data = load([resultsFolder, sceneName, '.mat']);
        LFsr_y = data.LF;
        [angRes, ~, H, W] = size(LFsr_y);        
        data = load([gtFolder, sceneName, '.mat']);
        LFgt_rgb = data.LF;
        LFgt_rgb = LFgt_rgb((11-angRes)/2:(9+angRes)/2, (11-angRes)/2:(9+angRes)/2, 1:H, 1:W, 1:3);        
        LFsr = zeros(size(LFgt_rgb));        
        
        for u = 1 : angRes
            for v = 1 : angRes                
                imgHR_rgb = squeeze(LFgt_rgb(u, v, :, :, :));
                imgLR_rgb = imresize(imgHR_rgb, 1/factor);
                imgLR_ycbcr = rgb2ycbcr(imgLR_rgb);
                imgSR_ycbcr = imresize(imgLR_ycbcr, factor);
                imgSR_ycbcr(:,:,1) = LFsr_y(u, v, :, :);
                imgSR_rgb = ycbcr2rgb(imgSR_ycbcr);
                LFsr(u, v, :, :, :) = imgSR_rgb;                
              
                SavePath = ['./SRimages/', DatasetName, '/', sceneName, '/'];
                if exist(SavePath, 'dir')==0
                    mkdir(SavePath);
                end
                imwrite(uint8(255*imgSR_rgb), [SavePath, num2str(u,'%02d'), '_', num2str(v,'%02d'), '.png' ]);
            end
        end        
    end
end
