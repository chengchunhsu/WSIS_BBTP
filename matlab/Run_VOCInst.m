clear all
addpath(genpath([pwd '/DenseCRF']))

% path of VOCcode
% (the code can be found in https://github.com/weiliu89/VOCdevkit/tree/master/VOCcode)
addpath(genpath('../VOCdevkit/VOCcode/'))

% path of output
IsSaveResult = false; %true;
MaskSaveDir = '../WSIS_BBTP/Mask/';

% path of mask.mat
InstRestDir = '../WSIS_BBTP/';

% path of VOC Dataset
VOCdevkitPath = '../VOCdevkit/VOC2012/';

VocSegFile = [ VOCdevkitPath '/ImageSets/Segmentation/val.json' ];
BBoxDir = [ VOCdevkitPath '/Annotations/' ];
ImageDir = [ VOCdevkitPath '/JPEGImages/' ];
InstGTDir = [ VOCdevkitPath '/SegmentationObject/' ];
ClassGTDir = [ VOCdevkitPath '/SegmentationClass/' ];

if ~exist(MaskSaveDir, 'dir')
    mkdir(MaskSaveDir)
end

NMSThreshold = 1;
MaskRatio = [0 1];

Classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};
ClassMap=containers.Map(Classes,1:length(Classes));

Threhold = [0.25 0.5 0.7 0.75];

f=fopen([VOCdevkitPath '/ImageSets/Segmentation/val.txt']);
is=textscan(f,'%s %*s');
ImageNameList = is{1};


InstResult = load([InstRestDir 'mask.mat']);


Res = fieldnames(InstResult);
Step = 4;
ImgID = Res(1:Step:end);
ImgID = strrep(ImgID,'img_', '');
ImgID = strrep(ImgID,'_masks', '');
ImageID = false(1, length(ImageNameList));
for i = 1:length(ImgID)
    Index = str2double(ImgID{i});
    ImageID(Index) = true;
end

InstResultCell = struct2cell(InstResult);
InstLabel = InstResultCell(3:Step:end);
InstScore = InstResultCell(2:Step:end);
InstMask = InstResultCell(1:Step:end);
GTInst = struct('InstClass', [], 'InstSegMap', []);
GTInst = repmat(GTInst, [1 length(InstMask)]);
InstSegRes = struct('Scores', [], 'InstSegMap', []);
InstSegRes = repmat(InstSegRes, [length(ImageNameList) length(Classes)]);
Count = 1;

D = Densecrf();
D.iterations = 5;
D.bilateral_weight = 0.005;
D.gaussian_weight = 0.05;
Time = nan(1, length(ImageNameList));
for k = 1:length(ImageNameList)
    disp([num2str(k) '/' num2str(length(InstLabel))])
    VOCBBox = PASreadrecord([BBoxDir ImageNameList{k} '.xml']);
    GTImageSize = VOCBBox.imgsize(2:-1:1);
    
    VOCBBox = VOCBBox.objects;
    
    bbs = cat(1,VOCBBox.bbox);
    ignore = false(size(bbs,1), 1);
    bbs = bbs(~ignore,:);
    t = ClassMap.values({VOCBBox.class});
    catIds=[t{:}];
    catIds = catIds(~ignore);
    GTInstMask = imread([InstGTDir ImageNameList{k} '.png']);
    GTClassMask = imread([ClassGTDir ImageNameList{k} '.png']);
    InstSegMap = false([size(GTInstMask), length(catIds)]);
    GTInst(k).InstClass = catIds;
    for l = 1:length(catIds)
        TempGTInstMask = GTInstMask;
        TempGTInstMask(TempGTInstMask ~= l  & TempGTInstMask ~= 255) = 0;
        if islogical(InstSegMap(1))
            TempGTInstMask(TempGTInstMask == 255) = 0;
        end
        InstSegMap(:,:,l) = TempGTInstMask;
    end
    GTInst(k).InstSegMap = InstSegMap;
    Img = imread([ImageDir '/' ImageNameList{k} '.jpg']);
    if ImageID(k)
        tic
        D.SetImage(Img);
        Prob = permute(InstMask{Count}, [3 4 1 2]);
        ImageSize = size(Prob);
        assert(all(ImageSize(1:2) == GTImageSize(1:2)))
        NumInst = size(Prob, 3);
        TempInstMask = false([ImageSize(1:2) NumInst]);
        for i = 1:NumInst
            TempProb = Prob(:,:,i);
            Unary = cat(3, -log(max(single(1-TempProb), 10^-5)), -log(max(single(TempProb), 10^-5)));
            D.SetUnary(Unary);
            D.mean_field;
            DenseCRFMask = D.segmentation == 2;
            TempInstMask(:,:,i) = DenseCRFMask;
        end
        MaskResRatio = sum(sum(TempInstMask, 1), 2) / (ImageSize(1) * ImageSize(2));
        MaskResRatio = MaskResRatio(:);
        KeepIndex = MaskResRatio > MaskRatio(1) & MaskResRatio < MaskRatio(2);
        TempTime = toc;
        Time(k) = TempTime;
        TempInstScore = InstScore{Count}(KeepIndex);
        TempInstLabel = InstLabel{Count}(KeepIndex);
        InstClass = TempInstLabel;
        TempInstMask = TempInstMask(:,:,KeepIndex);
        
        if any(KeepIndex)
            PredClass = unique(InstClass);
            NewTempInstScore = [];
            NewTempInstMask = [];
            NewTempInstLabel = [];
            for l = PredClass
                if NMSThreshold ~= 1
                    [Proposals, Score, SelectID] = NMS(TempInstMask(:,:,TempInstLabel == l), TempInstScore(TempInstLabel == l), NMSThreshold);
                else
                    Proposals = TempInstMask(:,:,TempInstLabel == l);
                    Score = TempInstScore(TempInstLabel == l);
                end
                NewTempInstScore = cat(2, NewTempInstScore, Score);
                NewTempInstMask = cat(3, NewTempInstMask, Proposals);
                NewTempInstLabel = cat(2, NewTempInstLabel, double(l) * ones(1,length(Score)));
                InstSegRes(k,l).InstSegMap = Proposals;
                InstSegRes(k,l).Scores = Score;
            end
            
            ResSaveName = [MaskSaveDir '/' ImageNameList{k} '.mat'];
            GTInstMasks = GTInst(k).InstSegMap;
            SelectProposals = NewTempInstMask;
            SelectScores = NewTempInstScore;
            SelectProposalLabel = NewTempInstLabel;
            Images = Img;
            InstClass = GTInst(k).InstClass;
            ImageName = ImageNameList{k};
            if IsSaveResult
                save(ResSaveName, 'GTInstMasks', 'SelectProposals', 'SelectScores', ...
                    'Images', 'InstClass','ImageName', 'SelectProposalLabel', ...
                    'GTInstMask', 'GTClassMask');
            end
        end
        Count = Count + 1;
    end
  
end

AP = EvalVOCInstSeg(InstSegRes, GTInst, 1:length(Classes), Threhold);
mean(AP(~isnan(AP(:,1)),:))
