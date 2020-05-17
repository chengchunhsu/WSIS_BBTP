close all; 

% path of VOCcode
% (the code can be found in https://github.com/weiliu89/VOCdevkit/tree/master/VOCcode)
addpath(genpath('../VOCdevkit/VOCcode/'))

% path of mask.mat
InstRestDir = '../WSIS_BBTP/';

% path of VOC Dataset
VOCdevkitPath = '../VOCdevkit/VOC2012/';
VocSegFile = [ VOCdevkitPath '/ImageSets/Segmentation/val.json' ];
BBoxDir = [ VOCdevkitPath '/Annotations/' ];
ImageDir = [ VOCdevkitPath '/JPEGImages/' ];
InstGTDir = [ VOCdevkitPath '/SegmentationObject/' ];
ClassGTDir = [ VOCdevkitPath '/SegmentationClass/' ];


FileName = mfilename;
if contains(FileName, 'copy')
    Temp = strfind(FileName, '_');
    TestID = str2double(FileName(Temp(1)+1:Temp(2)-1));
else
    TestID = 0;
end
if TestID == 0
    NMSThreshold = 1;
else
    NMSThreshold = TestID * 0.1;
end
MaskRatio = [-inf inf];
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
        if length(size(InstMask{Count})) == 4
            TempInstMask = logical(permute(InstMask{Count} >= 0.5, [3 4 1 2]));
        else
            TempInstMask = logical(InstMask{Count} >= 0.5);
        end
        if k == 30
            A= 1;
        end
        ImageSize = size(TempInstMask);
        assert(all(ImageSize(1:2) == GTImageSize(1:2)))
        
        MaskResRatio = sum(sum(TempInstMask, 1), 2) ./ (GTImageSize(1) * GTImageSize(2));
        KeepIndex = MaskResRatio(:) > MaskRatio(1) & MaskResRatio(:) < MaskRatio(2);
        
        if any(KeepIndex)
        TempInstScore = InstScore{Count}(KeepIndex);
        TempInstLabel = InstLabel{Count}(KeepIndex);
        TempInstMask = TempInstMask(:,:,KeepIndex);
         
        InstClass = TempInstLabel;
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
        
        
        
        GTInstMasks = GTInst(k).InstSegMap;
        SelectProposals = NewTempInstMask;
        SelectScores = NewTempInstScore;
        SelectProposalLabel = NewTempInstLabel;
        Images = Img;
        InstClass = GTInst(k).InstClass;
        ImageName = ImageNameList{k};
        end
        Count = Count + 1;
    end
    
   
    


end

AP = EvalVOCInstSeg(InstSegRes, GTInst, 1:length(Classes), Threhold);
mean(AP(~isnan(AP(:,1)),:))
