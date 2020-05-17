function AP = EvalVOCInstSeg(InstSegRes, GTInst, EvalCLassID, Threhold)
% InstSegRes = struct; [NumImgs X NumClass]
% InstSegRes = struct; [NumImgs X 1]

% first compute all overlaps
overlaps=cell(numel(GTInst),length(EvalCLassID));
gt = cell(numel(GTInst), 1);
Scores = cell(numel(GTInst),length(EvalCLassID));
for i=1:numel(GTInst)
    
    GTClass = GTInst(i).InstClass;
    GTInstSegMap = GTInst(i).InstSegMap;
    
    
    for j = 1:numel(EvalCLassID)
        PredInstSegMap = InstSegRes(i,j).InstSegMap;
        PredInstScores = InstSegRes(i,j).Scores;
        if ~isempty(PredInstSegMap)
            overlaps{i,j} = GetOverlap(PredInstSegMap, GTInstSegMap);
            Scores{i,j} = PredInstScores;
        else
            overlaps{i,j} = [];
            Scores{i,j} = [];
        end
    end
    gt{i}=GTClass;
end


% now run the evaluation. This is relatively fast once overlaps are precomputed
% ap_vol=zeros(9,numel(EvalCLassID));
AP = zeros(numel(EvalCLassID), length(Threhold));
for j=1:numel(EvalCLassID)
    for t = 1:length(Threhold)
        AP(j,t) = EvalInstSegAP(gt, Scores(:, EvalCLassID(j)), overlaps(:, EvalCLassID(j)), gt, EvalCLassID(j), Threhold(t));
%         ap_vol(t,j)=outputs(t,j).PR.ap;
%         fprintf('Evaluated threshold:%f for category:%d\n', 0.1*t, EvalCLassID(j));
    end
end
end

function [Overlap, NumInsts] = GetOverlap(PredInstSegMap, GTInstSegMap)

if isempty(PredInstSegMap)
    Overlap = [];
    NumInsts = 0;
    return
end

GTInstsSize = size(GTInstSegMap);
if iscell(GTInstSegMap)
    InstFun = @(Index)CellFun(GTInstSegMap,Index);
    NumInsts = max(GTInstsSize);
else
    if length(GTInstsSize) == 2
        NumInsts = 1;
    else
        NumInsts = GTInstsSize(3);
    end
    InstFun = @(Index)ThreeDMatFun(GTInstSegMap,Index);
     
end
NumProposals = size(PredInstSegMap, 3);

Overlap = zeros(NumInsts, NumProposals);
for i = 1:NumProposals
    Proposal = PredInstSegMap(:,:,i);
    for j = 1:NumInsts
        TempProposal = Proposal;
        TempGTInst = InstFun(j);
        Ignore = TempGTInst == 255;
        TempProposal(Ignore) = [];
        TempGTInst(Ignore) = [];
        TempGTInst = logical(TempGTInst);
        Overlap(j,i) = sum(TempProposal & TempGTInst) / sum(TempProposal | TempGTInst);
    end
end
end

function InstMask = CellFun(InstList, Index)
InstMask = InstList{Index};
end
function InstMask = ThreeDMatFun(InstList, Index)
InstMask = InstList(:,:,Index);
end
function InstMask = OneDMatFun(InstList, Index)
InstMask = InstList == Index;
end