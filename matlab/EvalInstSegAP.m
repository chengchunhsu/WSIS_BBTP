function [AP, Prec, Rec] = EvalInstSegAP(imglist, scoresperimg, overlapsall, gtlabels, categ, high_ov_thresh)

cnt=0;
for k=1:numel(scoresperimg)
	cnt=cnt+numel(scoresperimg{k});
end

AllData =zeros(cnt, 2);
cnt=0;
numgt=0;
for k=1:numel(imglist)

	%add things to diagnostic
	numdets=numel(scoresperimg{k});

	%get all overlaps
    Index = gtlabels{k}==categ;
    %record number of ground truth
	numgt=numgt+sum(gtlabels{k}==categ);

    overlaps=overlapsall{k};
    if(isempty(overlaps)) 
        continue; 
    end
    if sum(gtlabels{k}==categ) > 0
        overlaps=overlaps(Index,:);
        %compute labels using overlaps
        labels = OverlapsToLabels(scoresperimg{k}, overlaps, high_ov_thresh);
        AllData(cnt+1:cnt+numdets,1)=scoresperimg{k};
        AllData(cnt+1:cnt+numdets,2)=labels;
    else
        AllData(cnt+1:cnt+numdets,1)= scoresperimg{k};
        AllData(cnt+1:cnt+numdets,2) = zeros(size(scoresperimg{k}));
    end
	cnt=cnt+numdets;
end

%prec rec
scores = AllData(:,1);
labels = AllData(:,2);
[AP, Prec, Rec] = CalAP(scores, labels, numgt);
end


function Labels= OverlapsToLabels(Scores, Overlaps, Thresh)
NumInsts = size(Overlaps, 1);
NumDets = size(Overlaps, 2);

Covered=false(NumInsts,1);
Labels=zeros(NumDets,1);
[~, SortIndex]=sort(Scores, 'descend');


%assign
for k=1:numel(SortIndex)
    if(all(Covered))
        break;
    end
    Idx = find(~Covered);
    [Overlap, AssignID] = max(Overlaps(~Covered,SortIndex(k)));
    if(Overlap > Thresh)
        Labels(SortIndex(k))=1;
        Covered(Idx(AssignID))=true;
    end
end
end

function [ap, prec, rec] = CalAP(Scores, Labels, NumGTs)
Scores=Scores(:);
Labels=Labels(:);
[~, SortID]=sort(Scores, 'descend');
tp = Labels(SortID);
fp = 1-Labels(SortID);
tp = cumsum(tp);
fp = cumsum(fp);
prec = tp./(tp+fp);

rec = tp./NumGTs;

ap = VOCap(rec,prec);
end

function ap = VOCap(rec,prec)
mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
end

