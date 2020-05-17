# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals)
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


def project_boxes_on_boxes(matched_bboxes, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        matched_bboxes: an instance of BoxList
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    original_size = proposals.size
    assert matched_bboxes.size == proposals.size, "{}, {}".format(
        matched_bboxes, proposals)

    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    matched_bboxes = matched_bboxes.bbox.to(torch.device("cpu"))

    # Generate segmentation masks based on matched_bboxes
    polygons = []
    for matched_bbox in matched_bboxes:
        x1, y1, x2, y2 = matched_bbox[0], matched_bbox[1], matched_bbox[
            2], matched_bbox[3]
        p = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        polygons.append(p)
    segmentation_masks = SegmentationMask(polygons, original_size)

    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        mask_h = mask_w = self.discretization_size

        self.center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.],
                                           [0., 0., 0.]])  #, device=device)

        # TODO: modified this as one conv with 8 channels for efficiency
        self.pairwise_weights_list = [
            torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [0., 0., 0.]]),  #, device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 1.],
                          [0., 0., 0.]]),  #, device=device),
            torch.tensor([[0., 1., 0.], [0., 0., 0.],
                          [0., 0., 0.]]),  #, device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 0.],
                          [0., 1., 0.]]),  #, device=device),
            torch.tensor([[1., 0., 0.], [0., 0., 0.],
                          [0., 0., 0.]]),  #, device=device),
            torch.tensor([[0., 0., 1.], [0., 0., 0.],
                          [0., 0., 0.]]),  #, device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 0.],
                          [1., 0., 0.]]),  #, device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 0.],
                          [0., 0., 1.]]),  #, device=device),
        ]

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])

        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(segmentation_masks,
                                                     positive_proposals,
                                                     self.discretization_size)

            labels.append(labels_per_image)
            masks.append(masks_per_image)

        return labels, masks

    # 2019.4.16 Add for negative sample part for MIL.
    def prepare_targets_labels(self, proposals, targets):
        # Sample both negative and positive proposals
        # Only with image labels (without mask)
        labels = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            labels.append(labels_per_image)
        return labels

    # 2019.4.18 Add for per col/row MIL.
    def prepare_targets_cr(self, proposals, targets):
        # Sample both negative and positive proposals
        # Only with per col/row labels (without mask)
        labels = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            device = proposals_per_image.bbox.device

            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            pos_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            # delete field "mask"
            new_matched_targets = matched_targets.copy_with_fields(
                ["matched_idxs", "labels"])
            # generate bbox corresponding proposals
            pos_masks_per_image = project_boxes_on_boxes(
                new_matched_targets[pos_inds], proposals_per_image[pos_inds],
                self.discretization_size)

            # generate label per image
            # initialize as zeros, and thus all labels of negative sample is zeros
            M = self.discretization_size
            labels_per_image = torch.zeros(
                (len(proposals_per_image.bbox), M + M),
                device=device)  # (n_proposal, 56)

            # generate label of positive sample
            pos_labels = []
            for mask in pos_masks_per_image:
                label_col = [
                    torch.any(mask[col, :] > 0) for col in range(mask.size(0))
                ]
                label_row = [
                    torch.any(mask[:, row] > 0) for row in range(mask.size(1))
                ]
                label = torch.stack(label_col + label_row)
                pos_labels.append(label)
            pos_labels = torch.stack(pos_labels).float()
            labels_per_image[pos_inds] = pos_labels

            # save
            labels.append(labels_per_image)
        return labels

    def __call__(self, proposals, mask_logits, mil_score, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        # MIL term
        # Stack mil score and mask_logits
        if len(mil_score.shape) > 2 or mask_logits.size(1) > 1:  # multi-class
            class_labels, _ = self.prepare_targets(proposals, targets)
            class_labels = cat(class_labels, dim=0)
            if len(mil_score.shape) > 2:
                mil_score = [s[c] for s, c in zip(mil_score, class_labels)]
                mil_score = torch.stack(mil_score)
            if mask_logits.size(1) > 1:
                mask_logits = [m[c] for m, c in zip(mask_logits, class_labels)]
                mask_logits = torch.stack(mask_logits).unsqueeze(1)

        # Prepare target labels for mil loss of each col/row.
        labels = self.prepare_targets_cr(
            proposals, targets)  # for both positive/negative samples
        labels = cat(labels, dim=0)

        # Compute MIL term for each col/row MIL
        mil_loss = F.binary_cross_entropy_with_logits(mil_score, labels)

        # Pairwise term
        device = mask_logits.device
        mask_h, mask_w = mask_logits.size(2), mask_logits.size(3)
        pairwise_loss = []

        # Sigmoid transform to [0, 1]
        mask_logits_normalize = mask_logits.sigmoid()

        # Compute pairwise loss for each col/row MIL
        for w in self.pairwise_weights_list:
            conv = torch.nn.Conv2d(1, 1, 3, bias=False, padding=(1, 1))
            weights = self.center_weight - w
            weights = weights.view(1, 1, 3, 3).to(device)
            conv.weight = torch.nn.Parameter(weights)
            for param in conv.parameters():
                param.requires_grad = False
            aff_map = conv(mask_logits_normalize)

            cur_loss = (aff_map**2)
            cur_loss = torch.mean(cur_loss)
            pairwise_loss.append(cur_loss)
        pairwise_loss = torch.mean(torch.stack(pairwise_loss))

        return 1.0 * mil_loss, 0.05 * pairwise_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION)

    return loss_evaluator
