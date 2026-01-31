import numpy as np
# [IMPORTS] Import ESA class names dictionary for metric interpretation
from terramindFunctions import ESA_CLASSES

def calculate_precision(pred_map, target_map):
    """
    Calculates Precision metric: "What percentage of predictions are correct?"
    Formula: TP / (TP + FP)

    Args:
        pred_map: numpy array with predicted class codes
        target_map: numpy array with ground truth class codes

    Returns:
        tuple of (mean_precision_percent, class_details_dict)
    """
    classes_in_target = np.unique(target_map)
    precision_list = []
    details = {}

    for cls in classes_in_target:
        if cls == 0: continue

        p_mask = (pred_map == cls)
        t_mask = (target_map == cls)

        true_positives = np.logical_and(p_mask, t_mask).sum()
        false_positives = np.logical_and(p_mask, ~t_mask).sum()  # [FP] Model predicted: YES, Truth: NO

        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0  # [NOTE] If model didn't detect this class, precision = 0 (safe default)

        precision_list.append(precision)

        class_name = ESA_CLASSES.get(cls, f"Class {cls}")
        details[class_name] = precision * 100.0

    mean_precision = np.mean(precision_list) * 100.0 if precision_list else 0.0
    return mean_precision, details


def calculate_recall(pred_map, target_map):
    """
    Calculates Recall metric: "What percentage of true objects did the model find?"
    Formula: TP / (TP + FN)

    Args:
        pred_map: numpy array with predicted class codes
        target_map: numpy array with ground truth class codes

    Returns:
        tuple of (mean_recall_percent, class_details_dict)
    """
    classes_in_target = np.unique(target_map)
    recall_list = []
    details = {}

    for cls in classes_in_target:
        if cls == 0: continue

        p_mask = (pred_map == cls)
        t_mask = (target_map == cls)

        true_positives = np.logical_and(p_mask, t_mask).sum()
        false_negatives = np.logical_and(~p_mask, t_mask).sum()  # [FN] Model predicted: NO, Truth: YES

        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0  # [NOTE] Shouldn't happen since iterating over target classes

        recall_list.append(recall)

        class_name = ESA_CLASSES.get(cls, f"Class {cls}")
        details[class_name] = recall * 100.0

    mean_recall = np.mean(recall_list) * 100.0 if recall_list else 0.0
    return mean_recall, details

def calculate_dice_score(pred_map, target_map):
    """
    Calculates Dice coefficient (F1-Score for pixels).
    Returns mean Dice score across all classes.
    Dice score is typically higher than IoU for the same data.

    Args:
        pred_map: numpy array with predicted class codes
        target_map: numpy array with ground truth class codes

    Returns:
        float - mean Dice score as percentage
    """
    classes_in_target = np.unique(target_map)
    dice_list = []

    for cls in classes_in_target:
        # [SKIP] Skip class 0 (No data / No Data)
        if cls == 0:
            continue

        # [MASKS] Binary masks for current class
        p_mask = (pred_map == cls)
        t_mask = (target_map == cls)

        intersection = np.logical_and(p_mask, t_mask).sum()

        # [AREAS] Number of pixels in prediction and target
        area_pred = p_mask.sum()
        area_target = t_mask.sum()

        # [SAFE] Prevent division by zero
        if area_pred + area_target == 0:
            dice = 1.0  # Both empty = perfect match
        else:
            dice = (2.0 * intersection) / (area_pred + area_target)

        dice_list.append(dice)

    if len(dice_list) == 0:
        return 0.0

    return np.mean(dice_list) * 100.0


def calculate_accuracy(pred_map, target_map):
    """
    Calculates Pixel Accuracy (percentage of correctly classified pixels).

    Args:
        pred_map: numpy array with predicted class codes
        target_map: numpy array with ground truth class codes

    Returns:
        float - pixel accuracy as percentage
    """
    p = pred_map.flatten()
    t = target_map.flatten()

    # [VALID] Only compare pixels where target doesn't have "No data" (class 0)
    valid_mask = (t != 0)

    if np.sum(valid_mask) == 0:
        return 0.0

    correct_pixels = np.sum((p == t) & valid_mask)
    total_pixels = np.sum(valid_mask)

    return (correct_pixels / total_pixels) * 100.0

def calculate_miou(pred_map, target_map, verbose=False):
    """
    Calculates mean Intersection over Union (mIoU) metric.
    Optionally prints per-class results if verbose=True.

    Args:
        pred_map: numpy array with predicted class codes
        target_map: numpy array with ground truth class codes
        verbose: if True, print per-class IoU scores

    Returns:
        tuple of (mean_iou_percent, class_iou_dict)
    """
    # [FIND_CLASSES] Find all classes present in ground truth
    classes_in_target = np.unique(target_map)
    iou_list = []
    class_report = {}

    for cls in classes_in_target:
        # [SKIP] Skip class 0 (No data / No Data)
        if cls == 0:
            continue

        # [MASKS] Create binary masks (where this class appears)
        p_mask = (pred_map == cls)
        t_mask = (target_map == cls)

        intersection = np.logical_and(p_mask, t_mask).sum()
        union = np.logical_or(p_mask, t_mask).sum()

        if union > 0:
            iou = intersection / union
            iou_list.append(iou)

            # [CLASS_NAME] Get class name from dictionary
            class_name = ESA_CLASSES.get(cls, f"Class {cls}")
            class_report[class_name] = iou * 100.0

    if len(iou_list) == 0:
        return 0.0, {}

    miou = np.mean(iou_list) * 100.0

    # [VERBOSE] If verbose=True, print details to console
    if verbose:
        print("\n[IoU] IoU details per class")
        for name, score in class_report.items():
            print(f"{name:<20}: {score:.2f}%")

    return miou, class_report

def calculate_fw_iou(pred_map, target_map):
    """
    Calculates Frequency Weighted IoU.
    Weights depend on class frequency in target image.
    Provides "fairer" visual results (large forest weighted more than small river).

    Args:
        pred_map: numpy array with predicted class codes
        target_map: numpy array with ground truth class codes

    Returns:
        float - frequency weighted IoU as percentage
    """
    classes_in_target = np.unique(target_map)

    fw_iou_sum = 0.0
    total_valid_pixels = 0

    # [TOTAL] Sum all valid pixels (excluding class 0 - no data)
    for cls in classes_in_target:
        if cls == 0: continue
        total_valid_pixels += np.sum(target_map == cls)

    if total_valid_pixels == 0:
        return 0.0

    # [WEIGHTED] Compute weighted IoU for each class
    for cls in classes_in_target:
        if cls == 0: continue

        # [IOU] IoU for this class
        p_mask = (pred_map == cls)
        t_mask = (target_map == cls)

        intersection = np.logical_and(p_mask, t_mask).sum()
        union = np.logical_or(p_mask, t_mask).sum()

        iou = 0.0
        if union > 0:
            iou = intersection / union

        # [FREQUENCY] Class frequency (weight)
        frequency = np.sum(t_mask) / total_valid_pixels

        # [ADD] Add to sum: Weight * IoU
        fw_iou_sum += (frequency * iou)

    return fw_iou_sum * 100.0
