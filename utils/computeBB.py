import numpy as np

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].

    Modified from: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
    - Changes: unique shape consideered, boxes coordinates +1
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    # for i in range(mask.shape[-1]):
    #     m = mask[:, :, i]
    #     # Bounding box.
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    boxes = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)