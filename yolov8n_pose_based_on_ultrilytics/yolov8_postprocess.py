from pathlib import Path

import numpy as np
import cv2


def open_bin(bin_path):
    with open(bin_path, "rb") as f:
        data = f.read()
        data = np.frombuffer(data, dtype=np.float32)
    return data

def infernece_fpk(bin_path, file_name):

    cls_bin = f"{bin_path}/{file_name}_out_0.bin"
    det_bin = f"{bin_path}/{file_name}_out_1.bin"
    point_bin = f"{bin_path}/{file_name}_out_2.bin"
    
    cls_pred = open_bin(cls_bin).reshape(2100, 1).transpose(1, 0)
    det_pred = open_bin(det_bin).reshape(2100, 4).transpose(1, 0)
    point_pred = open_bin(point_bin).reshape(2100, 51).transpose(1, 0)
    # output_data = np.concatenate([det_pred,cls_pred,point_pred], 0)
    return det_pred,cls_pred,point_pred


def xywh2xyxy_denormalize(boxes, img_width, img_height):
    # Denormalize coordinates
    center_x = boxes[:, 0] * img_width
    center_y = boxes[:, 1] * img_height
    width = boxes[:, 2] * img_width
    height = boxes[:, 3] * img_height

    # Convert to XYXY
    x1 = center_x - (width / 2)
    y1 = center_y - (height / 2)
    x2 = center_x + (width / 2)
    y2 = center_y + (height / 2)

    # Stack the coordinates back into a 2D array
    denormalized_boxes = np.vstack((x1, y1, x2, y2)).T

    return denormalized_boxes


def iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > iou_threshold)[0]))
        )
    return pick


def draw_bbox_on_image(image: np.ndarray, x, y, w, h) -> np.ndarray:
    x = int(x * image.shape[1])
    y = int(y * image.shape[0])
    w = int(w * image.shape[1])
    h = int(h * image.shape[0])
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x + w // 2, y + h // 2
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def plot_keypoints_on_image(image: np.ndarray, keypoints, t):
    for keypoint in keypoints:
        x, y, visibility = keypoint
        if visibility > t:
            x = int(x / 320 * image.shape[1])
            y = int(y / 320 * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    return image


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def keypoints_plot(
    image: np.ndarray, kpts, kpt_threshold, shape=[320, 320], radius=6
) -> np.ndarray:

    for k in kpts:
        color_k = [0, 255, 0]
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < kpt_threshold:
                    continue
            x_coord = int(x_coord / shape[1] * image.shape[1])
            y_coord = int(y_coord / shape[0] * image.shape[0])
            cv2.circle(
                image, (x_coord, y_coord), radius, color_k, -1, lineType=cv2.LINE_AA
            )
    return image

def main():
    iou_threshold = 0.4,
    score_threshold = 0.4,
    kpt_threshold = 0.4,

    # get model results
    fpk_output_path = "yolo_pose_5rd_fix_outputs"   # fpk output bin dir
    image_path = "sss_demo/valid_keypoints/000000122046.jpg"  # image dir
    image = cv2.imread(image_path)

    file_name = Path(image_path).stem
    det_pred, cls_pred, point_pred = infernece_fpk(fpk_output_path, file_name)
    
    # additional postprocess
    output_data = np.concatenate([det_pred,cls_pred,point_pred], 0)
    output_data = output_data.T
    # cls sigmoid
    output_data[:, 4] = sigmoid(output_data[:, 4])
    # keypoints score sigmoid
    output_data[:, 7::3] = sigmoid(output_data[:, 7::3])

    # original postprocess
    boxes_xywh = output_data[:, :4]
    
    boxes_xyxy = xywh2xyxy_denormalize(boxes_xywh, image.shape[1], image.shape[0])
    indices = nms(boxes_xyxy, output_data[:, 4], iou_threshold)
    selected_output = output_data[indices]

    for output in selected_output:
        if output[4] < score_threshold:
            continue
        x, y, w, h = output[:4]
        keypoints = output[5:].reshape(-1, 3)

        # visualize bbox
        image = draw_bbox_on_image(image, x, y, w, h)
        image = keypoints_plot(
            image, keypoints, kpt_threshold, shape=(320, 320), radius=6
        )

    cv2.imwrite("test.jpg", image)


if __name__ == "__main__":
    main()
