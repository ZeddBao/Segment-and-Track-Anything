import gc

import cv2
import torch
import numpy as np
import pyrealsense2 as rs

from SegTracker import SegTracker
from model_args import segtracker_args,sam_args,aot_args
from seg_track_anything import draw_mask

def annotate(frame, annotated_masks):
    annotated_frame = frame.copy()
    for mask in annotated_masks:
        if mask['bbox'] is None:
            continue
        x1, y1, x2, y2 = mask['bbox']
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(annotated_frame, f"{mask['id']} {mask['phrase']} {mask['logit']:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated_frame

def extract_mask(mask, bbox):
    x1, y1, x2, y2 = bbox
    mask_in_box = mask[y1:y2, x1:x2].flatten()
    mask_in_box = mask_in_box[mask_in_box != 0]
    if len(mask_in_box) == 0:
        return mask == 0, 0
    id = np.bincount(mask_in_box).argmax()
    return mask == id, id

def get_bbox_from_mask(mask):
    y_indices, x_indices = np.where(mask == 1)
    if len(x_indices) == 0:
        return None
    x1 = x_indices.min()
    x2 = x_indices.max()
    y1 = y_indices.min()
    y2 = y_indices.max()
    return x1, y1, x2, y2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# cap = cv2.VideoCapture(1)
seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)

grounding_caption = tmp_caption = "phone.tissue"
box_threshold = 0.5
text_threshold = 0.5
visualize = False

frame_idx = 0

with torch.cuda.amp.autocast():
    while True:
        if grounding_caption != tmp_caption:
            frame_idx = 0
        # ret, frame = cap.read()
        # if not ret:
        #     break
        color_frame = pipeline.wait_for_frames().get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not color_frame:
            continue
        if frame_idx == 0:
            pred_mask, _, annotated_masks = seg_tracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold)
            for mask in annotated_masks:
                mask['mask'], mask['id'] = extract_mask(pred_mask, mask['bbox'])
                if visualize:
                    frame = draw_mask(frame, mask['mask'])
            # Reset the first frame's mask
            seg_tracker.restart_tracker()
            seg_tracker.add_reference(frame, pred_mask, frame_idx)
            seg_tracker.first_frame_mask = pred_mask
            if visualize:
                annotated_frame = annotate(frame, annotated_masks)
        elif (frame_idx % seg_tracker.sam_gap) == 0:
            seg_mask, _, annotated_masks = seg_tracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold)
            torch.cuda.empty_cache()
            gc.collect()
            track_mask = seg_tracker.track(frame)
            # find new objects, and update tracker with new objects
            new_obj_mask = seg_tracker.find_new_objs(track_mask, seg_mask)
            pred_mask = track_mask + new_obj_mask
            for mask in annotated_masks:
                mask['mask'], mask['id'] = extract_mask(pred_mask, mask['bbox'])
                if visualize:
                    frame = draw_mask(frame, mask['mask'])
            seg_tracker.add_reference(frame, pred_mask)
            annotated_frame = annotate(frame, annotated_masks)
        else:
            pred_mask = track_mask = seg_tracker.track(frame, update_memory=True)
            for mask in annotated_masks:
                mask['mask'] = pred_mask == mask['id']
                mask['bbox'] = get_bbox_from_mask(mask['mask'])
                if visualize:
                    frame = draw_mask(frame, mask['mask'])
            if visualize:
                annotated_frame = annotate(frame, annotated_masks)
        
        frame_idx += 1
        if visualize:
            masked_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', masked_frame)
        torch.cuda.empty_cache()
        gc.collect()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        tmp_caption = grounding_caption