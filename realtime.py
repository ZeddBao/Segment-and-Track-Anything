import gc

import cv2
import torch
import numpy as np
import pyrealsense2 as rs

from SegTracker import SegTracker
from model_args import segtracker_args,sam_args,aot_args
from seg_track_anything import draw_mask

def SegTracker_add_first_frame(seg_tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        seg_tracker.restart_tracker()
        seg_tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        seg_tracker.first_frame_mask = predicted_mask

    return seg_tracker

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# cap = cv2.VideoCapture(1)
seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)

grounding_caption = "mouse.cup.tissue.charger"
box_threshold = 0.25
text_threshold = 0.25

frame_idx = 0

with torch.cuda.amp.autocast():
    while True:
        # ret, frame = cap.read()
        # if not ret:
        #     break
        color_frame = pipeline.wait_for_frames().get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not color_frame:
            continue
        if frame_idx == 0:
            pred_mask, annotated_frame = seg_tracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold)
            seg_Tracker = SegTracker_add_first_frame(seg_tracker, frame, pred_mask)
        elif (frame_idx % seg_tracker.sam_gap) == 0:
            seg_mask, annotated_frame = seg_tracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold)
            torch.cuda.empty_cache()
            gc.collect()
            track_mask = seg_tracker.track(frame)
            # find new objects, and update tracker with new objects
            new_obj_mask = seg_tracker.find_new_objs(track_mask, seg_mask)
            # save_prediction(new_obj_mask, output_mask_dir, str(
            #     frame_idx+frame_num).zfill(5) + '_new.png')
            pred_mask = track_mask + new_obj_mask
            # segtracker.restart_tracker()
            seg_tracker.add_reference(frame, pred_mask)
        else:
            pred_mask = seg_tracker.track(frame, update_memory=True)
            
        
        frame_idx += 1
        masked_frame = draw_mask(frame, pred_mask)
        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        torch.cuda.empty_cache()
        gc.collect()
        cv2.imshow('frame', masked_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break