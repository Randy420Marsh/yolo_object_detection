"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        # Original SORT convert_x_to_bbox did not handle score in output
        # If needed, modify this to include score, but for tracking it's usually input only
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    # Modified __init__ to accept class_id
    def __init__(self, bbox, class_id=None):
        """
        Initialises a tracker using initial bounding box and optional class_id.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        # Store the class ID associated with this tracker
        self.class_id = class_id # <-- Store class ID here

    # Modified update to accept class_id
    def update(self, bbox, class_id=None):
        """
        Updates the state vector with observed bbox and its optional class ID.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        # Update class_id if a new one is provided (e.g., from a matched detection)
        if class_id is not None:
             self.class_id = class_id

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        # history stores predicted states, class ID is associated with the tracker object
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    # detections: numpy array [[x1,y1,x2,y2,score], ...]
    # trackers: numpy array [[x1,y1,x2,y2], ...] # Note: trackers here are just their predicted bboxes

    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 0), dtype=int) # Return empty unmatched_trackers as per original SORT format

    iou_matrix = iou_batch(detections[:, :4], trackers) # Calculate IOU using only bbox coordinates

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # Use linear_assignment for matching, fallback to where if possible for speed (less common case)
        # Using linear_assignment consistently is generally safer.
        matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU (handled by linear_assignment with -iou_matrix and cost)
    # The check below is a redundant check if linear_assignment is used and iou_threshold applied in cost
    # but keeping it for compatibility with original SORT logic if iou_threshold is used differently.
    # However, let's rely on linear_assignment's output directly based on cost matrix.
    # The matches are those from linear_assignment whose IOU is >= iou_threshold

    matches = []
    # Filter matches based on IOU threshold after linear assignment
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))


    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    # Convert unmatched_trackers list to numpy array
    unmatched_trackers = np.array(unmatched_trackers)

    return matches, np.array(unmatched_detections), unmatched_trackers # Return numpy arrays

class sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0 # Reset tracker ID counter

    # Modified update to accept class_ids and return class_id in output
    def update(self, dets=np.empty((0, 5)), class_ids=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
          class_ids - an optional numpy array or list of integer class IDs corresponding to dets.
                      If provided, must have the same number of elements as dets.
                      Class IDs are expected to be integers.
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns a numpy array where the last two columns are the object ID and its class ID.
        Format: [[x1,y1,x2,y2,track_id,class_id], ...]

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 4)) # Only need bbox for association
        to_del = []
        ret = [] # List to store output tracks

        # Predict step for existing trackers
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0] # Predicted bbox [x1,y1,x2,y2]
            trks[t, :] = pos

            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Filter out trackers marked for deletion (due to NaN in prediction)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # Update self.trackers list in reverse order to handle pop correctly
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Perform association between current detections and predicted tracker bboxes
        # associate_detections_to_trackers expects detections [[x1,y1,x2,y2,score],...]
        # and tracker bboxes [[x1,y1,x2,y2],...]
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)


        # update matched trackers with assigned detections and their class IDs
        for m in matched:
            detection_index = m[0] # Index in the current frame's detections (dets)
            tracker_index = m[1]   # Index in the current list of trackers (self.trackers)

            detection_box = dets[detection_index, :] # Get the bbox and score for this detection

            # Get the class ID for the matched detection
            # Ensure class_ids is provided and index is valid
            matched_class_id = None
            if class_ids is not None and len(class_ids) > detection_index:
                 # Use .item() to get the scalar value if it's a numpy array or torch tensor
                 matched_class_id = int(class_ids[detection_index].item()) if hasattr(class_ids[detection_index], 'item') else int(class_ids[detection_index])


            # Update the matched tracker with the detection bbox and class ID
            self.trackers[tracker_index].update(detection_box, class_id=matched_class_id) # Pass bbox and class_id

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            detection_box = dets[i, :] # Get the bbox and score for this unmatched detection

            # Get the class ID for the new tracker
            new_class_id = None
            if class_ids is not None and len(class_ids) > i:
                 # Use .item() to get the scalar value if it's a numpy array or torch tensor
                 new_class_id = int(class_ids[i].item()) if hasattr(class_ids[i], 'item') else int(class_ids[i])

            # Create a new tracker with the detection bbox and class ID
            trk = KalmanBoxTracker(detection_box, class_id=new_class_id) # Pass bbox and class_id
            self.trackers.append(trk)


        # Build the return array with tracked objects (min_hits filter applied) and their class IDs
        # Iterate through current trackers to build the output list
        for trk in self.trackers:
            # Check if the track meets the minimum hits criteria or is new (within min_hits frames)
            # and has been updated in the current frame (time_since_update < 1)
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()[0] # Get current bbox estimate [x1,y1,x2,y2]
                track_id = trk.id + 1 # Use tracker ID + 1 as per MOT benchmark

                # Append bbox, track_id, and class_id
                # Ensure class_id is an integer, default to -1 if None
                class_id_output = trk.class_id if trk.class_id is not None else -1

                # Append the track state in the format [x1,y1,x2,y2,track_id,class_id]
                ret.append(np.concatenate((d, [track_id], [class_id_output])).reshape(1, -1))


        # Remove dead tracklets (iterating in reverse to handle pops correctly)
        # Need a separate loop for removal after building the return list
        to_del_final = []
        for i in range(len(self.trackers) - 1, -1, -1):
             if (self.trackers[i].time_since_update > self.max_age):
                 to_del_final.append(i)

        for i in to_del_final:
             self.trackers.pop(i)


        if (len(ret) > 0):
            return np.concatenate(ret)
        # Return empty array with the expected number of columns (bbox + track_id + class_id)
        return np.empty((0, 6)) # Return empty array with 6 columns
