"""Keep the key grid locked to the keyboard when the camera or laptop moves.

At calibration we remember the reference frame and where the keyboard was in it.
For every new frame we find features (ORB) that match the reference, fit the
transform between the two views with RANSAC, and chain it onto the calibration:

    H_now = H_reference_to_now @ H_calibration

Matching against the fixed reference (not the previous frame) means errors
don't accumulate, so there is no drift.
"""
import cv2
import numpy as np

import layout

MIN_MATCHES = 15   # fewer good matches than this -> don't trust the frame
MIN_INLIERS = 12   # fewer RANSAC inliers than this -> don't trust the fit


class KeyboardTracker:
    def __init__(self, ref_frame, H_calib):
        self.ref_frame = ref_frame.copy()
        self.H_calib = H_calib
        self.H_current = H_calib

        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        # Only learn features that sit on the keyboard, not the desk or screen.
        block = cv2.perspectiveTransform(
            np.array(layout.BLOCK_CORNERS, dtype=np.float32).reshape(-1, 1, 2),
            H_calib,
        ).astype(np.int32)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, block, 255)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, mask)

    def update(self, frame):
        """Returns (H, inliers). H is held at its last good value on a bad frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 2:
            return self.H_current, 0

        # Ratio test: keep a match only if it is clearly better than the runner-up.
        pairs = self.matcher.knnMatch(self.ref_des, des, k=2)
        good = [p[0] for p in pairs
                if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < MIN_MATCHES:
            return self.H_current, 0

        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H_rel, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H_rel is None:
            return self.H_current, 0

        inliers = int(inlier_mask.sum())
        if inliers < MIN_INLIERS:
            return self.H_current, inliers

        self.H_current = H_rel @ self.H_calib
        return self.H_current, inliers
