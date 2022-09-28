import cv2


class ROI_SOS:
    def __init__(self, size) -> None:
        self.size = size

    def smooth(self, roi_array):
        """morphological image operations to region of interest"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.size, self.size))
        smoothed_roi = cv2.morphologyEx(roi_array, cv2.MORPH_CLOSE, kernel)
        return smoothed_roi


class ROI_Carla:
    def __init__(self, size) -> None:
        self.size = size

    def smooth(self, roi_array):
        """morphological image operations to region of interest"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.size+30, self.size+30))
        smoothed_roi = cv2.morphologyEx(roi_array, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.size, self.size)) 
        smoothed_roi = cv2.morphologyEx(smoothed_roi, cv2.MORPH_OPEN, kernel)
        return smoothed_roi

