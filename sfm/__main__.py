import cv2
import numpy as np


class Frame:
    def __init__(self, image):
        self.image = image
        self.features = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Frame<image={self.image.shape}, features={len(self.features)}>"


def get_frames_from_video():
    cap = cv2.VideoCapture("VID.mp4")
    while not cap.isOpened():
        cap = cv2.VideoCapture("VID.mp4")
        cv2.waitKey(100)
        print("Wait for the header")

    frames = []
    counter = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        counter += 1
        if counter % 5 == 0:
            continue
        if counter > 100:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(Frame(gray))

    cap.release()
    return frames


def calculate_feature_points(frames: list):
    """
    Goes through all frames from the given list and calculates features points on each frame.
    The resulting feature points are then added to the frame together with their descriptors.
    """
    detector = cv2.BRISK_create()
    for frame in frames:
        blurred = cv2.GaussianBlur(frame.image, (3, 3), 0)
        keypoints, descriptors = detector.detectAndCompute(blurred, None)
        keypoint_image = cv2.drawKeypoints(blurred, keypoints, None)
        frame.features = (keypoints, descriptors)

        # Show keypoints on image
        # cv2.imshow('keypoint_image', keypoint_image)
        # cv2.waitKey(10)

    return frames


def main():
    frames = get_frames_from_video()
    frames = calculate_feature_points(frames)

    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)

    # matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matcher = cv2.BFMatcher_create()

    for i in range(len(frames) - 1):
        frame = frames[i]
        next_frame = frames[i + 1]
        descriptors = frame.features[1]
        next_descriptors = next_frame.features[1]
        if not descriptors.any() or not next_descriptors.any():
            continue

        matches = matcher.knnMatch(descriptors, next_descriptors, k=1)

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           flags=0)
        matches_image = cv2.drawMatchesKnn(frame.image, frame.features[0], next_frame.image,
                                           next_frame.features[0], matches, None, **draw_params)
        cv2.imshow('matches_image', matches_image)
        cv2.waitKey(10)

    cv2.waitKey()


if __name__ == "__main__":
    main()
