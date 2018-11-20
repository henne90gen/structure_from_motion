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
        if counter % 2 == 0:
            continue
        # if counter > 10:
        #     break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(Frame(gray))

    cap.release()
    return frames


def calculate_feature_points(frames: list):
    for frame in frames:
        blurred = cv2.GaussianBlur(frame.image, (3, 3), 0)
        keypoints, descriptors = cv2.BRISK_create().detectAndCompute(blurred, None)
        keypoint_image = cv2.drawKeypoints(blurred, keypoints, None)
        frame.features = list(zip(keypoints, descriptors))

        # Show keypoints on image
        # cv2.imshow('keypoint_image', keypoint_image)
        # cv2.waitKey(10)

    return frames

def main():
    frames = get_frames_from_video()
    frames = calculate_feature_points(frames)
    print(frames)
    cv2.waitKey()


if __name__ == "__main__":
    main()
