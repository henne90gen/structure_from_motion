import cv2


def calculate_feature_points(frames: list):
    """
    Goes through all frames from the given list and calculates features points on each frame.
    The resulting feature points are then added to the frame together with their descriptors.
    """
    detector = cv2.BRISK_create()
    for frame in frames:
        blurred = cv2.GaussianBlur(frame.image, (3, 3), 0)
        keypoints, descriptors = detector.detectAndCompute(blurred, None)
        frame.features = (keypoints, descriptors)

        # Show keypoints on image
        # keypoint_image = cv2.drawKeypoints(blurred, keypoints, None)
        # cv2.imshow('keypoint_image', keypoint_image)
        # cv2.waitKey(10)

    return frames
