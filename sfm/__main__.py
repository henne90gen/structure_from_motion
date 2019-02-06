"""
With this module it will be possible to create 3D models from videos
"""

from functools import partial
import math
import cv2
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


class Frame:
    def __init__(self, image):
        self.image = image
        self.features = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Frame<image={self.image.shape}, features={len(self.features)}>"


def get_frames_from_video(filename: str, max_frame_count: int = None, only_record_every_n_frames: int = 1):
    """
    Extracts frames from a video file.
    Starting from the beginning, this function extracts max_frame_count frames or all frames if max_frame_count is None.
    Optionally an interval for capturing frames can be specified.
    Example (x = capture, _ = skip): max_frame_count = 6, only_record_every_n_frames = 3
    x _ _ x _ _ x _ _ x _ _ x _ _ x
    """
    cap = cv2.VideoCapture(filename)
    while not cap.isOpened():
        cap = cv2.VideoCapture(filename)
        cv2.waitKey(100)
        print("Wait for the header")

    frames = []
    counter = 0
    while max_frame_count is None or len(frames) < max_frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        counter += 1
        if counter % only_record_every_n_frames != 0:
            continue

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
        frame.features = (keypoints, descriptors)

        # Show keypoints on image
        # keypoint_image = cv2.drawKeypoints(blurred, keypoints, None)
        # cv2.imshow('keypoint_image', keypoint_image)
        # cv2.waitKey(10)

    return frames


def is_outlier(query_key_points, train_key_points, match):
    """
    Returns False if the distance between two keypoints is bigger than 100 pixels.
    """
    match = match[0]
    p1 = query_key_points[match.queryIdx].pt
    p2 = train_key_points[match.trainIdx].pt
    distance_vector = (p1[0] - p2[0], p1[1] - p2[1])
    distance = math.sqrt(
        distance_vector[0] * distance_vector[0] +
        distance_vector[1] * distance_vector[1]
    )
    return distance < 100


def draw_lines(img1, img2, lines, pts1, pts2):
    """
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    """
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def draw_matches(image1, image2, keypoints1, keypoints2, matches):
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       flags=0)
    matches_image = cv2.drawMatchesKnn(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches,
        None,
        **draw_params
    )
    cv2.imshow('matches_image', matches_image)


def draw_points(img, points: list):
    """
    Draws a list of two dimensional points on an image
    """
    for point in points:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img = cv2.circle(img, tuple(point), 5, color, -1)
    return img


def draw_epilines_and_epipoles(image1, image2, pts1, pts2, lines1, lines2, left_epipole, right_epipole):
    left_img, img4 = draw_lines(image1, image2, lines1, pts1, pts2)

    right_img, img6 = draw_lines(image2, image1, lines2, pts2, pts1)

    left_epipole = list(map(int, left_epipole[:-1]))
    left_img = draw_points(left_img, [left_epipole])

    right_epipole = list(map(int, right_epipole[:-1]))
    right_img = draw_points(right_img, [right_epipole])

    cv2.imshow('left', left_img)
    cv2.imshow('right', right_img)


def calculate_epipoles(lines):
    epipoles = []
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j or i < j:
                continue
            l1 = lines[i]
            l2 = lines[j]
            A = np.array([l1, l2, [0.0, 0.0, 1.0]])
            try:
                A_inv = np.linalg.inv(A)
            except:
                continue
            epipole = A_inv[:, 2]
            epipoles.append(epipole)
    return epipoles


def main():
    frames = get_frames_from_video(
        "VID.mp4",
        max_frame_count=20,
        only_record_every_n_frames=5
    )
    frames = calculate_feature_points(frames)

    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)

    # matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matcher = cv2.BFMatcher_create()

    total_matches = 0
    total_good_matches = 0

    for i in range(len(frames) - 1):
        frame = frames[i]
        next_frame = frames[i + 1]
        keypoints = frame.features[0]
        next_keypoints = next_frame.features[0]
        descriptors = frame.features[1]
        next_descriptors = next_frame.features[1]

        if not descriptors.any() or not next_descriptors.any():
            print("Could not find any descriptors")
            continue

        matches = matcher.knnMatch(descriptors, next_descriptors, k=1)

        good_matches = list(
            filter(partial(is_outlier, keypoints, next_keypoints), matches))

        total_matches += len(matches)
        total_good_matches += len(good_matches)

        good_matches_percentage = len(good_matches) * 100 / len(matches)
        print(len(matches), len(good_matches), good_matches_percentage)

        pts1 = []
        pts2 = []
        for (match,) in good_matches:
            pts1.append(keypoints[match.queryIdx].pt)
            pts2.append(next_keypoints[match.trainIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        fundamental_matrix, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_LMEDS)

        # We select only inlier points (FIXME find out what this does and why we are doing it)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(
            pts2.reshape(-1, 1, 2),
            2,
            fundamental_matrix
        )
        lines1 = lines1.reshape(-1, 3)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(
            pts1.reshape(-1, 1, 2),
            1,
            fundamental_matrix
        )
        lines2 = lines2.reshape(-1, 3)

        # the kernel (or nullspace) of the fundamental matrix is the epipole in the left image
        left_epipole = scipy.linalg.null_space(fundamental_matrix)
        left_epipole = left_epipole.reshape(3)
        left_epipole = left_epipole / left_epipole[-1]

        # the kernel (or nullspace) of the transposed fundamental matrix is the epipole in the right image
        right_epipole = scipy.linalg.null_space(fundamental_matrix.transpose())
        right_epipole = right_epipole.reshape(3)
        right_epipole = right_epipole / right_epipole[-1]

        draw_epilines_and_epipoles(frame.image, next_frame.image,
                                   pts1, pts2, lines1, lines2, left_epipole, right_epipole)

        a1 = right_epipole[0]
        a2 = right_epipole[1]
        a3 = right_epipole[2]
        partial_camera_matrix = np.array([
            [0, -1 * a3, a2],
            [a3, 0, -1 * a1],
            [-1 * a2, a1, 0]
        ]) * fundamental_matrix
        right_camera_matrix = np.ndarray((3, 4))
        right_camera_matrix[0:3, 0:3] = partial_camera_matrix
        right_camera_matrix[:, 3] = right_epipole

        left_camera_matrix = np.ndarray((3, 4))
        left_camera_matrix[0:3, 0:3] = np.identity(3)
        left_camera_matrix[:, 3] = [0, 0, 0]

        print(left_camera_matrix)
        print(right_camera_matrix)
        print(pts1.size)
        print(pts2.size)
        result = cv2.triangulatePoints(
            left_camera_matrix,
            right_camera_matrix,
            pts1.reshape(2, pts1.size // 2).astype(float),
            pts2.reshape(2, pts2.size // 2).astype(float)
        )
        result = result.reshape(result.size // 4, 4)
        for row in result:
            row = row / row[-1]
            print(row)

        cv2.waitKey(10)

    print(total_matches, total_good_matches,
          total_good_matches * 100 / total_matches)

    cv2.waitKey()


if __name__ == "__main__":
    main()
