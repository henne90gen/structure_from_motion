"""
With this module it will be possible to create 3D models from videos
"""

from functools import partial
import math
import cv2
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from .features import calculate_feature_points
from .drawing import draw_epilines_and_epipoles, draw_lines, draw_matches, draw_points


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


def triangulate(pts1, pts2, K, Kinv, P, P1):
    # triangulate
    pts1_pt_2r = pts1.reshape(2, pts1.size // 2).astype(float)
    pts2_pt_2r = pts2.reshape(2, pts2.size // 2).astype(float)
    pt_3d_h = cv2.triangulatePoints(P, P1, pts1_pt_2r, pts2_pt_2r)

    # calculate reprojection
    error = []
    pt_3d = cv2.convertPointsFromHomogeneous(
        pt_3d_h.reshape(pt_3d_h.size // 4, 4))
    R = P[0:3, 0:3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = P[:, 3]
    reprojected_pt_set, _ = cv2.projectPoints(
        pt_3d, rvec, tvec, K, np.array([]))
    reprojected_pt_set = reprojected_pt_set.reshape(
        reprojected_pt_set.size // 2, 2)

    for point, reprojected_point in zip(pts1, reprojected_pt_set):
        error.append(cv2.norm(point - reprojected_point))

    reprojection_error = cv2.mean(np.array(error))[0]
    print(reprojection_error)
    return reprojection_error


def test_triangulation():
    pass
    # vector<Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud);
    # vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());

    # Matx44d P4x4 = Matx44d::eye();
    # for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];

    # perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);

    # status.resize(pcloud.size(),0);
    # for (int i=0; i<pcloud.size(); i++) {
    #     status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
    # }
    # int count = countNonZero(status);

    # double percentage = ((double)count / (double)pcloud.size());
    # cout << count << "/" << pcloud.size() << " = " << percentage*100.0 << "% are in front of camera" << endl;
    # if(percentage < 0.75)
    #     return false; //less than 75% of the points are in front of the camera


def process_adjacent_frames(frame, next_frame):
    keypoints = frame.features[0]
    next_keypoints = next_frame.features[0]
    descriptors = frame.features[1]
    next_descriptors = next_frame.features[1]

    if not descriptors.any() or not next_descriptors.any():
        print("Could not find any descriptors")
        return

    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)

    # matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matcher = cv2.BFMatcher_create()

    matches = matcher.knnMatch(descriptors, next_descriptors, k=1)

    good_matches = list(
        filter(partial(is_outlier, keypoints, next_keypoints), matches))

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

    K = np.identity(3)
    K[0, 0] = 1000
    K[1, 1] = 1000
    K[0, 2] = frame.image.shape[1] / 2
    K[1, 2] = frame.image.shape[0] / 2
    essential_matrix = K.T * fundamental_matrix * K
    R1 = np.identity(3)
    R2 = np.identity(3)
    t = np.ndarray((3))
    cv2.decomposeEssentialMat(essential_matrix, R1, R2, t)

    P = np.ndarray((3, 4))
    P[0:3, 0:3] = np.identity(3)
    P[:, 3] = [0, 0, 0]

    P1 = np.ndarray((3, 4))
    P1[0:3, 0:3] = R1
    P1[:, 3] = t

    triangulate(pts1, pts2, K, np.linalg.inv(K), P, P1)
    # if test_triangulation():
    #     print("Triangulation succeeded")

    P1[0:3, 0:3] = R1
    P1[:, 3] = t * -1
    triangulate(pts1, pts2, K, np.linalg.inv(K), P, P1)
    # if test_triangulation():
    #     print("Triangulation succeeded")

    P1[0:3, 0:3] = R2
    P1[:, 3] = t
    triangulate(pts1, pts2, K, np.linalg.inv(K), P, P1)
    # if test_triangulation():
    #     print("Triangulation succeeded")

    P1[0:3, 0:3] = R2
    P1[:, 3] = t * -1
    triangulate(pts1, pts2, K, np.linalg.inv(K), P, P1)
    # if test_triangulation():
    #     print("Triangulation succeeded")

    # a1 = right_epipole[0]
    # a2 = right_epipole[1]
    # a3 = right_epipole[2]
    # partial_camera_matrix = np.array([
    #     [0, -1 * a3, a2],
    #     [a3, 0, -1 * a1],
    #     [-1 * a2, a1, 0]
    # ]) * fundamental_matrix
    # right_camera_matrix = np.ndarray((3, 4))
    # right_camera_matrix[0:3, 0:3] = partial_camera_matrix
    # right_camera_matrix[:, 3] = right_epipole

    # left_camera_matrix = np.ndarray((3, 4))
    # left_camera_matrix[0:3, 0:3] = np.identity(3)
    # left_camera_matrix[:, 3] = [0, 0, 0]

    # print(left_camera_matrix)
    # print(right_camera_matrix)

    # res, left_homography_matrix, right_homography_matrix = cv2.stereoRectifyUncalibrated(
    #     pts1, pts2, fundamental_matrix, frame.image.shape)
    # print(res)
    # print(left_homography_matrix)
    # print(right_homography_matrix)

    # left_rotation = np.linalg.inv(np.identity(3)) * left_homography_matrix * \
    #     np.identity(3)
    # right_rotation = np.linalg.inv(partial_camera_matrix) * right_homography_matrix * \
    #     partial_camera_matrix
    # print(left_rotation)
    # print(right_rotation)
    # left_pose = cameraPoseFromHomography(left_homography_matrix)
    # right_pose = cameraPoseFromHomography(right_homography_matrix)
    # print(left_pose)
    # print(right_pose)

    # print(pts1.size)
    # print(pts2.size)
    # result = cv2.triangulatePoints(
    #     left_camera_matrix,
    #     right_camera_matrix,
    #     pts1.reshape(2, pts1.size // 2).astype(float),
    #     pts2.reshape(2, pts2.size // 2).astype(float)
    # )
    # result = result.reshape(result.size // 4, 4)
    # for row in result:
    #     row = row / row[-1]
    #     print(row)

    cv2.waitKey(10)


def main():
    frames = get_frames_from_video(
        "VID.mp4",
        max_frame_count=20,
        only_record_every_n_frames=5
    )
    frames = calculate_feature_points(frames)

    for i in range(len(frames) - 1):
        frame = frames[i]
        next_frame = frames[i + 1]
        process_adjacent_frames(frame, next_frame)

    cv2.waitKey()


if __name__ == "__main__":
    main()
