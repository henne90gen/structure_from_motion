import cv2
import numpy as np


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
