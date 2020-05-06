#!/usr/bin/env python
######################################################################
# page_dewarp.py - Proof-of-concept of page-dewarping based on a
# "cubic sheet" model. Requires OpenCV (version 3 or greater),
# PIL/Pillow, and scipy.optimize.
######################################################################
# Author:  Matt Zucker
# Date:    July 2016
# License: MIT License (see LICENSE.txt)
######################################################################

import os
import sys
import datetime
import cv2
from PIL import Image
import numpy as np
import scipy.optimize
from multiprocessing import Process, Queue
import time


# for some reason pylint complains about cv2 members being undefined :(
# pylint: disable=E1101

# PAGE_INCREASE_X = 0     # 캔버스 크기 늘리기. 수치는 퍼센트
# PAGE_INCREASE_Y = 0
PAGE_MARGIN_X = 20       # reduced px to ignore near L/R edge
PAGE_MARGIN_Y = 0     # reduced px to ignore near T/B edge


OUTPUT_ZOOM = 1.0        # how much to zoom output relative to *original* image
OUTPUT_DPI = 300         # just affects stated DPI of PNG, not appearance
REMAP_DECIMATE = 16      # downscaling factor for remapping image

ADAPTIVE_WINSZ = 55      # window size for adaptive threshold in reduced px

TEXT_MIN_WIDTH = 300      # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 30      # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 100  # max reduced px thickness of detected text contour

EDGE_MAX_OVERLAP = 1.0   # max reduced px horiz. overlap of contours in span
EDGE_MAX_LENGTH = 100.0  # max reduced px length of edge connecting contours
EDGE_ANGLE_COST = 10.0   # cost of angles in edges (tradeoff vs. length)
EDGE_MAX_ANGLE = 7.5     # maximum change in angle allowed between contours

RVEC_IDX = slice(0, 3)   # index of rvec in params vector
TVEC_IDX = slice(3, 6)   # index of tvec in params vector
CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector

SPAN_MIN_WIDTH = 30      # minimum reduced px width for span
SPAN_PX_PER_STEP = 20    # reduced px spacing for sampling along spans
FOCAL_LENGTH = 1.2       # normalized focal length of camera

DEBUG_LEVEL = 0   # 0=none, 1=some, 2=lots, 3=all, 4=all+maxAdded
DEBUG_OUTPUT = 'screen'    # file, screen, both

WINDOW_NAME = 'Dewarp'   # Window name for visualization

OPTI_SANS_3LINE = True  # True, False 라인 최적화 여부, IMAGE_TYPE이 text 또는 mix일때 동작

IS_RASPI = False   # 라즈베리파이 보드인지 여부.

IMAGE_TYPE = 'auto' # mix(text + paint), text, paint, redLine, auto

THRESH_NUM = 50 # 쓰레쉬홀드

IS_HAND = False  # 손이 있을때

COLOR_MODE = 'color'    # color, thresh, gray

IS_HALF = False # True, False 이미지가 책 전체(2쪽)인지 반(1쪽)인지 여부

CUT_SIDE = True # True, False 이미지 가장자리를 보정처리 할 것인가 여부, 그림일땐 안하는걸 추천.

# nice color palette for visualizing contours, etc.
CCOLORS = [
    (255, 0, 0),
    (255, 63, 0),
    (255, 127, 0),
    (255, 191, 0),
    (255, 255, 0),
    (191, 255, 0),
    (127, 255, 0),
    (63, 255, 0),
    (0, 255, 0),
    (0, 255, 63),
    (0, 255, 127),
    (0, 255, 191),
    (0, 255, 255),
    (0, 191, 255),
    (0, 127, 255),
    (0, 63, 255),
    (0, 0, 255),
    (63, 0, 255),
    (127, 0, 255),
    (191, 0, 255),
    (255, 0, 255),
    (255, 0, 191),
    (255, 0, 127),
    (255, 0, 63),
]

# default intrinsic parameter matrix
K = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]], dtype=np.float32)

CORNERS_G = np.array([], dtype=np.float32)

if IS_RASPI == True:
    from picamera.array import PIRGBArray
    from picamera import PiCamera
    


def debug_show(name, step, text, display):

    if DEBUG_OUTPUT != 'screen':
        filetext = text.replace(' ', '_')
        outfile = name + '_debug_' + str(step) + '_' + filetext + '.png'
        cv2.imwrite(outfile, display)

    if DEBUG_OUTPUT != 'file':

        image = display.copy()
        height = image.shape[0]

        cv2.putText(image, text, (16, height-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 3, cv2.LINE_AA)

        cv2.putText(image, text, (16, height-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, image)

        if IS_RASPI == True:
            cv2.waitKey(0)
        else:
            while cv2.waitKey(5) < 0:
                pass

def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    if not rem:
        return i
    else:
        return i + factor - rem


def pix2norm(shape, pts):
    height, width = shape[:2]
    scl = 2.0/(max(height, width))
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2))*0.5

    return (pts - offset) * scl


def norm2pix(shape, pts, as_integer):
    height, width = shape[:2]
    scl = max(height, width)*0.5
    offset = np.array([0.5*width, 0.5*height],
                      dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval


def fltp(point):
    return tuple(point.astype(int).flatten())


def draw_correspondences(img, dstpoints, projpts):

    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)

    for pts, color in [(projpts, (255, 0, 0)),
                       (dstpoints, (0, 0, 255))]:

        for point in pts:
            cv2.circle(display, fltp(point), 3, color, -1, cv2.LINE_AA)

    for point_a, point_b in zip(projpts, dstpoints):
        cv2.line(display, fltp(point_a), fltp(point_b),
                 (255, 255, 255), 1, cv2.LINE_AA)

    return display


def get_default_params(corners, ycoords, xcoords):

    # page width and height
    page_width = np.linalg.norm(corners[1] - corners[0])
    page_height = np.linalg.norm(corners[-1] - corners[0])
    rough_dims = (page_width, page_height)

    # our initial guess for the cubic has no slope
    cubic_slopes = [0.0, 0.0]

    # object points of flat page in 3D coordinates
    corners_object3d = np.array([
        [0, 0, 0],
        [page_width, 0, 0],
        [page_width, page_height, 0],
        [0, page_height, 0]])

#    camera_matrix = np.load('camera_mat.npy')
#    dist_coefs = np.load('dist_coefs.npy')

    # estimate rotation and translation from four 2D-to-3D point
    # correspondences
    _, rvec, tvec = cv2.solvePnP(corners_object3d,
                                 corners, K, np.zeros(5))
                                #  corners, camera_matrix, dist_coefs)    


    span_counts = [len(xc) for xc in xcoords]

    params = np.hstack((np.array(rvec).flatten(),
                        np.array(tvec).flatten(),
                        np.array(cubic_slopes).flatten(),
                        ycoords.flatten()) +
                       tuple(xcoords))


    # print("params ori = ", params)

    return rough_dims, span_counts, params


def project_xy(xy_coords, pvec):

    # get cubic polynomial coefficients given
    #
    #  f(0) = 0, f'(0) = alpha
    #  f(1) = 0, f'(1) = beta

    alpha, beta = tuple(pvec[CUBIC_IDX])

    poly = np.array([
        alpha + beta,
        -2*alpha - beta,
        alpha,
        0])

    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))

#    camera_matrix = np.load('camera_mat.npy')
#    dist_coefs = np.load('dist_coefs.npy')

    image_points, _ = cv2.projectPoints(objpoints,
                                        pvec[RVEC_IDX],
                                        pvec[TVEC_IDX],
                                        # camera_matrix, dist_coefs)                                        
                                        K, np.zeros(5))
                                        

    return image_points


def project_keypoints(pvec, keypoint_index):

    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0

    return project_xy(xy_coords, pvec)


def resize_to_screen(src, maxw=1280, maxh=700, copy=False):

    # src.shape는 이미지의 높이와 넓이를 취득 함수. 0=높이 1=넓이
    height, width = src.shape[:2]

    # 기준 크기(1280x700)과 이미지의 크기를 비교해 스케일 값을 구한다.
    scl_x = float(width)/maxw
    scl_y = float(height)/maxh

    # np.ceil 요소별 올림
    # max 요소중 최대값을 돌려줌
    # scl은 scl값중 최대값을 받아 올림 작업을 해서 정수형으로 만듬
    scl = int(np.ceil(max(scl_x, scl_y)))

    # 스케일이 1이상이면(원본 소스가 크면) 리사이즈 작업
    if scl > 1.0:
        # 기준크기(1280x700)로 만들기 위한 비율 계산
        inv_scl = 1.0/scl
        # resize 작업. 상대 크기로 inv_scl 비율만큼 x, y의 사이즈를 줄인다.
        # cv2.INTER_AREA 영역 보간법 : 픽셀간의 관계를 고려해서 리샘플링
        # 즉, 결과 이미지의 픽셀 위치를 입력 이미지의 픽셀 위치에 배치하고 겹치는 영역의 평균을 구해 결과 이미지의 픽셀값으로 사용.
        img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
    elif copy:
        img = src.copy()
    else:
        img = src

    return img, scl


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


def get_page_extents(small):

    # 이미지의 높이, 넓이 취득
    height, width = small.shape[:2]

    if IMAGE_TYPE == 'text':
        xmin, xmax, ymin, ymax, _ = text_mask(small)
        # if xmin - PAGE_MARGIN_X > 0:
        xmin = xmin - PAGE_MARGIN_X
        # else:
        #     xmin = 0
        # if ymin - PAGE_MARGIN_Y > 0:            
        ymin = ymin - PAGE_MARGIN_Y
        # else :
        #     ymin = 0
        # if xmax + PAGE_MARGIN_X < width :
        xmax = xmax + PAGE_MARGIN_X
        # else:
        #     xmax = width
        # if ymax + PAGE_MARGIN_Y < height :
        ymax = ymax + PAGE_MARGIN_Y
        # else:
        #     ymax = height
        
    elif IMAGE_TYPE == 'paint' or IMAGE_TYPE == 'mix':
        xmin, xmax, ymin, ymax= paint_mask(small)
        xmin = xmin
        ymin = ymin
        xmax = xmax
        ymax = ymax
    
    global CORNERS_G
    CORNERS_G = np.array([[xmin,ymin], [xmax, ymin], [xmax,ymax], [xmin, ymax]], dtype=np.float32)
    
    # xmin = PAGE_MARGIN_X #50
    # ymin = PAGE_MARGIN_Y #20
    # xmax = width-PAGE_MARGIN_X
    # ymax = height-PAGE_MARGIN_Y

    # height, width 크기를 지닌 0으로 채워진 배열 생성.
    page = np.zeros((height, width), dtype=np.uint8)
    # page(도화지 느낌)에 흰색(255,255,255)의 선을 가진 좌측 상단 모서리(xmin, ymin)부터 우측 하단 모서리(xmax,ymax)까지 직사각형을 그림
    # 선 두께 -1은 선 두께 없이 모든 내용을 채우는 것을 의미?
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

    # print("page show")
    # cv2.imshow("page", page)

    # 그린 직사각형 4점의 좌표를 저장해 리턴
    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]])

    return page, outline


def get_mask(name, small, pagemask, masktype):
    # name : 파일 이름
    # small : resize(축소)된 이미지
    # pagemask : x축 50, y축 20만큼 마진을 가진 사각형(원래 크기에서)
    # masktype : 'text'

    if IMAGE_TYPE == 'redLine' :
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        red_upper = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255))
        red_lowwer = cv2.inRange(hsv, (170, 100, 100), (179, 255, 255))
        mix_color = cv2.addWeighted(red_upper, 1.0, red_lowwer, 1.0, 0.0)
        dst = cv2.bitwise_and(hsv, hsv, mask = mix_color)
        dst = cv2.cvtColor(dst, cv2.COLOR_HSV2BGR)
        sgray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    else :
        sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    

    if DEBUG_LEVEL >= 3:
        debug_show(name, 0.1, 'sgray', sgray)


    # masktype이 text
    if masktype == 'text':
        # adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None)
        # 적응형 이진화 알고리즘 함수 : 입력 이미지에 따라 임곗값이 스스로 다른값을 할당할 수 있도록 구성된 이미지 알고리즘
        # 이미지에 따라 어떠한 임곗값을 주더라도 이진화 처리가 어려운 이미지가 존재한다. 예를 들어 조명의 변화나 반사가 심한 경우 이미지내 밝기 분포가 달라 국소적으로 임곗값을 적용해야 원하는 결과를 얻을 수 있다.
        # 임곗값(thresh) : 임곗값 보다 낮은 픽셀값은 0(검정), 높은 픽셀값은 최댓값(maxval 여기선 255 흰색)으로 변경(cv2.THRESH_BINARY의 경우)
        # cv2.ADAPTIVE_THRESH_MEAN_C : blockSize 영역의 모든 픽셀에 평균 가중치를 적용
        # cv2.THRESH_BINARY_INV : 반전 이진화 코드. 임곗값을 초과할 경우 0, 아닐경우 maxval(255)로 변경.
        # ADAPTIVE_WINSZ : block size 여기서는 55 -> 55x55 크기 내의 영역을 분석해 적절한 임곗값을 설정
        # 상수값 25 : 상수값이 크면(양수) 밝아지고 작으면(음수) 어두워짐                 
        if IMAGE_TYPE != 'redLine':
            mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     25)
            # 세로선 지우기.
            edges = cv2.Canny(sgray,50,150,apertureSize = 3)
            lines = cv2.HoughLinesP(edges,1,np.pi/180,100, minLineLength=80, maxLineGap=5)
            for i in range(len(lines)):
                for x1,y1,x2,y2 in lines[i]:
                    gapY = np.abs(y2-y1)
                    gapX = np.abs(x2-x1)
                    if gapX < 5 and gapY > 10 :
                        cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),20)               
        
        else :
            _, mask = cv2.threshold(sgray, 1, 255, cv2.THRESH_BINARY)


        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.1, 'thresholded', mask)

        # 팽창 함수. x축9, y축 1만큼. x축을 팽창시켜서 가로줄을 만드는 것을 목적으로 함
        mask = cv2.dilate(mask, box(9, 1))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.2, 'dilated', mask)

        if IMAGE_TYPE != 'redLine' :
            # 침식 함수. x축 1, y축 3만큼 침식(축소) 시키는 걸 목적으로 한다. 블립 제거 목적
            mask = cv2.erode(mask, box(1, 3))

            if DEBUG_LEVEL >= 3:
                debug_show(name, 0.3, 'eroded', mask)
        
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(small, contours, -1, (0,0,255), 3)
        # for i in range(len(contours)):
        #     cv2.putText(small, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255, 0, 0), 1)
        #     print(i, hierarchy[0][i])
        
        # cv2.imshow("small", small)
        # while cv2.waitKey(5) < 0:
        #     pass

    # masktype이 line
    if masktype == 'line':

        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     7)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.4, 'thresholded', mask)

        mask = cv2.erode(mask, box(3, 1), iterations=3)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.5, 'eroded', mask)

        mask = cv2.dilate(mask, box(8, 2))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.6, 'dilated', mask)
    
    if masktype == 'paint':
        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     THRESH_NUM)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.4, 'thresholded', mask)

        mask = cv2.erode(mask, box(3, 1), iterations=3)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.5, 'eroded', mask)

        mask = cv2.dilate(mask, box(5, 1))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.6, 'dilated', mask)
        

    # np.minimum : 요소별 최소값을 돌려줌.
    # pagemask에서 마진의 외각부분을 검정으로 돌려줌. 밖의 하얀 부분을 없애기 위함.
    return np.minimum(mask, pagemask)


def interval_measure_overlap(int_a, int_b):
    return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])


def angle_dist(angle_b, angle_a):

    diff = angle_b - angle_a

    while diff > np.pi:
        diff -= 2*np.pi

    while diff < -np.pi:
        diff += 2*np.pi

    # np.abs : 요소별 절댓값
    return np.abs(diff)


def blob_mean_and_tangent(contour):
    # 윤곽선의 모멘트를 계산. 모멘트는 1XN 또는 Nx1의 형태, contour는 1xN
    moments = cv2.moments(contour)
    
    # 이미지 모멘트는 컨투어에 관한 특징값을 뜻한다. OpenCV에서는 moments 함수로 이미지 모멘트를 구한다. 컨투어 포인트 배열을 입력하면 해당 컨투어의 모멘트를 딕셔너리 타입으로 반환한다. 반환하는 모멘트는 총 24개로 10개의 위치 모멘트, 7개의 중심 모멘트, 7개의 정규화된 중심 모멘트로 이루어져 있다.
    # Spatial Moments : M00, M01, M02, M03, M10, M11, M12, M20, M21, M30
    # Central Moments : Mu02, Mu03, Mu11, Mu12, Mu20, Mu21, Mu30
    # Central Normalized Moments : Nu02, Nu03, Nu11, Nu12, Nu20, Nu21, Nu30
    
    area = moments['m00']   # 0차 모멘트 -> 폐곡선의 면적
    
    # 무게중심 구하기
    if area == 0:
        area = 1
        
    mean_x = moments['m10'] / area  
    mean_y = moments['m01'] / area

    # 2차 중심 모멘트
    moments_matrix = np.array([
        [moments['mu20'], moments['mu11']],
        [moments['mu11'], moments['mu02']]
    ]) / area
    # print ("moments_matrix = ",moments_matrix)

    # svd_u : calculated left singular vectors
    _, svd_u, _ = cv2.SVDecomp(moments_matrix)

    # print ("svd_u = ", svd_u)

    # 무게 중심점
    center = np.array([mean_x, mean_y])
    # tangent : 탄젠트
    # [:, 0] : 첫번째는 모두 두번째는 0번째, flatten() : 입력배열을 1차원 배열로 반환
    tangent = svd_u[:, 0].flatten().copy()

    # print("tangent = ", tangent)
    
    return center, tangent


class ContourInfo(object):

    def __init__(self, contour, rect, mask):

        self.contour = contour # 윤곽선
        self.rect = rect # 윤곽선을 둘러싼 사각형
        self.mask = mask # 윤곽선만을 담은 작은 그림(배경은 검정(0))
        # 윤곽선의 무게 중심점과 탄젠트 구하기.
        self.center, self.tangent = blob_mean_and_tangent(contour)

        # np.arctan2 : 아크탄젠트. 결과값은 각도(θ)
        # p = np.arctan2(y,x)
        self.angle = np.arctan2(self.tangent[1], self.tangent[0])

        clx = [self.proj_x(point) for point in contour]

        lxmin = min(clx)
        lxmax = max(clx)

        self.local_xrng = (lxmin, lxmax)

        # point0은 x축은 contour의 좌측끝, y축은 contour의 중간
        self.point0 = self.center + self.tangent * lxmin
        # point1은 x축은 contour의 우측끝, y축은 contour의 중간
        self.point1 = self.center + self.tangent * lxmax

        self.pred = None
        self.succ = None

    # np.dot은 행렬 곱 , 벡터의 내적을 구할때 쓰임. 
    # 탄젠트와 중심과 외곽선의 거리(?)의 내적?
    def proj_x(self, point):
        return np.dot(self.tangent, point.flatten()-self.center) 

    def local_overlap(self, other):
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))


def generate_candidate_edge(cinfo_a, cinfo_b):

    # we want a left of b (so a's successor will be b and b's
    # predecessor will be a) make sure right endpoint of b is to the
    # right of left endpoint of a.
    # x축을 비교해서 a에는 무조건 좌측(작은값), 우측에서는 b(큰값)이 오도록 수정.
    if cinfo_a.point0[0] > cinfo_b.point1[0]:
        tmp = cinfo_a
        cinfo_a = cinfo_b
        cinfo_b = tmp

    # a 또는 b를 포겠을때 측정 간격
    x_overlap_a = cinfo_a.local_overlap(cinfo_b)
    x_overlap_b = cinfo_b.local_overlap(cinfo_a)

    # 센터 차이
    overall_tangent = cinfo_b.center - cinfo_a.center
    # 각도 계산
    overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])

    # 두 윤곽선의 각도 차이
    delta_angle = max(angle_dist(cinfo_a.angle, overall_angle),
                      angle_dist(cinfo_b.angle, overall_angle)) * 180/np.pi

    # we want the largest overlap in x to be small
    x_overlap = max(x_overlap_a, x_overlap_b)

    # Norm(노름) : 벡터의 길이 혹은 크기를 측정하는 방법(함수)입니다. Norm이 측정한 벡터의 크기는 원점에서 벡터 좌표까지의 거리 혹은 Magnitude라고 합니다.    
    dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)

    # 거리가 100보다 크거나 제일큰 측정 간격 차이가 1보다 크거나 각도가 윤곽선 각도 차이가 7.5도 이상이면 패스
    if (dist > EDGE_MAX_LENGTH or
            x_overlap > EDGE_MAX_OVERLAP or
            delta_angle > EDGE_MAX_ANGLE):
        return None
    else:
        score = dist + delta_angle*EDGE_ANGLE_COST  # 점수를 매기고 리턴
        return (score, cinfo_a, cinfo_b)


def make_tight_mask(contour, xmin, ymin, width, height):

    # 높이, 넓이 만큼의 0으로 채워진 배열을 생성(윤곽선을 감싼 사각형 만큼의 캔버스 느낌)
    tight_mask = np.zeros((height, width), dtype=np.uint8)
    
    # reshape : 새로운 차원의 배열을 생성, 새로운 형태의 배열은 데이터의 총 갯수가 같아야함.
    # -1의 의미는 다른 요소를 먼저 설정하고 거기에 최적화된 값으로 자동 선택 된다. 여기서는 (x, 1, 2)가 선정되고 x값은 자동 선택된다. 
    # tight_contour는 윤곽선을 좌측 상단(0,0)을 기준으로 한 좌표로 최대한 이동해서 그린 선의 좌표가 된다.
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))

    # 윤곽선 그리기 함수
    # drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
    # contourIdx : 지정된 윤곽선 번호만 그림. 음수면 모든 윤곽선을 그린다. 여기서는 0이기 때문에 첫번째 윤곽선을 그린다.
    cv2.drawContours(tight_mask, [tight_contour], 0,
                     (1, 1, 1), -1) 

    return tight_mask


def get_contours(name, small, pagemask, masktype):
    # name : 파일 이름
    # small : resize(축소)된 이미지
    # pagemask : x축 50, y축 20만큼 마진을 가진 사각형(원래 크기에서)
    # masktype : 'text'
    s_height, s_width = small.shape[:2]

    mask = get_mask(name, small, pagemask, masktype)

    if DEBUG_LEVEL >= 3:
        debug_show(name, 0.7, 'get_mask', mask)

    # 윤곽선 검출 함수
    # contours, hierarchy = findContours(image, mode, method, contours=None, hierarchy=None, offset=None)
    # contours : 검출된 윤곽선, hierarchy : 계층 구조
    # mode : 윤곽선을 검출해 어떤 계층 구조의 형태를 사용할지 설정
    # cv2.RETR_EXTERNAL : 최외곽 윤곽선만 검색
    # method : 윤곽점의 표시 방법을 설정
    # cv2.CHAIN_APPROX_NONE : 검출된 윤곽선의 모든 윤곽점들을 좌푯값으로 반환한다. 반환된 좌푯값을 중심으로 8개의 이웃 중 하나 이상의 윤곽점들이 포함되 있다.
    if IS_RASPI == False:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    else :
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_out = []

    # contours(윤곽선의 윤곽점 좌표들)를 하나씩 빼서 처리
    for contour in contours:
        # 윤곽선의 경계 사각형
        # 윤곽선의 경계면을 둘러싸는 사각형을 구한다. 반환되는 결과는 회전이 고려되지 않은 직사각형 형태를 띠는데, 경계면의 윤곽점들을 둘러싸는 최소 사각형의 형태를 띈다.
        # openCV에서 좌표는 좌측 상단이 0,0 x축이 늘어나면 오른쪽으로, y축이 늘어나면 아래쪽으로 좌표가 이동된다.
        # xmin, ymin : 사각형 좌측 상단 좌표 , xmin + width, ymin + height : 사각형 우측 하단 좌표
        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect

        # 외곽선을 분석해서 너무 길거나(15이상 또는 높이의 1.5배 이상) 텍스트가 되기엔 너무 두꺼운(2이상) 얼룩은 무시합니다.
        if (width < TEXT_MIN_WIDTH or
                height < TEXT_MIN_HEIGHT or
                width < TEXT_MIN_ASPECT*height):
            continue
        
        if masktype == 'paint':
            contours_min = np.argmin(contour, axis = 0)
            contours_max = np.argmax(contour, axis = 0)
            # 그림일때 윤곽선이 상하 20% 가운데 있으면 버림
            if (((s_height//10)*8 > contour[contours_min[0][1]][0][1] and
                (s_height//10)*2 < contour[contours_min[0][1]][0][1]) or
                ((s_height//10)*8 > contour[contours_max[0][1]][0][1] and
                (s_height//10)*2 < contour[contours_max[0][1]][0][1])):
                continue
            
        # 윤곽선만을 담은 그림을 생성한다.
        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)

        # sum(axis=0)의 뜻은 x축의 모든 요소를 더해서 배열로 만든다. 
        # -> x=0의 모든 y값을 더함. x=1의 모든 y값을 더함 .... -> 결국 같은 x축의 y값의 합이므로 두께가 된다.
        # max는 요소중 최댓값을 구한다. -> 제일 두꺼운 값을 구한다.
        # text의 제일 두꺼운 길이를 10을 기준으로 했기 때문에 이보다 두꺼우면 text라 보지 않고 무시한다.
        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue
        
        # append : 자료 추가.
        contours_out.append(ContourInfo(contour, rect, tight_mask))
        
    if DEBUG_LEVEL >= 2:
        visualize_contours(name, small, contours_out)
        
    return contours_out

# 폭을 모아준다. 선을 잇는것을 뜻하는 듯.
def assemble_spans(name, small, pagemask, cinfo_list):
    # sorted는 정렬된 리스트를 돌려줌(원본 유지)
    # key=lambda cinfo: cinfo.rect[1] -> cinfo.rect[1](ymin : 좌측 상단의 y값)에 대해서 정렬(내림차순)
    cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])

    # generate all candidate edges
    candidate_edges = []

    # cinfo_list 값을 하나씩 꺼냄. i는 몇번째인지.
    # edge가 될 수 있는 후보군을 선출하고 각각의 점수를 매긴다.
    for i, cinfo_i in enumerate(cinfo_list):
        for j in range(i):  # range(i)는 0~(i-1)까지
            # note e is of the form (score, left_cinfo, right_cinfo)
            edge = generate_candidate_edge(cinfo_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)

    # sort candidate edges by score (lower is better)
    # 점수를 정렬한다. 작을 수록 높은 점수이다.
    candidate_edges.sort()

    # for each candidate edge
    for _, cinfo_a, cinfo_b in candidate_edges:
        # if left and right are unassigned, join them
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a

    # generate list of spans as output
    spans = []

    # until we have removed everything from the list
    while cinfo_list:

        # get the first on the list
        # while문을 돌면서 첫번째 것만 계속 탐색(결국은 다 탐색?)
        cinfo = cinfo_list[0]

        # keep following predecessors until none exists
        # 전의 윤곽선 정보를 저장
        while cinfo.pred:
            cinfo = cinfo.pred

        # start a new span
        cur_span = []

        width = 0.0

        # follow successors til end of span
        while cinfo:
            # remove from list (sadly making this loop *also* O(n^2)
            cinfo_list.remove(cinfo)
            # add to span
            cur_span.append(cinfo)
            # xmin과 xmax의 차이로 폭을 얻음.
            width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
            # set successor(계승)
            cinfo = cinfo.succ

        # add if long enough
        if width > SPAN_MIN_WIDTH:
            spans.append(cur_span)

    if DEBUG_LEVEL >= 2:
        visualize_spans(name, small, pagemask, spans)

    return spans


def sample_spans(shape, spans):

    span_points = []

    for span in spans:

        contour_points = []

        for cinfo in span:
            # mask(윤곽선만 있는 그림)의 x축을 Nx1의 차원으로 변형
            yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
            
            # 요소끼리 곱하고 x축 기준으로 합.
            totals = (yvals * cinfo.mask).sum(axis=0)
            #means의 뜻은? y축의 좌표인듯
            means = totals // cinfo.mask.sum(axis=0)
            # 윤곽선을 둘러싸는 사각형의 좌측 상단 포인트 좌표 취득
            xmin, ymin = cinfo.rect[:2]

            # 20포인트 마다 키포인트 선택
            step = SPAN_PX_PER_STEP
            start = ((len(means)-1) % step) // 2

            # xmin과 ymin을 합쳐서 실제 좌표에 대입해 실제 윤곽선의 키포인트 좌표를 구한다.(mask는 윤곽선만 그린 0,0을 기준으로 한 작은 그림이기에)
            contour_points += [(x+xmin, means[x]+ymin)
                               for x in range(start, len(means), step)]

        # 윤곽선 좌표의 차원 배열을 수정한다.
        contour_points = np.array(contour_points,
                                  dtype=np.float32).reshape((-1, 1, 2))
       
        # 아래는 왜하는 것일까? scl과 offset 적용?
        contour_points = pix2norm(shape, contour_points)

        span_points.append(contour_points)

    return span_points


def keypoints_from_samples(name, small, pagemask, page_outline,
                           span_points):

    all_evecs = np.array([[0.0, 0.0]])
    all_weights = 0

    for points in span_points:
        # PCA의 고유 벡터 추출
        _, evec = cv2.PCACompute(points.reshape((-1, 2)),
                                 None, maxComponents=1)
        
        # 노름 계산을 통해 weight 얻음
        weight = np.linalg.norm(points[-1] - points[0])


        all_evecs += evec * weight
        all_weights += weight

    # span의 평균 고유벡터 추출
    evec = all_evecs / all_weights

    x_dir = evec.flatten()

    if x_dir[0] < 0:
        x_dir = -x_dir

    y_dir = np.array([-x_dir[1], x_dir[0]])


    pagecoords = cv2.convexHull(page_outline)

    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
    pagecoords = pagecoords.reshape((-1, 2))

    px_coords = np.dot(pagecoords, x_dir)
    py_coords = np.dot(pagecoords, y_dir)

    px0 = px_coords.min()
    px1 = px_coords.max()

    py0 = py_coords.min()
    py1 = py_coords.max()

    p00 = px0 * x_dir + py0 * y_dir
    p10 = px1 * x_dir + py0 * y_dir
    p11 = px1 * x_dir + py1 * y_dir
    p01 = px0 * x_dir + py1 * y_dir

    if IMAGE_TYPE == 'paint' or OPTI_SANS_3LINE == True :        
        corners = np.vstack((pix2norm(pagemask.shape, CORNERS_G[0]), pix2norm(pagemask.shape, CORNERS_G[1]), pix2norm(pagemask.shape,CORNERS_G[2]),pix2norm(pagemask.shape, CORNERS_G[3]))).reshape((-1, 1, 2))
    else:
        corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))
    # corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))    

    ycoords = []
    xcoords = []

    for points in span_points:
        pts = points.reshape((-1, 2))
        px_coords = np.dot(pts, x_dir)
        py_coords = np.dot(pts, y_dir)
        ycoords.append(py_coords.mean() - py0)
        xcoords.append(px_coords - px0)

    if DEBUG_LEVEL >= 2:
        visualize_span_points(name, small, span_points, corners)

    return corners, np.array(ycoords), xcoords


def visualize_contours(name, small, cinfo_list):

    # samll과 동일한 크기의 0으로 채워진 배열 생성
    regions = np.zeros_like(small)

    # enumerate : 반복문을 사용할때 몇번째 반복문인지 알 필요가 있을때. j값은 0~ 순서를 나타냄.
    for j, cinfo in enumerate(cinfo_list):
        #윤곽선을 다양한 색깔로 그림.
        cv2.drawContours(regions, [cinfo.contour], 0,
                         CCOLORS[j % len(CCOLORS)], -1)

    # max(axis=2) : z축 기준으로 각열의 요소를 그룹으로 해서 그중 제일 큰 값.
    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    # 색깔선과 아닌것 반반씩 섞어서 합침
    display[mask] = (display[mask]/2) + (regions[mask]/2)


    for j, cinfo in enumerate(cinfo_list):
        color = CCOLORS[j % len(CCOLORS)]
        color = tuple([c/4 for c in color])

        # 무게중심은 하얀 점으로 찍음
        cv2.circle(display, fltp(cinfo.center), 3,
                   (255, 255, 255), 1, cv2.LINE_AA)

        # point0부터 point1까지 하얀 선을 그림
        cv2.line(display, fltp(cinfo.point0), fltp(cinfo.point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    debug_show(name, 1, 'contours', display)


def visualize_spans(name, small, pagemask, spans):

    regions = np.zeros_like(small)

    for i, span in enumerate(spans):
        contours = [cinfo.contour for cinfo in span]
        cv2.drawContours(regions, contours, -1,
                         CCOLORS[i*3 % len(CCOLORS)], -1)
    
    

    mask = (regions.max(axis=2) != 0)

    display = small.copy()

    # 색깔선과 아닌것 반반씩 섞어서 합침
    display[mask] = (display[mask]/2) + (regions[mask]/2)

    # 테두리 검은색으로 칠함
    display[pagemask == 0] //= 4

    debug_show(name, 2, 'spans', display)


def visualize_span_points(name, small, span_points, corners):

    display = small.copy()

    for i, points in enumerate(span_points):

        points = norm2pix(small.shape, points, False)

        mean, small_evec = cv2.PCACompute(points.reshape((-1, 2)),
                                          None,
                                          maxComponents=1)

        dps = np.dot(points.reshape((-1, 2)), small_evec.reshape((2, 1)))
        dpm = np.dot(mean.flatten(), small_evec.flatten())

        point0 = mean + small_evec * (dps.min()-dpm)
        point1 = mean + small_evec * (dps.max()-dpm)

        for point in points:
            cv2.circle(display, fltp(point), 3,
                       CCOLORS[i % len(CCOLORS)], -1, cv2.LINE_AA)

        cv2.line(display, fltp(point0), fltp(point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    cv2.polylines(display, [norm2pix(small.shape, corners, True)],
                  True, (0, 0, 255))

    debug_show(name, 3, 'span points', display)


def imgsize(img):
    height, width = img.shape[:2]
    return '{}x{}'.format(width, height)


def make_keypoint_index(span_counts):

    nspans = len(span_counts)
    npts = sum(span_counts)
    keypoint_index = np.zeros((npts+1, 2), dtype=int)
    start = 1

    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start:start+end, 1] = 8+i
        start = end

    keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans

    return keypoint_index


def optimize_params(name, small, dstpoints, span_counts, params):

    keypoint_index = make_keypoint_index(span_counts)

    def objective(pvec):
        ppts = project_keypoints(pvec, keypoint_index)
        return np.sum((dstpoints - ppts)**2)

    print ("initial objective is ", objective(params))

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 4, 'keypoints before', display)

    print ("optimizing", len(params), "parameters...")
    start_opti = datetime.datetime.now()
    # res = scipy.optimize.root(objective, params, method='lm')
    res = scipy.optimize.minimize(objective, params,
                                  method='Powell')
    end_opti = datetime.datetime.now()
    print ("optimization took", round((end_opti-start_opti).total_seconds(), 2), "sec.")
    print ("final objective is", res.fun)
    params = res.x

    # print("params opti = ", params)

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 5, 'keypoints after', display)

    return params


def get_page_dims(corners, rough_dims, params):

    dst_br = corners[2].flatten()

    dims = np.array(rough_dims)
    def objective(dims):
        proj_br = project_xy(dims, params)
        return np.sum((dst_br - proj_br.flatten())**2)

    res = scipy.optimize.minimize(objective, dims, method='Powell')
    dims = res.x
    print ("got page dims", dims[0], 'x', dims[1])

    return dims


def remap_image(name, img, small, page_dims, params, page_hof):

    height = 0.5 * page_dims[1] * OUTPUT_ZOOM * img.shape[0]
    height = round_nearest_multiple(height, REMAP_DECIMATE)

    width = round_nearest_multiple(height * page_dims[0] / page_dims[1],
                                   REMAP_DECIMATE)

    print ("output will be {}x{}".format(width, height))

    height_small = height // REMAP_DECIMATE
    width_small = width // REMAP_DECIMATE

    page_x_range = np.linspace(0, page_dims[0], width_small)
    page_y_range = np.linspace(0, page_dims[1], height_small)

    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)

    page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)),
                                page_y_coords.flatten().reshape((-1, 1))))

    page_xy_coords = page_xy_coords.astype(np.float32)

    image_points = project_xy(page_xy_coords, params)
    image_points = norm2pix(img.shape, image_points, False)
    
    image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
    image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)

    image_x_coords = cv2.resize(image_x_coords, (width, height),
                                interpolation=cv2.INTER_CUBIC)

    image_y_coords = cv2.resize(image_y_coords, (width, height),
                                interpolation=cv2.INTER_CUBIC)
    mapX2, mapY2 = cv2.convertMaps(image_x_coords, image_y_coords, cv2.CV_16SC2)

    remapped = cv2.remap(img, image_x_coords, image_y_coords,
                        cv2.INTER_CUBIC,
                        # None, cv2.BORDER_REPLICATE)
                        None, cv2.BORDER_CONSTANT, borderValue = [255,255,255])
    
    if CUT_SIDE == True:
        remapped = cut_side(remapped, page_hof, image_x_coords, img)

    if COLOR_MODE == 'color':
        if IMAGE_TYPE == 'paint' or IMAGE_TYPE == 'mix':
            f_img = cut_edge(remapped)
        else:
            f_img = remapped    

        # pil_image = Image.fromarray(thresh)
        # pil_image = pil_image.convert('1')

        file_name = name + page_hof + '_ckbs_color.png'
        # pil_image.save(threshfile, dpi=(OUTPUT_DPI, OUTPUT_DPI))
        cv2.imwrite(file_name, f_img)
        
    if COLOR_MODE == 'thresh':
        if IMAGE_TYPE == 'paint' or IMAGE_TYPE == 'mix':
            remapped = cut_edge(remapped)
        img_gray = cv2.cvtColor(remapped, cv2.COLOR_RGB2GRAY)                            
        f_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, ADAPTIVE_WINSZ, 25)

        pil_image = Image.fromarray(f_img)
        pil_image = pil_image.convert('1')

        file_name = name + page_hof + '_ckbs_thresh.png'
        pil_image.save(file_name, dpi=(OUTPUT_DPI, OUTPUT_DPI))
    
    if COLOR_MODE == 'gray':
        if IMAGE_TYPE == 'paint' or IMAGE_TYPE == 'mix':
            remapped = cut_edge(remapped)                
        f_img = cv2.cvtColor(remapped, cv2.COLOR_RGB2GRAY)
        pil_image = Image.fromarray(f_img)
        pil_image = pil_image.convert('1')

        file_name = name + page_hof + '_ckbs_gray.png'
        pil_image.save(file_name, dpi=(OUTPUT_DPI, OUTPUT_DPI))
        
    if DEBUG_LEVEL >= 1:
        height = small.shape[0]
        width = int(round(height * float(f_img.shape[1])/f_img.shape[0]))
        display = cv2.resize(f_img, (width, height),
                             interpolation=cv2.INTER_AREA)
        debug_show(name, 6, 'output', display)

    return file_name

# 문자의 끝을 구해 마스킹
def text_mask(small):
    global DEBUG_LEVEL
    global TEXT_MIN_WIDTH
    global TEXT_MIN_HEIGHT
    global TEXT_MAX_THICKNESS
    
    tmp_debug_level = DEBUG_LEVEL
    DEBUG_LEVEL = 0
    tmp_text_min_width = TEXT_MIN_WIDTH
    TEXT_MIN_WIDTH = 10
    tmp_text_min_height = TEXT_MIN_HEIGHT
    TEXT_MIN_HEIGHT = 1
    tmp_text_max_thickness = TEXT_MAX_THICKNESS
    TEXT_MAX_THICKNESS = 10
    
    name = 'mask_temp'
    spans = []
    height, width = small.shape[:2]
    pagemask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(pagemask, (0, 0), (width, height), (255, 255, 255), -1)
    
    # 윤곽선 정보 검출
    cinfo_list = get_contours(name, small, pagemask, 'text')
    # 윤곽선을 길게 합침
    spans_text = assemble_spans(name, small, pagemask, cinfo_list)

    # 텍스트가 없어서 span 갯수가 부족시 line을 디텍팅 하도록 설계
    if len(spans_text) < 3:
        print ('  detecting lines because only', len(spans_text), 'text spans')
        cinfo_list = get_contours(name, small, pagemask, 'line')
        spans2 = assemble_spans(name, small, pagemask, cinfo_list)
        if len(spans2) > len(spans_text):
            spans_text = spans2
    
    # 윤곽선 각각의 넓이와 y축 위치 구하기.
    span_with_min = []
    span_with_max = []
    span_height_min = []
    span_height_max = []
    
    for i, span in enumerate(spans_text):
        # 한줄 빼오기
        contours = [cinfo.contour for cinfo in span]
        # 빼온거에서 젤 왼쪽과 오른쪽 을 빼서 길이 재고 y축 값 구하기.
        for i, _ in enumerate(range(len(contours))):
            contours_min = np.argmin(contours[i], axis = 0)
            contours_max = np.argmax(contours[i], axis = 0)
            span_with_min.append(contours[i][contours_min[0][0]][0][0])
            span_with_max.append(contours[i][contours_max[0][0]][0][0])
            span_height_min.append(contours[i][contours_min[0][1]][0][1])
            span_height_max.append(contours[i][contours_max[0][1]][0][1])

    DEBUG_LEVEL = tmp_debug_level
    TEXT_MIN_WIDTH = tmp_text_min_width
    TEXT_MIN_HEIGHT = tmp_text_min_height
    TEXT_MAX_THICKNESS = tmp_text_max_thickness
    
    # 리턴 값은 x축 최소값, x축 최대값, y축 최소값, y축 최대값
    return np.min(span_with_min), np.max(span_with_max), np.min(span_height_min), np.max(span_height_max), len(spans_text)

# 엣지의 끝을 구해 마스킹
def paint_mask(small):
    global DEBUG_LEVEL
    global TEXT_MIN_WIDTH
    global TEXT_MIN_HEIGHT
    global TEXT_MAX_THICKNESS
    global THRESH_NUM
    
    tmp_debug_level = DEBUG_LEVEL
    DEBUG_LEVEL = 0
    tmp_text_min_width = TEXT_MIN_WIDTH
    TEXT_MIN_WIDTH = 300
    tmp_text_min_height = TEXT_MIN_HEIGHT
    TEXT_MIN_HEIGHT = 30
    tmp_text_max_thickness = TEXT_MAX_THICKNESS
    TEXT_MAX_THICKNESS = 100
    tmp_thresh_num = THRESH_NUM
    THRESH_NUM = 50
    
    if IS_HAND == True:
        # BGR -> YCrCb 변환
        YCrCb = cv2.cvtColor(small,cv2.COLOR_BGR2YCrCb)
        # 피부 검출
        mask_hand = cv2.inRange(YCrCb,np.array([0,133,77]),np.array([255,173,127]))
        mask_hand = cv2.bitwise_not(mask_hand)
        # 피부 색 나오도록 연산
        small = cv2.bitwise_and(small,small,mask=mask_hand)    
    
    name = 'mask_temp'
    spans = []
    height, width = small.shape[:2]
    pagemask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(pagemask, (0, 0), (width, height), (255, 255, 255), -1)
    
    while True:
        cinfo_list = get_contours(name, small, pagemask, 'paint')
        spans = assemble_spans(name, small, pagemask, cinfo_list)            
        THRESH_NUM = THRESH_NUM - 1            
        if len(spans) == 2:
            break
        if THRESH_NUM == 0:
            THRESH_NUM = 100
            continue
        if THRESH_NUM == 50:
            break
    
    # 윤곽선 각각의 넓이와 y축 위치 구하기.
    span_with_min = []
    span_with_max = []
    span_height_min = []
    span_height_max = []
    for i, span in enumerate(spans):
        # 한줄 빼오기
        contours = [cinfo.contour for cinfo in span]
        # 빼온거에서 젤 왼쪽과 오른쪽 을 빼서 길이 재고 y축 값 구하기.
        contours_min = np.argmin(contours[0], axis = 0)
        contours_max = np.argmax(contours[0], axis = 0)
        span_with_min.append(contours[0][contours_min[0][0]][0][0])
        span_with_max.append(contours[0][contours_max[0][0]][0][0])
        span_height_min.append(contours[0][contours_min[0][1]][0][1])
        span_height_max.append(contours[0][contours_max[0][1]][0][1])

    DEBUG_LEVEL = tmp_debug_level
    TEXT_MIN_WIDTH = tmp_text_min_width
    TEXT_MIN_HEIGHT = tmp_text_min_height
    TEXT_MAX_THICKNESS = tmp_text_max_thickness
    THRESH_NUM = tmp_thresh_num
    
    # 리턴 값은 x축 최소값, x축 최대값, y축 최소값, y축 최대값
    return np.min(span_with_min), np.max(span_with_max), np.min(span_height_min), np.max(span_height_max)

# 상하 부분 자르기
def cut_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    # 선을 추출
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100, minLineLength=80, maxLineGap=5)
    tmp_candy = []
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            gapY = np.abs(y2-y1)
            gapX = np.abs(x2-x1)
            # 선의 x축의 차이가 30이상 y축의 차이가 10이하면 가로줄로 판별
            if gapX > 30 and gapY < 10 :
                tmp_candy.append(y1)

    _, width = img.shape[:2]
    # 제일 처음 가로줄과 마지막 가로줄이 엣지라고 판단
    img = img[np.min(tmp_candy):np.max(tmp_candy),0:width]
    return img

def cut_side(src, side, image_x_coords, img):

    height, width = src.shape[:2]

    corners_x1 = np.where(image_x_coords[0,:].astype(int) ==img.shape[1])
    corners_x2 = np.where(image_x_coords[height-1,:].astype(int) ==img.shape[1])
    
    cut_L = False
    
    if corners_x1[0].size == 0 and corners_x2[0].size == 0:
        print("no side")
    elif corners_x1[0].size == 0:
        corners_x1 = [[width]]
        # corners_x1[0][0] = width
        print("corners_x1 is None")
        cut_L = True
    elif corners_x2[0].size == 0:
        corners_x2 = [[width]]
        # corners_x2[0][0] = width
        print("corners_x2 is None")
        cut_L = True
    else:
        cut_L = True
    if cut_L == True:
        topLeft = [0,0]
        bottomLeft = [0, height]
        topRight = [corners_x1[0][0], 0]
        bottomRight = [corners_x2[0][0], height]
        src_pts = np.float32([topLeft, topRight, bottomLeft, bottomRight])
        if corners_x1[0][0] >= corners_x2[0][0]:
            dst_pts = np.float32([[0,0], [corners_x1[0][0],0], [0, height], [corners_x1[0][0], height]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            src = cv2.warpPerspective(src, M, (corners_x1[0][0], height))
        else:
            dst_pts = np.float32([[0,0], [corners_x2[0][0],0], [0, height], [corners_x2[0][0], height]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            src = cv2.warpPerspective(src, M, (corners_x2[0][0], height))
        print("cut side _L")
        
    corners_x1 = np.where(image_x_coords[0,:].astype(int) == 0)
    corners_x2 = np.where(image_x_coords[height-1,:].astype(int) == 0)
        
    if corners_x1[0].size == 0 and corners_x2[0].size == 0:
        print("no side")
        return src
    elif corners_x1[0].size == 0:
        corners_x1 = [[0]]
    elif corners_x2[0].size == 0:
        corners_x2 = [[0]]
    topLeft = [corners_x1[0][0],0]
    bottomLeft = [corners_x2[0][0], height]
    topRight = [width, 0]
    bottomRight = [width, height]
    src_pts = np.float32([topLeft, topRight, bottomLeft, bottomRight])
    if corners_x1[0][0] <= corners_x2[0][0]:
        dst_pts = np.float32([[corners_x1[0][0],0], [width,0], [corners_x1[0][0], height], [width, height]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        src = cv2.warpPerspective(src, M, (width - corners_x1[0][0], height))
    else:
        dst_pts = np.float32([[corners_x2[0][0],0], [width,0], [corners_x2[0][0], height], [width, height]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        src = cv2.warpPerspective(src, M, (width -corners_x2[0][0], height))        
    print("cut side _R")


    return src
    # topLeft = None
    
    # small, scl = resize_to_screen(src)
    
    # if DEBUG_LEVEL >= 4:
    #     debug_show('before_cutSide', 7, 'before_cutSide', small)    
        
    # img_height, img_width = small.shape[:2]
        
    # gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    
    # edges = cv2.Canny(gray,50,50,apertureSize = 3)
    
    # if DEBUG_LEVEL >= 4:
    #     debug_show('Canny', 7, 'Canny', edges)

    # lines = cv2.HoughLines(edges,1,np.pi/180,150)
    # if lines is None:
    #     return src
    # for line in lines:
    #     rho,theta = line[0]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a))
    #     if DEBUG_LEVEL >= 4:
    #         cv2.line(small,(x1,y1),(x2,y2),(0,0,255),3)
    #     angle = np.arctan2(y1 - y2, x1 - x2)
    #     delta_angle = np.abs(angle*180/np.pi)

    #     # https://sijoo.tistory.com/34
    #     # y=0일때, y=height일때 교점 x1,x2
    #     x1 = int(np.ceil(rho / np.cos(theta)))
    #     x2 = int(np.ceil((rho - img_height*np.sin(theta) ) / np.cos(theta)))

    #     if delta_angle > 80 and delta_angle < 100:
    #         if side == '_R'and x1 < img_width//10 * 3 :
    #             topLeft = [x1*scl ,0]
    #             bottomLeft = [x2*scl , height]
    #             topRight = [width, 0]        
    #             bottomRight = [width, height]
    #             break                
    #         elif side == '_L' and x1 > img_width//10 * 7:
    #             topLeft = [0 ,0]
    #             bottomLeft = [0 , height]
    #             topRight = [x1*scl, 0]        
    #             bottomRight = [x2*scl, height]                
    #             break
    #         # elif side == '' and x1 < img_width//10 * 3 :
    #         #     topLeft = [x1*scl ,0]
    #         #     bottomLeft = [x2*scl , height]
    #         #     topRight = [width, 0]        
    #         #     bottomRight = [width, height]
    #         #     break
    #         # elif x1 > img_width//10 * 7:
    #         #     topLeft = [0 ,0]
    #         #     bottomLeft = [0 , height]
    #         #     topRight = [x1*scl, 0]        
    #         #     bottomRight = [x2*scl, height]                
    #         #     break                
   
    # if topLeft is None:
    #     return src

    # if DEBUG_LEVEL >= 4:
    #     debug_show('HoughLines', 7, 'HoughLines', small)    
    
    # src_pts = np.float32([topLeft, topRight, bottomLeft, bottomRight])
    # dst_pts = np.float32([[0,0], [width,0], [0, height], [width, height]])
    
    # M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # dst = cv2.warpPerspective(src, M, (width, height))

    # return dst

def cut_half(src):
    
    height, width = src.shape[:2]

    small, scl = resize_to_screen(src)
       
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    for i,_ in enumerate(range(3)):
        edges = cv2.Canny(gray,50,150-(i*50),apertureSize = 3)

        # 선을 추출
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100, minLineLength=50, maxLineGap=10)
        tmp_candy = []
        tmp_abs = []
        for k in range(len(lines)):
            for x1,y1,x2,y2 in lines[k]:
                if DEBUG_LEVEL >= 4:
                    cv2.line(small,(x1,y1),(x2,y2),(0,0,255),3)
                gapY = np.abs(y2-y1)
                angle = np.arctan2(y1 - y2, x1 - x2)
                delta_angle = np.abs(angle*180/np.pi)
                # 세로줄 판별
                if delta_angle > 85 and delta_angle < 95 and gapY > 30 :
                    tmp_candy.append([x1,x2])
                    tmp_abs.append(np.abs(x1- width//2))

        # 세로선이 없거나 40%~60% 사이에 없으면 이미지의 반으로 자름.
        if (tmp_candy == [] or
            (tmp_candy[np.argmin(tmp_abs)][0]+tmp_candy[np.argmin(tmp_abs)][1])//2 < (width//10)*4 or 
            (tmp_candy[np.argmin(tmp_abs)][0]+tmp_candy[np.argmin(tmp_abs)][1])//2 > (width//10)*6) :
            if i != 2:
                continue
            left_img = src[0:height,0:width//2]
            right_img = src[0:height,width//2+1:width]

        else :
            if scl > 1.0:
                left_img = src[0:height,0:((tmp_candy[np.argmin(tmp_abs)][0]+tmp_candy[np.argmin(tmp_abs)][1])*scl)//2]
                right_img = src[0:height,((tmp_candy[np.argmin(tmp_abs)][0]+tmp_candy[np.argmin(tmp_abs)][1])*scl)//2 + 1:width]
            else:
                left_img = src[0:height,0:(tmp_candy[np.argmin(tmp_abs)][0]+tmp_candy[np.argmin(tmp_abs)][1])//2]
                right_img = src[0:height,(tmp_candy[np.argmin(tmp_abs)][0]+tmp_candy[np.argmin(tmp_abs)][1])//2 + 1:width]
    
    if DEBUG_LEVEL >= 4:
        debug_show('cut_half', 7, 'cut_half', small)  

    return left_img, right_img
    
# 색상으로 그림인지 판별
def is_paint(img):
    small, _ = resize_to_screen(img, maxw=600, maxh=600)
    pixels = np.float32(small.reshape(-1, 3))
    n_colors = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    
    indices = np.argsort(counts)[::-1]  
    
    img_paint = False
    
    for i in range(3):
        if np.all(palette[indices[i+1]] < 140):
            img_paint = True
              
    if DEBUG_LEVEL >= 4:
        import matplotlib.pyplot as plt
        average = small.mean(axis=0).mean(axis=0)
        avg_patch = np.ones(shape=small.shape, dtype=np.uint8)*np.uint8(average)         
        freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
        rows = np.int_(small.shape[0]*freqs)
        dom_patch = np.zeros(shape=small.shape, dtype=np.uint8)
        for i in range(len(rows) - 1):
            dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))
        ax0.imshow(avg_patch)
        ax0.set_title('Average color')
        ax0.axis('off')
        ax1.imshow(dom_patch)
        ax1.set_title('Dominant colors')
        ax1.axis('off')
        plt.show(fig)
    
    return img_paint

def img_process(num, start, total_len, img, page_hof, basename, img_paint):
    global THRESH_NUM, TEXT_MIN_WIDTH, TEXT_MIN_HEIGHT, TEXT_MAX_THICKNESS, IMAGE_TYPE
    # 결과를 파일이 아닌 스크린으로 보여줄때 윈도우 창 이름 설정
    if DEBUG_LEVEL > 0 and DEBUG_OUTPUT != 'file':
        cv2.namedWindow(WINDOW_NAME)

    # if PAGE_INCREASE_X > 0:
    #     height, width = img.shape[:2]
    #     page_w = np.ones((height, (width * PAGE_INCREASE_X)//100, 3), dtype=np.uint8)
    #     _, page_w = cv2.threshold(page_w, 0, 255, cv2.THRESH_BINARY)        
    #     img = np.hstack((img, page_w))
    #     img = np.hstack((page_w, img))
    # if PAGE_INCREASE_Y > 0:
    #     height, width = img.shape[:2]
    #     page_h = np.ones(((height * PAGE_INCREASE_Y)//100, width, 3), dtype=np.uint8)
    #     _, page_h = cv2.threshold(page_h, 0, 255, cv2.THRESH_BINARY)        
    #     img = np.vstack((img, page_h))
    #     img = np.vstack((page_h, img))     
    
    # resize 기준크기(1280x700)의 크기로 보정한다.(작으면 보정 안함)
    small, _ = resize_to_screen(img)
    # name 확장자를 뺀 파일 이름
    name, _ = os.path.splitext(basename)
        
    print ("loaded", basename, "with size", imgsize(img),)
    print ('and resized to', imgsize(small))
        
    if DEBUG_LEVEL >= 3:
        debug_show(name, 0.0, 'original', small)
    
    if IMAGE_TYPE == 'auto':
        _ ,_ ,_ ,_ , tmp_len = text_mask(small)
        if tmp_len < 15 and img_paint == True:
            IMAGE_TYPE = 'paint'
        elif tmp_len < 3:
            IMAGE_TYPE = 'paint'
        else:
            IMAGE_TYPE = 'text'
        print("IMAGE_TYPE = ", IMAGE_TYPE)

    
    # 마진 x축 50, y축 20 만큼 있는 사각형을 그림.
    pagemask, page_outline = get_page_extents(small)

    spans = []
    
    if IMAGE_TYPE == 'paint' or IMAGE_TYPE == 'mix':
        while True:
            cinfo_list = get_contours(name, small, pagemask, 'paint')
            spans = assemble_spans(name, small, pagemask, cinfo_list)            
            THRESH_NUM = THRESH_NUM - 1
            if len(spans) == 2:
                break
            if THRESH_NUM == 0:
                THRESH_NUM = 100
                continue
            if THRESH_NUM == 50:
                break
                                
    if IMAGE_TYPE == 'text' or IMAGE_TYPE == 'mix':
        TEXT_MIN_WIDTH = 15
        TEXT_MIN_HEIGHT = 2
        TEXT_MAX_THICKNESS = 10        
        # 윤곽선 정보 검출
        cinfo_list = get_contours(name, small, pagemask, 'text')
        # 윤곽선을 길게 합침
        spans_text = assemble_spans(name, small, pagemask, cinfo_list)
    
        # 텍스트가 없어서 span 갯수가 부족시 line을 디텍팅 하도록 설계
        if len(spans_text) < 3:
            print ('  detecting lines because only', len(spans_text), 'text spans')
            cinfo_list = get_contours(name, small, pagemask, 'line')
            spans2 = assemble_spans(name, small, pagemask, cinfo_list)
            if len(spans2) > len(spans_text):
                spans_text = spans2
        
        # 3라인 선택. 모든 라인수가 3줄이상.
        if OPTI_SANS_3LINE == True and len(spans_text) > 3:
            # 윤곽선 각각의 넓이와 y축 위치 구하기.
            span_with = []
            span_height = []
            for i, span in enumerate(spans_text):
                # 한줄 빼오기
                contours = [cinfo.contour for cinfo in span]
                # 빼온거에서 젤 왼쪽과 오른쪽 을 빼서 길이 재고 y축 값 구하기.
                tmp_xmin = []
                tmp_xmax = []
                tmp_ymin = []
                tmp_ymax = []
                for i,_ in enumerate(range(len(contours))):
                    contours_min = np.argmin(contours[i], axis = 0)
                    contours_max = np.argmax(contours[i], axis = 0)
                    tmp_xmax.append(contours[i][contours_max[0][0]][0][0])
                    tmp_xmin.append(contours[i][contours_min[0][0]][0][0])
                    tmp_ymax.append(contours[i][contours_max[0][1]][0][1])
                    tmp_ymin.append(contours[i][contours_min[0][1]][0][1])
                
                span_with.append(np.max(tmp_xmax)-np.min(tmp_xmin))
                # 높이는 중간값
                span_height.append((np.max(tmp_ymax)+np.min(tmp_ymin))//2)

            # 이미지의 높이와 폭, 
            img_height, _ = small.shape[:2]

            # 1,2,3 분할 할 곳의 외곽선을 갯수 구함
            s0 = 0
            s1 = 0
            s2 = 0
            for i, s_height in enumerate(span_height):
                if s_height > 0 and s_height < (img_height//10)*3:
                    s0 = s0 + 1
                elif s_height > (img_height//10)*3 and s_height < (img_height//10)*7:
                    s1 = s1 + 1
                else :
                    s2 = s2 + 1            

            # 갯수 대비해서 외곽선의 정보를 각각의 리스트에 담는다.
            tmp_one = []
            tmp_two = []
            tmp_three = []
            for i in range(s0):
                tmp_one.append(span_with.pop(0))
            for i in range(s1):
                tmp_two.append(span_with.pop(0))
            for i in range(s2):
                tmp_three.append(span_with.pop(0))

            # 3분할 영역에서 가장 긴것을 정해서 spans에 추가.
            if len(tmp_one) != 0:
                spans.append(spans_text[np.argmax(tmp_one)])
            if len(tmp_two) != 0:
                spans.append(spans_text[len(tmp_one)+np.argmax(tmp_two)])
            if len(tmp_three) != 0:
                spans.append(spans_text[len(tmp_one)+len(tmp_two)+np.argmax(tmp_three)])
        
        # 3라인 선택이 아니면
        else :
            for tmp_spans in spans_text:
                spans.append(tmp_spans)
        
    # 윤곽선이 부족할때 처리를 할 수 없기 때문에 이미지 처리를 하지 않음
    if len(spans) < 1:
        print ('skipping', name, 'because only', len(spans), 'spans')
        return
    
    # shape는 axb 처럼 몇차원인지 반환
    # 윤곽선의 키 포인트를 반환.
    span_points = sample_spans(small.shape, spans)
    
    print ('  got', len(spans), 'spans',)
    print ('with', sum([len(pts) for pts in span_points]), 'points.')
        
    corners, ycoords, xcoords = keypoints_from_samples(name, small,
                                                    pagemask,
                                                    page_outline,
                                                    span_points)
    
    rough_dims, span_counts, params = get_default_params(corners,
                                                        ycoords, xcoords)
    
    dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) +
                        tuple(span_points))
    params = optimize_params(name, small,
                            dstpoints,
                            span_counts, params)
    
    page_dims = get_page_dims(corners, rough_dims, params)
    
    outfile = remap_image(name, img, small, page_dims, params, page_hof)
    
    print ('  wrote', outfile)
 
    if num+1 == total_len:
        end = datetime.datetime.now()
        print ("total took", round((end-start).total_seconds(), 2), "sec.")

def main_raspi():
    start = datetime.datetime.now()

    if IS_HALF == True:
        total_len = 1
    else:
        total_len = 2

    th = []
    l = 0

    camera = PiCamera()
    camera.resolution(480,320)
    camera.framerate = 30
    rawCapture = PIRGBArray(camera, size =(480,320))
    time.sleep(0.1)
    
    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        image = frame.array
        
        cv2.imshow('frame', image)
        key = cv2.waitKey(1) & 0xff
        rawCapture.truncate(0)
        if key==88:
            break
                
    th.append(0)
    # 파일 이름으로 openCV에서 파일을 읽음
    img = cv2.imread(frame)
    img_paint = is_paint(img)
    
    # basename 확장자 까지의 이름
    basename = "book"
    if IS_HALF == False:
        l_img, r_img = cut_half(img)
        th[l] = Process(target=img_process, args=(0, start, total_len, l_img,'_L', basename, img_paint))
        th[l].start()
        l = l+1
        th.append(0)
        th[l] = Process(target=img_process, args=(1, start, total_len, r_img, '_R', basename, img_paint))
        th[l].start()
        l = l+1
    else:            
        th[l] = Process(target=img_process, args=(0, start, total_len, img, '', basename, img_paint))
        th[l].start()
        l = l+1

    for k,_ in enumerate(range(l)):
        th[k].join()    

def main():
    start = datetime.datetime.now()
    # 파라미터 값은 이미지 파일 하나 이상이 들어와야 함
    if len(sys.argv) < 2:
        print ("usage:", sys.argv[0], "IMAGE1 [IMAGE2 ...]")
        sys.exit(0)
    
    total_len = len(sys.argv) - 1

    th = []
    l = 0
    # input 파일 개수만큼 for문을 돌림 (한개일땐 한번만...)
    for i, imgfile in enumerate(sys.argv[1:]):
        th.append(0)
        # 파일 이름으로 openCV에서 파일을 읽음
        img = cv2.imread(imgfile)
        if IMAGE_TYPE == 'auto':
            img_paint = is_paint(img)
        else:
            img_paint = False
        
        # basename 확장자 까지의 이름
        basename = os.path.basename(imgfile)
        if IS_HALF == False:
            l_img, r_img = cut_half(img)
            th[l] = Process(target=img_process, args=(i, start, total_len, l_img,'_L', basename, img_paint))
            th[l].start()
            l = l+1
            th.append(0)
            th[l] = Process(target=img_process, args=(i, start, total_len, r_img, '_R', basename, img_paint))
            th[l].start()
            l = l+1
        else:            
            th[l] = Process(target=img_process, args=(i, start, total_len, img, '', basename, img_paint))
            th[l].start()
            l = l+1

    for k,_ in enumerate(range(l)):
        th[k].join()

if __name__ == '__main__':
    # if IS_RASPI == True:
    #     main_raspi()
    # else:
    #     main()
    main()