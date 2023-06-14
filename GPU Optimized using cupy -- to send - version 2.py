import cupy as cp
import cv2
import os, psutil
import glob
from moviepy.editor import VideoFileClip
from moviepy import *
import time

# Initialize variables for calculating average memory usage and process time
total_memory_usage = 0
total_process_time = 0
process_time = 0 
k = 2

for i in range(k):
    # Color selection (HSL)
    def convert_hsl(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    def HSL_color_selection(image):
        converted_image = convert_hsl(image)
        lower_threshold = cp.uint8([0, 200, 0])
        upper_threshold = cp.uint8([255, 255, 255])
        white_mask = cv2.inRange(cp.asnumpy(converted_image), cp.asnumpy(lower_threshold), cp.asnumpy(upper_threshold))

        lower_threshold = cp.uint8([10, 0, 100])
        upper_threshold = cp.uint8([40, 255, 255])
        yellow_mask = cv2.inRange(cp.asnumpy(converted_image), cp.asnumpy(lower_threshold), cp.asnumpy(upper_threshold))

        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        return masked_image

    # Canny edge detection
    def gray_scale(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def gaussian_smoothing(image, kernel_size=13):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def canny_detector(image, low_threshold=50, high_threshold=150):
        return cv2.Canny(image, low_threshold, high_threshold)

    # Region of interest
    def region_selection(image):
        mask = cp.zeros_like(image)
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        rows, cols = image.shape[:2]
        bottom_left = [cols * 0.1, rows * 0.95]
        top_left = [cols * 0.4, rows * 0.6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right = [cols * 0.6, rows * 0.6]
        vertices = cp.array([[bottom_left, top_left, top_right, bottom_right]], dtype=cp.int32)
        cv2.fillPoly(mask, cp.asnumpy(vertices), ignore_mask_color)
        masked_image = cv2.bitwise_and(image, cp.asnumpy(mask))
        return masked_image

    # Hough transform
    def hough_transform(image):
        rho = 1
        theta = cp.pi / 180
        threshold = 20
        minLineLength = 20
        maxLineGap = 300
        return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                               minLineLength=minLineLength, maxLineGap=maxLineGap)

    def draw_lines(image, lines, color=[0, 255, 0], thickness=2):
        image = cp.copy(image)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(cp.asnumpy(image), (x1, y1), (x2, y2), color, thickness)
        return image

    # Averaging and extrapolating the lane lines
    def average_slope_intercept(lines):
        left_lines = []
        left_weights = []
        right_lines = []
        right_weights = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                length = cp.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
        left_lane = cp.dot(left_weights, left_lines) / cp.sum(left_weights) if len(left_weights) > 0 else None
        right_lane = cp.dot(right_weights, right_lines) / cp.sum(right_weights) if len(right_weights) > 0 else None
        return left_lane, right_lane

    def pixel_points(y1, y2, line):
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))

    def lane_lines(image, lines):
        left_lane, right_lane = average_slope_intercept(lines)
        y1 = image.shape[0]
        y2 = y1 * 0.6
        left_line = pixel_points(y1, y2, left_lane)
        right_line = pixel_points(y1, y2, right_lane)
        return left_line, right_line

    def draw_lane_lines(image, lines, color=[0, 255, 0], thickness=12):
        line_image = cp.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(cp.asnumpy(line_image), *line, color, thickness)
        return cv2.addWeighted(image, 1.0, cp.asnumpy(line_image), 1.0, 0.0)

    def frame_processor(image):
        color_select = HSL_color_selection(image)
        gray = gray_scale(color_select)
        smooth = gaussian_smoothing(gray)
        edges = canny_detector(smooth)
        region = region_selection(edges)
        hough = hough_transform(region)
        result = draw_lane_lines(image, lane_lines(image, hough))
        return result

    def process_video(test_video, output_video):
        start_time = time.time()
        input_video = VideoFileClip(os.path.join('test_videos', test_video), audio=False)
        processed = input_video.fl_image(frame_processor)
        processed.write_videofile(os.path.join('output_videos', output_video), audio=False)
        end_time = time.time()
        global process_time
        process_time = end_time - start_time
        process_time_out = process_time
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 ** 2

    def optimize_code():
        cp = cp.get_array_module(masked_image)
        cp.asnumpy = cp.asarray

        total_memory_usage = 0
        total_process_time = 0
        process_time = 0
        k = 2

        for i in range(k):
            process_video('solidYellowLeft.mp4', 'solidYellowLeft_output.mp4')

            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 ** 2
            total_memory_usage += memory_usage

            total_process_time += process_time

        print("Total Memory usage: ", total_memory_usage / k, " MB")  # in MB
        print('Total process time: {} seconds'.format(total_process_time / k))

    optimize_code()