#Importing some useful packages
import numpy as np
import cv2
import os, psutil
import glob
from moviepy.editor import VideoFileClip
from moviepy import *
import time
import cupy as cp
import cupyx.scipy.ndimage

#color selection (HSL)
def convert_hsl(image):
    return cv2.cuda.cvtColor(image, cv2.COLOR_RGB2HLS)

def HSL_color_selection(image):
    #Convert the input image to HSL on the GPU
    converted_image_gpu = cv2.cuda_GpuMat(convert_hsl(image))
    
    #White color mask
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask_gpu = cv2.cuda.inRange(converted_image_gpu, cv2.cuda_GpuMat(lower_threshold), cv2.cuda_GpuMat(upper_threshold))
    
    #Yellow color mask
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask_gpu = cv2.cuda.inRange(converted_image_gpu, cv2.cuda_GpuMat(lower_threshold), cv2.cuda_GpuMat(upper_threshold))
    
    #Combine white and yellow masks on the GPU
    mask_gpu = cv2.cuda.bitwise_or(white_mask_gpu, yellow_mask_gpu)
    masked_image_gpu = cv2.cuda.bitwise_and(cv2.cuda_GpuMat(image), cv2.cuda_GpuMat(image), mask = mask_gpu)
    
    #Download the masked image from the GPU to the CPU
    masked_image = masked_image_gpu.download()
    
    return masked_image

#Canny edge detection
def gray_scale(image):
    return cv2.cuda.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_smoothing(image, kernel_size = 13):
    #Create a CUDA device object
    d_image = cv2.cuda_GpuMat(image)
    
    #Create a CUDA Gaussian filter object
    d_filter = cv2.cuda.createGaussianFilter(d_image.type(), d_image.type(), (kernel_size, kernel_size), 0)
    
    #Apply the Gaussian filter on the GPU
    smoothed_image_gpu = d_filter.apply(d_image)
    
    #Download the smoothed image from the GPU to the CPU
    smoothed_image = smoothed_image_gpu.download()
    
    return smoothed_image

def canny_detector(image, low_threshold = 50, high_threshold = 150):
    #Create a CUDA device object
    d_image = cv2.cuda_GpuMat(image)
    
    #Create a CUDA Canny edge detector object
    d_canny = cv2.cuda.createCannyEdgeDetector(low_threshold, high_threshold)
    
    #Apply the Canny edge detector on the GPU
    edge_image_gpu = d_canny.detect(d_image)
    
    #Download the edge image from the GPU to the CPU
    edge_image = edge_image_gpu.download()
    
    return edge_image

#region of interest
def region_selection(image):
    #Create a CUDA device object
    d_image = cv2.cuda_GpuMat(image)
    
    mask = np.zeros_like(image)   
    #Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    #We could have used fixed numbers as the vertices of the polygon,
    #but they will not be applicable to images with different dimesnions.
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    
    #Create a CUDA device array for the vertices
    d_vertices = cv2.cuda_GpuMat(vertices)
    
    #Create a CUDA mask object
    d_mask = cv2.cuda.createMaskFromPoints(d_vertices, d_image.size(), 255)
    
    #Apply the mask on the GPU
    masked_image_gpu = cv2.cuda.bitwise_and(d_image, d_mask)
    
    #Download the masked image from the GPU to the CPU
    masked_image = masked_image_gpu.download()
    
    return masked_image

def hough_transform(image):
    """
    Determine and cut the region of interest in the input image.
        Parameters:
            image: The output of a Canny transform.
    """
    rho = 1              #Distance resolution of the accumulator in pixels.
    theta = cp.pi/180    #Angle resolution of the accumulator in radians.
    threshold = 20       #Only lines that are greater than threshold will be returned.
    minLineLength = 20   #Line segments shorter than that are rejected.
    maxLineGap = 300     #Maximum allowed gap between points on the same line to link them
    
    # Convert the input image to a cupy array
    image = cp.array(image)
    
    # Use the Canny edge detector from cupyx.scipy.ndimage
    edges = cupyx.scipy.ndimage.filters.canny(image, sigma=1)
    
    # Run the Hough transform on the edges
    lines = cv2.HoughLinesP(cp.asnumpy(edges), rho=rho, theta=theta, threshold=threshold,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    # Convert the output to cupy array and return
    return cp.array(lines)

def draw_lines(image, lines, color=[0, 255, 0], thickness=2):
    """
    Draw lines onto the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
            lines: The lines we want to draw.
            color (Default = red): Line color.
            thickness (Default = 2): Line thickness.
    """
    image = cp.array(image)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(cp.asnumpy(image), (x1, y1), (x2, y2), color, thickness)
    return cp.asnumpy(image)

import cupy as cp

def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
    """
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    
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
    left_weights = cp.array(left_weights, dtype='float32')
    right_weights = cp.array(right_weights, dtype='float32')
    left_lines = cp.array(left_lines, dtype='float32')
    right_lines = cp.array(right_lines, dtype='float32')
    
    left_lane  = cp.dot(left_weights,  left_lines) / cp.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = cp.dot(right_weights, right_lines) / cp.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = cp.int((y1 - intercept)/slope)
    x2 = cp.int((y2 - intercept)/slope)
    y1 = cp.int(y1)
    y2 = cp.int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    
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
    
    left_lane  = cp.dot(left_weights,  left_lines) / cp.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = cp.dot(right_weights, right_lines) / cp.sum(right_weights) if len(right_weights) > 0 else None
    
    y1 = image.shape[0]
    y2 = cp.int(y1 * 0.6)
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

    
def draw_lane_lines(image, lines, color=[0, 255, 0], thickness=12):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness. 
    """
    line_image = cp.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, cp.asnumpy(line_image), 1.0, 0.0)


def frame_processor(image):
    """
    Process the input frame to detect lane lines.
        Parameters:
            image: Single video frame.
    """
    color_select = cp.asarray(HSL_color_selection(image))
    gray = gray_scale_cuda(color_select)
    smooth = gaussian_smoothing_cuda(gray)
    edges = canny_detector_cuda(smooth)
    region = region_selection_cuda(edges)
    hough = hough_transform_cuda(region)
    result = draw_lane_lines(image, lane_lines(image, cp.asnumpy(hough)))
    return result 

def process_video(test_video, output_video):
    """
    Read input video stream and produce a video file with detected lane lines.
        Parameters:
            test_video: Input video.
            output_video: A video file with detected lane lines.
    """
    start_time = time.time()
    input_video = VideoFileClip(os.path.join('test_videos', test_video), audio=False)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(os.path.join('output_videos', output_video), audio=False)
    #calculating the total time to process the video and write it
    end_time = time.time()
    print('Total process time: {} seconds'.format(end_time - start_time))


#video input
process_video('solidYellowLeft.mp4', 'solidYellowLeft_output.mp4')

#Memory usage calculation
process = psutil.Process()
print("Total Memory usage: ",process.memory_info().rss/1024**2," MB") #in MB