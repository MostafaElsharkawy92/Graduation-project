# Importing some useful packages
import cupy as cp
import cv2
import os, psutil
import glob
from moviepy.editor import VideoFileClip
from moviepy import *
import time

# Initialize variable for calculating average memory usage and process time
total_memory_usage = 0
total_process_time = 0
process_time = 0 
k=1

for i in range(k):
    # Color selection (HSL)
    def convert_hsl(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    def HSL_color_selection(image):
        # Convert the input image to HSL
        converted_image = convert_hsl(image)

        # White color mask
        lower_threshold = cp.uint8([0, 200, 0])
        upper_threshold = cp.uint8([255, 255, 255])
        white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

        # Yellow color mask
        lower_threshold = cp.uint8([10, 0, 100])
        upper_threshold = cp.uint8([40, 255, 255])
        yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

        # Combine white and yellow masks
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        return masked_image

    # Canny edge detection
    def gray_scale(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    def gaussian_smoothing(image, kernel_size = 13):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def canny_detector(image, low_threshold = 50, high_threshold = 150):
        image_gpu = cv2.cuda_GpuMat()
        image_gpu.upload(image)
        edges_gpu = cv2.cuda.createCannyEdgeDetector(low_threshold, high_threshold).detect(image_gpu)
        edges = edges_gpu.download()
        return edges

    # Region of interest
    def region_selection(image):
        mask = cp.zeros_like(image)

        # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # We could have used fixed numbers as the vertices of the polygon,
        # but they will not be applicable to images with different dimensions.
        rows, cols = image.shape[:2]
        bottom_left  = [cols * 0.1, rows * 0.95]
        top_left     = [cols * 0.4, rows * 0.6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right    = [cols * 0.6, rows * 0.6]
        vertices = cp.array([[bottom_left, top_left, top_right, bottom_right]], dtype=cp.int32)
        cv2.fillPoly(mask, vertices.get(), ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask.get())
        return masked_image

    # Hough transform
    def hough_transform(image):
        """
        Determine and cut the region of interest in the input image.
            Parameters:
                image: The output of a Canny transform.
        """
        rho = 1              # Distance resolution of the accumulator in pixels.
        theta = cp.pi/180    # Angle resolution of the accumulator in radians.
        threshold = 20       # Only lines that are greater than threshold will be returned.
        minLineLength = 20   # Line segments shorter than that are rejected.
        maxLineGap = 300     # Maximum allowed gap between points on the same line to link them
        return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                               minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    def draw_lines(image, lines, color=[0, 255, 0], thickness=2):
        """
        Draw lines onto the input image.
            Parameters:
                image: An np.array compatible with plt.imshow.
                lines: The lines we want to draw.
                color (Default = red): Line color.
                thickness (Default = 2): Line thickness.
        """
        image = cp.copy(image)
        for line in lines:
            for x1, y1, x2, y2 in line:
               cv2.line(image.get(), (x1, y1), (x2, y2), color, thickness)
        return image

    # Run the pipeline
    def process_image(image):
        # Convert the image to GPU memory
        image_gpu = cv2.cuda_GpuMat()
        image_gpu.upload(image)

        # Apply the HSL color selection
        hsl_selected = HSL_color_selection(image_gpu.download())

        # Apply the Canny edge detection
        gray = gray_scale(hsl_selected)
        smooth = gaussian_smoothing(gray)
        edges = canny_detector(smooth)

        # Apply the region selection
        masked_edges = region_selection(edges)

        # Apply the Hough transform
        lines = hough_transform(masked_edges)

        # Draw the lines on the original image
        result = draw_lines(image, lines)

        return result

    # Process the video
    process = psutil.Process(os.getpid())
    process_time_start = time.time()
    input_video = '/content/solidYellowLeft.mp4'
    output_video = '/content/solidYellowLeft_Output.mp4'
    clip = VideoFileClip(input_video)
    processed_clip = clip.fl_image(process_image)
    processed_clip.write_videofile(output_video, audio=False)
    process_time_end = time.time()

    # Update the process time and memory usage
    process_time = process_time_end - process_time_start
    total_process_time += process_time
    total_memory_usage += process.memory_info().rss

average_process_time = total_process_time / k
average_memory_usage = total_memory_usage / k

print("The average process time: {0:.2f} seconds".format(average_process_time))
print("The average memory usage: {0:.2f} MB".format(average_memory_usage / (1024 * 1024)))