#include <opencv2/opencv.hpp> // opencv
#include <iostream> // cout and cin
#include <math.h> // power, tan
#include <stdlib.h>  // absolute value
#include <algorithm> // for sort

using namespace std;
double median(vector<double> vec) {
	int vecSize = vec.size();
	if (vecSize == 0) {
		throw domain_error("median of empty vector");
	}
	sort(vec.begin(), vec.end());
	int middle;
	double median;
	if (vecSize % 2 == 0) {
		middle = vecSize/2;
		median = (vec[middle-1] + vec[middle]) / 2;
	}
	else {
		middle = vecSize/2;
		median = vec[middle];
	}
	return median;
}

int main(int argc, char** argv) {

	// Path a video file
    string video_input = "/Users/yakup/Test_Input_Files/UD/testvideo640480.mp4";

	void VideoProcessor(string video_input, string video_output) {
		cv::VideoCapture inputVideo(video_input);



	}




    while (true) {
        cv::Mat frame;
        inputVideo >> frame;

        if (frame.empty()) {
            break;  // No more frames to process
        }

        cv::Mat grayscaleFrame;
        cv::cvtColor(frame, grayscaleFrame, cv::COLOR_BGR2GRAY);  // Convert to grayscale

        outputVideo.write(grayscaleFrame);  // Write the grayscale frame to the output video
    }
}

	void processVideo(string videoFilename) {

		cv::VideoCapture capture(videoFilename);    // Create the capture object
		cv::VideoWriter outputVideo; // Create the writer object
		int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC)); // Get Codec Type- Int form

		// Create the output video file
		outputVideo.open("/Users/yakup/Test_output_Files/output_test640480.mp4" , ex, capture.get(CV_CAP_PROP_FPS), Size(xsize, ysize), true);
    
		// Create variables
		Mat frame, result;

		// Always reading frame
		while( capture.read(frame) ){

		}
	}

return 0;
}
