#include <opencv2/opencv.hpp> // opencv
#include <iostream> // cout and cin
#include <math.h> // power, tan
#include <stdlib.h>  // absolute value
#include <algorithm> // for sort

using namespace std;

// Function for finding median
double median(vector<double> vec) {

	// get size of vector
	int vecSize = vec.size();

	// if vector is empty throw error
	if (vecSize == 0) {
		throw domain_error("median of empty vector");
	}

	// sort vector
	sort(vec.begin(), vec.end());

	// define middle and median
	int middle;
	double median;

	// if even number of elements in vec, take average of two middle values
	if (vecSize % 2 == 0) {
		// a value representing the middle of the array. If array is of size 4 this is 2
		// if it's 8 then middle is 4
		middle = vecSize / 2;

		// take average of middle values, so if vector is [1, 2, 3, 4] we want average of 2 and 3
		// since we index at 0 middle will be the higher one vec[2] in the above vector is 3, and vec[1] is 2
		median = (vec[middle - 1] + vec[middle]) / 2;
	}

	// odd number of values in the vector
	else {
		middle = vecSize / 2; // take the middle again

		// if vector is 1 2 3 4 5, middle will be 5/2 = 2, and vec[2] = 3, the middle value
		median = vec[middle];
	}

	return median;
}

cv::Mat convertHSL(const cv::Mat& image) {
	cv::Mat convertedImage;
	cv::cvtColor(image, convertedImage, cv::COLOR_BGR2HLS);  // BGR2HLS instead of RGB2HLS
	return convertedImage;
}

cv::Mat HSLColorSelection(const cv::Mat& image) {
	// Convert the input image to HSL
	cv::Mat convertedImage = convertHSL(image);

	// White color mask
	cv::Scalar lowerThresholdWhite(0, 200, 0);
	cv::Scalar upperThresholdWhite(255, 255, 255);
	cv::Mat whiteMask;
	cv::inRange(convertedImage, lowerThresholdWhite, upperThresholdWhite, whiteMask);

	// Yellow color mask
	cv::Scalar lowerThresholdYellow(10, 0, 100);
	cv::Scalar upperThresholdYellow(40, 255, 255);
	cv::Mat yellowMask;
	cv::inRange(convertedImage, lowerThresholdYellow, upperThresholdYellow, yellowMask);

	// Combine white and yellow masks
	cv::Mat mask = whiteMask | yellowMask;
	cv::Mat maskedImage;
	cv::bitwise_and(image, image, maskedImage, mask);

	return maskedImage;
}

int main(int argc, char** argv) {

	cv::VideoCapture inputVideo("D:/test/test_videos/input2.mp4");
	//cv::VideoCapture inputVideo(0); //for live streaming

	if (!inputVideo.isOpened()) {
		std::cerr << "Error: Unable to open video file." << std::endl;
		return -1;
	}

	int frameWidth = static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_WIDTH));
	int frameHeight = static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT));
	double frameRate = inputVideo.get(cv::CAP_PROP_FPS);
	
	// Define the codec and create the output video
	cv::VideoWriter outputVideo("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), frameRate, cv::Size(frameWidth, frameHeight));
	
	// Create a vector to store frames
	std::vector<cv::Mat> frames;
	
	// Loop through the video and extract frames
	cv::Mat frame;
	while (inputVideo.read(frame)) {
		frames.push_back(frame);

		//--------------HLS Color Selection-----------------
		cv::Mat Converted_HLS = HSLColorSelection(frame);

		//--------------GRAYSCALE frame-----------------
		cv::Mat frameGray;
		cv::cvtColor(Converted_HLS,frameGray, cv::COLOR_BGRA2GRAY);

		//--------------GAUSSIAN SMOOTHING-----------------
		int kernelSize = 9; // bigger kernel = more smoothing
		cv::Mat smoothedIm;
		cv::GaussianBlur(frameGray, smoothedIm, cv::Size(kernelSize, kernelSize), 0, 0);

		//---------------EDGE DETECTION---------------------
		int minVal = 60;
		int maxVal = 150;
		cv::Mat edgesIm;
		cv::Canny(smoothedIm, edgesIm, minVal, maxVal);

		//---------------CREATE MASK---------------------------
		cv::Mat mask(frame.size().height, frame.size().width, CV_8UC1, cv::Scalar(0));
		// Define the points for the mask
		cv::Point p1 = cv::Point(0, frame.size().height);
		cv::Point p2 = cv::Point(455, 320);
		cv::Point p3 = cv::Point(505, 320);
		cv::Point p4 = cv::Point(frame.size().width, frame.size().height);
		// create vector from array with points we just defined
		cv::Point vertices1[] = { p1,p2,p3,p4 };
		std::vector<cv::Point> vertices(vertices1, vertices1 + sizeof(vertices1) / sizeof(cv::Point));
		// Create vector of vectors, add the vertices we defined above
		std::vector<std::vector<cv::Point> > verticesToFill;
		verticesToFill.push_back(vertices);
		// Fill in the vertices on the blank frame, showing what the mask is
		cv::fillPoly(mask, verticesToFill, cv::Scalar(255, 255, 255));
		
		//----------------APPLY MASK TO frame----------------------
		// Create masked im, which takes input1, input2, and output. Only keeps where two frames overlap
		cv::Mat maskedIm = edgesIm.clone();
		cv::bitwise_and(edgesIm, mask, maskedIm);

		//------------------------HOUGH LINES----------------------------
		float rho = 2;
		float pi = 3.14159265358979323846;
		float theta = pi / 180;
		float threshold = 45;
		int minLineLength = 40;
		int maxLineGap = 100;
		vector<cv::Vec4i> lines; // A Vec4i is a type holding 4 integers
		cv::HoughLinesP(maskedIm, lines, rho, theta, threshold, minLineLength, maxLineGap);
		// Check if we got more than one line
		if (!lines.empty() && lines.size() > 2) {
			// Initialize lines frame
			cv::Mat allLinesIm(frame.size().height, frame.size().width, CV_8UC3, cv::Scalar(0, 0, 0)); // CV_8UC3 to make it a 3 channel)
			// Loop through lines
			for (size_t i = 0; i != lines.size(); ++i) {
				// Draw line onto frame
				cv::line(allLinesIm, cv::Point(lines[i][0], lines[i][1]),
					cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
			}
			//---------------Separate Lines Into Positive/Negative Slope--------------------
			// Define arrays for positive/negative lines
			vector< vector<double> > slopePositiveLines; // format will be [x1 y1 x2 y2 slope]
			vector< vector<double> > slopeNegativeLines;
			vector<float> yValues;
			// keep track of if we added one of each, want at least one of each to proceed
			bool addedPos = false;
			bool addedNeg = false;
			// array counter for appending new lines
			int negCounter = 0;
			int posCounter = 0;
			// Loop through all lines
			for (size_t i = 0; i != lines.size(); ++i) {
				// Get points for current line
				float x1 = lines[i][0];
				float y1 = lines[i][1];
				float x2 = lines[i][2];
				float y2 = lines[i][3];
				// get line length
				float lineLength = pow(pow(x2 - x1, 2) + pow(y2 - y1, 2), .5);
				// if line is long enough
				if (lineLength > 30) {
					// dont divide by zero
					if (x2 != x1) {
						// get slope
						float slope = (y2 - y1) / (x2 - x1);
						// Check if slope is positive
						if (slope > 0) {
							// Find angle of line wrt x axis.
							float tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta) value
							float angle = atan(tanTheta) * 180 / pi;
							// Only pass good line angles,  dont want verticalish/horizontalish lines
							if (abs(angle) < 85 && abs(angle) > 20) {
								// Add a row to the matrix
								slopeNegativeLines.resize(negCounter + 1);
								// Reshape current row to 5 columns [x1, y1, x2, y2, slope]
								slopeNegativeLines[negCounter].resize(5);
								// Add values to row
								slopeNegativeLines[negCounter][0] = x1;
								slopeNegativeLines[negCounter][1] = y1;
								slopeNegativeLines[negCounter][2] = x2;
								slopeNegativeLines[negCounter][3] = y2;
								slopeNegativeLines[negCounter][4] = -slope;
								// add yValues
								yValues.push_back(y1);
								yValues.push_back(y2);
								// Note that we added a positive slope line
								addedPos = true;
								// iterate the counter
								negCounter++;
							}
						}
						// Check if slope is Negative
						if (slope < 0) {
							// Find angle of line wrt x axis.
							float tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta) value
							float angle = atan(tanTheta) * 180 / pi;
							// Only pass good line angles,  dont want verticalish/horizontalish lines
							if (abs(angle) < 85 && abs(angle) > 20) {
								// Add a row to the matrix
								slopePositiveLines.resize(posCounter + 1);
								// Reshape current row to 5 columns [x1, y1, x2, y2, slope]
								slopePositiveLines[posCounter].resize(5);
								// Add values to row
								slopePositiveLines[posCounter][0] = x1;
								slopePositiveLines[posCounter][1] = y1;
								slopePositiveLines[posCounter][2] = x2;
								slopePositiveLines[posCounter][3] = y2;
								slopePositiveLines[posCounter][4] = -slope;
								// add yValues
								yValues.push_back(y1);
								yValues.push_back(y2);
								// Note that we added a positive slope line
								addedNeg = true;
								// iterate counter
								posCounter++;
							}
						} // if slope < 0
					} // if x2 != x1
				}// if lineLength > 30
			} // looping though all lines

			// If we didn't get any positive lines, go though again and just add any positive slope lines
			if (addedPos == false) { // if we didnt add any positive lines
				// loop through lines
				for (size_t i = 0; i != lines.size(); ++i) {
					// Get points for current line
					float x1 = lines[i][0];
					float y1 = lines[i][1];
					float x2 = lines[i][2];
					float y2 = lines[i][3];
					// Get slope
					float slope = (y2 - y1) / (x2 - x1);
					// Check if slope is positive
					if (slope > 0 && x2 != x1) {
						// Find angle of line wrt x axis.
						float tanTheta = tan((abs(y2 - y1)) / (abs(x2 - x1))); // tan(theta) value
						float angle = atan(tanTheta) * 180 / pi;
						// Only pass good line angles,  dont want verticalish/horizontalish lines
						if (abs(angle) < 85 && abs(angle) > 15) {
							// Add a row to the matrix
							slopeNegativeLines.resize(negCounter + 1);
							// Reshape current row to 5 columns [x1, y1, x2, y2, slope]
							slopeNegativeLines[negCounter].resize(5);
							// Add values to row
							slopeNegativeLines[negCounter][0] = x1;
							slopeNegativeLines[negCounter][1] = y1;
							slopeNegativeLines[negCounter][2] = x2;
							slopeNegativeLines[negCounter][3] = y2;
							slopeNegativeLines[negCounter][4] = -slope;
							// add yValues
							yValues.push_back(y1);
							yValues.push_back(y2);
							// Note that we added a positive slope line
							addedPos = true;
							// iterate the counter
							negCounter++;
						}
					}
				}
			} // if addedPos == false
			// If we still dont have lines then fuck
			if (addedPos == false || addedNeg == false) {
				cout << "Not enough lines found" << endl;
			}

			//-----------------GET POSITIVE/NEGATIVE SLOPE AVERAGES-----------------------
			// Average the position of lines and extrapolate to the top and bottom of the lane.
			// Add positive slopes from slopePositiveLines into a vector positive slopes
			vector<float> positiveSlopes;
			for (unsigned int i = 0; i != slopePositiveLines.size(); ++i) {
				positiveSlopes.push_back(slopePositiveLines[i][4]);
			}
			// Get median of positiveSlopes
			sort(positiveSlopes.begin(), positiveSlopes.end()); // sort vec
			int middle; // define middle value
			double posSlopeMedian; // define positive slope median
			// if even number of elements in vec, take average of two middle values
			if (positiveSlopes.size() % 2 == 0) {
				// a value representing the middle of the array. If array is of size 4 this is 2
				// if it's 8 then middle is 4
				middle = positiveSlopes.size() / 2;
				// take average of middle values, so if vector is [1, 2, 3, 4] we want average of 2 and 3
				// since we index at 0 middle will be the higher one vec[2] in the above vector is 3, and vec[1] is 2
				posSlopeMedian = (positiveSlopes[middle - 1] + positiveSlopes[middle]) / 2;
			}
			// odd number of values in the vector
			else {
				middle = positiveSlopes.size() / 2; // take the middle again
				// if vector is 1 2 3 4 5, middle will be 5/2 = 2, and vec[2] = 3, the middle value
				posSlopeMedian = positiveSlopes[middle];
			}
			// Define vector of 'good' slopes, slopes that are drastically different than the others are thrown out
			vector<float> posSlopesGood;
			float posSum = 0.0; // sum so we'll be able to get mean
			// Loop through positive slopes and add the good ones
			for (size_t i = 0; i != positiveSlopes.size(); ++i) {
				// check difference between current slope and the median. If the difference is small enough it's good
				if (abs(positiveSlopes[i] - posSlopeMedian) < posSlopeMedian * .2) {
					posSlopesGood.push_back(positiveSlopes[i]); // Add slope to posSlopesGood
					posSum += positiveSlopes[i]; // add to sum
				}
			}
			// Get mean of good positive slopes
			float posSlopeMean = posSum / posSlopesGood.size();

			////////////////////////////////////////////////////////////////////////
			// Add negative slopes from slopeNegativeLines into a vector negative slopes
			vector<float> negativeSlopes;
			for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
				negativeSlopes.push_back(slopeNegativeLines[i][4]);
			}
			// Get median of negativeSlopes
			sort(negativeSlopes.begin(), negativeSlopes.end()); // sort vec
			int middleNeg; // define middle value
			double negSlopeMedian; // define negative slope median
			// if even number of elements in vec, take average of two middle values
			if (negativeSlopes.size() % 2 == 0) {
				// a value representing the middle of the array. If array is of size 4 this is 2
				// if it's 8 then middle is 4
				middleNeg = negativeSlopes.size() / 2;
				// take average of middle values, so if vector is [1, 2, 3, 4] we want average of 2 and 3
				// since we index at 0 middle will be the higher one vec[2] in the above vector is 3, and vec[1] is 2
				negSlopeMedian = (negativeSlopes[middleNeg - 1] + negativeSlopes[middleNeg]) / 2;
			}
			// odd number of values in the vector
			else {
				middleNeg = negativeSlopes.size() / 2; // take the middle again
				// if vector is 1 2 3 4 5, middle will be 5/2 = 2, and vec[2] = 3, the middle value
				negSlopeMedian = negativeSlopes[middle];
			}
			// Define vector of 'good' slopes, slopes that are drastically different than the others are thrown out
			vector<float> negSlopesGood;
			float negSum = 0.0; // sum so we'll be able to get mean
			// Loop through positive slopes and add the good ones
			for (size_t i = 0; i != negativeSlopes.size(); ++i) {
				// check difference between current slope and the median. If the difference is small enough it's good
				if (abs(negativeSlopes[i] - negSlopeMedian) < .9) { // < negSlopeMedian*.2
					negSlopesGood.push_back(negativeSlopes[i]); // Add slope to negSlopesGood
					negSum += negativeSlopes[i]; // add to sum
				}
			}
			// Get mean of good positive slopes
			float negSlopeMean = negSum / negSlopesGood.size();

			//----------------GET AVERAGE X COORD WHEN Y COORD OF LINE = 0--------------------
			// Positive Lines
			vector<double> xInterceptPos; // define vector for x intercepts of positive slope lines
			// Loop through positive slope lines, find and store x intercept values
			for (size_t i = 0; i != slopePositiveLines.size(); ++i) {
				double x1 = slopePositiveLines[i][0]; // x value
				double y1 = frame.rows - slopePositiveLines[i][1]; // y value...yaxis is flipped
				double slope = slopePositiveLines[i][4];
				double yIntercept = y1 - slope * x1; // yintercept of line
				double xIntercept = -yIntercept / slope; // find x intercept based off y = mx+b
				if (isnan(xIntercept) == 0) { // check for nan
					xInterceptPos.push_back(xIntercept); // add value
				}
			}
			// Get median of x intercepts for positive slope lines
			double xIntPosMed = median(xInterceptPos);
			// Define vector storing 'good' x intercept values, same concept as the slope calculations before
			vector<double> xIntPosGood;
			double xIntSum = 1; // for finding avg
			// Now that we got median, loop through lines again and compare values against median
			for (size_t i = 0; i != slopePositiveLines.size(); ++i) {
				double x1 = slopePositiveLines[i][0]; // x value
				double y1 = frame.rows - slopePositiveLines[i][1]; // y value...yaxis is flipped
				double slope = slopePositiveLines[i][4];
				double yIntercept = y1 - slope * x1; // yintercept of line
				double xIntercept = -yIntercept / slope; // find x intercept based off y = mx+b
				// check for nan and check if it's close enough to the median
				if (isnan(xIntercept) == 0 && abs(xIntercept - xIntPosMed) < .35 * xIntPosMed) {
					xIntPosGood.push_back(xIntercept); // add to 'good' vector
					xIntSum += xIntercept;
				}
			}
			// Get mean x intercept value for positive slope lines
			double xInterceptPosMean = xIntSum / xIntPosGood.size();

			/////////////////////////////////////////////////////////////////
			// Negative Lines
			vector<double> xInterceptNeg; // define vector for x intercepts of negative slope lines
			// Loop through negative slope lines, find and store x intercept values
			for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
				double x1 = slopeNegativeLines[i][0]; // x value
				double y1 = frame.rows - slopeNegativeLines[i][1]; // y value...yaxis is flipped
				double slope = slopeNegativeLines[i][4];
				double yIntercept = y1 - slope * x1; // yintercept of line
				double xIntercept = -yIntercept / slope; // find x intercept based off y = mx+b
				if (isnan(xIntercept) == 0) { // check for nan
					xInterceptNeg.push_back(xIntercept); // add value
				}
			}
			// Get median of x intercepts for negative slope lines
			double xIntNegMed = median(xInterceptNeg);
			// Define vector storing 'good' x intercept values, same concept as the slope calculations before
			vector<double> xIntNegGood;
			double xIntSumNeg = 1; // for finding avg
			// Now that we got median, loop through lines again and compare values against median
			for (size_t i = 0; i != slopeNegativeLines.size(); ++i) {
				double x1 = slopeNegativeLines[i][0]; // x value
				double y1 = frame.rows - slopeNegativeLines[i][1]; // y value...yaxis is flipped
				double slope = slopeNegativeLines[i][4];
				double yIntercept = y1 - slope * x1; // yintercept of line
				double xIntercept = -yIntercept / slope; // find x intercept based off y = mx+b
				// check for nan and check if it's close enough to the median
				if (isnan(xIntercept) == 0 && abs(xIntercept - xIntNegMed) < .35 * xIntNegMed) {
					xIntNegGood.push_back(xIntercept); // add to 'good' vector
					xIntSumNeg += xIntercept;
				}
			}
			// Get mean x intercept value for negative slope lines
			double xInterceptNegMean = xIntSumNeg / xIntNegGood.size();

			//-----------------------PLOT LANE LINES------------------------
			// Need end points of line to draw in. Have x1,y1 (xIntercept,im.shape[1]) where
			// im.shape[1] is the bottom of the frame. take y2 as some num (min/max y in the good lines?)
			// then find corresponding x
			// Create frame, lane lines on real frame
			cv::Mat laneLineframe = frame.clone();
			cv::Mat laneFill = frame.clone();
			// Positive Slope Line
			float slope = posSlopeMean;
			double x1 = xInterceptPosMean;
			int y1 = 0;
			double y2 = frame.size().height - (frame.size().height - frame.size().height * .35);
			double x2 = (y2 - y1) / slope + x1;
			// Add positive slope line to frame
			x1 = int(x1 + .5);
			x2 = int(x2 + .5);
			y1 = int(y1 + .5);
			y2 = int(y2 + .5);
			cv::line(laneLineframe, cv::Point(x1, frame.size().height - y1), cv::Point(x2, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 3, 8);
			// Negative Slope Line
			slope = negSlopeMean;
			double x1N = xInterceptNegMean;
			int y1N = 0;
			double x2N = (y2 - y1N) / slope + x1N;
			// Add negative slope line to frame
			x1N = int(x1N + .5);
			x2N = int(x2N + .5);
			y1N = int(y1N + .5);
			cv::line(laneLineframe, cv::Point(x1N, frame.size().height - y1N), cv::Point(x2N, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 3, 8);

			// -----------------BLEND frame-----------------------
			// Use cv::Point type for x,y points
			cv::Point v1 = cv::Point(x1, frame.size().height - y1);
			cv::Point v2 = cv::Point(x2, frame.size().height - y2);
			cv::Point v3 = cv::Point(x1N, frame.size().height - y1N);
			cv::Point v4 = cv::Point(x2N, frame.size().height - y2);
			// create vector from array of corner points of lane
			cv::Point verticesBlend[] = { v1,v3,v4,v2 };
			std::vector<cv::Point> verticesVecBlend(verticesBlend, verticesBlend + sizeof(verticesBlend) / sizeof(cv::Point));
			// Create vector of vectors to be used in fillPoly, add the vertices we defined above
			std::vector<std::vector<cv::Point> > verticesfp;
			verticesfp.push_back(verticesVecBlend);
			// Fill area created from vector points
			cv::fillPoly(laneFill, verticesfp, cv::Scalar(0, 255, 255));
			// Blend frame
			float opacity = .25;
			cv::Mat blendedIm;
			cv::addWeighted(laneFill, opacity, frame, 1 - opacity, 0, blendedIm);
			// Plot lane lines
			cv::line(blendedIm, cv::Point(x1, frame.size().height - y1), cv::Point(x2, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 8, 8);
			cv::line(blendedIm, cv::Point(x1N, frame.size().height - y1N), cv::Point(x2N, frame.size().height - y2),
				cv::Scalar(0, 255, 0), 8, 8);
			outputVideo.write(blendedIm);
		} // end if we got more than one line
			// We do none of that if we don't see enough lines
		else {
			cout << "Not enough lines found" << endl;
		}
	}

	// Close the video file
	inputVideo.release();
	outputVideo.release();
	return 0;
}