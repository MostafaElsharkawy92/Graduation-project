#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>




#include <iostream>
#include <math.h> // power, tan
#include <stdlib.h>  // absolute value
#include <algorithm> // for sort
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <math.h> // power, tan
#include <stdlib.h>  // absolute value
#include <algorithm> // for sort

using namespace std;
using namespace cv;


cv::String folderpath = "/mnt/small_group/*.jpg";
std::vector <cv::String> filenames;

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

int main(int argc, char** argv) {

	int status = 0;
		clock_t start, end;
		double cpu_time_used;

		//clock_t start_1, end_1;
		//double cpu_time_used_1;

		//clock_t start_2, end_2;
		//double cpu_time_used_2;

		start = clock();

	//std::vector<cv::Mat> out_frames;
	int frameCount = 0;
	std::string outputDir = "/mnt/output_frames/";
	cv::glob(folderpath, filenames);

	for (size_t i = 0; i < filenames.size(); i++)
	{
		cv::Mat image = cv::imread(filenames[i]);

		// Load the image
		//cv::Mat image = cv::imread("image.jpg");

		// Check if the image was loaded successfully
		if (image.empty()) {
			std::cout << "Failed to load the image." << std::endl;
			return -1;
		}

		//--------------GRAYSCALE IMAGE-----------------
// Define grayscale image
		cv::Mat imageGray;

		// Convert image to grayscale
		cv::cvtColor(image, imageGray, cv::COLOR_BGRA2GRAY);

		// window for grayscaled image
		cv::String grayscaleWindowName = "Grayscaled image";

		// Show grayscale image
		////cv::imshow(grayscaleWindowName, imageGray);
		//cv::waitKey(0)(0); // wait for a key press


		//--------------GAUSSIAN SMOOTHING-----------------
		// Use low pass filter to remove noise, removes high freq stuff like edges
		int kernelSize = 9; // bigger kernel = more smoothing

		// Define smoothed image
		cv::Mat smoothedIm;
		cv::GaussianBlur(imageGray, smoothedIm, cv::Size(kernelSize, kernelSize), 0, 0);

		// window for smoothed image
		cv::String smoothedWindowName = "Smoothed image";

		// Show smoothed image
		//cv::imshow(smoothedWindowName, smoothedIm);
		//cv::waitKey(0)(0);//cv::waitKey(0)(0); // wait for a key press


		//---------------EDGE DETECTION---------------------
		// finds gradient in x,y direction, gradient direction is perpendicular to edges
		// Define values for edge detection
		int minVal = 60;
		int maxVal = 150;

		// Define edge detection image, do edge detection
		cv::Mat edgesIm;
		cv::Canny(smoothedIm, edgesIm, minVal, maxVal);

		// window for edge detection image
		cv::String edgeWindowName = "edge detection image";

		// Show edge detection image
		//cv::imshow(edgeWindowName, edgesIm);
		//cv::waitKey(0)(0); // wait for a key press

		//--------------------CREATE MASK---------------------------
		// Create mask to only keep area defined by four corners
		// Black out every area outside area

		// Define masked image
		// Create all black image with same dimensions as original image
		// 3rd arg is CV_<bit-depth>{U|S|F}C(<number_of_channels>), so this is 8bit, unsigned int, channels: 1
		cv::Mat mask(image.size().height, image.size().width, CV_8UC1, cv::Scalar(0)); // CV_8UC3 to make it a 3 channel

		// Define the points for the mask
		// Use cv::Point type for x,y points
		cv::Point p1 = cv::Point(0, image.size().height);
		cv::Point p2 = cv::Point(455, 320);
		cv::Point p3 = cv::Point(505, 320);
		cv::Point p4 = cv::Point(image.size().width, image.size().height);

		// create vector from array with points we just defined
		cv::Point vertices1[] = { p1,p2,p3,p4 };
		std::vector<cv::Point> vertices(vertices1, vertices1 + sizeof(vertices1) / sizeof(cv::Point));

		// Create vector of vectors, add the vertices we defined above
		// (you could add multiple other similar contours to this vector)
		std::vector<std::vector<cv::Point> > verticesToFill;
		verticesToFill.push_back(vertices);

		// Fill in the vertices on the blank image, showing what the mask is
		cv::fillPoly(mask, verticesToFill, cv::Scalar(255, 255, 255));

		// Show the mask
		//cv::imshow("Mask", mask);
		//cv::waitKey(0)(0);

		//---------------------APPLY MASK TO IMAGE----------------------
		// create image only where mask and edge Detection image are the same

		// Create masked im, which takes input1, input2, and output. Only keeps where two images overlap
		cv::Mat maskedIm = edgesIm.clone();
		cv::bitwise_and(edgesIm, mask, maskedIm);

		// Show masked image
		//cv::imshow("Masked Image", maskedIm);
		//cv::waitKey(0)(0);


		//------------------------HOUGH LINES----------------------------
		float rho = 2;
		float pi = 3.14159265358979323846;
		float theta = pi / 180;
		float threshold = 45;
		int minLineLength = 40;
		int maxLineGap = 100;
		//bool gotLines = false;

		// Variables for once we have line averages
		//float posSlopeMean = 0;
		//double xInterceptPosMean = 0;
		//float negSlopeMean = 0;
		//double xInterceptNegMean = 0;



		vector<cv::Vec4i> lines; // A Vec4i is a type holding 4 integers
		cv::HoughLinesP(maskedIm, lines, rho, theta, threshold, minLineLength, maxLineGap);

		// Check if we got more than one line
		if (!lines.empty() && lines.size() > 2) {

			// Initialize lines image
			cv::Mat allLinesIm(image.size().height, image.size().width, CV_8UC3, cv::Scalar(0, 0, 0)); // CV_8UC3 to make it a 3 channel)

			// Loop through lines
			// std::size_t can store the maximum size of a theoretically possible object of any type
			for (size_t i = 0; i != lines.size(); ++i) {

				// Draw line onto image
				cv::line(allLinesIm, cv::Point(lines[i][0], lines[i][1]),
					cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
			}

			// Display images
			//cv::imshow("Hough Lines", allLinesIm);
			//cv::waitKey(0)(0);


			//---------------Separate Lines Into Positive/Negative Slope--------------------
			// Separate line segments by their slope to decide left line vs. the right line

			// Define arrays for positive/negative lines
			vector< vector<double> > slopePositiveLines; // format will be [x1 y1 x2 y2 slope]
			vector< vector<double> > slopeNegativeLines;
			vector<float> yValues;
			//float slopePositiveLines[] = {};
			//float slopeNegativeLines[] = {};

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
		   // cout << endl;
			} // looping though all lines


			// If we didn't get any positive lines, go though again and just add any positive slope lines
			// Be less strict
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


			// If we didn't get any negative lines, go though again and just add any positive slope lines
			// Be less strict
			if (addedNeg == false) { // if we didnt add any positive lines

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
							slopePositiveLines.resize(posCounter + 1);

							// Reshape current row to 5 columns [x1, y1, x2, y2, slope]
							slopePositiveLines[posCounter].resize(5);

							// Add values to row
							slopeNegativeLines[posCounter][0] = x1;
							slopeNegativeLines[posCounter][1] = y1;
							slopeNegativeLines[posCounter][2] = x2;
							slopeNegativeLines[posCounter][3] = y2;
							slopeNegativeLines[posCounter][4] = -slope;

							// add yValues
							yValues.push_back(y1);
							yValues.push_back(y2);

							// Note that we added a positive slope line
							addedNeg = true;

							// iterate the counter
							posCounter++;
						}
					}

				}
			} // if addedNeg == false

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

			//std::cout << "negativeSlopes.size(): " << negativeSlopes.size() << endl;
			//std::cout << "condition: " << negSlopeMedian*.2 << endl;

			// Loop through positive slopes and add the good ones
			for (size_t i = 0; i != negativeSlopes.size(); ++i) {

				//cout << "check: " << negativeSlopes[i]  << endl;

				// check difference between current slope and the median. If the difference is small enough it's good
				if (abs(negativeSlopes[i] - negSlopeMedian) < .9) { // < negSlopeMedian*.2
					negSlopesGood.push_back(negativeSlopes[i]); // Add slope to negSlopesGood
					negSum += negativeSlopes[i]; // add to sum
				}
			}

			//cout << endl;
			// Get mean of good positive slopes
			float negSlopeMean = negSum / negSlopesGood.size();
			//std::cout << "negSum: " << negSum << " negSlopesGood.size(): " << negSlopesGood.size() << " negSlopeMean: " << negSlopeMean << endl;


			//----------------GET AVERAGE X COORD WHEN Y COORD OF LINE = 0--------------------
			// Positive Lines
			vector<double> xInterceptPos; // define vector for x intercepts of positive slope lines

			// Loop through positive slope lines, find and store x intercept values
			for (size_t i = 0; i != slopePositiveLines.size(); ++i) {
				double x1 = slopePositiveLines[i][0]; // x value
				double y1 = image.rows - slopePositiveLines[i][1]; // y value...yaxis is flipped
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
				double y1 = image.rows - slopePositiveLines[i][1]; // y value...yaxis is flipped
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
				double y1 = image.rows - slopeNegativeLines[i][1]; // y value...yaxis is flipped
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
				double y1 = image.rows - slopeNegativeLines[i][1]; // y value...yaxis is flipped
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
			//gotLines = true;


			//-----------------------PLOT LANE LINES------------------------
			// Need end points of line to draw in. Have x1,y1 (xIntercept,im.shape[1]) where
			// im.shape[1] is the bottom of the image. take y2 as some num (min/max y in the good lines?)
			// then find corresponding x

			// Create image, lane lines on real image
			cv::Mat laneLineImage = image.clone();
			cv::Mat laneFill = image.clone();

			// Positive Slope Line
			float slope = posSlopeMean;
			double x1 = xInterceptPosMean;
			int y1 = 0;
			double y2 = image.size().height - (image.size().height - image.size().height * .35);
			double x2 = (y2 - y1) / slope + x1;

			// Add positive slope line to image
			x1 = int(x1 + .5);
			x2 = int(x2 + .5);
			y1 = int(y1 + .5);
			y2 = int(y2 + .5);
			cv::line(laneLineImage, cv::Point(x1, image.size().height - y1), cv::Point(x2, image.size().height - y2),
				cv::Scalar(0, 255, 0), 3, 8);


			// Negative Slope Line
			slope = negSlopeMean;
			double x1N = xInterceptNegMean;
			int y1N = 0;
			double x2N = (y2 - y1N) / slope + x1N;

			// Add negative slope line to image
			x1N = int(x1N + .5);
			x2N = int(x2N + .5);
			y1N = int(y1N + .5);
			cv::line(laneLineImage, cv::Point(x1N, image.size().height - y1N), cv::Point(x2N, image.size().height - y2),
				cv::Scalar(0, 255, 0), 3, 8);

			// Plot positive and negative lane lines
			//cv::imshow("Lane lines on image", laneLineImage);
			//cv::waitKey(0)(0); // wait for a key press


			// -----------------BLEND IMAGE-----------------------
			// Use cv::Point type for x,y points
			cv::Point v1 = cv::Point(x1, image.size().height - y1);
			cv::Point v2 = cv::Point(x2, image.size().height - y2);
			cv::Point v3 = cv::Point(x1N, image.size().height - y1N);
			cv::Point v4 = cv::Point(x2N, image.size().height - y2);

			// create vector from array of corner points of lane
			cv::Point verticesBlend[] = { v1,v3,v4,v2 };
			std::vector<cv::Point> verticesVecBlend(verticesBlend, verticesBlend + sizeof(verticesBlend) / sizeof(cv::Point));

			// Create vector of vectors to be used in fillPoly, add the vertices we defined above
			std::vector<std::vector<cv::Point> > verticesfp;
			verticesfp.push_back(verticesVecBlend);

			// Fill area created from vector points
			cv::fillPoly(laneFill, verticesfp, cv::Scalar(0, 255, 255));

			// Blend image
		/*	float opacity = .25;
			cv::Mat blendedIm;
			cv::addWeighted(laneFill, opacity, image, 1 - opacity, 0, blendedIm);*/

			cv::Mat blendedIm;
			blendedIm = image.clone();

			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					//
					//									Kernel Part Execution
					//
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					//------------------Setup OpenCL environement-------------------------------------------

					unsigned int DATA_SIZE = image.rows * image.cols;
					size_t size_in_bytes = DATA_SIZE * sizeof(unsigned char);
					std::cout << " size_in_bytes = '" << size_in_bytes << "'\n";

					if (argc != 2) {
						std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
						return EXIT_FAILURE;
					}
					char* xclbinFilename = argv[1];

					//----------------------------Setting up Target Platform and Device--------------------------------------

					std::vector<cl::Device> devices;
					cl::Device device;
					std::vector<cl::Platform> platforms;
					bool found_device = false;

					//traversing all Platforms To find Xilinx Platform and targeted
					//Device in Xilinx Platform
					cl::Platform::get(&platforms);
					for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
						cl::Platform platform = platforms[i];
						std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
						if (platformName == "Xilinx") {
							devices.clear();
							platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
							if (devices.size()) {
								device = devices[0];
								found_device = true;
								break;
							}
						}
					}
					if (found_device == false) {
						std::cout << "Error: Unable to find Target Device "
							<< device.getInfo<CL_DEVICE_NAME>() << std::endl;
						return EXIT_FAILURE;
					}

					//------------------------------Create Context and Command Queue------------------------------------

					cl::Context context(device);
					cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

					//--------------------------------------Load xclbin--------------------------------------------------

					std::cout << "Loading: '" << xclbinFilename << "'\n";
					std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
					bin_file.seekg(0, bin_file.end);
					unsigned nb = bin_file.tellg();
					bin_file.seekg(0, bin_file.beg);
					char* buf = new char[nb];
					bin_file.read(buf, nb);

					//-----------------------------Creating Program from Binary File-------------------------------------

					cl::Program::Binaries bins;
					bins.push_back({ buf,nb });
					devices.resize(1);
					cl::Program program(context, devices, bins);

					//----------------------------------including kernel program-------------------------------------------------

					cl::Kernel krnl(program, "krnl_v");

					//----------------------------------Defining Buffers-------------------------------------------------

					cl::Buffer buffer_in_1(context, CL_MEM_READ_ONLY, size_in_bytes);
					cl::Buffer buffer_in_2(context, CL_MEM_READ_ONLY, size_in_bytes);
					cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY, size_in_bytes);

					//----------------------------------Kernel arguements-------------------------------------------------

					int narg = 0;
					krnl.setArg(narg++, buffer_in_1);
					krnl.setArg(narg++, buffer_in_2);
					krnl.setArg(narg++, buffer_out);
					krnl.setArg(narg++, image.rows);
					krnl.setArg(narg++, image.cols);

					//----------------------------------Data Setting #1-------------------------------------------------

					//q.enqueueWriteBuffer(buffer_in_1, CL_TRUE, 0, size_in_bytes, laneFill.data);
					//q.enqueueWriteBuffer(buffer_in_2, CL_TRUE, 0, size_in_bytes, image.data);
					//q.enqueueWriteBuffer(buffer_out, CL_TRUE, 0, size_in_bytes, blendedIm.data);

					unsigned char* ptr_in_1 = (unsigned char*)q.enqueueMapBuffer(buffer_in_1, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes/4);
					unsigned char* ptr_in_2= (unsigned char*)q.enqueueMapBuffer(buffer_in_2, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes/4);
					unsigned char* ptr_out1 = (unsigned char*)q.enqueueMapBuffer(buffer_out, CL_TRUE, CL_MAP_READ, 0, size_in_bytes/4);

					for (unsigned int i = 0; i < 129600; i++) {
						ptr_in_1[i] = laneFill.data[i];
						ptr_in_2[i] = image.data[i];
					}

					for (unsigned int i = 0; i < 129600; i++) {
						blendedIm.data[i] = ptr_out1[i];
					}

					//-------------------------------------Kernel Execution on FPGA #1-------------------------------------------------

					q.enqueueMigrateMemObjects({ buffer_in_1 }, 0/* 0 means from host*/);
					q.enqueueMigrateMemObjects({ buffer_in_2 }, 0/* 0 means from host*/);
					q.enqueueTask(krnl);
					q.enqueueMigrateMemObjects({ buffer_out }, CL_MIGRATE_MEM_OBJECT_HOST);
					q.finish();

					//----------------------------------Data Setting #2-------------------------------------------------

					unsigned char* ptr_in_3 = (unsigned char*)q.enqueueMapBuffer(buffer_in_1, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes/4);
					unsigned char* ptr_in_4= (unsigned char*)q.enqueueMapBuffer(buffer_in_2, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes/4);
					unsigned char* ptr_out2 = (unsigned char*)q.enqueueMapBuffer(buffer_out, CL_TRUE, CL_MAP_READ, 0, size_in_bytes/4);

					for (unsigned int i = 129600; i < 259200; i++) {
						ptr_in_3[i] = laneFill.data[i];
						ptr_in_4[i] = image.data[i];
					}

					for (unsigned int i = 129600; i < 259200; i++) {
						blendedIm.data[i] = ptr_out2[i];
					}

					//-------------------------------------Kernel Execution on FPGA #2-------------------------------------------------

					q.enqueueMigrateMemObjects({ buffer_in_1 }, 0/* 0 means from host*/);
					q.enqueueMigrateMemObjects({ buffer_in_2 }, 0/* 0 means from host*/);
					q.enqueueTask(krnl);
					q.enqueueMigrateMemObjects({ buffer_out }, CL_MIGRATE_MEM_OBJECT_HOST);
					q.finish();

					//----------------------------------Data Setting #3-------------------------------------------------

					unsigned char* ptr_in_5 = (unsigned char*)q.enqueueMapBuffer(buffer_in_1, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes/4);
					unsigned char* ptr_in_6= (unsigned char*)q.enqueueMapBuffer(buffer_in_2, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes/4);
					unsigned char* ptr_out3 = (unsigned char*)q.enqueueMapBuffer(buffer_out, CL_TRUE, CL_MAP_READ, 0, size_in_bytes/4);

					for (unsigned int i = 259200; i < 388800; i++) {
						ptr_in_5[i] = laneFill.data[i];
						ptr_in_6[i] = image.data[i];
					}

					for (unsigned int i = 259200; i < 388800; i++) {
						blendedIm.data[i] = ptr_out3[i];
					}

					//-------------------------------------Kernel Execution on FPGA #3-------------------------------------------------

					q.enqueueMigrateMemObjects({ buffer_in_1 }, 0/* 0 means from host*/);
					q.enqueueMigrateMemObjects({ buffer_in_2 }, 0/* 0 means from host*/);
					q.enqueueTask(krnl);
					q.enqueueMigrateMemObjects({ buffer_out }, CL_MIGRATE_MEM_OBJECT_HOST);
					q.finish();

					//----------------------------------Data Setting #4-------------------------------------------------

					unsigned char* ptr_in_7 = (unsigned char*)q.enqueueMapBuffer(buffer_in_1, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes/4);
					unsigned char* ptr_in_8= (unsigned char*)q.enqueueMapBuffer(buffer_in_2, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes/4);
					unsigned char* ptr_out4 = (unsigned char*)q.enqueueMapBuffer(buffer_out, CL_TRUE, CL_MAP_READ, 0, size_in_bytes/4);

					for (unsigned int i = 388800; i < 518400; i++) {
						ptr_in_7[i] = laneFill.data[i];
						ptr_in_8[i] = image.data[i];
					}

					for (unsigned int i = 388800; i < 518400; i++) {
						blendedIm.data[i] = ptr_out4[i];
					}

					//-------------------------------------Kernel Execution on FPGA #4-------------------------------------------------

					q.enqueueMigrateMemObjects({ buffer_in_1 }, 0/* 0 means from host*/);
					q.enqueueMigrateMemObjects({ buffer_in_2 }, 0/* 0 means from host*/);
					q.enqueueTask(krnl);
					q.enqueueMigrateMemObjects({ buffer_out }, CL_MIGRATE_MEM_OBJECT_HOST);
					q.finish();

					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					///
					///											Kernel Part finish
					///
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




			// Plot lane lines
			cv::line(blendedIm, cv::Point(x1, image.size().height - y1), cv::Point(x2, image.size().height - y2),
				cv::Scalar(0, 255, 0), 8, 8);
			cv::line(blendedIm, cv::Point(x1N, image.size().height - y1N), cv::Point(x2N, image.size().height - y2),
				cv::Scalar(0, 255, 0), 8, 8);

			// Show final frame
			//cv::imshow("Final Output", blendedIm);
			//cv::waitKey(0);

			//out_frames.push_back(blendedIm);

			//cv::Mat blendedIm;
			std::string outputFilename = outputDir + "frame_" + std::to_string(frameCount) + ".jpg";
			cv::imwrite(outputFilename, blendedIm);
			frameCount++;
		}
	}
	end = clock();
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	cout << "execution time is  : " << cpu_time_used << endl;

					//cout << "---------------------------------------" << endl;

					//cout << "Total Execution time is " << cpu_time_used + cpu_time_used_1+ cpu_time_used_2 << endl;
					cout << "All frames are created" << endl;
	return 0;
}
