// Exercise the upstream array and image-processing code in the guest.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

int main() {
    cv::Mat image(3, 3, CV_8UC1, cv::Scalar(42));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(6, 6), 0, 0, cv::INTER_NEAREST);
    if (resized.rows != 6 || resized.cols != 6 || resized.at<unsigned char>(5, 5) != 42)
        return 1;
    if (cv::mean(resized)[0] != 42) return 2;
    cv::Mat flipped;
    cv::flip(resized, flipped, 1);
    if (cv::countNonZero(flipped != resized) != 0) return 3;
    return 0;
}
