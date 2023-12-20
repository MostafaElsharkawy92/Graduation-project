#include "xf_canny_config.h"

extern "C" {
void canny_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                 ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                 int rows,
                 int cols,
                 int low_threshold,
                 int high_threshold) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2

// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=rows     bundle=control
    #pragma HLS INTERFACE s_axilite port=cols     bundle=control
    #pragma HLS INTERFACE s_axilite port=low_threshold     bundle=control
    #pragma HLS INTERFACE s_axilite port=high_threshold     bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, INTYPE> in_mat(rows, cols);
// clang-format off
    #pragma HLS stream variable=in_mat.data depth=2
    // clang-format on

    xf::cv::Mat<XF_2UC1, HEIGHT, WIDTH, XF_NPPC32> dst_mat(rows, cols);
// clang-format off
    #pragma HLS stream variable=dst_mat.data depth=2
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, INTYPE>(img_inp, in_mat);
    xf::cv::Canny<FILTER_WIDTH, NORM_TYPE, XF_8UC1, XF_2UC1, HEIGHT, WIDTH, INTYPE, XF_NPPC32, XF_USE_URAM>(
        in_mat, dst_mat, low_threshold, high_threshold);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_2UC1, HEIGHT, WIDTH, XF_NPPC32>(dst_mat, img_out);
}
}
