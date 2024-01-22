extern "C" {
	void krnl_v(
		unsigned char* input_image_1,
		unsigned char* input_image_2,
		unsigned char* output_image,
		unsigned int   m,
		unsigned int   n)

	{
#pragma HLS INTERFACE m_axi port=input_image_1 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=input_image_2 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output_image offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=input_image_1 bundle=control
#pragma HLS INTERFACE s_axilite port=input_image_2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_image bundle=control

#pragma HLS INTERFACE s_axilite port=m bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

		unsigned char input_pixel_1;
		unsigned char input_pixel_2;
		unsigned char output_pixel;

		unsigned char IN1[960][135];
		unsigned char IN2[960][135];
		unsigned char OUT[960][135];
		int index1 = 0;
		int index2 = 0;
		int index3 = 0;


#pragma HLS array_partition variable=IN1 dim=2 cyclic factor=5
#pragma HLS array_partition variable=IN2 dim=2 cyclic factor=5
#pragma HLS array_partition variable=OUT dim=2 cyclic factor=5

		for (int i = 0; i < 960; ++i) {
			for (int j = 0; j < 135; ++j) {
				//#pragma HLS UNROLL factor =5
				IN1[i][j] = input_image_1[index1++];
			}
		}

		for (int i = 0; i < 960; ++i) {
			for (int j = 0; j < 135; ++j) {
				//#pragma HLS UNROLL factor =5
				IN2[i][j] = input_image_2[index2++];
			}
		}

		for (int q = 0; q < 960; ++q) {
			for (int w = 0; w < 135; ++w) {
				//#pragma HLS UNROLL factor =5
				input_pixel_1 = IN1[q][w];
				input_pixel_2 = IN2[q][w];
				output_pixel = input_pixel_1 * 0.25 + input_pixel_2 * 0.75;
				OUT[q][w] = output_pixel;
			}
		}

		for (int i = 0; i < 960; ++i) {
			for (int j = 0; j < 135; ++j) {
				//#pragma HLS UNROLL factor =5
				output_image[index3++] = OUT[i][j];
			}
		}

	}
}
