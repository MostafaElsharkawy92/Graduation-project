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

		for (unsigned int i = 0; i < m * n; i++) {
			input_pixel_1 = input_image_1[i];
			input_pixel_2 = input_image_2[i];
			output_pixel = input_pixel_1 * 0.25 + input_pixel_2 * 0.75;
			output_image[i] = output_pixel;
		}

	}
}
