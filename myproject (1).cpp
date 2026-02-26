#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_layer[1*55],
    result_t layer7_out[1]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_layer complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_layer,layer7_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<model_default_t, 14080>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 16384>(wr2, "wr2.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(br2, "br2.txt");
        nnet::load_weights_from_txt<model_default_t, 8192>(w4, "w4.txt");
        nnet::load_weights_from_txt<model_default_t, 4096>(wr4, "wr4.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(br4, "br4.txt");
        nnet::load_weights_from_txt<model_default_t, 2048>(w5, "w5.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(wr5, "wr5.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b5, "b5.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(br5, "br5.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b6, "b6.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[1*64];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer4_t layer4_out[1*32];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    layer5_t layer5_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer6_t layer6_out[1];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    nnet::lstm_stack<input_t, layer2_t, config2>(input_layer, layer2_out, w2, wr2, b2, br2); // lstm

    nnet::lstm_stack<layer2_t, layer4_t, config4>(layer2_out, layer4_out, w4, wr4, b4, br4); // lstm_1

    nnet::lstm_stack<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, wr5, b5, br5); // lstm_2

    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // dense

    nnet::sigmoid<layer6_t, result_t, sigmoid_config7>(layer6_out, layer7_out); // dense_sigmoid

}

