#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <forward_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>
#include <fstream>
#include <cstring>

namespace sscma {

using namespace std;

// MARK: - Types
namespace types {

struct anchor_bbox_t {
    float    x1;
    float    y1;
    float    x2;
    float    y2;
    float    score;
    uint8_t  anchor_class;
    uint16_t anchor_index;
};

template <typename T> struct pt_t {
    T x;
    T y;
};

template <typename T> struct pt3_t {
    T x;
    T y;
    T z;
};

template <typename T, size_t N> struct pt3_set_t { pt3_t<T> data[N]; };

struct anchor_stride_t {
    size_t stride;
    size_t split;
    size_t size;
    size_t start;
};

}  // namespace types

typedef struct el_shape_t {
    size_t size;
    int*   dims;
} el_shape_t;

typedef struct el_quant_param_t {
    float   scale;
    int32_t zero_point;
} el_quant_param_t;

typedef struct el_box_t {
    uint16_t x;
    uint16_t y;
    uint16_t w;
    uint16_t h;
    uint8_t  score;
    uint16_t target;
} el_box_t;

typedef struct el_point_t {
    uint16_t x;
    uint16_t y;
    uint8_t  score;
    uint8_t  target;
} el_point_t;

typedef struct el_keypoint_t {
    el_box_t                box;
    std::vector<el_point_t> pts;
    uint8_t                 score;
    uint8_t                 target;
} el_keypoint_t;

// MARK: - Utils
namespace utils {

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

inline void softmax(float* data, size_t size) {
    float sum{0.f};
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::exp(data[i]);
        sum += data[i];
    }
    for (size_t i = 0; i < size; ++i) {
        data[i] /= sum;
    }
}

inline float dequant_value_i(size_t idx, const float* output_array, int32_t zero_point, float scale) {
    // return static_cast<float>(output_array[idx] - zero_point) * scale;
    return static_cast<float>(output_array[idx]);
}

decltype(auto) generate_anchor_strides(size_t input_size, std::vector<size_t> strides = {8, 16, 32}) {
    std::vector<types::anchor_stride_t> anchor_strides(strides.size());
    size_t                              nth_anchor = 0;
    for (size_t i = 0; i < strides.size(); ++i) {
        size_t stride     = strides[i];
        size_t split      = input_size / stride;
        size_t size       = split * split;
        anchor_strides[i] = {stride, split, size, nth_anchor};
        nth_anchor += size;
    }
    return anchor_strides;
}

decltype(auto) generate_anchor_matrix(const std::vector<types::anchor_stride_t>& anchor_strides,
                                      float                                      shift_right = 1.f,
                                      float                                      shift_down  = 1.f) {
    const auto                                   anchor_matrix_size = anchor_strides.size();
    std::vector<std::vector<types::pt_t<float>>> anchor_matrix(anchor_matrix_size);
    const float                                  shift_right_init = shift_right * 0.5f;
    const float                                  shift_down_init  = shift_down * 0.5f;

    for (size_t i = 0; i < anchor_matrix_size; ++i) {
        const auto& anchor_stride   = anchor_strides[i];
        const auto  split           = anchor_stride.split;
        const auto  size            = anchor_stride.size;
        auto&       anchor_matrix_i = anchor_matrix[i];

        anchor_matrix[i].resize(size);

        for (size_t j = 0; j < size; ++j) {
            float x            = static_cast<float>(j % split) * shift_right + shift_right_init;
            float y            = static_cast<float>(j / split) * shift_down + shift_down_init;
            anchor_matrix_i[j] = {x, y};
        }
    }

    return anchor_matrix;
}

inline float compute_iou(const types::anchor_bbox_t& l, const types::anchor_bbox_t& r, float epsilon = 1e-3) {
    float x1         = std::max(l.x1, r.x1);
    float y1         = std::max(l.y1, r.y1);
    float x2         = std::min(l.x2, r.x2);
    float y2         = std::min(l.y2, r.y2);
    float w          = std::max(0.0f, x2 - x1);
    float h          = std::max(0.0f, y2 - y1);
    float inter      = w * h;
    float l_area     = (l.x2 - l.x1) * (l.y2 - l.y1);
    float r_area     = (r.x2 - r.x1) * (r.y2 - r.y1);
    float union_area = l_area + r_area - inter;
    if (union_area < epsilon) {
        return 0.0f;
    }
    return inter / union_area;
}

inline void anchor_nms(std::forward_list<types::anchor_bbox_t>& bboxes,
                       float                                    nms_iou_thresh,
                       float                                    nms_score_thresh,
                       bool                                     soft_nms,
                       float                                    epsilon = 1e-3) {
    bboxes.sort([](const types::anchor_bbox_t& l, const types::anchor_bbox_t& r) { return l.score > r.score; });

    for (auto it = bboxes.begin(); it != bboxes.end(); it++) {
        if (it->score < epsilon) continue;

        for (auto it2 = std::next(it); it2 != bboxes.end(); it2++) {
            if (it2->score < epsilon) continue;

            auto iou = compute_iou(*it, *it2);
            if (iou > nms_iou_thresh) {
                if (soft_nms) {
                    it2->score = it2->score * (1.f - iou);
                    if (it2->score < nms_score_thresh) it2->score = 0.f;
                } else {
                    it2->score = 0.f;
                }
            }
        }
    }

    bboxes.remove_if([epsilon](const types::anchor_bbox_t& bbox) { return bbox.score < epsilon; });
}

}  // namespace utils

// MARK: - Misc
using KeyPointType = el_keypoint_t;
using ScoreType    = uint8_t;
using IoUType      = uint8_t;

// MARK: - Parameters
uint16_t image_inp_width  = 192;
uint16_t image_inp_height = 192;

uint16_t tensor_inp_width  = 192;
uint16_t tensor_inp_height = 192;

float _w_scale = tensor_inp_width / static_cast<float>(image_inp_width);
float _h_scale = tensor_inp_height / static_cast<float>(image_inp_height);

uint8_t _score_threshold = 60;
uint8_t _iou_threshold   = 45;

// MARK: - Temporary Variables (please do not modify)
std::vector<types::anchor_stride_t>          _anchor_strides;
std::vector<std::pair<float, float>>         _scaled_strides;
std::vector<std::vector<types::pt_t<float>>> _anchor_matrix;

static constexpr size_t _outputs         = 7;
static constexpr size_t _anchor_variants = 3;

size_t _output_scores_ids[_anchor_variants];
size_t _output_bboxes_ids[_anchor_variants];
size_t _output_keypoints_id;

el_shape_t       _output_shapes[_outputs];
el_quant_param_t _output_quant_params[_outputs];

std::forward_list<KeyPointType> _results;

// MARK: - Init functions

/*
 * @brief Init function for YOLOv8 Pose model
 * @param output_shapes: array of output shapes
 * @param output_quant_params: array of output quantization parameters
 * @return void
 * @note the two input arrays should have the same size as _outputs
 *       and the 'output_shapes' should be in the same order as the 'output_quant_params'
*/
void yolov8PoseInit(const el_shape_t* output_shapes, const el_quant_param_t* output_quant_params) {
    assert(output_shapes != nullptr);
    assert(output_quant_params != nullptr);

    _anchor_strides = utils::generate_anchor_strides(std::min(tensor_inp_width, tensor_inp_height));
    _anchor_matrix  = utils::generate_anchor_matrix(_anchor_strides);

    for (size_t i = 0; i < _outputs; ++i) {
        _output_shapes[i]       = output_shapes[i];
        _output_quant_params[i] = output_quant_params[i];
    }

    _scaled_strides.reserve(_anchor_strides.size());
    for (const auto& anchor_stride : _anchor_strides) {
        _scaled_strides.emplace_back(std::make_pair(static_cast<float>(anchor_stride.stride) * _w_scale,
                                                    static_cast<float>(anchor_stride.stride) * _h_scale));
    }

    for (size_t i = 0; i < _outputs; ++i) {
        // assuimg all outputs has 3 dims and the first dim is 1 (actual validation is done in is_model_valid)
        auto dim_1 = _output_shapes[i].dims[1];
        auto dim_2 = _output_shapes[i].dims[2];

        switch (dim_2) {
        case 1:
            for (size_t j = 0; j < _anchor_variants; ++j) {
                if (dim_1 == static_cast<int>(_anchor_strides[j].size)) {
                    _output_scores_ids[j] = i;
                    break;
                }
            }
            break;
        case 64:
            for (size_t j = 0; j < _anchor_variants; ++j) {
                if (dim_1 == static_cast<int>(_anchor_strides[j].size)) {
                    _output_bboxes_ids[j] = i;
                    break;
                }
            }
            break;
        default:
            if (dim_2 % 3 == 0) {
                _output_keypoints_id = i;
            }
        }
    }

    // check if all outputs ids is unique and less than outputs (redandant)
    using CheckType       = uint8_t;
    size_t    check_bytes = sizeof(CheckType) * 8u;
    CheckType check       = 1 << (_output_keypoints_id % check_bytes);
    for (size_t i = 0; i < _anchor_variants; ++i) {
        CheckType f_s = 1 << (_output_scores_ids[i] % check_bytes);
        CheckType f_b = 1 << (_output_bboxes_ids[i] % check_bytes);
        assert(!(f_s & f_b));
        assert(!(f_s & check));
        assert(!(f_b & check));
        check |= f_s | f_b;
    }
    assert(!(check ^ 0b01111111));
}

// MARK: - Postprocess functions

/*
 * @brief Postprocess function for YOLOv8 Pose model
 * @param tensors_out_ptr: pointer to the output tensors
 *        it should be an 2d array of int8_t* with size of _outputs
 *        the 2nd dimension should be the output tensor flattened in a 1d array
 *        please see TF Lite Micro implementation for more details
 *        we will match the output tensor with the its shape automatically
 * @return list of keypoints
*/
std::forward_list<KeyPointType> yolov8PosePostprocess(float** tensors_out_ptr) {
    assert(tensors_out_ptr != nullptr);

    _results.clear();

    const float* output_data[_outputs];
    for (size_t i = 0; i < _outputs; ++i) {
        output_data[i] = static_cast<float*>(tensors_out_ptr[i]);
        assert(output_data[i] != nullptr);
    }

    // post-process
    const float score_threshold = static_cast<float>(_score_threshold) / 100.f;
    const float iou_threshold   = static_cast<float>(_iou_threshold) / 100.f;

    std::forward_list<types::anchor_bbox_t> anchor_bboxes;

    const auto anchor_matrix_size = _anchor_matrix.size();
    for (size_t i = 0; i < anchor_matrix_size; ++i) {
        const auto  output_scores_id         = _output_scores_ids[i];
        const auto* output_scores            = output_data[output_scores_id];
        const auto  output_scores_quant_parm = _output_quant_params[output_scores_id];

        const auto  output_bboxes_id           = _output_bboxes_ids[i];
        const auto* output_bboxes              = output_data[output_bboxes_id];
        const auto  output_bboxes_shape_dims_2 = _output_shapes[output_bboxes_id].dims[2];
        const auto  output_bboxes_quant_parm   = _output_quant_params[output_bboxes_id];

        const auto  stride  = _scaled_strides[i];
        const float scale_w = stride.first;
        const float scale_h = stride.second;

        const auto& anchor_array      = _anchor_matrix[i];
        const auto  anchor_array_size = anchor_array.size();

        for (size_t j = 0; j < anchor_array_size; ++j) {
            float score = utils::sigmoid(utils::dequant_value_i(
              j, output_scores, output_scores_quant_parm.zero_point, output_scores_quant_parm.scale));

            if (score < score_threshold) continue;

            // DFL
            float dist[4];
            float matrix[16];

            const auto pre = j * output_bboxes_shape_dims_2;
            for (size_t m = 0; m < 4; ++m) {
                const size_t offset = pre + m * 16;
                for (size_t n = 0; n < 16; ++n) {
                    matrix[n] = utils::dequant_value_i(
                      offset + n, output_bboxes, output_bboxes_quant_parm.zero_point, output_bboxes_quant_parm.scale);
                }

                utils::softmax(matrix, 16);

                float res = 0.f;
                for (size_t n = 0; n < 16; ++n) {
                    res += matrix[n] * static_cast<float>(n);
                }
                dist[m] = res;
            }

            const auto anchor = anchor_array[j];

            float x1 = (anchor.x - dist[0]) * scale_w;
            float y1 = (anchor.y - dist[1]) * scale_h;
            float x2 = (anchor.x + dist[2]) * scale_w;
            float y2 = (anchor.y + dist[3]) * scale_h;

            anchor_bboxes.emplace_front(types::anchor_bbox_t{
              .x1           = x1,
              .y1           = y1,
              .x2           = x2,
              .y2           = y2,
              .score        = score,
              .anchor_class = static_cast<decltype(types::anchor_bbox_t::anchor_class)>(i),
              .anchor_index = static_cast<decltype(types::anchor_bbox_t::anchor_index)>(j),
            });
        }
    }

    if (anchor_bboxes.empty()) return _results;

    utils::anchor_nms(anchor_bboxes, iou_threshold, score_threshold, false);

    const auto*  output_keypoints            = output_data[_output_keypoints_id];
    const auto   output_keypoints_dims_2     = _output_shapes[_output_keypoints_id].dims[2];
    const auto   output_keypoints_quant_parm = _output_quant_params[_output_keypoints_id];
    const size_t keypoint_nums               = output_keypoints_dims_2 / 3;

    std::vector<types::pt3_t<float>> n_keypoint(keypoint_nums);

    // extract keypoints from outputs and store all results
    for (const auto& anchor_bbox : anchor_bboxes) {
        const auto pre =
          (_anchor_strides[anchor_bbox.anchor_class].start + anchor_bbox.anchor_index) * output_keypoints_dims_2;

        auto       anchor = _anchor_matrix[anchor_bbox.anchor_class][anchor_bbox.anchor_index];
        const auto stride = _scaled_strides[anchor_bbox.anchor_class];

        anchor.x -= 0.5f;
        anchor.y -= 0.5f;
        const float scale_w = stride.first;
        const float scale_h = stride.second;

        for (size_t i = 0; i < keypoint_nums; ++i) {
            const auto offset = pre + i * 3;

            float x = utils::dequant_value_i(
              offset, output_keypoints, output_keypoints_quant_parm.zero_point, output_keypoints_quant_parm.scale);
            float y = utils::dequant_value_i(
              offset + 1, output_keypoints, output_keypoints_quant_parm.zero_point, output_keypoints_quant_parm.scale);
            float z = utils::dequant_value_i(
              offset + 2, output_keypoints, output_keypoints_quant_parm.zero_point, output_keypoints_quant_parm.scale);

            x = x * 2.f + anchor.x;
            y = y * 2.f + anchor.y;
            z = utils::sigmoid(z);

            n_keypoint[i] = {x, y, z};
        }

        // convert coordinates and rescale bbox
        float cx = (anchor_bbox.x1 + anchor_bbox.x2) * 0.5f;
        float cy = (anchor_bbox.y1 + anchor_bbox.y2) * 0.5f;
        float w  = (anchor_bbox.x2 - anchor_bbox.x1);
        float h  = (anchor_bbox.y2 - anchor_bbox.y1);
        float s  = anchor_bbox.score * 100.f;

        KeyPointType keypoint;
        keypoint.box = {
          .x      = static_cast<decltype(KeyPointType::box.x)>(std::round(cx)),
          .y      = static_cast<decltype(KeyPointType::box.y)>(std::round(cy)),
          .w      = static_cast<decltype(KeyPointType::box.w)>(std::round(w)),
          .h      = static_cast<decltype(KeyPointType::box.h)>(std::round(h)),
          .score  = static_cast<decltype(KeyPointType::box.score)>(std::round(s)),
          .target = static_cast<decltype(KeyPointType::box.target)>(0),
        };
        keypoint.pts.reserve(keypoint_nums);
        size_t target = 0;
        for (const auto& kp : n_keypoint) {
            float x = kp.x * scale_w;
            float y = kp.y * scale_h;
            float z = kp.z * 100.f;
            keypoint.pts.emplace_back(el_point_t{
              .x      = static_cast<decltype(el_point_t::x)>(std::round(x)),
              .y      = static_cast<decltype(el_point_t::y)>(std::round(y)),
              .score  = static_cast<decltype(el_point_t::score)>(std::round(z)),
              .target = static_cast<decltype(el_point_t::target)>(target++),
            });
        }
        keypoint.score  = keypoint.box.score;
        keypoint.target = keypoint.box.target;
        _results.emplace_front(std::move(keypoint));
    }

    return _results;
}

}  // namespace sscma

#include <filesystem>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core.hpp>


// MARK: - Main
int main() {
    using namespace sscma;

    // MARK: Tensor descriptions
    el_shape_t       output_shapes[7];
    el_quant_param_t output_quant_params[7];

    // 256x256x3
    // output_shapes[0] = {3, new int[3]{1, 256,  64}}; // 16384
    // output_shapes[1] = {3, new int[3]{1, 1024, 64}}; // 65536
    // output_shapes[2] = {3, new int[3]{1, 64,   1}};  // 64
    // output_shapes[3] = {3, new int[3]{1, 1344, 51}}; // 68864
    // output_shapes[4] = {3, new int[3]{1, 1024, 1}};  // 1024
    // output_shapes[5] = {3, new int[3]{1, 64,   64}}; // 4096
    // output_shapes[6] = {3, new int[3]{1, 256,  1}};  // 256

    // output_quant_params[0] = {0.07264699786901474d, -70};
    // output_quant_params[1] = {0.07905934751033783d, -66};
    // output_quant_params[2] = {0.14429759979248047d, +107};
    // output_quant_params[3] = {0.06099024787545204d, +7};
    // output_quant_params[4] = {0.09318135678768158d, +112};
    // output_quant_params[5] = {0.07246128469705582d, -66};
    // output_quant_params[6] = {0.18436530232429504d, +116};

    // 192x192x3
    output_shapes[0] = {3, new int[3]{1, 144, 64}}; // 9216
    output_shapes[1] = {3, new int[3]{1, 36,  64}}; // 2304
    output_shapes[2] = {3, new int[3]{1, 576, 1}};  // 576
    output_shapes[3] = {3, new int[3]{1, 144, 1}};  // 144
    output_shapes[4] = {3, new int[3]{1, 36,  1}};  // 36
    output_shapes[5] = {3, new int[3]{1, 756, 51}}; // 38556
    output_shapes[6] = {3, new int[3]{1, 576, 64}}; // 36864

    output_quant_params[0] = {0.0743677094578743d,  -72};
    output_quant_params[1] = {0.0694025531411171d,  -64};
    output_quant_params[2] = {0.09354444593191147d, +110};
    output_quant_params[3] = {0.1864520013332367d,  +116};
    output_quant_params[4] = {0.14908528327941895d, +107};
    output_quant_params[5] = {0.05130333825945854d, +5};
    output_quant_params[6] = {0.08505406230688095d, -64};

    yolov8PoseInit(output_shapes, output_quant_params);


    // MARK: Load tensors
    auto loadTensorFromFile = [](const std::string& filename) {
        std::ifstream     file(filename, std::ios::in | std::ios::binary);
        std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
        std::cout << "Loaded " << filename << " with size " << buffer.size() << std::endl;
        return buffer;
    };

    std::string prefix = std::filesystem::current_path().string();
    prefix += "/../../Reshaped/";

    std::string file_id = "005";

    std::vector<char> buffers[7] = {
        loadTensorFromFile(prefix + file_id + "_out_0.bin"),
        loadTensorFromFile(prefix + file_id + "_out_5.bin"),
        loadTensorFromFile(prefix + file_id + "_out_4.bin"),
        loadTensorFromFile(prefix + file_id + "_out_6.bin"),
        loadTensorFromFile(prefix + file_id + "_out_2.bin"),
        loadTensorFromFile(prefix + file_id + "_out_3.bin"),
        loadTensorFromFile(prefix + file_id + "_out_1.bin"),
    };

    // cast bytes to float32, hope we don't have to deal with endianness
    std::vector<float> tensors[7];
    for (size_t i = 0; i < 7; ++i) {
        tensors[i].resize(buffers[i].size() / sizeof(float));
        
        for (size_t j = 0; j < tensors[i].size(); ++j) {
            union {
                float f;
                char  b[4];
            } u;
            u.b[0] = buffers[i][j * sizeof(float) + 0];
            u.b[1] = buffers[i][j * sizeof(float) + 1];
            u.b[2] = buffers[i][j * sizeof(float) + 2];
            u.b[3] = buffers[i][j * sizeof(float) + 3];

            tensors[i][j] = u.f;
            std::cout << std::fixed << std::setprecision(4) << u.f << " ";
            static int k = 0;
            if (++k % 16 == 0) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;

        std::cout << "Tensor " << i << " size: " << tensors[i].size() << std::endl;
        
        assert(tensors[i].size() == output_shapes[i].dims[0] * output_shapes[i].dims[1] * output_shapes[i].dims[2]);
    }

    // quantize to int8
    // std::vector<int8_t> tensors_int8[7];
    // for (size_t i = 0; i < 7; ++i) {
    //     tensors_int8[i].resize(tensors[i].size());
    //     for (size_t j = 0; j < tensors[i].size(); ++j) {
    //         tensors_int8[i][j] = 
    //         static_cast<int8_t>(
    //             std::round(
    //                 (tensors[i][j] / output_quant_params[i].scale)
    //             )
    //         ) - output_quant_params[i].zero_point;
    //         // static_cast<int8_t>(
    //         //     std::round(
    //         //         tensors[i][j]
    //         //     )
    //         // );
    //     }
    // }

    float* quan_tensors[7] = {
        tensors[0].data(),
        tensors[1].data(),
        tensors[2].data(),
        tensors[3].data(),
        tensors[4].data(),
        tensors[5].data(),
        tensors[6].data(),
    };

    // MARK: Postprocess
    auto res = yolov8PosePostprocess(quan_tensors);

    for (const auto& kp : res) {
        std::cout << "Box: [" << kp.box.x << ", " << kp.box.y << ", " << kp.box.w << ", " << kp.box.h << "] " << 
            static_cast<int>(kp.box.score) << " " << static_cast<int>(kp.box.target) << std::endl;
        // for (const auto& pt : kp.pts) {
        //     std::cout << "Point: " << pt.x << ", " << pt.y << ", " << static_cast<int>(pt.score) << std::endl;
        // }
    }

    std::cout << "Results: " << std::distance(res.begin(), res.end()) << std::endl;


    // MARK: Visualize
    prefix = std::filesystem::current_path().string();
    prefix += "/../../DataInjection_Input_JPG/";

    cv::Mat img = cv::imread(prefix + file_id + ".jpg");

    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(image_inp_width, image_inp_height));

    size_t i = 0;
    for (const auto& kp : res) {
        cv::rectangle(img_resized, cv::Rect(kp.box.x - (kp.box.w / 2), kp.box.y - ((kp.box.h / 2)), kp.box.w, kp.box.h), cv::Scalar(0, 255, 0), 2);
        for (const auto& pt : kp.pts) {
            cv::circle(img_resized, cv::Point(pt.x, pt.y), 1, cv::Scalar(0, 255 * i, 255), 1);
        }
    }

    cv::namedWindow("Result", cv::WINDOW_NORMAL);
    cv::imshow("Result", img_resized);
    cv::waitKey(0);
}
