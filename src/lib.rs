use anyhow::Result;
use cv2::prelude::*;
use ndarray as nd;
use opencv as cv2;
use tvm_rt as trt;
use std::time::Instant;

use opencv::dnn::print;
use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
};

trait AsArray {
    fn try_as_array(&self) -> Result<nd::ArrayView3<u8>>;
}
impl AsArray for cv2::core::Mat {
    fn try_as_array(&self) -> Result<nd::ArrayView3<u8>> {
        let bytes = self.data_bytes()?;
        let size = self.size()?;
        let a = nd::ArrayView3::from_shape((size.height as usize, size.width as usize, 3), bytes)?;
        Ok(a)
    }
}

pub fn ndarray2_to_cv_8u(a: &nd::Array2<u8>) -> cv2::core::Mat {
    unsafe {
        cv2::core::Mat::new_rows_cols_with_data(
            a.shape()[0] as i32,
            a.shape()[1] as i32,
            cv2::core::CV_8U,
            std::mem::transmute(a.as_ptr()),
            cv2::core::Mat_AUTO_STEP,
        )
        .unwrap()
    }
}

pub fn ndarray3_to_cv2_8u(data: &nd::Array3<u8>) -> cv2::core::Mat {
    // assert the 3rd dimension is exactly 3
    assert_eq!(data.shape()[2], 3);
    // transforms an Array3 into a opencv Mat data type.
    unsafe {
        cv2::core::Mat::new_rows_cols_with_data(
            data.shape()[0] as i32,
            data.shape()[1] as i32,
            cv2::core::CV_8UC3,
            std::mem::transmute(data.as_ptr()),
            cv2::core::Mat_AUTO_STEP,
        )
        .unwrap()
    }
}

pub fn preprocess(img: nd::ArrayView3<u8>) -> nd::Array<f32, nd::IxDyn> {
    let avg = vec![0.485, 0.456, 0.406];
    let std = vec![0.229, 0.224, 0.225];

    let mut arr = img.clone().mapv(|elem| (elem as f32) / 255.0);
    for ((_x, _y, z), value) in arr.indexed_iter_mut() {
        let temp = (value.clone() - avg[z]) / std[z];
        *value = temp;
    }
    let arr = arr.permuted_axes([2, 0, 1]).into_dyn();
    let arr = arr.insert_axis(nd::Axis(0)).to_owned();
    return arr;
}

pub fn make_segmentation_visualisation_with_transparency(
    image_color: &nd::Array3<u8>,
    seg_mask: &nd::Array2<u8>,
    color_map: &nd::Array2<u8>,
) -> nd::Array3<u8> {
    let mut overlay: nd::Array3<u8> = image_color.clone();
    let shape = image_color.shape();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..3 {
                overlay[[i, j, k]] =
                    overlay[[i, j, k]] / 2 + color_map[[seg_mask[[i, j]] as usize, k]] / 2;
            }
        }
    }
    overlay
}

/// convert the BoundingBox to rect in opencv
pub fn bbox2rect(bbox: &Vec<f32>) -> cv2::core::Rect {
    let x1 = (bbox[0] - 0.5 * bbox[2]) * 848.0;
    let y1 = (bbox[1] - 0.5 * bbox[3]) * 480.0;
    let rect = cv2::core::Rect {
        x: x1 as i32,
        y: y1 as i32,
        width: (bbox[2] * 848.0) as i32,
        height: (bbox[3] * 480.0) as i32,
    };
    rect
}

/// wrapper function for rectangle
pub fn plot_rect_cv(
    image_vis: &mut cv2::core::Mat,
    rect: cv2::core::Rect,
    color: cv2::core::Scalar,
) {
    const THICKNESS: i32 = 5;
    const LINE_TYPE: i32 = 8;
    const SHIFT: i32 = 0;
    let _ = cv2::imgproc::rectangle(image_vis, rect, color, THICKNESS, LINE_TYPE, SHIFT).unwrap();
}

pub fn cv_bridge(img: &r2r::sensor_msgs::msg::Image) -> cv2::core::Mat {
    // kitti image size
    const kw: i32 = 1242;
    const kh: i32 = 375;

    // realsense image size
    const rw: i32 = 848;
    const rh: i32 = 480;

    let mut mat = cv2::core::Mat::zeros(kh, kw, cv2::core::CV_32FC3)
        .unwrap()
        .to_mat()
        .unwrap();
    for i in 0..kh {
        for j in 0..kw {
            let mut pix = cv2::core::Vec3f::default();
            pix[0] = img.data[((i * kw + j) * 3 + 0) as usize].into();
            pix[1] = img.data[((i * kw + j) * 3 + 1) as usize].into();
            pix[2] = img.data[((i * kw + j) * 3 + 2) as usize].into();
            *mat.at_2d_mut(i, j).unwrap() = pix;
        }
    }
    let mut mat_8u = Mat::default();
    mat.convert_to(&mut mat_8u, cv2::core::CV_8UC3, 1., 0.)
        .unwrap();
    let img_rgb = mat_8u.clone();

    cv2::imgproc::resize(
        &img_rgb,
        &mut mat_8u,
        cv2::core::Size {
            width: rw,
            height: rh,
        },
        0.,
        0.,
        cv2::imgproc::INTER_LINEAR,
    )
    .unwrap();

    return mat_8u;
}

pub fn detect_objects(img: &cv2::core::Mat) -> cv2::core::Mat {
    let mut img_display = img.clone();
    let image_arr: nd::ArrayView3<u8> = img.try_as_array().unwrap();
    let arr = preprocess(image_arr);

    let now = Instant::now();
    let dev = trt::Device::cpu(0);
    let input =
        trt::NDArray::from_rust_ndarray(&arr, dev, trt::DataType::float(32, 1)).unwrap();
    // load the built module
    let lib = trt::Module::load(&Path::new("./model/detection_lib.so")).unwrap();
    let mut graph_rt = trt::graph_rt::GraphRt::from_module(lib, dev).unwrap();
    graph_rt.set_input("input0", input).unwrap();
    graph_rt.run().unwrap();
    let elapsed = now.elapsed();
    println!("Time elapsed for object detection: {:.2?} ns", elapsed);
    println!("Frequcny for object detection: {:.2?}", 1.0/(elapsed.as_nanos() as f64 / 1e9));

    // prepare to get the output
    let labels_nd = graph_rt.get_output(0).unwrap();
    let bboxes_nd = graph_rt.get_output(1).unwrap();
    let probs_nd = graph_rt.get_output(2).unwrap();

    let labels: Vec<i32> = labels_nd.to_vec::<i32>().unwrap();
    // println!("labels {:?}", labels);
    let bboxes_flat: Vec<f32> = bboxes_nd.to_vec::<f32>().unwrap();
    // println!("bboxes_flat shape {}", bboxes_flat.len());
    let bboxes: Vec<Vec<f32>> = bboxes_flat.chunks(4).map(|x| x.to_vec()).collect();
    let probs: Vec<f32> = probs_nd.to_vec::<f32>().unwrap();

    for (i, bbox) in bboxes.iter().enumerate() {
        if probs[i] < 0.7 {
            continue;
        }
        // println!("probs {}", probs[i]);
        // println!("label {}", labels[i]);
        // println!("bbox {:?}", bbox);
        let rect = bbox2rect(bbox);
        let color = cv2::core::Scalar::new(255f64, 0f64, 0f64, -1f64);
        plot_rect_cv(&mut img_display, rect, color);
    }

    return img_display;
}

pub fn segment(img: &cv2::core::Mat) -> cv2::core::Mat {
    let mut img_display = img.clone();

    let mut reduced = img.clone();
    cv2::imgproc::resize(
        &img_display,
        &mut reduced,
        cv2::core::Size {
            width: 512,
            height: 256,
        },
        0.,
        0.,
        cv2::imgproc::INTER_LINEAR,
    )
        .unwrap();
    let image_arr: nd::ArrayView3<u8> = reduced.try_as_array().unwrap();
    // println!("arr shape {:?}", image_arr.shape());
    let arr = preprocess(image_arr);
    // println!("input {:?}", arr);

    let dev = trt::Device::cpu(0);

    let input =
        trt::NDArray::from_rust_ndarray(&arr, dev, trt::DataType::float(32, 1)).unwrap();

    // load the built module
    let lib = trt::Module::load(&Path::new("./model/segmentation_lib.so")).unwrap();
    let mut graph_rt = trt::graph_rt::GraphRt::from_module(lib, dev).unwrap();
    graph_rt.set_input("input0", input).unwrap();
    let now = Instant::now();
    graph_rt.run().unwrap();
    let elapsed = now.elapsed();
    println!("Time elapsed for image segmentation: {:.2?} ns", elapsed);
    println!("Frequcny for segmentation: {:.2?}", 1.0/(elapsed.as_nanos() as f64 / 1e9));

    // prepare to get the output
    let output_nd = graph_rt.get_output(0).unwrap();

    let output: Vec<i32> = output_nd.to_vec::<i32>().unwrap();
    let output: Vec<u8> = output.iter().map(|&e| e as u8).collect();
    // println!("{:?}", output);
    let seg_mask = nd::Array::from_shape_vec((480, 848), output)
        .unwrap()
        .to_owned();

    let color_map = nd::array![
            [128,0,128],
            [0,128,128],
            [0,0,128],
            [165,42,42],
            [178,34,34],
            [220,20,60],
            [0,206,209],
            [64,224,208],
            [175,238,238],
            [127,255,212],
            [176,224,230],
            [95,158,160],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [0,128,0],
            [128,128,0],
            [128,0,0],
            [128,128,128],
            [192,192,192],
        ];

    let img_display: nd::ArrayView3<u8> = img_display.try_as_array().unwrap();
    let image_seg: nd::Array3<u8> = make_segmentation_visualisation_with_transparency(
        &img_display.to_owned(),
        &seg_mask,
        &color_map,
    );
    let image_seg: cv2::core::Mat = ndarray3_to_cv2_8u(&image_seg);

    return image_seg;
}

pub fn process_image(img: &r2r::sensor_msgs::msg::Image) -> cv2::core::Mat {
    let mat = cv_bridge(img);
    let img_display = detect_objects(&mat);
    // let img_display = segment(&mat);
    cv2::highgui::imshow("kitti", &img_display).unwrap();
    cv2::highgui::wait_key(10).unwrap();
    return mat;
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_image() {
        // let path = "./data/2011_09_26-0056-0000000081-003157.png".to_string();
        // let img = cv2::imgcodecs::imread(&path, cv2::imgcodecs::IMREAD_COLOR).unwrap();
        //
        // cv2::highgui::imshow("img", &img).unwrap();
        // cv2::highgui::wait_key(100).unwrap();
    }

    #[test]
    fn test_detection() {
        let path = "./data/2011_09_26-0056-0000000081-003157.png".to_string();
        let img = cv2::imgcodecs::imread(&path, cv2::imgcodecs::IMREAD_COLOR).unwrap();
        let mut img_rgb = img.clone();
        cv2::imgproc::cvt_color(&img, &mut img_rgb, cv2::imgproc::COLOR_BGR2RGB, 0).unwrap();
        let mut img_display = img_rgb.clone();
        cv2::imgproc::resize(
            &img_rgb,
            &mut img_display,
            cv2::core::Size {
                width: 848,
                height: 480,
            },
            0.,
            0.,
            cv2::imgproc::INTER_LINEAR,
        )
        .unwrap();

        let image_arr: nd::ArrayView3<u8> = img_display.try_as_array().unwrap();
        // println!("arr shape {:?}", image_arr.shape());
        let arr = preprocess(image_arr);
        // println!("input {:?}", arr);

        let dev = trt::Device::cpu(0);

        let input =
            trt::NDArray::from_rust_ndarray(&arr, dev, trt::DataType::float(32, 1)).unwrap();
        // println!(
        //     "input shape is {:?}, len: {}, size: {}",
        //     input.shape(),
        //     input.len(),
        //     input.size(),
        // );

        // load the built module
        let lib = trt::Module::load(&Path::new("./model/detection_lib.so")).unwrap();
        let mut graph_rt = trt::graph_rt::GraphRt::from_module(lib, dev).unwrap();
        graph_rt.set_input("input0", input).unwrap();
        graph_rt.run().unwrap();

        // prepare to get the output
        let labels_nd = graph_rt.get_output(0).unwrap();
        let bboxes_nd = graph_rt.get_output(1).unwrap();
        let probs_nd = graph_rt.get_output(2).unwrap();

        // println!(
        //     "probs_nd shape is {:?}, len: {}, size: {}",
        //     probs_nd.shape(),
        //     probs_nd.len(),
        //     probs_nd.size(),
        // );
        //
        // println!(
        //     "output dtype {} {} {}",
        //     labels_nd.dtype(),
        //     bboxes_nd.dtype(),
        //     probs_nd.dtype(),
        // );

        let labels: Vec<i32> = labels_nd.to_vec::<i32>().unwrap();
        // println!("labels {:?}", labels);
        let bboxes_flat: Vec<f32> = bboxes_nd.to_vec::<f32>().unwrap();
        // println!("bboxes_flat shape {}", bboxes_flat.len());
        let bboxes: Vec<Vec<f32>> = bboxes_flat.chunks(4).map(|x| x.to_vec()).collect();
        // println!("bboxes shape {}", bboxes.len());
        // let bboxes: Vec<Vec<f32>> = bboxes
        //     .iter()
        //     .map(|x| x.iter().map(|v| v / ratio).collect())
        //     .collect();
        // println!("bboxes shape {}", bboxes.len());
        let probs: Vec<f32> = probs_nd.to_vec::<f32>().unwrap();
        // println!("scores {:?}", scores);

        for (i, bbox) in bboxes.iter().enumerate() {
            if probs[i] < 0.7 {
                continue;
            }
            println!("probs {}", probs[i]);
            println!("label {}", labels[i]);
            println!("bbox {:?}", bbox);
            let rect = bbox2rect(bbox);
            let color = cv2::core::Scalar::new(255f64, 0f64, 0f64, -1f64);
            plot_rect_cv(&mut img_display, rect, color);
        }

        cv2::highgui::imshow("detection", &img_display).unwrap();
        cv2::highgui::wait_key(0).unwrap();
    }

    #[test]
    fn test_segmentation() {
        let path = "./data/2011_09_26-0056-0000000081-003157.png".to_string();
        let img = cv2::imgcodecs::imread(&path, cv2::imgcodecs::IMREAD_COLOR).unwrap();
        let original_shape = img.size().unwrap();
        let mut img_rgb = img.clone();
        cv2::imgproc::cvt_color(&img, &mut img_rgb, cv2::imgproc::COLOR_BGR2RGB, 0).unwrap();
        let mut img_display = img_rgb.clone();
        cv2::imgproc::resize(
            &img_rgb,
            &mut img_display,
            cv2::core::Size {
                width: 848,
                height: 480,
            },
            0.,
            0.,
            cv2::imgproc::INTER_LINEAR,
        )
        .unwrap();
        let img_display: nd::ArrayView3<u8> = img_display.try_as_array().unwrap();

        let mut reduced = img_rgb.clone();
        cv2::imgproc::resize(
            &img_rgb,
            &mut reduced,
            cv2::core::Size {
                width: 512,
                height: 256,
            },
            0.,
            0.,
            cv2::imgproc::INTER_LINEAR,
        )
        .unwrap();
        let image_arr: nd::ArrayView3<u8> = reduced.try_as_array().unwrap();
        // println!("arr shape {:?}", image_arr.shape());
        let arr = preprocess(image_arr);
        // println!("input {:?}", arr);

        let dev = trt::Device::cpu(0);

        let input =
            trt::NDArray::from_rust_ndarray(&arr, dev, trt::DataType::float(32, 1)).unwrap();
        // println!(
        //     "input shape is {:?}, len: {}, size: {}",
        //     input.shape(),
        //     input.len(),
        //     input.size(),
        // );

        // load the built module
        let lib = trt::Module::load(&Path::new("./model/segmentation_lib.so")).unwrap();
        let mut graph_rt = trt::graph_rt::GraphRt::from_module(lib, dev).unwrap();
        graph_rt.set_input("input0", input).unwrap();
        graph_rt.run().unwrap();

        // prepare to get the output
        let output_nd = graph_rt.get_output(0).unwrap();
        // println!(
        //     "output shape is {:?}, len: {}, size: {}, dtype {}",
        //     output_nd.shape(),
        //     output_nd.len(),
        //     output_nd.size(),
        //     output_nd.dtype()
        // );

        let output: Vec<i32> = output_nd.to_vec::<i32>().unwrap();
        let output: Vec<u8> = output.iter().map(|&e| e as u8).collect();
        // println!("{:?}", output);
        let seg_mask = nd::Array::from_shape_vec((480, 848), output)
            .unwrap()
            .to_owned();

        let color_map = nd::array![
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
        ];

        let image_seg: nd::Array3<u8> = make_segmentation_visualisation_with_transparency(
            &img_display.to_owned(),
            &seg_mask,
            &color_map,
        );
        let image_seg: cv2::core::Mat = ndarray3_to_cv2_8u(&image_seg);

        // cv2::highgui::imshow("segmentation", &image_seg).unwrap();
        // cv2::highgui::wait_key(0).unwrap();
    }
}
