use anyhow::Result;
use ndarray as nd;
use opencv as cv;
use opencv::prelude::*;

use tvm_rt as trt;

use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
};

trait AsArray {
    fn try_as_array(&self) -> Result<nd::ArrayView3<u8>>;
}
impl AsArray for cv::core::Mat {
    fn try_as_array(&self) -> Result<nd::ArrayView3<u8>> {
        let bytes = self.data_bytes()?;
        let size = self.size()?;
        let a = nd::ArrayView3::from_shape((size.height as usize, size.width as usize, 3), bytes)?;
        Ok(a)
    }
}

pub fn ndarray2_to_cv_8u(a: &nd::Array2<u8>) -> cv::core::Mat {
    unsafe {
        cv::core::Mat::new_rows_cols_with_data(
            a.shape()[0] as i32,
            a.shape()[1] as i32,
            cv::core::CV_8U,
            std::mem::transmute(a.as_ptr()),
            cv::core::Mat_AUTO_STEP,
        )
        .unwrap()
    }
}

pub fn ndarray3_to_cv2_8u(data: &nd::Array3<u8>) -> cv::core::Mat {
    // assert the 3rd dimension is exactly 3
    assert_eq!(data.shape()[2], 3);
    // transforms an Array3 into a opencv Mat data type.
    unsafe {
        cv::core::Mat::new_rows_cols_with_data(
            data.shape()[0] as i32,
            data.shape()[1] as i32,
            cv::core::CV_8UC3,
            std::mem::transmute(data.as_ptr()),
            cv::core::Mat_AUTO_STEP,
        )
            .unwrap()
    }
}

pub fn preprocess(img: nd::ArrayView3<u8>) -> nd::Array<f32, nd::IxDyn> {
    let avg = vec![0.485, 0.456, 0.406];
    let std = vec![0.229, 0.224, 0.225];

    let mut arr = img.clone().mapv(|elem| (elem as f32) / 255.0);
    for ((x, y, z), value) in arr.indexed_iter_mut() {
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


fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_image() {
        let path = "./data/2011_09_26-0056-0000000081-003157.png".to_string();
        let img = cv::imgcodecs::imread(&path, cv::imgcodecs::IMREAD_COLOR).unwrap();

        cv::highgui::imshow("img", &img).unwrap();
        cv::highgui::wait_key(100).unwrap();
    }

    #[test]
    fn test_detection() {

    }

    #[test]
    fn test_segmentation() {
        let path = "./data/2011_09_26-0056-0000000081-003157.png".to_string();
        let img = cv::imgcodecs::imread(&path, cv::imgcodecs::IMREAD_COLOR).unwrap();
        let original_shape = img.size().unwrap();
        let mut img_rgb = img.clone();
        cv::imgproc::cvt_color(&img,
                                 &mut img_rgb,
                                 cv::imgproc::COLOR_BGR2RGB,
                                 0).unwrap();
        let mut img_display = img_rgb.clone();
        cv::imgproc::resize(
            &img_rgb,
            &mut img_display,
            cv::core::Size {
                width: 848,
                height: 480,
            },
            0.,
            0.,
            cv::imgproc::INTER_LINEAR,
        ).unwrap();
        let img_display: nd::ArrayView3<u8> = img_display.try_as_array().unwrap();

        let mut reduced = img_rgb.clone();
        cv::imgproc::resize(
            &img_rgb,
            &mut reduced,
            cv::core::Size {
                width: 512,
                height: 256,
            },
            0.,
            0.,
            cv::imgproc::INTER_LINEAR,
        ).unwrap();
        let image_arr: nd::ArrayView3<u8> = reduced.try_as_array().unwrap();
        println!("arr shape {:?}", image_arr.shape());
        let arr = preprocess(image_arr);
        // println!("input {:?}", arr);

        let dev = trt::Device::cpu(0);

        let input =
            trt::NDArray::from_rust_ndarray(&arr, dev, trt::DataType::float(32, 1)).unwrap();
        println!(
            "input shape is {:?}, len: {}, size: {}",
            input.shape(),
            input.len(),
            input.size(),
        );

        // load the built module
        let lib = trt::Module::load(&Path::new("./model/segmentation_lib.so")).unwrap();
        let mut graph_rt = trt::graph_rt::GraphRt::from_module(lib, dev).unwrap();
        graph_rt.set_input("input0", input).unwrap();
        graph_rt.run().unwrap();

        // prepare to get the output
        let output_nd = graph_rt.get_output(0).unwrap();
        println!(
            "output shape is {:?}, len: {}, size: {}, dtype {}",
            output_nd.shape(),
            output_nd.len(),
            output_nd.size(),
            output_nd.dtype()
        );

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
        let image_seg: cv::core::Mat = ndarray3_to_cv2_8u(&image_seg);

        cv::highgui::imshow("segmentation", &image_seg).unwrap();
        cv::highgui::wait_key(0).unwrap();
    }
}
