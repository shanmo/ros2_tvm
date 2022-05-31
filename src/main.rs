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
    fn test_segmentation() {
        let path = "./data/2011_09_26-0056-0000000081-003157.png".to_string();
        let img = cv::imgcodecs::imread(&path, cv::imgcodecs::IMREAD_COLOR).unwrap();
        let original_shape = img.size().unwrap();
        let mut img_rgb = img.clone();
        cv::imgproc::cvt_color(&img,
                                 &mut img_rgb,
                                 cv::imgproc::COLOR_BGR2RGB,
                                 0).unwrap();
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
        let arr: nd::ArrayView3<u8> = reduced.try_as_array().unwrap();
        println!("arr shape {:?}", arr.shape());
        let arr = preprocess(arr);

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

        let output: Vec<f32> = output_nd.to_vec::<f32>().unwrap();
        let output: Vec<u8> = output.iter().map(|&e| e as u8).collect();
        let seg_mask = nd::Array::from_shape_vec((256, 512), output)
            .unwrap()
            .to_owned();
        let seg_mask = ndarray2_to_cv_8u(&seg_mask);
        let mut mask = seg_mask.clone();
        cv::imgproc::resize(
            &seg_mask,
            &mut mask,
            cv::core::Size {
                width: original_shape.width,
                height: original_shape.height,
            },
            0.,
            0.,
            cv::imgproc::INTER_LINEAR,
        ).unwrap();

        cv::highgui::imshow("img", &img).unwrap();
        cv::highgui::imshow("segmentation", &mask).unwrap();
        cv::highgui::wait_key(0).unwrap();
    }
}
