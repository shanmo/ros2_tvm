use anyhow::Result;
use cv2::prelude::*;
use ndarray as nd;
use opencv as cv2;

use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::sleep;

use futures::executor::LocalPool;
use futures::future;
use futures::stream::StreamExt;
use futures::task::LocalSpawnExt;
use r2r::QosProfile;

use tvm_rt as trt;

use opencv::dnn::print;
use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
};

mod lib;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "camera_subscriber", "")?;
    let mut pool = LocalPool::new();
    let spawner = pool.spawner();

    let _camera_subscriber = node.subscribe("/kitti/image/color/left", QosProfile::default())?;

    spawner.spawn_local(async move {
        _camera_subscriber
            .for_each(|img: r2r::sensor_msgs::msg::Image| {
                lib::detect(&img);
                future::ready(())
            })
            .await
    })?;

    // loop indefinitely to ensure that the objects are not destroyed
    loop {
        node.spin_once(std::time::Duration::from_millis(100));
        pool.run_until_stalled();
    }
}
