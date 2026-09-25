#![recursion_limit = "256"]
use std::{
    thread,
    time::{Duration, Instant},
};

use burn::optim::AdamConfig;
use model::{ModelConfig, TrainingConfig};

pub mod data;
pub mod model;

fn main() {
    let backend = if cfg!(feature = "flex") {
        "Flex"
    } else if cfg!(feature = "cubecl") {
        "CubeCL CPU"
    } else if cfg!(feature = "wgpu") {
        "WGPU"
    } else {
        "NdArray"
    };

    println!("Using {backend} backend. Starting in 2 seconds...");
    thread::sleep(Duration::from_secs(2));
    let artifact_dir = "/tmp/mnist_test_guide";
    let start = Instant::now();

    let device: burn::tensor::Device = {
        #[cfg(feature = "flex")]
        {
            burn::tensor::Device::flex().autodiff()
        }
        #[cfg(all(not(feature = "flex"), feature = "cubecl"))]
        {
            burn::tensor::Device::cpu().autodiff()
        }
        #[cfg(all(not(feature = "flex"), not(feature = "cubecl"), feature = "wgpu"))]
        {
            let device: burn::tensor::Device = burn::backend::wgpu::WgpuDevice::default().into();
            device.autodiff()
        }
        #[cfg(all(not(feature = "flex"), not(feature = "cubecl"), not(feature = "wgpu"),))]
        {
            let device: burn::tensor::Device =
                burn::backend::ndarray::NdArrayDevice::default().into();
            device.autodiff()
        }
    };

    crate::model::train(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()).with_num_epochs(1),
        device,
    );

    println!("Total runtime: {:.2?}", start.elapsed());
}
