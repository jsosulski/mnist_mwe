#![recursion_limit = "256"]
use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use model::{ModelConfig, TrainingConfig};

pub mod data;
pub mod model;

fn main() {
    let artifact_dir = "/tmp/guide";
    #[cfg(feature = "wgpu")]
    {
        type MyBackend = burn::backend::Wgpu<f32, i32>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        type MyAutodiffBackend = Autodiff<MyBackend>;
        crate::model::train::<MyAutodiffBackend>(
            artifact_dir,
            TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()).with_num_epochs(1),
            device,
        );
    }
    #[cfg(not(feature = "wgpu"))]
    {
        type MyBackend = burn::backend::NdArray<f32, i32>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        type MyAutodiffBackend = Autodiff<MyBackend>;
        crate::model::train::<MyAutodiffBackend>(
            artifact_dir,
            TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()).with_num_epochs(1),
            device,
        );
    }
}
