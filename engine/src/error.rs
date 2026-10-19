use thiserror::Error;

#[derive(Debug, Error)]
pub enum KvasirError {
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),
    #[error("cuBLAS error: {0}")]
    Cublas(#[from] cudarc::cublas::result::CublasError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("config error: {0}")]
    Config(String),
    #[error("shape error: {0}")]
    Shape(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
}

pub type Result<T> = std::result::Result<T, KvasirError>;

#[macro_export]
macro_rules! ensure_shape {
    ($cond:expr, $($arg:tt)+) => {
        if !($cond) {
            return Err($crate::error::KvasirError::Shape(format!($($arg)+)));
        }
    };
}
