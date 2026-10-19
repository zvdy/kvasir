use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

use super::device::Gpu;
use crate::ensure_shape;
use crate::error::Result;

pub struct DeviceTensor<T> {
    pub buf: CudaSlice<T>,
    shape: Vec<usize>,
}

impl<T: DeviceRepr + ValidAsZeroBits> DeviceTensor<T> {
    pub fn zeros(gpu: &Gpu, shape: &[usize]) -> Result<Self> {
        let n: usize = shape.iter().product();
        Ok(Self {
            buf: gpu.stream.alloc_zeros::<T>(n)?,
            shape: shape.to_vec(),
        })
    }

    pub fn from_host(gpu: &Gpu, data: &[T], shape: &[usize]) -> Result<Self> {
        let n: usize = shape.iter().product();
        ensure_shape!(
            n == data.len(),
            "from_host: shape {shape:?} needs {n} elements, got {}",
            data.len()
        );
        Ok(Self {
            buf: gpu.stream.clone_htod(data)?,
            shape: shape.to_vec(),
        })
    }

    pub fn to_host(&self, gpu: &Gpu) -> Result<Vec<T>> {
        Ok(gpu.stream.clone_dtoh(&self.buf)?)
    }
}

impl<T> DeviceTensor<T> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn numel(&self) -> usize {
        self.buf.len()
    }
    pub fn rows(&self) -> usize {
        self.shape[0]
    }
    pub fn cols(&self) -> usize {
        self.shape[1..].iter().product()
    }
    pub fn from_slice(buf: CudaSlice<T>, shape: &[usize]) -> Result<Self> {
        let n: usize = shape.iter().product();
        ensure_shape!(
            n == buf.len(),
            "from_slice: shape {shape:?} needs {n} elements, buffer has {}",
            buf.len()
        );
        Ok(Self {
            buf,
            shape: shape.to_vec(),
        })
    }
}
