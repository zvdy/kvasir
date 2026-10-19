use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> PathBuf {
    if let Ok(p) = std::env::var("NVCC") {
        return PathBuf::from(p);
    }
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(root) = std::env::var(var) {
            let p = Path::new(&root).join("bin/nvcc");
            if p.exists() {
                return p;
            }
        }
    }
    for root in ["/opt/cuda", "/usr/local/cuda"] {
        let p = Path::new(root).join("bin/nvcc");
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("nvcc")
}

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let kernels = Path::new("kernels");
    let nvcc = find_nvcc();
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=NVCC");

    let entries = std::fs::read_dir(kernels).expect("cannot read engine/kernels");
    for entry in entries {
        let path = entry.expect("bad dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).expect("bad kernel name");
        let ptx = out_dir.join(format!("{stem}.ptx"));
        let status = Command::new(&nvcc)
            .args(["-ptx", "-arch=sm_89", "-O3", "-lineinfo"])
            .arg("-I")
            .arg(kernels)
            .arg("-o")
            .arg(&ptx)
            .arg(&path)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {}: {e} (set NVCC=/path/to/nvcc)", nvcc.display()));
        if !status.success() {
            panic!("nvcc failed on {}", path.display());
        }
    }
}
