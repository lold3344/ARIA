use std::process::Command;
use std::path::PathBuf;

fn main() {
    let cuda_root = std::env::var("CUDA_PATH")
        .unwrap_or_else(|_| r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9".to_string());

    // Compile CUDA kernels
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let kernel_src = "src/kernels.cu";
    let obj_out = PathBuf::from(&out_dir).join("kernels.o");
    let lib_out = PathBuf::from(&out_dir).join("kernels.lib");

    println!("cargo:rerun-if-changed={}", kernel_src);

    // nvcc compile to object
    let status = Command::new("nvcc")
        .args([
            kernel_src,
            "-c",
            "-O3",
            "--use_fast_math",
            "-arch=sm_89",          // RTX 4060 = Ada Lovelace sm_89
            "--ptxas-options=-v",
            "-Xcompiler", "/MD",    // MSVC runtime
            "-o", obj_out.to_str().unwrap(),
        ])
        .status()
        .expect("nvcc not found — install CUDA Toolkit");
    assert!(status.success(), "nvcc compilation failed");

    // lib from object (Windows)
    let lib_status = Command::new("lib.exe")
        .args([
            &format!("/OUT:{}", lib_out.to_str().unwrap()),
            obj_out.to_str().unwrap(),
        ])
        .status()
        .expect("lib.exe not found — run from MSVC Developer Command Prompt");
    assert!(lib_status.success(), "lib.exe failed");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=kernels");

    // Link CUDA runtime and cuBLAS
    let cuda_lib = format!(r"{}\lib\x64", cuda_root);
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
