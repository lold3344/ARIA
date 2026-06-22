use std::process::Command;
use std::path::PathBuf;

fn find_cl_exe() -> Option<String> {
    // Try PATH first
    if Command::new("cl.exe").arg("/?").output().map(|o| o.status.success()).unwrap_or(false) {
        return Some("cl.exe".to_string());
    }

    // Search common Visual Studio install locations
    let roots = [
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio",
    ];
    let years = ["2022", "2019", "2017"];
    let editions = ["BuildTools", "Community", "Professional", "Enterprise"];
    let host = "Hostx64";
    let target = "x64";

    for root in &roots {
        for year in &years {
            for ed in &editions {
                let cl = format!(
                    r"{}\{}\{}\VC\Tools\MSVC\*\bin\{}\{}\cl.exe",
                    root, year, ed, host, target
                );
                // Use glob via fs::read_dir
                let base = format!(r"{}\{}\{}\VC\Tools\MSVC", root, year, ed);
                if let Ok(entries) = std::fs::read_dir(&base) {
                    // Collect all versions, pick the newest (lexicographic sort works for MSVC versions)
                    let mut versions: Vec<_> = entries.flatten()
                        .map(|e| e.path().join("bin").join(host).join(target).join("cl.exe"))
                        .filter(|p| p.exists())
                        .collect();
                    versions.sort();
                    if let Some(newest) = versions.last() {
                        return Some(newest.to_str().unwrap().to_string());
                    }
                }
                let _ = cl; // suppress unused warning
            }
        }
    }
    None
}

fn main() {
    let cuda_root = std::env::var("CUDA_PATH")
        .unwrap_or_else(|_| r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9".to_string());

    let out_dir    = std::env::var("OUT_DIR").unwrap();
    let kernel_src = "src/kernels.cu";
    let ptx_out    = PathBuf::from(&out_dir).join("kernels.ptx");

    println!("cargo:rerun-if-changed={}", kernel_src);

    let cl = find_cl_exe();

    // ── 1. Compile kernels.cu → PTX (device kernels) ──────────────
    let mut args = vec![
        kernel_src.to_string(),
        "--ptx".to_string(),
        "-O3".to_string(),
        "--use_fast_math".to_string(),
        "-arch=sm_89".to_string(),
        "-o".to_string(), ptx_out.to_str().unwrap().to_string(),
    ];

    if let Some(ref cc) = cl {
        println!("cargo:warning=nvcc host compiler: {}", cc);
        args.push("-ccbin".to_string());
        args.push(cc.clone());
    } else {
        println!("cargo:warning=cl.exe not found, trying nvcc without host compiler flag");
    }

    let status = Command::new("nvcc")
        .args(&args)
        .status()
        .expect("nvcc not found — ensure CUDA Toolkit is installed and nvcc is in PATH");

    if !status.success() {
        if cl.is_none() {
            panic!(
                "nvcc PTX compilation failed.\n\
                 nvcc requires MSVC cl.exe on Windows.\n\
                 Install Visual Studio Build Tools from:\n\
                 https://visualstudio.microsoft.com/visual-cpp-build-tools/\n\
                 Make sure 'Desktop development with C++' workload is selected."
            );
        } else {
            panic!("nvcc PTX compilation failed");
        }
    }

    println!("cargo:rustc-env=OUT_DIR={}", out_dir);

    let cuda_lib = format!(r"{}\lib\x64", cuda_root);
    println!("cargo:rustc-link-search=native={}", cuda_lib);
}
