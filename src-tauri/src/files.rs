use std::path::Path;
use std::time::Duration;

use serde::Serialize;
use tokio::io::AsyncBufReadExt;
use tokio::time::sleep;

use crate::process::{data_dir, weights_dir, exe_dir};
use crate::AppState;

#[tauri::command]
pub async fn get_project_dir(state: tauri::State<'_, AppState>) -> Result<String, String> {
    Ok(state.project_dir.to_string_lossy().to_string())
}

#[derive(Debug, Serialize, Clone)]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub size_mb: f64,
    pub lines: Option<usize>,
}

#[tauri::command]
pub async fn list_gguf(state: tauri::State<'_, AppState>) -> Result<Vec<FileInfo>, String> {
    let dir = weights_dir(&state.project_dir);
    list_files(&dir, "gguf").await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_datasets(state: tauri::State<'_, AppState>) -> Result<Vec<FileInfo>, String> {
    let dir = data_dir(&state.project_dir);
    if !dir.exists() {
        return Err(format!("Data directory not found: {}", dir.display()));
    }
    // Do not count lines here; huge JSONL files make the GUI hang.
    list_files_no_lines(&dir, "jsonl").await.map_err(|e| format!("list_datasets failed: {}", e))
}

#[tauri::command]
pub async fn list_caches(state: tauri::State<'_, AppState>) -> Result<Vec<FileInfo>, String> {
    let dir = data_dir(&state.project_dir);
    list_files_prefix(&dir, "sequences_cache_").await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_binaries(state: tauri::State<'_, AppState>) -> Result<Vec<String>, String> {
    let dir = exe_dir(&state.project_dir);
    let mut names = Vec::new();
    if let Ok(entries) = tokio::fs::read_dir(dir).await {
        let mut entries = entries;
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".exe") {
                names.push(name.trim_end_matches(".exe").to_string());
            }
        }
    }
    names.sort();
    Ok(names)
}

async fn count_lines(path: &Path) -> Option<usize> {
    let file = tokio::fs::File::open(path).await.ok()?;
    let reader = tokio::io::BufReader::new(file);
    let mut lines = reader.lines();
    let mut count = 0usize;
    while let Ok(Some(_)) = lines.next_line().await {
        count += 1;
    }
    Some(count)
}

async fn list_files(dir: &Path, ext: &str) -> anyhow::Result<Vec<FileInfo>> {
    let mut infos = Vec::new();
    if !dir.exists() {
        anyhow::bail!("directory does not exist: {}", dir.display());
    }
    let mut entries = tokio::fs::read_dir(dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some(ext) {
            let meta = entry.metadata().await?;
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let lines = if ext == "jsonl" { count_lines(&path).await } else { None };
            infos.push(FileInfo {
                name: name.clone(),
                path: path.to_string_lossy().to_string(),
                size_mb: meta.len() as f64 / (1024.0 * 1024.0),
                lines,
            });
        }
    }
    infos.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(infos)
}

async fn list_files_no_lines(dir: &Path, ext: &str) -> anyhow::Result<Vec<FileInfo>> {
    let mut infos = Vec::new();
    if !dir.exists() {
        anyhow::bail!("directory does not exist: {}", dir.display());
    }
    let mut entries = tokio::fs::read_dir(dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some(ext) {
            let meta = entry.metadata().await?;
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            infos.push(FileInfo {
                name: name.clone(),
                path: path.to_string_lossy().to_string(),
                size_mb: meta.len() as f64 / (1024.0 * 1024.0),
                lines: None,
            });
        }
    }
    infos.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(infos)
}

async fn list_files_prefix(dir: &Path, prefix: &str) -> anyhow::Result<Vec<FileInfo>> {
    let mut infos = Vec::new();
    if !dir.exists() {
        anyhow::bail!("directory does not exist: {}", dir.display());
    }
    let mut entries = tokio::fs::read_dir(dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        if name.starts_with(prefix) && name.ends_with(".bin") {
            let meta = entry.metadata().await?;
            infos.push(FileInfo {
                name: name.clone(),
                path: path.to_string_lossy().to_string(),
                size_mb: meta.len() as f64 / (1024.0 * 1024.0),
                lines: None,
            });
        }
    }
    infos.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(infos)
}

#[tauri::command]
pub async fn delete_file(path: String) -> Result<(), String> {
    tokio::fs::remove_file(&path).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_all_gguf(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let dir = weights_dir(&state.project_dir);
    delete_by_ext(&dir, "gguf").await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_all_cache(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let dir = data_dir(&state.project_dir);
    delete_by_prefix(&dir, "sequences_cache_").await.map_err(|e| e.to_string())
}

async fn delete_by_ext(dir: &Path, ext: &str) -> anyhow::Result<()> {
    if let Ok(entries) = tokio::fs::read_dir(dir).await {
        let mut entries = entries;
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some(ext) {
                let _ = tokio::fs::remove_file(path).await;
            }
        }
    }
    Ok(())
}

async fn delete_by_prefix(dir: &Path, prefix: &str) -> anyhow::Result<()> {
    if let Ok(entries) = tokio::fs::read_dir(dir).await {
        let mut entries = entries;
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            if name.starts_with(prefix) {
                let _ = tokio::fs::remove_file(path).await;
            }
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn wait_for_file(path: String, timeout_ms: u64) -> Result<bool, String> {
    let path = Path::new(&path);
    let deadline = Duration::from_millis(timeout_ms);
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        if path.exists() {
            return Ok(true);
        }
        sleep(Duration::from_millis(200)).await;
    }
    Ok(false)
}
