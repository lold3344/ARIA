use serde::Deserialize;

use crate::process::{kill_process, spawn_process};
use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct TrainRequest {
    pub mode: String, // "fresh" | "debug" | "sft" | "tiny"
    pub datasets: String, // "all" or "1 3 7"
    pub max_seqs: Option<String>,
    pub lr: Option<String>,
    pub warmup: Option<String>,
    pub epochs: Option<String>,
    pub use_existing_cache: bool,
}

#[derive(Debug, Deserialize)]
pub struct ExportRequest {
    pub source: String,
    pub target: String,
}

#[tauri::command]
pub async fn start_training(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    req: TrainRequest,
) -> Result<String, String> {
    let id = format!("train-{}", uuid::Uuid::new_v4());

    let bin = match req.mode.as_str() {
        "debug" => "train_debug",
        "sft" => "sft_train",
        "tiny" => "tiny_train",
        _ => "train_fresh",
    };

    let mut envs = Vec::new();
    if let Some(v) = req.lr { envs.push(("ARIA_LR".to_string(), v)); }
    if let Some(v) = req.warmup { envs.push(("ARIA_WARMUP".to_string(), v)); }
    if let Some(v) = req.epochs { envs.push(("ARIA_EPOCHS".to_string(), v)); }
    if let Some(v) = req.max_seqs { envs.push(("ARIA_MAX_SEQS".to_string(), v)); }

    // Write dataset selection to a temp input file.
    let input_path = std::env::temp_dir().join(format!("aria_gui_input_{}.txt", id));
    let selection = if req.datasets.trim().is_empty() { "all" } else { &req.datasets };
    tokio::fs::write(&input_path, selection).await.map_err(|e| e.to_string())?;

    let args = vec!["data base".to_string()];

    // If use_existing_cache is true, we don't delete cache; if false, delete before start.
    if !req.use_existing_cache {
        let _ = crate::files::delete_all_cache(state.clone()).await;
    }

    // Spawn directly with stdin from the input file so the dataset prompt is answered.
    let id = spawn_process(
        state.processes.clone(),
        state.project_dir.clone(),
        id.clone(),
        bin,
        &args.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        envs,
        Some(input_path),
        Some(app),
    )
    .await
    .map_err(|e| e.to_string())?;

    Ok(id)
}

#[tauri::command]
pub async fn stop_training(state: tauri::State<'_, AppState>, id: String) -> Result<(), String> {
    kill_process(state.processes.clone(), &id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn start_inference(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    mode: String,
    weights: String,
    prompt: String,
) -> Result<String, String> {
    let id = format!("inference-{}", uuid::Uuid::new_v4());

    let bin = match mode.as_str() {
        "greedy" => "greedy_test",
        "sample" => "sample_test",
        "test_suite" => "test_suite",
        "debug_logits" => "debug_logits",
        _ => "inference",
    };

    let args = match bin {
        "inference" => vec![weights.clone(), prompt.clone()],
        "debug_logits" => vec![weights.clone()],
        _ => vec![weights.clone()],
    };

    let id_ret = spawn_process(
        state.processes.clone(),
        state.project_dir.clone(),
        id.clone(),
        bin,
        &args.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        vec![],
        None,
        Some(app),
    )
    .await
    .map_err(|e| e.to_string())?;

    Ok(id_ret)
}

#[tauri::command]
pub async fn export_gguf(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    req: ExportRequest,
) -> Result<String, String> {
    let id = format!("export-{}", uuid::Uuid::new_v4());
    let source = if std::path::Path::new(&req.source).is_absolute() {
        req.source.clone()
    } else {
        state.project_dir.join("aria json").join(&req.source).to_string_lossy().to_string()
    };
    let target = if std::path::Path::new(&req.target).is_absolute() {
        req.target.clone()
    } else {
        state.project_dir.join("aria json").join(&req.target).to_string_lossy().to_string()
    };

    let id_ret = spawn_process(
        state.processes.clone(),
        state.project_dir.clone(),
        id.clone(),
        "export_gguf",
        &[&source, &target],
        vec![],
        None,
        Some(app),
    )
    .await
    .map_err(|e| e.to_string())?;

    Ok(id_ret)
}

#[tauri::command]
pub async fn send_chat_message(
    state: tauri::State<'_, AppState>,
    id: String,
    message: String,
) -> Result<(), String> {
    // For interactive inference we would need a persistent stdin handle.
    // Not implemented in this minimal version.
    let _ = (state, id, message);
    Err("Interactive chat stdin not yet implemented".to_string())
}
