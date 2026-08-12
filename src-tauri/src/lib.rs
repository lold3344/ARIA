mod commands;
mod files;
mod process;

use process::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new().expect("failed to create app state"))
        .plugin(tauri_plugin_log::Builder::default().build())
        .invoke_handler(tauri::generate_handler![
            commands::start_training,
            commands::stop_training,
            commands::start_inference,
            commands::export_gguf,
            commands::send_chat_message,
            files::list_gguf,
            files::list_datasets,
            files::list_caches,
            files::list_binaries,
            files::delete_file,
            files::delete_all_gguf,
            files::delete_all_cache,
            files::wait_for_file,
            files::get_project_dir,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
