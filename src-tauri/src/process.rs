use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use chrono::Local;
use tauri::Emitter;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::process::{Child, Command};
use tokio::sync::{broadcast, Mutex};

pub type ProcessMap = Arc<Mutex<HashMap<String, (Child, broadcast::Sender<String>)>>>;

pub struct AppState {
    pub processes: ProcessMap,
    pub project_dir: PathBuf,
}

impl AppState {
    pub fn new() -> anyhow::Result<Self> {
        let project_dir = Self::find_project_dir()?;
        Ok(Self {
            processes: Arc::new(Mutex::new(HashMap::new())),
            project_dir,
        })
    }

    fn find_project_dir() -> anyhow::Result<PathBuf> {
        // Try CARGO_MANIFEST_DIR first (set during build, points to src-tauri folder).
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let manifest_dir = PathBuf::from(manifest_dir);
            // Go up from src-tauri to the project root.
            if let Some(root) = manifest_dir.parent() {
                if root.join("data base").exists() || root.join("Cargo.toml").exists() {
                    return Ok(root.to_path_buf());
                }
            }
        }

        // Otherwise walk up from the executable until we find both data base and aria json.
        let mut dir = std::env::current_exe()?.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        loop {
            if dir.join("data base").exists() && dir.join("aria json").exists() {
                return Ok(dir);
            }
            if dir.join("Cargo.toml").exists() {
                // Only accept the root Cargo.toml (the one for the aria package).
                if let Ok(content) = std::fs::read_to_string(dir.join("Cargo.toml")) {
                    if content.contains("name = \"aria\"") {
                        return Ok(dir);
                    }
                }
            }
            if !dir.pop() {
                break;
            }
        }

        anyhow::bail!("Could not locate ARIA project directory (need data base + aria json folders, or root Cargo.toml with name = \"aria\")")
    }
}

pub fn exe_dir(project_dir: &Path) -> PathBuf {
    project_dir.join("target").join("release")
}

pub fn data_dir(project_dir: &Path) -> PathBuf {
    project_dir.join("data base")
}

pub fn weights_dir(project_dir: &Path) -> PathBuf {
    project_dir.join("aria json")
}

/// Spawn a CLI binary and stream stdout/stderr lines to a broadcast channel.
pub async fn spawn_process(
    processes: ProcessMap,
    project_dir: PathBuf,
    id: String,
    name: &str,
    args: &[&str],
    envs: Vec<(String, String)>,
    stdin_input: Option<PathBuf>,
    app: Option<tauri::AppHandle>,
) -> anyhow::Result<String> {
    let exe = if cfg!(windows) && !name.ends_with(".exe") {
        exe_dir(&project_dir).join(format!("{}.exe", name))
    } else {
        exe_dir(&project_dir).join(name)
    };
    if !exe.exists() {
        anyhow::bail!("Executable not found: {}", exe.display());
    }

    let (tx, _rx) = broadcast::channel::<String>(4096);
    let tx2 = tx.clone();

    let stdin = if let Some(path) = &stdin_input {
        let file = std::fs::File::open(path)?;
        Stdio::from(file)
    } else {
        Stdio::null()
    };

    let mut cmd = Command::new(&exe);
    cmd.args(args)
        .current_dir(&project_dir)
        .stdin(stdin)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .envs(envs.into_iter().map(|(k, v)| (k, v)).collect::<HashMap<_, _>>());

    // On Windows, disable creation of a console window for the child.
    #[cfg(windows)]
    {
        #[allow(unused_imports)]
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    let mut child = cmd.spawn()?;

    let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("stdout missing"))?;
    let stderr = child.stderr.take().ok_or_else(|| anyhow::anyhow!("stderr missing"))?;

    // stdout reader
    let tx_stdout = tx.clone();
    tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let line = format!("[{}] {}", Local::now().format("%H:%M:%S"), line);
            let _ = tx_stdout.send(line);
        }
    });

    // stderr reader
    let tx_err = tx2.clone();
    tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let line = format!("[{}] [ERR] {}", Local::now().format("%H:%M:%S"), line);
            let _ = tx_err.send(line);
        }
    });

    // Forward broadcast events to the frontend if an AppHandle is provided.
    if let Some(app) = app {
        let id2 = id.clone();
        let mut rx = tx.subscribe();
        tokio::spawn(async move {
            while let Ok(line) = rx.recv().await {
                let _ = app.emit("process-log", serde_json::json!({"id": &id2, "line": line }));
            }
        });
    }

    processes.lock().await.insert(id.clone(), (child, tx));

    Ok(id)
}

/// Kill a running process.
pub async fn kill_process(processes: ProcessMap, id: &str) -> anyhow::Result<()> {
    let mut map = processes.lock().await;
    if let Some((mut child, _)) = map.remove(id) {
        let _ = child.kill().await;
    }
    Ok(())
}
