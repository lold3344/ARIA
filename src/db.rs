use anyhow::Result;
use std::fs;
use std::path::Path;
use serde_json::json;
use crate::storage::DialogEntry;

pub fn init_db(db_path: &str) -> Result<()> {
    let path = Path::new(db_path);
    if !path.exists() {
        fs::write(db_path, "[]")?;
    }
    Ok(())
}

pub fn insert_dialog(
    db_path: &str,
    entry: &DialogEntry,
    session_id: &str,
) -> Result<()> {
    let content = fs::read_to_string(db_path)?;
    let mut dialogs: Vec<serde_json::Value> = serde_json::from_str(&content).unwrap_or_default();
    
    let mut dialog_json = serde_json::to_value(entry)?;
    dialog_json["session_id"] = json!(session_id);
    
    dialogs.push(dialog_json);
    fs::write(db_path, serde_json::to_string_pretty(&dialogs)?)?;
    
    Ok(())
}

pub fn update_dialog_reward(
    db_path: &str,
    dialog_id: &str,
    reward: f32,
) -> Result<()> {
    let content = fs::read_to_string(db_path)?;
    let mut dialogs: Vec<serde_json::Value> = serde_json::from_str(&content)?;
    
    for dialog in &mut dialogs {
        if dialog["id"].as_str() == Some(dialog_id) {
            dialog["reward"] = json!(reward);
            break;
        }
    }
    
    fs::write(db_path, serde_json::to_string_pretty(&dialogs)?)?;
    Ok(())
}