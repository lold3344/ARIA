use anyhow::Result;
use sqlx::sqlite::SqlitePool;
use serde_json::json;
use crate::storage::{DialogEntry, CipherEnvelope};

pub async fn init_db(database_url: &str) -> Result<SqlitePool> {
    let pool = SqlitePool::connect(database_url).await?;
    
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS dialogs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            nonce_b64 TEXT NOT NULL,
            ciphertext_b64 TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            reward REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        "#,
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        "#,
    )
    .execute(&pool)
    .await?;

    Ok(pool)
}

pub async fn create_session(pool: &SqlitePool, session_id: &str) -> Result<()> {
    sqlx::query("INSERT INTO sessions (id) VALUES (?)")
        .bind(session_id)
        .execute(pool)
        .await?;
    
    Ok(())
}

pub async fn insert_dialog(
    pool: &SqlitePool,
    entry: &DialogEntry,
    session_id: &str,
) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO dialogs (id, session_id, role, nonce_b64, ciphertext_b64, timestamp, reward)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&entry.id)
    .bind(session_id)
    .bind(&entry.role)
    .bind(&entry.encrypted_message.nonce_b64)
    .bind(&entry.encrypted_message.ciphertext_b64)
    .bind(entry.timestamp)
    .bind(entry.reward)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn get_session_dialogs(
    pool: &SqlitePool,
    session_id: &str,
) -> Result<Vec<DialogEntry>> {
    let rows = sqlx::query_as::<_, (String, String, String, String, i64, Option<f32>)>(
        r#"
        SELECT id, role, nonce_b64, ciphertext_b64, timestamp, reward
        FROM dialogs
        WHERE session_id = ?
        ORDER BY timestamp ASC
        "#,
    )
    .bind(session_id)
    .fetch_all(pool)
    .await?;

    let entries = rows.into_iter()
        .map(|(id, role, nonce_b64, ciphertext_b64, timestamp, reward)| {
            DialogEntry {
                id,
                role,
                encrypted_message: CipherEnvelope {
                    nonce_b64,
                    ciphertext_b64,
                },
                timestamp,
                reward,
            }
        })
        .collect();

    Ok(entries)
}

pub async fn update_dialog_reward(
    pool: &SqlitePool,
    dialog_id: &str,
    reward: f32,
) -> Result<()> {
    sqlx::query("UPDATE dialogs SET reward = ? WHERE id = ?")
        .bind(reward)
        .bind(dialog_id)
        .execute(pool)
        .await?;

    Ok(())
}

pub async fn get_all_sessions(pool: &SqlitePool) -> Result<Vec<String>> {
    let rows = sqlx::query_as::<_, (String,)>("SELECT DISTINCT session_id FROM dialogs")
        .fetch_all(pool)
        .await?;

    Ok(rows.iter().map(|(id,)| id.clone()).collect())
}
