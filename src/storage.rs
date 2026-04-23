use chacha20poly1305::{
    aead::{Aead, KeyInit},
    Key, XChaCha20Poly1305, XNonce,
};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CipherEnvelope {
    pub nonce_b64: String,
    pub ciphertext_b64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogEntry {
    pub id: String,
    pub role: String,
    pub encrypted_message: CipherEnvelope,
    pub timestamp: i64,
    pub reward: Option<f32>,
}

pub struct EncryptionManager {
    key: [u8; 32],
}

impl EncryptionManager {
    pub fn new(master_key_hex: &str) -> Result<Self> {
        let decoded = hex::decode(master_key_hex)
            .context("Invalid hex key")?;
        
        if decoded.len() != 32 {
            anyhow::bail!("Key must be 32 bytes");
        }

        let mut key = [0u8; 32];
        key.copy_from_slice(&decoded);

        Ok(EncryptionManager { key })
    }

    pub fn generate_key() -> String {
        let mut key = [0u8; 32];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut key);
        hex::encode(key)
    }

    pub fn encrypt(&self, plaintext: &str) -> Result<CipherEnvelope> {
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.key));
        let mut nonce = [0u8; 24];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut nonce);

        let ciphertext = cipher
            .encrypt(XNonce::from_slice(&nonce), plaintext.as_bytes())
            .map_err(|_| anyhow::anyhow!("Encryption failed"))?;

        Ok(CipherEnvelope {
            nonce_b64: B64.encode(nonce),
            ciphertext_b64: B64.encode(ciphertext),
        })
    }

    pub fn decrypt(&self, envelope: &CipherEnvelope) -> Result<String> {
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.key));
        
        let nonce_raw = B64.decode(&envelope.nonce_b64)
            .context("Invalid nonce")?;
        let ciphertext_raw = B64.decode(&envelope.ciphertext_b64)
            .context("Invalid ciphertext")?;

        if nonce_raw.len() != 24 {
            anyhow::bail!("Nonce must be 24 bytes");
        }

        let plaintext = cipher
            .decrypt(XNonce::from_slice(&nonce_raw), ciphertext_raw.as_ref())
            .map_err(|_| anyhow::anyhow!("Decryption failed"))?;

        String::from_utf8(plaintext)
            .context("Invalid UTF-8 in decrypted text")
    }
}

pub fn get_current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
