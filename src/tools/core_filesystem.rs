use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use tokio::fs;

fn canonical_workspace_root(workspace_root: &Path) -> Result<PathBuf> {
    let root = workspace_root.canonicalize().map_err(|err| {
        anyhow!(
            "Failed to resolve workspace root '{}': {}",
            workspace_root.display(),
            err
        )
    })?;
    Ok(root)
}

fn normalize_candidate_path(workspace_root: &Path, path: &str) -> Result<PathBuf> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("Path is required"));
    }

    let candidate = PathBuf::from(trimmed);
    if candidate.is_absolute() {
        Ok(candidate)
    } else {
        Ok(workspace_root.join(candidate))
    }
}

fn resolve_with_existing_ancestor(candidate: &Path) -> Result<PathBuf> {
    if candidate.exists() {
        return candidate
            .canonicalize()
            .map_err(|err| anyhow!("Failed to resolve path '{}': {}", candidate.display(), err));
    }

    let mut cursor = candidate.to_path_buf();
    let mut missing_suffix: Vec<OsString> = Vec::new();

    while !cursor.exists() {
        let Some(file_name) = cursor.file_name() else {
            return Err(anyhow!(
                "Unable to resolve path '{}': no existing ancestor found",
                candidate.display()
            ));
        };
        missing_suffix.push(file_name.to_os_string());
        let Some(parent) = cursor.parent() else {
            return Err(anyhow!(
                "Unable to resolve path '{}': no parent directory found",
                candidate.display()
            ));
        };
        cursor = parent.to_path_buf();
    }

    let mut resolved = cursor
        .canonicalize()
        .map_err(|err| anyhow!("Failed to resolve path '{}': {}", cursor.display(), err))?;
    for part in missing_suffix.iter().rev() {
        resolved.push(part);
    }
    Ok(resolved)
}

fn ensure_inside_workspace(resolved: &Path, workspace_root: &Path) -> Result<()> {
    let canonical_root = canonical_workspace_root(workspace_root)?;
    if resolved.starts_with(&canonical_root) {
        return Ok(());
    }

    Err(anyhow!(
        "Path '{}' is outside allowed workspace '{}'",
        resolved.display(),
        canonical_root.display()
    ))
}

pub fn resolve_workspace_path(workspace_root: &Path, raw_path: &str) -> Result<PathBuf> {
    let candidate = normalize_candidate_path(workspace_root, raw_path)?;
    let resolved = resolve_with_existing_ancestor(&candidate)?;
    ensure_inside_workspace(&resolved, workspace_root)?;
    Ok(resolved)
}

pub async fn read_file(workspace_root: &Path, path: &str) -> Result<String> {
    let resolved = resolve_workspace_path(workspace_root, path)?;
    if !resolved.exists() {
        return Err(anyhow!("File not found: {}", path));
    }
    if !resolved.is_file() {
        return Err(anyhow!("Not a file: {}", path));
    }

    let content = fs::read_to_string(&resolved)
        .await
        .map_err(|err| anyhow!("Failed to read file '{}': {}", path, err))?;
    Ok(content)
}

pub async fn write_file(workspace_root: &Path, path: &str, content: &str) -> Result<String> {
    let resolved = resolve_workspace_path(workspace_root, path)?;
    let parent = resolved
        .parent()
        .ok_or_else(|| anyhow!("Cannot determine parent directory for '{}'", path))?;
    fs::create_dir_all(parent)
        .await
        .map_err(|err| anyhow!("Failed to create parent directory for '{}': {}", path, err))?;

    let file_name = resolved
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| "temp_file".to_string());
    let temp_suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_path = resolved.with_file_name(format!("{file_name}.tmp.{temp_suffix}"));

    fs::write(&temp_path, content)
        .await
        .map_err(|err| anyhow!("Failed writing temp file for '{}': {}", path, err))?;
    fs::rename(&temp_path, &resolved)
        .await
        .map_err(|err| anyhow!("Failed replacing '{}': {}", path, err))?;

    Ok(format!(
        "Successfully wrote {} bytes to {}",
        content.len(),
        resolved.display()
    ))
}

pub async fn edit_file(
    workspace_root: &Path,
    path: &str,
    old_text: &str,
    new_text: &str,
) -> Result<String> {
    if old_text.is_empty() {
        return Err(anyhow!("old_text cannot be empty"));
    }

    let resolved = resolve_workspace_path(workspace_root, path)?;
    if !resolved.exists() {
        return Err(anyhow!("File not found: {}", path));
    }
    if !resolved.is_file() {
        return Err(anyhow!("Not a file: {}", path));
    }

    let content = fs::read_to_string(&resolved)
        .await
        .map_err(|err| anyhow!("Failed to read file '{}': {}", path, err))?;
    let occurrences = content.match_indices(old_text).count();
    if occurrences == 0 {
        return Err(anyhow!(
            "old_text not found in '{}'. Provide the exact text to replace.",
            path
        ));
    }
    if occurrences > 1 {
        return Err(anyhow!(
            "old_text appears {} times in '{}'. Provide more unique context.",
            occurrences,
            path
        ));
    }

    let updated = content.replacen(old_text, new_text, 1);
    fs::write(&resolved, updated)
        .await
        .map_err(|err| anyhow!("Failed writing updated file '{}': {}", path, err))?;
    Ok(format!("Successfully edited {}", resolved.display()))
}
