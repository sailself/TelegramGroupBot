//! Best-effort, bounded retention of generated sprite sheets for troubleshooting.
use std::{
    fs::{self, Metadata, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, Ordering},
        Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_SHEETS: usize = 16;
const MAX_TOTAL_BYTES: u64 = 128 * 1024 * 1024;
const MAX_SHEET_BYTES: usize = 32 * 1024 * 1024;
static LOCK: Mutex<()> = Mutex::new(());
static SEQUENCE: AtomicU32 = AtomicU32::new(0);

fn is_reparse(meta: &Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        meta.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        meta.file_type().is_symlink()
    }
}

fn ensure_directory(path: &Path) -> io::Result<()> {
    match fs::create_dir(path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_dir() || is_reparse(&meta) {
        return Err(io::Error::other(
            "GIF artifact directory is not a regular directory",
        ));
    }
    Ok(())
}

fn owned_name(name: &str) -> bool {
    let Some(inner) = name.strip_prefix("gif-sheet-").and_then(|s| {
        [".png", ".jpg", ".webp", ".bin"]
            .iter()
            .find_map(|suffix| s.strip_suffix(suffix))
    }) else {
        return false;
    };
    let parts: Vec<_> = inner.split('-').collect();
    parts.len() == 3
        && parts[0].len() == 32
        && parts[1].len() == 8
        && parts[2].len() == 8
        && parts
            .iter()
            .all(|part| part.bytes().all(|c| c.is_ascii_hexdigit()))
}

fn save_to(root: &Path, bytes: &[u8], max_files: usize, max_bytes: u64) -> io::Result<PathBuf> {
    if bytes.len() > MAX_SHEET_BYTES || bytes.len() as u64 > max_bytes || max_files == 0 {
        return Err(io::Error::other(
            "GIF sheet exceeds artifact storage limits",
        ));
    }
    let _guard = LOCK
        .lock()
        .map_err(|_| io::Error::other("GIF artifact lock poisoned"))?;
    ensure_directory(root)?;
    let media = root.join("media");
    ensure_directory(&media)?;
    let dir = media.join("gif");
    ensure_directory(&dir)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let extension = match image::guess_format(bytes) {
        Ok(image::ImageFormat::Png) => "png",
        Ok(image::ImageFormat::Jpeg) => "jpg",
        Ok(image::ImageFormat::WebP) => "webp",
        _ => "bin",
    };
    let name = format!(
        "gif-sheet-{timestamp:032x}-{:08x}-{sequence:08x}.{extension}",
        std::process::id()
    );
    let path = dir.join(name);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    if let Err(error) = file.write_all(bytes) {
        drop(file);
        let _ = fs::remove_file(&path);
        return Err(error);
    }
    drop(file);
    let mut entries = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !owned_name(&name) {
            continue;
        }
        let meta = fs::symlink_metadata(entry.path())?;
        if meta.is_file() && !is_reparse(&meta) {
            entries.push((name, entry.path(), meta.len()));
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut total: u64 = entries.iter().map(|entry| entry.2).sum();
    let mut remaining = entries.len();
    for (_, candidate, size) in entries {
        if candidate == path {
            continue;
        }
        if remaining <= max_files && total <= max_bytes {
            break;
        }
        // Check parent directories and the entry again before deletion.
        ensure_directory(root)?;
        ensure_directory(&media)?;
        ensure_directory(&dir)?;
        let meta = fs::symlink_metadata(&candidate)?;
        if meta.is_file() && !is_reparse(&meta) {
            fs::remove_file(candidate)?;
            remaining -= 1;
            total = total.saturating_sub(size);
        }
    }
    Ok(path)
}

pub(super) fn save_sheet(bytes: &[u8]) -> io::Result<()> {
    let path = save_to(Path::new("data"), bytes, MAX_SHEETS, MAX_TOTAL_BYTES)?;
    tracing::info!(path = %path.display(), bytes = bytes.len(), "Saved GIF sprite sheet");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retains_only_bounded_owned_regular_files() {
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("gif-artifact-test-{}", std::process::id()));
        for _ in 0..4 {
            save_to(&root, b"sheet", 2, 100).unwrap();
        }
        let dir = root.join("media/gif");
        let owned_directory = dir.join(format!("gif-sheet-{:032x}-{:08x}-{:08x}.png", 0, 0, 0));
        fs::create_dir(&owned_directory).unwrap();
        let unrelated = dir.join("other.bin");
        fs::write(&unrelated, b"keep").unwrap();
        save_to(&root, b"data", 2, 100).unwrap();
        assert_eq!(
            fs::read_dir(&dir)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| owned_name(&e.file_name().to_string_lossy())
                    && e.file_type().unwrap().is_file())
                .count(),
            2
        );
        assert_eq!(fs::read(&unrelated).unwrap(), b"keep");
        assert!(owned_directory.is_dir());
        save_to(&root, b"12345", 4, 8).unwrap();
        let retained: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| {
                owned_name(&e.file_name().to_string_lossy()) && e.file_type().unwrap().is_file()
            })
            .collect();
        assert_eq!(retained.len(), 1);
        assert_eq!(fs::metadata(retained[0].path()).unwrap().len(), 5);
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgba8(1, 1)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let path = save_to(&root, &png.into_inner(), 4, 4096).unwrap();
        assert_eq!(path.extension().unwrap(), "png");
        assert!(owned_name(&path.file_name().unwrap().to_string_lossy()));
        assert!(save_to(&root, b"too long", 2, 3).is_err());
        let blocked = root.join("file");
        fs::write(&blocked, b"not directory").unwrap();
        assert!(save_to(&blocked, b"sheet", 2, 100).is_err());
        let canonical_root = root.canonicalize().unwrap();
        let canonical_target = std::env::current_dir()
            .unwrap()
            .join("target")
            .canonicalize()
            .unwrap();
        assert!(canonical_root.starts_with(canonical_target));
        fs::remove_dir_all(canonical_root).unwrap();
    }
}
