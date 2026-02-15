use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use tracing::{debug, info, warn};

use crate::skills::types::{SkillDoc, SkillFrontmatter, SkillMeta};

fn parse_frontmatter(raw: &str) -> Result<(SkillFrontmatter, String)> {
    let normalized = raw.replace("\r\n", "\n");
    if !normalized.starts_with("---\n") {
        return Err(anyhow!("Skill file is missing YAML frontmatter"));
    }

    let rest = &normalized[4..];
    let Some(closing_index) = rest.find("\n---\n") else {
        return Err(anyhow!("Skill file has unterminated YAML frontmatter"));
    };

    let yaml_text = &rest[..closing_index];
    let body = rest[(closing_index + 5)..].trim().to_string();
    let frontmatter: SkillFrontmatter = serde_yaml::from_str(yaml_text)
        .map_err(|err| anyhow!("Failed to parse skill frontmatter: {}", err))?;
    Ok((frontmatter, body))
}

fn normalize_skill_meta(frontmatter: SkillFrontmatter) -> Result<SkillMeta> {
    let name = frontmatter.name.trim().to_string();
    if name.is_empty() {
        return Err(anyhow!("Skill name cannot be empty"));
    }
    let description = frontmatter.description.trim().to_string();
    if description.is_empty() {
        return Err(anyhow!("Skill description cannot be empty"));
    }

    let tags = frontmatter
        .tags
        .into_iter()
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let triggers = frontmatter
        .triggers
        .into_iter()
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let allowed_tools = frontmatter
        .allowed_tools
        .into_iter()
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let risk_level = frontmatter
        .risk_level
        .unwrap_or_else(|| "safe_read".to_string())
        .trim()
        .to_lowercase();

    Ok(SkillMeta {
        name,
        description,
        tags,
        triggers,
        allowed_tools,
        risk_level,
        version: frontmatter.version.map(|value| value.trim().to_string()),
        enabled: frontmatter.enabled,
    })
}

fn parse_skill_file(path: &Path) -> Result<SkillDoc> {
    let raw = fs::read_to_string(path)
        .map_err(|err| anyhow!("Failed to read skill file '{}': {}", path.display(), err))?;
    let (frontmatter, body) = parse_frontmatter(&raw)?;
    let meta = normalize_skill_meta(frontmatter)?;
    Ok(SkillDoc {
        meta,
        body,
        source_path: Some(path.to_path_buf()),
        always_active: false,
    })
}

pub fn built_in_core_workspace_skill() -> SkillDoc {
    let body = [
        "When to use:",
        "- Always available as foundational workspace capability.",
        "",
        "Procedure:",
        "1. Use `read_file` before modifying files.",
        "2. Prefer `edit_file` for targeted changes.",
        "3. Use `write_file` for full rewrites or new files.",
        "4. Use `exec` for build/test/check commands scoped to workspace.",
        "",
        "Failure handling:",
        "- If an edit target is ambiguous, request more context before editing.",
        "- If command output is empty or truncated, run narrower commands.",
    ]
    .join("\n");

    SkillDoc {
        meta: SkillMeta {
            name: "core-workspace".to_string(),
            description:
                "Built-in core tools for reading/writing/editing files and executing shell commands."
                    .to_string(),
            tags: vec![
                "filesystem".to_string(),
                "editing".to_string(),
                "shell".to_string(),
            ],
            triggers: vec![
                "read".to_string(),
                "write".to_string(),
                "edit".to_string(),
                "command".to_string(),
                "shell".to_string(),
                "bash".to_string(),
                "powershell".to_string(),
            ],
            allowed_tools: vec![
                "read_file".to_string(),
                "write_file".to_string(),
                "edit_file".to_string(),
                "exec".to_string(),
            ],
            risk_level: "mixed".to_string(),
            version: Some("1".to_string()),
            enabled: true,
        },
        body,
        source_path: None,
        always_active: true,
    }
}

fn sorted_skill_paths(skills_dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = fs::read_dir(skills_dir).map_err(|err| {
        anyhow!(
            "Failed to read skills directory '{}': {}",
            skills_dir.display(),
            err
        )
    })?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|err| anyhow!("Failed to read skills directory entry: {}", err))?;
        let path = entry.path();

        let is_top_level_markdown = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("md"))
            .unwrap_or(false);
        if path.is_file() && is_top_level_markdown {
            paths.push(path);
            continue;
        }

        if path.is_dir() {
            let direct_skill = path.join("SKILL.md");
            if direct_skill.is_file() {
                paths.push(direct_skill);
                continue;
            }

            let nested_entries = fs::read_dir(&path).map_err(|err| {
                anyhow!("Failed to read skill folder '{}': {}", path.display(), err)
            })?;
            for nested in nested_entries {
                let nested =
                    nested.map_err(|err| anyhow!("Failed to read nested skill entry: {}", err))?;
                let nested_path = nested.path();
                let is_skill_file = nested_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.eq_ignore_ascii_case("SKILL.md"))
                    .unwrap_or(false);
                if nested_path.is_file() && is_skill_file {
                    paths.push(nested_path);
                    break;
                }
            }
        }
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

pub fn load_skills(skills_dir: &Path) -> Vec<SkillDoc> {
    let mut docs = Vec::new();
    docs.push(built_in_core_workspace_skill());

    if !skills_dir.exists() {
        debug!(
            "Skills directory '{}' not found; only built-in skills will be used",
            skills_dir.display()
        );
        return docs;
    }

    let paths = match sorted_skill_paths(skills_dir) {
        Ok(value) => value,
        Err(err) => {
            warn!("{}", err);
            return docs;
        }
    };

    for path in paths {
        match parse_skill_file(&path) {
            Ok(doc) => {
                if !doc.meta.enabled {
                    debug!("Skipping disabled skill '{}'", doc.meta.name);
                    continue;
                }
                docs.push(doc);
            }
            Err(err) => warn!("Skipping invalid skill '{}': {}", path.display(), err),
        }
    }

    info!("Loaded {} skill(s)", docs.len());
    docs
}
