use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SkillFrontmatter {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub triggers: Vec<String>,
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    #[serde(default)]
    pub risk_level: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

const fn default_enabled() -> bool {
    true
}

#[derive(Debug, Clone)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub triggers: Vec<String>,
    pub allowed_tools: Vec<String>,
    pub risk_level: String,
    pub version: Option<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct SkillDoc {
    pub meta: SkillMeta,
    pub body: String,
    pub source_path: Option<PathBuf>,
    pub always_active: bool,
}

#[derive(Debug, Clone)]
pub struct ActiveSkillSet {
    pub selected: Vec<SkillDoc>,
    pub selected_names: Vec<String>,
    pub allowed_tools: Vec<String>,
}
