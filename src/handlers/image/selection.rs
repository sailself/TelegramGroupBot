use super::*;

pub(super) const IMAGE_RESOLUTION_OPTIONS: [&str; 3] = ["2K", "4K", "1K"];
pub(super) const IMAGE_ASPECT_RATIO_OPTIONS: [&str; 14] = [
    "4:3", "3:4", "16:9", "9:16", "1:1", "21:9", "3:2", "2:3", "5:4", "4:5", "4:1", "1:4", "8:1",
    "1:8",
];
pub const IMAGE_RESOLUTION_CALLBACK_PREFIX: &str = "image_res:";
pub const IMAGE_ASPECT_RATIO_CALLBACK_PREFIX: &str = "image_aspect:";
pub const IMAGE_MODEL_CALLBACK_PREFIX: &str = "image_model:";
pub const IMAGE_CODEX_SIZE_CALLBACK_PREFIX: &str = "image_codex_size:";
pub(super) const IMAGE_DEFAULT_RESOLUTION: &str = "2K";
pub(super) const IMAGE_ASPECT_RATIO_AUTO_CALLBACK: &str = "auto";
pub(super) fn build_resolution_keyboard(request_key: &str) -> InlineKeyboardMarkup {
    let buttons = IMAGE_RESOLUTION_OPTIONS
        .iter()
        .map(|res| {
            InlineKeyboardButton::callback(
                res.to_string(),
                format!(
                    "{}{}|{}",
                    IMAGE_RESOLUTION_CALLBACK_PREFIX, request_key, res
                ),
            )
        })
        .collect::<Vec<_>>();

    let rows = buttons
        .chunks(3)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    InlineKeyboardMarkup::new(rows)
}

pub(super) fn image_model_callback_data(request_key: &str, model: ImageGenerationModel) -> String {
    let token = match model {
        ImageGenerationModel::Gemini => "gemini",
        ImageGenerationModel::CodexGptImage2 => "codex",
    };
    format!("{}{}|{}", IMAGE_MODEL_CALLBACK_PREFIX, request_key, token)
}

pub(super) fn parse_image_generation_model(value: &str) -> Option<ImageGenerationModel> {
    match value.trim() {
        "gemini" => Some(ImageGenerationModel::Gemini),
        "codex" => Some(ImageGenerationModel::CodexGptImage2),
        _ => None,
    }
}

pub(super) fn parse_default_image_generation_model(value: &str) -> Option<ImageGenerationModel> {
    match value.trim().to_lowercase().as_str() {
        "gemini" => Some(ImageGenerationModel::Gemini),
        "codex" | "openai-codex" | "openai-codex:selected" => {
            Some(ImageGenerationModel::CodexGptImage2)
        }
        _ => None,
    }
}

pub(super) fn resolve_default_image_generation_model(
    default_model: &str,
    gemini_available: bool,
    codex_available: bool,
) -> std::result::Result<ImageGenerationModel, String> {
    let Some(model) = parse_default_image_generation_model(default_model) else {
        return Err(format!(
            "Default image model {} is not supported. Set DEFAULT_IMAGE_MODEL to gemini or codex.",
            default_model.trim()
        ));
    };

    if model == ImageGenerationModel::Gemini && !gemini_available {
        if codex_available {
            return Ok(ImageGenerationModel::CodexGptImage2);
        }
        return Err(
            "No image model is configured. Enable Gemini or complete Codex setup with /codexlogin."
                .to_string(),
        );
    }

    if model == ImageGenerationModel::CodexGptImage2 && !codex_available {
        return Err(format!(
            "Default image model {} is unavailable. Complete Codex setup with /codexlogin or set DEFAULT_IMAGE_MODEL=gemini.",
            default_model.trim()
        ));
    }

    Ok(model)
}

pub(super) fn build_image_model_keyboard(
    request_key: &str,
    include_gemini: bool,
    include_codex: bool,
    default_model: ImageGenerationModel,
) -> InlineKeyboardMarkup {
    let mut buttons = Vec::new();
    if include_gemini {
        buttons.push(InlineKeyboardButton::callback(
            CONFIG.gemini.image_model.clone(),
            image_model_callback_data(request_key, ImageGenerationModel::Gemini),
        ));
    }

    if include_codex {
        buttons.push(InlineKeyboardButton::callback(
            crate::llm::codex_image::codex_image_display_model(),
            image_model_callback_data(request_key, ImageGenerationModel::CodexGptImage2),
        ));
    }

    if let Some(default_index) = buttons.iter().position(|button| match default_model {
        ImageGenerationModel::Gemini => {
            matches!(
                &button.kind,
                teloxide::types::InlineKeyboardButtonKind::CallbackData(data)
                    if data == &image_model_callback_data(request_key, ImageGenerationModel::Gemini)
            )
        }
        ImageGenerationModel::CodexGptImage2 => {
            matches!(
                &button.kind,
                teloxide::types::InlineKeyboardButtonKind::CallbackData(data)
                    if data == &image_model_callback_data(request_key, ImageGenerationModel::CodexGptImage2)
            )
        }
    }) {
        let default_button = buttons.remove(default_index);
        buttons.insert(0, default_button);
    }

    InlineKeyboardMarkup::new(vec![buttons])
}

pub(super) fn build_aspect_ratio_keyboard(request_key: &str) -> InlineKeyboardMarkup {
    let mut buttons = vec![InlineKeyboardButton::callback(
        "Auto",
        format!(
            "{}{}|{}",
            IMAGE_ASPECT_RATIO_CALLBACK_PREFIX, request_key, IMAGE_ASPECT_RATIO_AUTO_CALLBACK
        ),
    )];
    buttons.extend(
        IMAGE_ASPECT_RATIO_OPTIONS
            .iter()
            .map(|aspect| {
                InlineKeyboardButton::callback(
                    aspect.to_string(),
                    format!(
                        "{}{}|{}",
                        IMAGE_ASPECT_RATIO_CALLBACK_PREFIX, request_key, aspect
                    ),
                )
            })
            .collect::<Vec<_>>(),
    );

    let rows = buttons
        .chunks(3)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    InlineKeyboardMarkup::new(rows)
}

pub(super) fn build_codex_size_keyboard(request_key: &str) -> InlineKeyboardMarkup {
    let buttons = crate::llm::codex_image::CODEX_IMAGE_SUPPORTED_SIZES
        .iter()
        .map(|size| {
            InlineKeyboardButton::callback(
                size.to_string(),
                format!(
                    "{}{}|{}",
                    IMAGE_CODEX_SIZE_CALLBACK_PREFIX, request_key, size
                ),
            )
        })
        .collect::<Vec<_>>();

    let rows = buttons
        .chunks(3)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    InlineKeyboardMarkup::new(rows)
}

pub(super) fn resolve_image_request_settings(
    request: &PendingImageRequest,
    resolution: Option<&str>,
    aspect_ratio: Option<&str>,
) -> (String, Option<String>) {
    let final_resolution = resolution
        .or(request.resolution.as_deref())
        .unwrap_or(IMAGE_DEFAULT_RESOLUTION)
        .to_string();
    let final_aspect = aspect_ratio
        .filter(|value| *value != IMAGE_ASPECT_RATIO_AUTO_CALLBACK && !value.trim().is_empty())
        .or(request.aspect_ratio.as_deref())
        .filter(|value| *value != IMAGE_ASPECT_RATIO_AUTO_CALLBACK && !value.trim().is_empty())
        .map(|value| value.to_string());

    (final_resolution, final_aspect)
}

pub(super) fn valid_selection(
    request: &PendingImageRequest,
    user: i64,
    chat: i64,
    message: i32,
    stage: ImageSelectionStage,
) -> bool {
    request.user_id == user
        && request.chat_id == chat
        && request.selection_message_id == i64::from(message)
        && request.stage == stage
        && request.deadline > tokio::time::Instant::now()
}

pub async fn image_selection_callback(
    bot: Bot,
    state: AppState,
    query: CallbackQuery,
) -> Result<()> {
    let _ = bot.answer_callback_query(query.id.clone()).await;
    let Some(data) = &query.data else {
        return Ok(());
    };
    let Some(selection_message) = query.message.as_ref() else {
        return Ok(());
    };
    let query_user_id = i64::try_from(query.from.id.0).unwrap_or_default();
    let Some((_, payload)) = data.split_once(':') else {
        return Ok(());
    };
    let Some((key, value)) = payload.split_once('|') else {
        return Ok(());
    };
    if value.contains('|') {
        return Ok(());
    }
    let gate = {
        let entry = state.pending_image_requests.entry(key);
        entry.get().map(|r| r.selection_gate.clone())
    };
    let Some(gate) = gate else {
        return Ok(());
    };
    let selection_guard = gate.lock().await;

    if data.starts_with(IMAGE_MODEL_CALLBACK_PREFIX) {
        let payload = data.trim_start_matches(IMAGE_MODEL_CALLBACK_PREFIX);
        let mut parts = payload.split('|');
        let request_key = parts.next().unwrap_or("");
        let model_token = parts.next().unwrap_or("");
        let Some(model) = parse_image_generation_model(model_token) else {
            return Ok(());
        };
        if model == ImageGenerationModel::Gemini && !CONFIG.gemini_api_available() {
            return Ok(());
        }
        if model == ImageGenerationModel::CodexGptImage2
            && !crate::llm::codex_image::codex_image_available()
        {
            return Ok(());
        }

        let (next_command, ready_request) = {
            let mut entry = state.pending_image_requests.entry(request_key);
            let Some(request) = entry.get_mut() else {
                return Ok(());
            };
            if !valid_selection(
                request,
                query_user_id,
                selection_message.chat().id.0,
                selection_message.id().0,
                ImageSelectionStage::Model,
            ) {
                return Ok(());
            }
            request.model = Some(model);
            request.stage = match model {
                ImageGenerationModel::Gemini => ImageSelectionStage::Resolution,
                ImageGenerationModel::CodexGptImage2 => ImageSelectionStage::CodexSize,
            };
            request.deadline = tokio::time::Instant::now()
                + Duration::from_secs(CONFIG.limits.model_selection_timeout);
            let command = request.command;
            // /img has nothing more to ask: take the request now so the
            // timeout task can no longer race this selection.
            let ready_request = match command {
                PendingImageCommand::Img => entry.take(),
                PendingImageCommand::Image => None,
            };
            (command, ready_request)
        };

        match (next_command, model) {
            (PendingImageCommand::Img, _) => {
                if let Some(request) = ready_request {
                    drop(selection_guard);
                    finalize_image_request(&bot, &state, request, None, None).await?;
                }
            }
            (PendingImageCommand::Image, ImageGenerationModel::Gemini) => {
                if let Some(message) = &query.message {
                    bot.edit_message_text(
                        message.chat().id,
                        message.id(),
                        format!(
                            "Choose a resolution for {} (default: {}).",
                            CONFIG.gemini.image_model, IMAGE_DEFAULT_RESOLUTION
                        ),
                    )
                    .reply_markup(build_resolution_keyboard(request_key))
                    .await?;
                }
            }
            (PendingImageCommand::Image, ImageGenerationModel::CodexGptImage2) => {
                if let Some(message) = &query.message {
                    bot.edit_message_text(
                        message.chat().id,
                        message.id(),
                        format!(
                            "Choose a size for {}, or wait to let the model decide.",
                            crate::llm::codex_image::codex_image_display_model()
                        ),
                    )
                    .reply_markup(build_codex_size_keyboard(request_key))
                    .await?;
                }
            }
        }
        return Ok(());
    }

    if data.starts_with(IMAGE_CODEX_SIZE_CALLBACK_PREFIX) {
        let payload = data.trim_start_matches(IMAGE_CODEX_SIZE_CALLBACK_PREFIX);
        let mut parts = payload.split('|');
        let request_key = parts.next().unwrap_or("");
        let size = parts.next().unwrap_or("");
        if !crate::llm::codex_image::is_supported_codex_image_size(size) {
            return Ok(());
        }

        let ready_request = {
            let mut entry = state.pending_image_requests.entry(request_key);
            match entry.get_mut() {
                Some(request)
                    if !valid_selection(
                        request,
                        query_user_id,
                        selection_message.chat().id.0,
                        selection_message.id().0,
                        ImageSelectionStage::CodexSize,
                    ) =>
                {
                    return Ok(())
                }
                Some(request) => {
                    request.model = Some(ImageGenerationModel::CodexGptImage2);
                    request.codex_size = Some(size.to_string());
                }
                None => {}
            }
            entry.take()
        };
        if let Some(request) = ready_request {
            drop(selection_guard);
            finalize_image_request(&bot, &state, request, None, None).await?;
        }
        return Ok(());
    }

    if data.starts_with(IMAGE_RESOLUTION_CALLBACK_PREFIX) {
        let payload = data.trim_start_matches(IMAGE_RESOLUTION_CALLBACK_PREFIX);
        let mut parts = payload.split('|');
        let request_key = parts.next().unwrap_or("");
        let resolution = parts.next().unwrap_or("");
        if !IMAGE_RESOLUTION_OPTIONS.contains(&resolution) {
            return Ok(());
        }

        if let Some(request) = state.pending_image_requests.entry(request_key).get_mut() {
            if !valid_selection(
                request,
                query_user_id,
                selection_message.chat().id.0,
                selection_message.id().0,
                ImageSelectionStage::Resolution,
            ) {
                return Ok(());
            }
            request.resolution = Some(resolution.to_string());
            request.stage = ImageSelectionStage::AspectRatio;
            request.deadline = tokio::time::Instant::now()
                + Duration::from_secs(CONFIG.limits.model_selection_timeout);
        } else {
            return Ok(());
        }

        if let Some(message) = &query.message {
            bot.edit_message_text(
                message.chat().id,
                message.id(),
                format!(
                    "Resolution set to {}. Choose an aspect ratio, or Auto to let the model decide.",
                    resolution
                ),
            )
            .reply_markup(build_aspect_ratio_keyboard(request_key))
            .await?;
        }
        return Ok(());
    }

    if data.starts_with(IMAGE_ASPECT_RATIO_CALLBACK_PREFIX) {
        let payload = data.trim_start_matches(IMAGE_ASPECT_RATIO_CALLBACK_PREFIX);
        let mut parts = payload.split('|');
        let request_key = parts.next().unwrap_or("");
        let aspect = parts.next().unwrap_or("");
        if aspect != IMAGE_ASPECT_RATIO_AUTO_CALLBACK
            && !IMAGE_ASPECT_RATIO_OPTIONS.contains(&aspect)
        {
            return Ok(());
        }

        let ready_request = {
            let mut entry = state.pending_image_requests.entry(request_key);
            match entry.get_mut() {
                Some(request)
                    if !valid_selection(
                        request,
                        query_user_id,
                        selection_message.chat().id.0,
                        selection_message.id().0,
                        ImageSelectionStage::AspectRatio,
                    ) =>
                {
                    return Ok(())
                }
                Some(request) => {
                    request.aspect_ratio = if aspect == IMAGE_ASPECT_RATIO_AUTO_CALLBACK {
                        None
                    } else {
                        Some(aspect.to_string())
                    };
                }
                None => {}
            }
            entry.take()
        };

        let selected_aspect = if aspect == IMAGE_ASPECT_RATIO_AUTO_CALLBACK {
            None
        } else {
            Some(aspect)
        };
        if let Some(request) = ready_request {
            drop(selection_guard);
            finalize_image_request(&bot, &state, request, None, selected_aspect).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{response_with_status, ExpectedRequest, TestServer};
    fn request(stage: ImageSelectionStage) -> PendingImageRequest {
        PendingImageRequest {
            selection_gate: Default::default(),
            stage,
            deadline: tokio::time::Instant::now() + Duration::from_secs(10),
            user_id: 1,
            chat_id: 2,
            message_id: 3,
            command: PendingImageCommand::Image,
            prompt: "prompt".into(),
            image_urls: vec![],
            telegraph_contents: vec![],
            selection_message_id: 4,
            llm_invocation_id: None,
            model: Some(ImageGenerationModel::Gemini),
            codex_size: None,
            resolution: Some("4K".into()),
            aspect_ratio: None,
        }
    }
    fn callback(data: &str, user: i64, message: i32) -> CallbackQuery {
        serde_json::from_value(serde_json::json!({"id":"callback","from":{"id":user,"is_bot":false,"first_name":"fixture"},"message":{"message_id":message,"date":1,"chat":{"id":2,"type":"private"},"text":"picker"},"chat_instance":"test","data":data})).unwrap()
    }
    fn acknowledgement() -> ExpectedRequest {
        ExpectedRequest::new(
            "POST",
            "/bot123:test/AnswerCallbackQuery",
            response_with_status(200, br#"{"ok":true,"result":true}"#.to_vec()),
        )
    }

    #[tokio::test]
    async fn callbacks_do_not_redraw_unauthorized_expired_duplicate_or_missing_menus() {
        let mut expected = (0..6).map(|_| acknowledgement()).collect::<Vec<_>>();
        expected.push(ExpectedRequest::new("POST","/bot123:test/EditMessageText",response_with_status(200,br#"{"ok":true,"result":{"message_id":4,"date":1,"chat":{"id":2,"type":"private"}}}"#.to_vec())));
        expected.extend((0..2).map(|_| acknowledgement()));
        let server = TestServer::new(expected);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let db = crate::db::test_support::init_test_db("image-callbacks").await;
        let state = AppState::new(db.clone(), 123, "test".into());
        state
            .pending_image_requests
            .insert("req".into(), request(ImageSelectionStage::Resolution));
        for (data, user, message) in [
            ("image_aspect:req|1:1", 1, 4),
            ("image_res:req|4K", 9, 4),
            ("image_res:req|4K", 1, 9),
            ("image_res:req|invalid", 1, 4),
        ] {
            image_selection_callback(bot.clone(), state.clone(), callback(data, user, message))
                .await
                .unwrap();
        }
        assert_eq!(
            state
                .pending_image_requests
                .entry("req")
                .get()
                .unwrap()
                .stage,
            ImageSelectionStage::Resolution
        );
        state
            .pending_image_requests
            .entry("req")
            .get_mut()
            .unwrap()
            .deadline = tokio::time::Instant::now();
        image_selection_callback(
            bot.clone(),
            state.clone(),
            callback("image_res:req|4K", 1, 4),
        )
        .await
        .unwrap();
        state
            .pending_image_requests
            .entry("req")
            .get_mut()
            .unwrap()
            .deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        image_selection_callback(
            bot.clone(),
            state.clone(),
            callback("image_res:req|4K", 1, 4),
        )
        .await
        .unwrap();
        assert_eq!(
            state
                .pending_image_requests
                .entry("req")
                .get()
                .unwrap()
                .stage,
            ImageSelectionStage::AspectRatio
        );
        image_selection_callback(
            bot.clone(),
            state.clone(),
            callback("image_res:req|1K", 1, 4),
        )
        .await
        .unwrap();
        let saved = state.pending_image_requests.entry("req").take().unwrap();
        assert_eq!(saved.resolution.as_deref(), Some("4K"));
        image_selection_callback(bot, state, callback("image_res:req|4K", 1, 4))
            .await
            .unwrap();
        server.join().unwrap();
        db.shutdown().await;
    }

    #[tokio::test(start_paused = true)]
    async fn every_picker_stage_times_out_once_with_saved_choices() {
        for stage in [
            ImageSelectionStage::Model,
            ImageSelectionStage::Resolution,
            ImageSelectionStage::AspectRatio,
            ImageSelectionStage::CodexSize,
        ] {
            let pending = crate::state::PendingRequests::new();
            let (send, receive) = tokio::sync::oneshot::channel();
            pending.insert_with_deadline(
                "req".into(),
                request(stage),
                |r| r.deadline,
                move |r| async move {
                    send.send(r).unwrap();
                },
            );
            tokio::time::advance(Duration::from_secs(11)).await;
            let expired = receive.await.unwrap();
            assert_eq!(expired.stage, stage);
            assert_eq!(expired.resolution.as_deref(), Some("4K"));
            assert_eq!(pending.count(), 0);
            assert!(pending.entry("req").take().is_none());
        }
    }
}
