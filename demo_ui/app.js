const form = document.getElementById("qa-form");
const sourcesOutput = document.getElementById("sourcesOutput");
const statusBadge = document.getElementById("statusBadge");
const memoryText = document.getElementById("memoryText");
const retrievalText = document.getElementById("retrievalText");
const conversationOutput = document.getElementById("conversationOutput");
const submitButton = document.getElementById("submitButton");
const clearChatButton = document.getElementById("clearChat");
const ingestStatus = document.getElementById("ingestStatus");
const modeNewButton = document.getElementById("modeNewButton");
const modeExistingButton = document.getElementById("modeExistingButton");
const ingestNewPanel = document.getElementById("ingestNewPanel");
const ingestExistingPanel = document.getElementById("ingestExistingPanel");
const uploadButtonNew = document.getElementById("uploadButtonNew");
const uploadButtonExisting = document.getElementById("uploadButtonExisting");
const pickFilesButtonNew = document.getElementById("pickFilesButtonNew");
const pickFilesButtonExisting = document.getElementById("pickFilesButtonExisting");
const filePickerTextNew = document.getElementById("filePickerTextNew");
const filePickerTextExisting = document.getElementById("filePickerTextExisting");

const conversationState = {
  summary: "",
  recentTurns: [],
  currentTopic: "",
};

let courseOptions = [];

const appDefaults = {
  target: "both",
  topK: 8,
  contextK: 6,
  candidateK: 4,
  embeddingModel: "text-embedding-3-small",
  retrievalMethod: "dense_rerank",
  rrfK: 60,
  faissWeight: 1,
  bm25Weight: 1,
  densePoolMultiplier: 4,
  denseRerankDenseWeight: 0.65,
  denseRerankBm25Weight: 0.35,
};

const fields = {
  courseId: document.getElementById("courseId"),
  query: document.getElementById("query"),
  generationModel: document.getElementById("generationModel"),
  memoryWindow: document.getElementById("memoryWindow"),
  rrfK: document.getElementById("rrfK"),
  faissWeight: document.getElementById("faissWeight"),
  bm25Weight: document.getElementById("bm25Weight"),
  ingestCourseId: document.getElementById("ingestCourseId"),
  ingestCourseName: document.getElementById("ingestCourseName"),
  ingestFilesNew: document.getElementById("ingestFilesNew"),
  ingestFilesExisting: document.getElementById("ingestFilesExisting"),
  ingestMode: document.getElementById("ingestMode"),
  ingestExistingCourse: document.getElementById("ingestExistingCourse"),
};

let activeIngestJobId = null;
let ingestPollTimer = null;

function setStatus(text) {
  statusBadge.textContent = text;
}

function setIngestStatus(text, className = "empty-state") {
  ingestStatus.className = `ingest-status ${className}`;
  ingestStatus.textContent = text;
}

function updateFilePickerText(fileInput, textNode) {
  const files = Array.from(fileInput.files || []);
  if (!files.length) {
    textNode.textContent = "No files selected";
    return;
  }
  if (files.length === 1) {
    textNode.textContent = files[0].name;
    return;
  }
  textNode.textContent = `${files.length} files selected`;
}

function ingestSelectedCourse() {
  const selectedId = fields.ingestExistingCourse.value;
  return courseOptions.find((course) => course.id === selectedId) || null;
}

function setIngestMode(mode) {
  fields.ingestMode.value = mode;
  const isExisting = mode === "existing";
  ingestNewPanel.hidden = isExisting;
  ingestExistingPanel.hidden = !isExisting;
  modeNewButton.classList.toggle("active", !isExisting);
  modeExistingButton.classList.toggle("active", isExisting);
  modeNewButton.setAttribute("aria-pressed", String(!isExisting));
  modeExistingButton.setAttribute("aria-pressed", String(isExisting));
}

function normalizeInlineMath(text) {
  return String(text)
    .replace(/\\\((.*?)\\\)/g, "$1")
    .replace(/\\\[(.*?)\\\]/g, "$1")
    .replace(/\\_/g, "_")
    .replace(/\bH_0\b/g, "H0")
    .replace(/\bH₀\b/g, "H0");
}

function softenOverBold(text) {
  return String(text).replace(/(\*\*|__)(.+?)\1/g, (match, marker, content) => {
    const plain = content.trim();
    const wordCount = plain.split(/\s+/).filter(Boolean).length;
    if (wordCount > 5 || plain.length > 42) {
      return plain;
    }
    return `${marker}${plain}${marker}`;
  });
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatInline(text) {
  return escapeHtml(softenOverBold(normalizeInlineMath(text)))
    .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/_(.+?)_/g, "<em>$1</em>")
    .replace(/\[S(\d+)\]/g, '<span class="citation">[S$1]</span>');
}

function normalizeAnswerMarkdown(text) {
  const normalized = softenOverBold(normalizeInlineMath(text))
    .replace(/\r\n/g, "\n")
    .replace(/^\s*#{2,6}\s*$/gm, "")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/([^\n])\s+(Short answer\b[: -]*)/gi, "$1\n\n$2")
    .replace(/([^\n])\s+(Examples\b[: -]*)/gi, "$1\n\n$2")
    .replace(/([^\n])\s+(Example or Intuition\b[: -]*)/gi, "$1\n\n$2")
    .replace(/([^\n])\s+(Examples or Intuition\b[: -]*)/gi, "$1\n\n$2")
    .replace(/([^\n])\s+(Intuition\b[: -]*)/gi, "$1\n\n$2")
    .replace(/([^\n])\s+(Sources\b[: -]*)/gi, "$1\n\n$2")
    .replace(/^(Short answer|Examples|Example or Intuition|Examples or Intuition|Intuition|Sources)\b[: -]*/gim, "### $1\n\n")
    .replace(/\s+-\s+(?=[A-Z[])/g, "\n- ")
    .replace(/\n- (?=\[S\d+\])/g, "\n- ")
    .replace(/([^\n])\s+([-*]\s+\[S\d+\])/g, "$1\n$2")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  const rawLines = normalized.split("\n");
  const mergedLines = [];
  let inSources = false;

  for (const rawLine of rawLines) {
    const line = rawLine.trim();

    if (!line) {
      mergedLines.push("");
      continue;
    }

    const isHeading = /^#{1,6}\s+/.test(line);
    const isListItem = /^[-*]\s+/.test(line) || /^\d+\.\s+/.test(line);

    if (isHeading) {
      inSources = /^#{1,6}\s+Sources$/i.test(line);
      mergedLines.push(line);
      continue;
    }

    let nextLine = line;
    if (inSources && /^\[S\d+\]/.test(nextLine)) {
      nextLine = `- ${nextLine}`;
    }

    const previous = mergedLines[mergedLines.length - 1] || "";
    const previousIsHeading = /^#{1,6}\s+/.test(previous);
    const previousIsList = /^[-*]\s+/.test(previous) || /^\d+\.\s+/.test(previous);
    const canMergeIntoPrevious =
      previous &&
      !previousIsHeading &&
      !previousIsList &&
      !isListItem &&
      !/[:.!?]$/.test(previous);

    if (canMergeIntoPrevious) {
      mergedLines[mergedLines.length - 1] = `${previous} ${nextLine}`;
    } else {
      mergedLines.push(nextLine);
    }
  }

  return mergedLines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function renderList(lines, ordered) {
  const tag = ordered ? "ol" : "ul";
  const pattern = ordered ? /^\d+\.\s+/ : /^[-*]\s+/;
  const items = lines
    .map((line) => `<li>${formatInline(line.replace(pattern, ""))}</li>`)
    .join("");
  return `<${tag}>${items}</${tag}>`;
}

function renderParagraph(lines) {
  const paragraph = lines.map((line) => formatInline(line)).join(" ");
  return `<p>${paragraph}</p>`;
}

function renderBlock(block) {
  const trimmed = block.trim();
  if (!trimmed) {
    return "";
  }

  const lines = trimmed.split("\n").map((line) => line.trim()).filter(Boolean);
  if (!lines.length) {
    return "";
  }

  if (lines.every((line) => /^```/.test(line))) {
    return "";
  }

  if (lines.length === 1 && /^#{1,6}\s+/.test(lines[0])) {
    const headingMatch = lines[0].match(/^(#{1,6})\s+(.*)$/);
    const level = Math.min((headingMatch?.[1]?.length || 1) + 2, 6);
    const content = headingMatch?.[2] || lines[0];
    return `<h${level}>${formatInline(content)}</h${level}>`;
  }

  if (lines.length > 1 && /^#{1,6}\s+/.test(lines[0])) {
    const headingMatch = lines[0].match(/^(#{1,6})\s+(.*)$/);
    const level = Math.min((headingMatch?.[1]?.length || 1) + 2, 6);
    const content = headingMatch?.[2] || lines[0];
    const rest = lines.slice(1);

    if (rest.every((line) => /^[-*]\s+/.test(line))) {
      return `<h${level}>${formatInline(content)}</h${level}>${renderList(rest, false)}`;
    }

    if (rest.every((line) => /^\d+\.\s+/.test(line))) {
      return `<h${level}>${formatInline(content)}</h${level}>${renderList(rest, true)}`;
    }

    return `<h${level}>${formatInline(content)}</h${level}>${renderParagraph(rest)}`;
  }

  if (lines.length === 1 && /^(-{3,}|\*{3,})$/.test(lines[0])) {
    return "<hr>";
  }

  if (lines.every((line) => /^>\s?/.test(line))) {
    const quote = lines.map((line) => formatInline(line.replace(/^>\s?/, ""))).join("<br>");
    return `<blockquote>${quote}</blockquote>`;
  }

  if (lines.every((line) => /^[-*]\s+/.test(line))) {
    return renderList(lines, false);
  }

  if (lines.every((line) => /^\d+\.\s+/.test(line))) {
    return renderList(lines, true);
  }

  if (lines.length === 1 && /^[A-Za-z][A-Za-z\s]{0,40}$/.test(lines[0])) {
    return `<h4>${formatInline(lines[0])}</h4>`;
  }

  if (lines.length === 1 && /:$/.test(lines[0])) {
    return `<h4>${formatInline(lines[0].replace(/:$/, ""))}</h4>`;
  }

  if (lines.length > 1 && /:$/.test(lines[0])) {
    const heading = `<h4>${formatInline(lines[0].replace(/:$/, ""))}</h4>`;
    const rest = lines.slice(1);

    if (rest.every((line) => /^[-*]\s+/.test(line))) {
      return `${heading}${renderList(rest, false)}`;
    }

    if (rest.every((line) => /^\d+\.\s+/.test(line))) {
      return `${heading}${renderList(rest, true)}`;
    }

    return `${heading}${renderParagraph(rest)}`;
  }

  return renderParagraph(lines);
}

function renderAnswer(text) {
  const normalized = normalizeAnswerMarkdown(text);
  if (!normalized) {
    return "<p>No answer returned.</p>";
  }

  const htmlBlocks = normalized
    .split(/\n\s*\n/)
    .map((block) => renderBlock(block))
    .filter(Boolean);

  return htmlBlocks.join("");
}

function setSourcesEmpty(text, className = "empty-state") {
  sourcesOutput.innerHTML = "";
  sourcesOutput.className = `sources-list ${className}`;
  sourcesOutput.textContent = text;
}

function setConversationEmpty(text, className = "empty-state") {
  conversationOutput.innerHTML = "";
  conversationOutput.className = `conversation-output ${className}`;
  conversationOutput.textContent = text;
}

function renderSources(sources) {
  if (!sources.length) {
    setSourcesEmpty("No context sources were returned.");
    return;
  }

  sourcesOutput.className = "sources-list";
  sourcesOutput.innerHTML = sources
    .map(
      (source) => {
        const sourceMeta =
          source.level && source.chunkType && source.level !== source.chunkType
            ? `${escapeHtml(source.level)} / ${escapeHtml(source.chunkType || "unknown")}`
            : escapeHtml(source.level || source.chunkType || "unknown");

        return `
        <article class="source-item">
          <div class="source-topline">
            <div class="source-label">
              <span class="pill">${escapeHtml(source.label)}</span>
              <span>${escapeHtml(source.location)}</span>
            </div>
            <div class="source-meta">${sourceMeta}</div>
          </div>
          <div class="source-preview">${escapeHtml(source.preview || "No preview available.")}</div>
        </article>
      `
      }
    )
    .join("");
}

function renderConversation() {
  const turns = conversationState.recentTurns || [];
  const summary = (conversationState.summary || "").trim();

  if (!turns.length && !summary) {
    memoryText.textContent = "No conversation memory yet";
    setConversationEmpty("Start a conversation to see recent turns and rolling memory.");
    return;
  }

  memoryText.textContent = summary
    ? "Older turns are compressed into a running summary"
    : "Currently showing only recent turns";

  const parts = [];
  if (summary) {
    parts.push(`
      <article class="memory-summary">
        <div class="memory-label">Running Summary</div>
        <p>${escapeHtml(summary)}</p>
      </article>
    `);
  }

  turns.forEach((turn) => {
    const role = turn.role === "user" ? "You" : "Assistant";
    const roleClass = turn.role === "user" ? "user-turn" : "assistant-turn";
    const content =
      turn.role === "assistant"
        ? `<div class="rich-answer">${renderAnswer(turn.content)}</div>`
        : `<p>${escapeHtml(turn.content)}</p>`;
    parts.push(`
      <article class="chat-turn ${roleClass}">
        <div class="chat-role">${role}</div>
        <div class="chat-content">${content}</div>
      </article>
    `);
  });

  conversationOutput.className = "conversation-output";
  conversationOutput.innerHTML = parts.join("");
}

function resetConversationState() {
  conversationState.summary = "";
  conversationState.recentTurns = [];
  conversationState.currentTopic = "";
  fields.query.value = "";
  retrievalText.textContent = "Context snippets used for generation";
  setSourcesEmpty("No source results yet.");
  renderConversation();
}

function applyDefaults(config) {
  const defaults = config.defaults;
  courseOptions = Array.isArray(config.courseOptions) ? config.courseOptions : [];
  fields.generationModel.value = defaults.generationModel;
  fields.memoryWindow.value = defaults.memoryWindow;
  fields.rrfK.value = defaults.rrfK;
  fields.faissWeight.value = defaults.faissWeight;
  fields.bm25Weight.value = defaults.bm25Weight;
  appDefaults.target = defaults.target;
  appDefaults.topK = defaults.topK;
  appDefaults.contextK = defaults.contextK;
  appDefaults.candidateK = defaults.candidateK;
  appDefaults.embeddingModel = defaults.embeddingModel;
  appDefaults.retrievalMethod = defaults.retrievalMethod;
  appDefaults.rrfK = defaults.rrfK;
  appDefaults.faissWeight = defaults.faissWeight;
  appDefaults.bm25Weight = defaults.bm25Weight;
  appDefaults.densePoolMultiplier = defaults.densePoolMultiplier;
  appDefaults.denseRerankDenseWeight = defaults.denseRerankDenseWeight;
  appDefaults.denseRerankBm25Weight = defaults.denseRerankBm25Weight;

  fields.courseId.innerHTML = courseOptions.length
    ? courseOptions
        .map((course) => `<option value="${escapeHtml(course.id)}">${escapeHtml(course.name)}</option>`)
        .join("")
    : `<option value="">No indexed courses yet</option>`;

  fields.ingestExistingCourse.innerHTML = courseOptions.length
    ? courseOptions
        .map((course) => `<option value="${escapeHtml(course.id)}">${escapeHtml(course.id)}</option>`)
        .join("")
    : `<option value="">No indexed courses yet</option>`;

  if (config.courses.includes(defaults.courseId)) {
    fields.courseId.value = defaults.courseId;
  } else if (config.courses.length) {
    fields.courseId.value = config.courses[0];
  }

  if (courseOptions.length) {
    fields.ingestExistingCourse.value = courseOptions[0].id;
  }

  setIngestMode(fields.ingestMode.value || "new");
}

async function loadConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) {
    throw new Error("Failed to load config.");
  }
  const config = await response.json();
  applyDefaults(config);
  return config;
}

function buildIngestPayload() {
  const mode = fields.ingestMode.value;
  if (mode === "existing") {
    const selectedCourse = ingestSelectedCourse();
    return {
      courseId: selectedCourse?.id || "",
      courseName: selectedCourse?.name || "",
      files: Array.from(fields.ingestFilesExisting.files || []),
    };
  }

  return {
    courseId: fields.ingestCourseId.value.trim(),
    courseName: fields.ingestCourseName.value.trim(),
    files: Array.from(fields.ingestFilesNew.files || []),
  };
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      const base64 = result.includes(",") ? result.split(",")[1] : result;
      resolve(base64);
    };
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

function stopIngestPolling() {
  if (ingestPollTimer) {
    clearTimeout(ingestPollTimer);
    ingestPollTimer = null;
  }
}

async function pollIngestJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to fetch ingestion status.");
    }

    const step = data.current_step && data.current_step !== "done" ? ` Current step: ${data.current_step}.` : "";
    const files = Array.isArray(data.uploaded_files) && data.uploaded_files.length
      ? ` Files: ${data.uploaded_files.join(", ")}.`
      : "";

    if (data.status === "completed") {
      setIngestStatus(`Index ready for ${data.course_id}.${files}`, "success-state");
      activeIngestJobId = null;
      stopIngestPolling();
      const config = await loadConfig();
      if (config.courses.includes(data.course_id)) {
        fields.courseId.value = data.course_id;
      }
      fields.ingestCourseId.value = data.course_id;
      fields.ingestCourseName.value = data.course_name || fields.ingestCourseName.value;
      fields.ingestFilesNew.value = "";
      fields.ingestFilesExisting.value = "";
      updateFilePickerText(fields.ingestFilesNew, filePickerTextNew);
      updateFilePickerText(fields.ingestFilesExisting, filePickerTextExisting);
      return;
    }

    if (data.status === "failed") {
      const errorText = data.error ? ` Error: ${data.error}` : "";
      setIngestStatus(`Ingestion failed for ${data.course_id}.${step}${errorText}`, "error-state");
      activeIngestJobId = null;
      stopIngestPolling();
      return;
    }

    setIngestStatus(`Building index for ${data.course_id}.${step}${files}`, "loading-state");
    ingestPollTimer = setTimeout(() => pollIngestJob(jobId), 2500);
  } catch (error) {
    setIngestStatus("Could not refresh ingestion status. Check the server logs.", "error-state");
    activeIngestJobId = null;
    stopIngestPolling();
  }
}

async function startIngestion() {
  const { courseId, courseName, files } = buildIngestPayload();

  if (!courseId) {
    setIngestStatus("Choose or enter a course before uploading.", "error-state");
    return;
  }
  if (!courseName) {
    setIngestStatus("Enter a course name before uploading.", "error-state");
    return;
  }
  if (!files.length) {
    setIngestStatus("Choose at least one PDF file.", "error-state");
    return;
  }

  uploadButtonNew.disabled = true;
  uploadButtonExisting.disabled = true;
  setIngestStatus("Preparing files for upload...", "loading-state");

  try {
    const serializedFiles = await Promise.all(
      files.map(async (file) => ({
        name: file.name,
        contentBase64: await fileToBase64(file),
      }))
    );

    const response = await fetch("/api/course-ingest", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        courseId,
        courseName,
        files: serializedFiles,
      }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Upload failed.");
    }

    activeIngestJobId = data.jobId;
    setIngestStatus(`Upload complete. Starting pipeline for ${data.courseId}.`, "loading-state");
    stopIngestPolling();
    ingestPollTimer = setTimeout(() => pollIngestJob(data.jobId), 1200);
  } catch (error) {
    setIngestStatus(error.message || "Upload failed.", "error-state");
  } finally {
    uploadButtonNew.disabled = false;
    uploadButtonExisting.disabled = false;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fields.courseId.value) {
    setStatus("No indexed course");
    setSourcesEmpty("Upload and index a course before asking questions.", "error-state");
    return;
  }

  const payload = {
    courseId: fields.courseId.value,
    query: fields.query.value.trim(),
    target: appDefaults.target,
    topK: Number(appDefaults.topK),
    contextK: Number(appDefaults.contextK),
    candidateK: Number(appDefaults.candidateK),
    generationModel: fields.generationModel.value.trim(),
    embeddingModel: appDefaults.embeddingModel,
    retrievalMethod: appDefaults.retrievalMethod,
    rrfK: Number(fields.rrfK.value),
    faissWeight: Number(fields.faissWeight.value),
    bm25Weight: Number(fields.bm25Weight.value),
    densePoolMultiplier: Number(appDefaults.densePoolMultiplier),
    denseRerankDenseWeight: Number(appDefaults.denseRerankDenseWeight),
    denseRerankBm25Weight: Number(appDefaults.denseRerankBm25Weight),
    memoryWindow: Number(fields.memoryWindow.value),
    conversationSummary: conversationState.summary,
    recentTurns: conversationState.recentTurns,
    currentTopic: conversationState.currentTopic,
  };

  submitButton.disabled = true;
  setStatus("Generating...");
  retrievalText.textContent = "Retrieving course content and preparing context";
  setSourcesEmpty("Preparing source results...");

  try {
    const response = await fetch("/api/answer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Request failed.");
    }

    setStatus("Completed");
    retrievalText.textContent = data.retrievalQuery
      ? `Retrieved ${data.retrievedResultsCount} candidates and used ${data.usedSourcesCount} of them as sources with query: ${data.retrievalQuery}`
      : "Context snippets used for generation";
    renderSources(data.sources || []);
    conversationState.summary = data.conversationSummary || "";
    conversationState.recentTurns = data.recentTurns || [];
    conversationState.currentTopic = data.currentTopic || "";
    renderConversation();
    fields.query.value = "";
  } catch (error) {
    setStatus("Error");
    retrievalText.textContent = "Check your index files, API key, or parameter settings";
    setSourcesEmpty("No source results were returned for this request.", "error-state");
  } finally {
    submitButton.disabled = false;
  }
});

clearChatButton.addEventListener("click", resetConversationState);
uploadButtonNew.addEventListener("click", startIngestion);
uploadButtonExisting.addEventListener("click", startIngestion);
pickFilesButtonNew.addEventListener("click", () => fields.ingestFilesNew.click());
pickFilesButtonExisting.addEventListener("click", () => fields.ingestFilesExisting.click());
fields.ingestFilesNew.addEventListener("change", () => updateFilePickerText(fields.ingestFilesNew, filePickerTextNew));
fields.ingestFilesExisting.addEventListener("change", () => updateFilePickerText(fields.ingestFilesExisting, filePickerTextExisting));
modeNewButton.addEventListener("click", () => setIngestMode("new"));
modeExistingButton.addEventListener("click", () => setIngestMode("existing"));

loadConfig()
  .then(() => {
    resetConversationState();
    setIngestStatus("No ingestion job running.");
    updateFilePickerText(fields.ingestFilesNew, filePickerTextNew);
    updateFilePickerText(fields.ingestFilesExisting, filePickerTextExisting);
  })
  .catch(() => {
    setStatus("Initialization failed");
    setSourcesEmpty("The config endpoint is unavailable.", "error-state");
    setConversationEmpty("The page could not initialize conversation memory.", "error-state");
    setIngestStatus("The page could not initialize upload controls.", "error-state");
  });
