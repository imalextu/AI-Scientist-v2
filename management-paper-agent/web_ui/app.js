const configInput = document.getElementById("configInput");
const topicInput = document.getElementById("topicInput");
const titleInput = document.getElementById("titleInput");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const historyDirInput = document.getElementById("historyDirInput");
const cacheSelect = document.getElementById("cacheSelect");
const refreshCacheBtn = document.getElementById("refreshCacheBtn");
const loadCacheBtn = document.getElementById("loadCacheBtn");
const resumeStageSelect = document.getElementById("resumeStageSelect");
const resumeTargetSelect = document.getElementById("resumeTargetSelect");
const resumeBtn = document.getElementById("resumeBtn");
const statusLine = document.getElementById("statusLine");
const pathLine = document.getElementById("pathLine");
const logOutput = document.getElementById("logOutput");
const literatureOutput = document.getElementById("literatureOutput");
const reviewOutput =
  document.getElementById("reviewOutput") ||
  document.getElementById("researchOutput");
const ideaOutput = document.getElementById("ideaOutput");
const outlineOutput = document.getElementById("outlineOutput");
const paperOutput = document.getElementById("paperOutput");
const copyButtons = Array.from(document.querySelectorAll(".copy-btn"));
const cacheButtons = Array.from(document.querySelectorAll(".cache-btn"));

const stageSequence = ["literature", "idea", "review", "outline", "paper"];
const stageOutputMap = {
  literature: literatureOutput,
  review: reviewOutput,
  idea: ideaOutput,
  outline: outlineOutput,
  paper: paperOutput,
};
const stageLabelMap = {
  literature: "00 文献检索 JSON",
  review: "00 文献综述 Markdown",
  idea: "01 选题设计 JSON",
  outline: "02 论文大纲 JSON",
  paper: "03 论文正文 Markdown",
};
const copyTargetLabelMap = {
  literatureOutput: stageLabelMap.literature,
  reviewOutput: stageLabelMap.review,
  researchOutput: stageLabelMap.review,
  ideaOutput: stageLabelMap.idea,
  outlineOutput: stageLabelMap.outline,
  paperOutput: stageLabelMap.paper,
};

let currentSource = null;
let streamClosedByClient = false;
let currentJobId = null;
let cacheManifestMap = new Map();
let activeCacheId = "";

function nowTag() {
  return new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

function logLine(message) {
  logOutput.textContent += `[${nowTag()}] ${message}\n`;
  logOutput.scrollTop = logOutput.scrollHeight;
}

function setStatus(message) {
  statusLine.textContent = `状态：${message}`;
}

function setPath(pathText) {
  pathLine.textContent = `输出目录：${pathText || "-"}`;
}

function setRunning(running) {
  startBtn.disabled = running;
  stopBtn.disabled = !running;
  historyDirInput.disabled = running;
  refreshCacheBtn.disabled = running;
  if (running) {
    cacheSelect.disabled = true;
    loadCacheBtn.disabled = true;
    resumeStageSelect.disabled = true;
    resumeTargetSelect.disabled = true;
    resumeBtn.disabled = true;
  } else {
    const hasCache = cacheManifestMap.size > 0;
    cacheSelect.disabled = !hasCache;
    loadCacheBtn.disabled = !hasCache;
    resumeStageSelect.disabled = !hasCache;
    resumeTargetSelect.disabled = !hasCache;
    resumeBtn.disabled = !hasCache;
  }
  if (!running) {
    stopBtn.textContent = "停止任务";
  }
}

function clearOutputs(options = {}) {
  const clearLog = options.clearLog !== false;
  if (clearLog) {
    logOutput.textContent = "";
  }
  stageSequence.forEach((stage) => {
    replaceStageText(stage, "");
  });
  setPath("-");
}

function parseEventData(evt) {
  try {
    return JSON.parse(evt.data || "{}");
  } catch (_error) {
    return {};
  }
}

function safePrettyJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (_error) {
    return String(value || "");
  }
}

function inferRunDirFromFilePath(filePath) {
  const raw = String(filePath || "").trim();
  if (!raw) {
    return "";
  }
  const normalized = raw.replace(/\\/g, "/");
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length <= 1) {
    return "";
  }
  parts.pop();
  return normalized.startsWith("/") ? `/${parts.join("/")}` : parts.join("/");
}

function normalizeStage(stage) {
  const value = String(stage || "").trim().toLowerCase();
  if (Object.prototype.hasOwnProperty.call(stageOutputMap, value)) {
    return value;
  }
  return "";
}

function resolveStageTarget(stage) {
  if (!stage) {
    return null;
  }
  const normalized = normalizeStage(stage);
  if (normalized) {
    return stageOutputMap[normalized];
  }
  return null;
}

function appendStageText(stage, text) {
  const target = resolveStageTarget(stage);
  if (!target || !text) {
    return;
  }
  target.textContent += text;
  target.scrollTop = target.scrollHeight;
}

function replaceStageText(stage, text) {
  const target = resolveStageTarget(stage);
  if (!target) {
    return;
  }
  target.textContent = text || "";
  target.scrollTop = target.scrollHeight;
}

function fallbackCopyText(text) {
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "readonly");
  area.style.position = "fixed";
  area.style.left = "-9999px";
  document.body.appendChild(area);
  area.focus();
  area.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } finally {
    document.body.removeChild(area);
  }
  return copied;
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const copied = fallbackCopyText(text);
  if (!copied) {
    throw new Error("当前浏览器环境不支持复制");
  }
}

function flashCopyButton(button) {
  const defaultLabel = button.dataset.labelDefault || "复制";
  const copiedLabel = button.dataset.labelCopied || "已复制";
  button.textContent = copiedLabel;
  window.setTimeout(() => {
    button.textContent = defaultLabel;
  }, 1200);
}

async function copyOutputContent(button) {
  const targetId = button.dataset.copyTarget || "";
  const target = document.getElementById(targetId);
  const targetLabel = copyTargetLabelMap[targetId] || "模块内容";
  if (!target) {
    setStatus(`复制失败：未找到 ${targetLabel}`);
    return;
  }

  const text = target.textContent || "";
  if (!text.trim()) {
    setStatus(`复制失败：${targetLabel} 为空`);
    return;
  }

  try {
    await copyTextToClipboard(text);
    flashCopyButton(button);
    setStatus(`已复制 ${targetLabel}`);
    logLine(`已复制 ${targetLabel}`);
  } catch (error) {
    setStatus(`复制失败：${targetLabel}`);
    logLine(`复制失败（${targetLabel}）：${error.message}`);
  }
}

function selectedFilePath(file) {
  return (file.webkitRelativePath || file.name || "").replace(/\\/g, "/");
}

function selectedDirLabel(files) {
  if (!files.length) {
    return "-";
  }
  const parts = selectedFilePath(files[0]).split("/").filter(Boolean);
  if (parts.length > 1) {
    return parts[0];
  }
  return files[0].name || "已选择目录";
}

function findHistoryFile(files, fileName) {
  const target = fileName.toLowerCase();
  return files.find((file) => {
    const fullPath = selectedFilePath(file).toLowerCase();
    return (
      fullPath === target ||
      fullPath.endsWith(`/${target}`) ||
      file.name.toLowerCase() === target
    );
  });
}

function prettyJsonText(text) {
  if (!text) {
    return "";
  }
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch (_error) {
    return text;
  }
}

function outputSnapshot() {
  return {
    literature: literatureOutput.textContent || "",
    review: reviewOutput.textContent || "",
    idea: ideaOutput.textContent || "",
    outline: outlineOutput.textContent || "",
    paper: paperOutput.textContent || "",
  };
}

function updateResumeStageOptions(manifest) {
  const allowed = new Set(
    Array.isArray(manifest?.resumable_stages) && manifest.resumable_stages.length
      ? manifest.resumable_stages
      : stageSequence
  );
  Array.from(resumeStageSelect.options).forEach((option) => {
    option.disabled = !allowed.has(option.value);
  });
  if (resumeStageSelect.value && !allowed.has(resumeStageSelect.value)) {
    const fallback = Array.from(allowed)[0] || "literature";
    resumeStageSelect.value = fallback;
  }
}

function suggestResumeStage(manifest) {
  if (!manifest) {
    return "literature";
  }
  const cacheStage = normalizeStage(manifest.stage);
  const resumable = Array.isArray(manifest.resumable_stages)
    ? manifest.resumable_stages.filter((item) => !!normalizeStage(item))
    : [];
  if (!cacheStage) {
    return resumable[0] || "literature";
  }
  const cacheIdx = stageSequence.indexOf(cacheStage);
  const nextStage = stageSequence[Math.min(cacheIdx + 1, stageSequence.length - 1)];
  if (resumable.includes(nextStage)) {
    return nextStage;
  }
  if (resumable.includes(cacheStage)) {
    return cacheStage;
  }
  return resumable[0] || "literature";
}

function renderCacheOptions(caches, preferredCacheId = "") {
  cacheManifestMap = new Map();
  cacheSelect.innerHTML = "";

  if (!Array.isArray(caches) || !caches.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "暂无缓存";
    cacheSelect.appendChild(option);
    cacheSelect.disabled = true;
    loadCacheBtn.disabled = true;
    resumeStageSelect.disabled = true;
    resumeTargetSelect.disabled = true;
    resumeBtn.disabled = true;
    activeCacheId = "";
    updateResumeStageOptions(null);
    return;
  }

  caches.forEach((cache) => {
    if (!cache || !cache.cache_id) {
      return;
    }
    cacheManifestMap.set(cache.cache_id, cache);
    const option = document.createElement("option");
    const createdAt = cache.created_at || "-";
    const stageLabel = stageLabelMap[cache.stage] || cache.stage || "-";
    const title = cache.title || cache.topic_preview || "";
    option.value = cache.cache_id;
    option.textContent = `${createdAt} | ${stageLabel}${title ? ` | ${title}` : ""}`;
    cacheSelect.appendChild(option);
  });

  cacheSelect.disabled = false;
  loadCacheBtn.disabled = false;
  resumeStageSelect.disabled = false;
  resumeTargetSelect.disabled = false;
  resumeBtn.disabled = false;

  const targetCacheId =
    (preferredCacheId && cacheManifestMap.has(preferredCacheId) && preferredCacheId) ||
    (activeCacheId && cacheManifestMap.has(activeCacheId) && activeCacheId) ||
    caches[0].cache_id;
  cacheSelect.value = targetCacheId;
  activeCacheId = targetCacheId;
  updateResumeStageOptions(cacheManifestMap.get(targetCacheId));
}

async function refreshCacheList(preferredCacheId = "") {
  try {
    const response = await fetch("/api/cache/list");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "获取缓存列表失败");
    }
    renderCacheOptions(data.caches || [], preferredCacheId);
  } catch (error) {
    logLine(`刷新缓存列表失败：${error.message}`);
  }
}

async function loadSelectedCache() {
  if (currentJobId) {
    setStatus("任务进行中，无法恢复缓存");
    return;
  }
  const cacheId = String(cacheSelect.value || "").trim();
  if (!cacheId) {
    setStatus("请先选择缓存");
    return;
  }

  setStatus("恢复缓存中");
  try {
    const response = await fetch(`/api/cache/${encodeURIComponent(cacheId)}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "恢复缓存失败");
    }

    configInput.value = data.config_text || "";
    topicInput.value = data.topic_text || "";
    const outputs = data.outputs || {};
    stageSequence.forEach((stage) => {
      replaceStageText(stage, outputs[stage] || "");
    });

    const manifest = data.cache || {};
    activeCacheId = manifest.cache_id || cacheId;
    setPath(`缓存/${activeCacheId}`);
    updateResumeStageOptions(manifest);
    resumeStageSelect.value = suggestResumeStage(manifest);
    setStatus("缓存已恢复");
    logLine(
      `已恢复缓存：${activeCacheId}（阶段：${
        stageLabelMap[manifest.stage] || manifest.stage || "-"
      }）`
    );
  } catch (error) {
    setStatus("恢复缓存失败");
    logLine(`恢复缓存失败：${error.message}`);
  }
}

async function saveStageCache(stage) {
  const normalizedStage = normalizeStage(stage);
  if (!normalizedStage) {
    setStatus("缓存失败：阶段参数无效");
    return;
  }

  const configText = configInput.value.trim();
  const topicText = topicInput.value.trim();
  if (!configText) {
    setStatus("缓存失败：config 为空");
    return;
  }
  if (!topicText) {
    setStatus("缓存失败：topic 为空");
    return;
  }

  const outputs = outputSnapshot();
  if (!String(outputs[normalizedStage] || "").trim()) {
    setStatus(`缓存失败：${stageLabelMap[normalizedStage]} 为空`);
    return;
  }

  try {
    const response = await fetch("/api/cache/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stage: normalizedStage,
        config_text: configText,
        topic_text: topicText,
        outputs,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "缓存保存失败");
    }
    const cache = data.cache || {};
    const cacheId = cache.cache_id || "";
    activeCacheId = cacheId;
    setStatus(`缓存成功：${cacheId}`);
    logLine(
      `缓存成功：${cacheId}（包含阶段：${(cache.stages_included || []).join(", ")}）`
    );
    await refreshCacheList(cacheId);
  } catch (error) {
    setStatus("缓存失败");
    logLine(`缓存失败：${error.message}`);
  }
}

async function loadHistoryDir(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) {
    setStatus("未选择历史目录");
    return;
  }
  if (currentJobId) {
    setStatus("任务进行中，无法加载历史目录");
    return;
  }

  clearOutputs();
  const folderLabel = selectedDirLabel(files);
  setPath(folderLabel);
  setStatus("加载历史目录中");
  logLine(`开始加载历史目录：${folderLabel}`);

  const targets = [
    {
      stage: "literature",
      fileName: "00_literature.json",
      format: prettyJsonText,
    },
    { stage: "idea", fileName: "01_idea.json", format: prettyJsonText },
    {
      stage: "review",
      fileName: "00_literature_review.md",
      format: (text) => text || "",
    },
    { stage: "outline", fileName: "02_outline.json", format: prettyJsonText },
    { stage: "paper", fileName: "03_thesis.md", format: (text) => text || "" },
  ];

  let loadedCount = 0;

  for (const target of targets) {
    const file = findHistoryFile(files, target.fileName);
    if (!file) {
      logLine(`未找到 ${target.fileName}`);
      continue;
    }
    try {
      const rawText = await file.text();
      replaceStageText(target.stage, target.format(rawText));
      loadedCount += 1;
      logLine(`已加载 ${target.fileName}`);
    } catch (error) {
      logLine(`读取 ${target.fileName} 失败：${error.message}`);
    }
  }

  if (loadedCount === 0) {
    setStatus("未在目录中找到 00_literature/01_idea/00_literature_review/02/03 文件");
    return;
  }

  activeCacheId = "";
  updateResumeStageOptions(null);
  setStatus(`历史结果已加载（${loadedCount}/5）`);
}

function closeSource() {
  if (!currentSource) {
    return;
  }
  streamClosedByClient = true;
  currentSource.close();
  currentSource = null;
}

function finishJob() {
  closeSource();
  setRunning(false);
  currentJobId = null;
}

function handleEvent(name, payload) {
  if (name === "status") {
    logLine(payload.message || "任务已启动");
    return;
  }

  if (name === "resume_started") {
    logLine(
      `从缓存继续：${payload.cache_id || "-"}，起始阶段=${
        payload.resume_from_stage || "-"
      }，结束阶段=${payload.resume_to_stage || "-"}`
    );
    return;
  }

  if (name === "workflow_started") {
    setStatus(`进行中（run_id: ${payload.run_id || "-"}）`);
    logLine(`工作流开始，模型：${payload.model || "-"}`);
    return;
  }

  if (name === "literature_started") {
    logLine(`文献检索开始（provider: ${payload.provider || "-"}）`);
    return;
  }

  if (name === "literature_query_expanded") {
    logLine(
      `检索词扩展完成（${payload.count || 0} 条）：${(payload.queries || []).join(
        " | "
      )}`
    );
    return;
  }

  if (name === "literature_completed") {
    logLine(`文献检索完成，命中 ${payload.count || 0} 条`);
    replaceStageText("literature", JSON.stringify(payload.items || [], null, 2));
    const runDir = inferRunDirFromFilePath(payload.path || "");
    if (runDir) {
      setPath(runDir);
    }
    return;
  }

  if (name === "stage_started") {
    logLine(`阶段开始：${payload.stage}`);
    return;
  }

  if (name === "stage_round_started") {
    logLine(`${payload.stage} 第 ${payload.round} 轮开始`);
    return;
  }

  if (name === "stage_round_completed") {
    logLine(
      `${payload.stage} 第 ${payload.round} 轮完成（finish_reason: ${
        payload.finish_reason || "-"
      }）`
    );
    return;
  }

  if (name === "llm_request") {
    const stage = payload.stage || "-";
    const operation = payload.operation || "-";
    const model = payload.model || "-";
    const roundSuffix = payload.round ? `，round=${payload.round}` : "";
    logLine(`LLM 调用入参：stage=${stage}，operation=${operation}，model=${model}${roundSuffix}`);
    logLine(safePrettyJson(payload));
    return;
  }

  if (name === "stage_delta") {
    appendStageText(payload.stage, payload.text || "");
    return;
  }

  if (name === "stage_restored") {
    replaceStageText(payload.stage, payload.content || "");
    const runDir = inferRunDirFromFilePath(payload.path || "");
    if (runDir) {
      setPath(runDir);
    }
    logLine(`已从缓存恢复阶段：${payload.stage}`);
    return;
  }

  if (name === "stage_completed") {
    replaceStageText(payload.stage, payload.content || "");
    const usage = payload.usage || {};
    const runDir = inferRunDirFromFilePath(payload.path || "");
    if (runDir) {
      setPath(runDir);
    }
    logLine(
      `${payload.stage} 阶段完成（tokens: ${usage.total_tokens || 0}, rounds: ${
        usage.rounds || 1
      }）`
    );
    return;
  }

  if (name === "workflow_completed") {
    if (payload.run_dir) {
      setPath(payload.run_dir);
    }
    logLine("工作流已完成");
    return;
  }

  if (name === "done") {
    setStatus("完成");
    if (payload.run_dir) {
      setPath(payload.run_dir);
      logLine(`生成完成：${payload.run_dir}`);
    }
    finishJob();
    return;
  }

  if (name === "job_cancelled") {
    setStatus("已停止");
    logLine(payload.message || "任务已停止");
    finishJob();
    return;
  }

  if (name === "job_error") {
    const message = payload.message || "未知错误";
    setStatus("失败");
    logLine(`任务失败：${message}`);
    finishJob();
  }
}

function bindSource(jobId) {
  streamClosedByClient = false;
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  currentSource = source;

  const eventNames = [
    "status",
    "resume_started",
    "workflow_started",
    "literature_started",
    "literature_query_expanded",
    "literature_completed",
    "stage_started",
    "stage_round_started",
    "stage_round_completed",
    "llm_request",
    "stage_delta",
    "stage_restored",
    "stage_completed",
    "workflow_completed",
    "done",
    "job_cancelled",
    "job_error",
  ];

  eventNames.forEach((name) => {
    source.addEventListener(name, (evt) => {
      handleEvent(name, parseEventData(evt));
    });
  });

  source.onerror = () => {
    if (streamClosedByClient) {
      return;
    }
    logLine("连接中断，请检查后端日志后重试。");
    setStatus("连接中断");
    finishJob();
  };
}

async function createJob(options = {}) {
  const configText = configInput.value.trim();
  const topicText = topicInput.value.trim();
  const title = titleInput.value.trim();
  const resumeCacheId = String(options.resumeCacheId || "").trim();
  const resumeFromStage = String(options.resumeFromStage || "").trim();
  const resumeToStage = String(options.resumeToStage || "").trim();
  const preserveOutputs = options.preserveOutputs === true;

  if (!configText) {
    setStatus("config 不能为空");
    return;
  }
  if (!topicText) {
    setStatus("topic 不能为空");
    return;
  }

  if (currentSource) {
    closeSource();
  }
  currentJobId = null;

  if (!preserveOutputs) {
    clearOutputs();
  }
  setRunning(true);
  setStatus("创建任务中");

  try {
    const requestPayload = {
      config_text: configText,
      topic_text: topicText,
      title,
    };
    if (resumeCacheId) {
      requestPayload.resume_cache_id = resumeCacheId;
      requestPayload.resume_from_stage = resumeFromStage;
      requestPayload.resume_to_stage = resumeToStage;
    }

    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "创建任务失败");
    }

    setStatus("进行中");
    logLine(`任务已创建：${data.job_id}`);
    currentJobId = data.job_id;
    bindSource(data.job_id);
  } catch (error) {
    setStatus("失败");
    finishJob();
    logLine(`任务创建失败：${error.message}`);
  }
}

async function resumeFromCache() {
  if (currentJobId) {
    setStatus("任务进行中，无法继续");
    return;
  }
  const cacheId = String(cacheSelect.value || "").trim();
  if (!cacheId) {
    setStatus("请先选择缓存");
    return;
  }

  const manifest = cacheManifestMap.get(cacheId);
  if (manifest) {
    const allowed = Array.isArray(manifest.resumable_stages)
      ? manifest.resumable_stages
      : [];
    if (allowed.length && !allowed.includes(resumeStageSelect.value)) {
      setStatus(`该缓存不可从 ${resumeStageSelect.value} 开始`);
      return;
    }
  }

  const resumeFromStage = resumeStageSelect.value;
  const resumeToStage =
    resumeTargetSelect.value === "same" ? resumeFromStage : "paper";
  const fromLabel = stageLabelMap[resumeFromStage] || resumeFromStage;
  const toLabel = stageLabelMap[resumeToStage] || resumeToStage;
  logLine(`准备从缓存继续：${cacheId}（${fromLabel} -> ${toLabel}）`);

  await createJob({
    resumeCacheId: cacheId,
    resumeFromStage,
    resumeToStage,
    preserveOutputs: true,
  });
}

async function stopJob() {
  if (!currentJobId) {
    setStatus("无运行中的任务");
    return;
  }

  stopBtn.disabled = true;
  stopBtn.textContent = "停止中...";
  setStatus("停止中");

  try {
    const response = await fetch(`/api/jobs/${currentJobId}/cancel`, {
      method: "POST",
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "停止任务失败");
    }

    logLine(data.message || "已发送停止请求，正在终止任务。");
    if (data.status === "cancelled") {
      setStatus("已停止");
      finishJob();
    }
  } catch (error) {
    setStatus("停止失败");
    stopBtn.disabled = false;
    stopBtn.textContent = "停止任务";
    logLine(`停止任务失败：${error.message}`);
  }
}

async function loadInitial() {
  try {
    const response = await fetch("/api/initial");
    const data = await response.json();
    configInput.value = data.config_text || "";
    topicInput.value = data.topic_text || "";
    logLine(`已加载默认文件：${data.config_path || "-"}`);
    logLine(`已加载默认文件：${data.topic_path || "-"}`);
  } catch (error) {
    logLine(`加载默认内容失败：${error.message}`);
  }
}

startBtn.addEventListener("click", () => {
  createJob();
});

stopBtn.addEventListener("click", () => {
  stopJob();
});

historyDirInput.addEventListener("change", () => {
  loadHistoryDir(historyDirInput.files);
});

copyButtons.forEach((button) => {
  button.addEventListener("click", () => {
    copyOutputContent(button);
  });
});

cacheButtons.forEach((button) => {
  button.addEventListener("click", () => {
    saveStageCache(button.dataset.cacheStage || "");
  });
});

refreshCacheBtn.addEventListener("click", () => {
  refreshCacheList();
});

loadCacheBtn.addEventListener("click", () => {
  loadSelectedCache();
});

cacheSelect.addEventListener("change", () => {
  activeCacheId = String(cacheSelect.value || "").trim();
  updateResumeStageOptions(cacheManifestMap.get(activeCacheId));
});

resumeBtn.addEventListener("click", () => {
  resumeFromCache();
});

async function bootstrap() {
  await loadInitial();
  await refreshCacheList();
}

bootstrap();
