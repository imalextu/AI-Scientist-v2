const configInput = document.getElementById("configInput");
const topicInput = document.getElementById("topicInput");
const titleInput = document.getElementById("titleInput");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const historyDirInput = document.getElementById("historyDirInput");
const statusLine = document.getElementById("statusLine");
const pathLine = document.getElementById("pathLine");
const logOutput = document.getElementById("logOutput");
const literatureOutput = document.getElementById("literatureOutput");
const researchOutput = document.getElementById("researchOutput");
const ideaOutput = document.getElementById("ideaOutput");
const outlineOutput = document.getElementById("outlineOutput");
const paperOutput = document.getElementById("paperOutput");
const copyButtons = Array.from(document.querySelectorAll(".copy-btn"));

const stageOutputMap = {
  literature: literatureOutput,
  research: researchOutput,
  idea: ideaOutput,
  outline: outlineOutput,
  paper: paperOutput,
};

const copyTargetLabelMap = {
  literatureOutput: "00 文献检索 JSON",
  researchOutput: "00 研究树搜索 JSON",
  ideaOutput: "01 选题设计 JSON",
  outlineOutput: "02 论文大纲 JSON",
  paperOutput: "03 论文正文 Markdown",
};

let currentSource = null;
let streamClosedByClient = false;
let currentJobId = null;

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
  if (!running) {
    stopBtn.textContent = "停止任务";
  }
}

function clearOutputs() {
  logOutput.textContent = "";
  literatureOutput.textContent = "";
  researchOutput.textContent = "";
  ideaOutput.textContent = "";
  outlineOutput.textContent = "";
  paperOutput.textContent = "";
  setPath("-");
}

function parseEventData(evt) {
  try {
    return JSON.parse(evt.data || "{}");
  } catch (_error) {
    return {};
  }
}

function resolveStageTarget(stage) {
  if (!stage) {
    return null;
  }
  if (stageOutputMap[stage]) {
    return stageOutputMap[stage];
  }
  if (stage.startsWith("research:")) {
    return researchOutput;
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
    {
      stage: "research",
      fileName: "00_research_tree.json",
      format: prettyJsonText,
    },
    { stage: "idea", fileName: "01_idea.json", format: prettyJsonText },
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
    setStatus("未在目录中找到 00_literature/00_research_tree/01/02/03 文件");
    return;
  }
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

  if (name === "workflow_started") {
    setStatus(`进行中（run_id: ${payload.run_id || "-"}）`);
    logLine(`工作流开始，模型：${payload.model || "-"}`);
    return;
  }

  if (name === "literature_started") {
    logLine(`文献检索开始（provider: ${payload.provider || "-"}）`);
    return;
  }

  if (name === "literature_completed") {
    logLine(`文献检索完成，命中 ${payload.count || 0} 条`);
    replaceStageText(
      "literature",
      JSON.stringify(payload.items || [], null, 2)
    );
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

  if (name === "stage_delta") {
    appendStageText(payload.stage, payload.text || "");
    return;
  }

  if (name === "stage_completed") {
    replaceStageText(payload.stage, payload.content || "");
    const usage = payload.usage || {};
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
    "workflow_started",
    "literature_started",
    "literature_completed",
    "stage_started",
    "stage_round_started",
    "stage_round_completed",
    "stage_delta",
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

async function createJob() {
  const configText = configInput.value.trim();
  const topicText = topicInput.value.trim();
  const title = titleInput.value.trim();

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

  clearOutputs();
  setRunning(true);
  setStatus("创建任务中");

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        config_text: configText,
        topic_text: topicText,
        title,
      }),
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

loadInitial();
