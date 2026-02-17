const configInput = document.getElementById("configInput");
const topicInput = document.getElementById("topicInput");
const titleInput = document.getElementById("titleInput");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusLine = document.getElementById("statusLine");
const pathLine = document.getElementById("pathLine");
const logOutput = document.getElementById("logOutput");
const ideaOutput = document.getElementById("ideaOutput");
const outlineOutput = document.getElementById("outlineOutput");
const paperOutput = document.getElementById("paperOutput");

const stageOutputMap = {
  idea: ideaOutput,
  outline: outlineOutput,
  paper: paperOutput,
};

let currentSource = null;
let streamClosedByClient = false;

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
}

function clearOutputs() {
  logOutput.textContent = "";
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

function appendStageText(stage, text) {
  const target = stageOutputMap[stage];
  if (!target || !text) {
    return;
  }
  target.textContent += text;
  target.scrollTop = target.scrollHeight;
}

function replaceStageText(stage, text) {
  const target = stageOutputMap[stage];
  if (!target) {
    return;
  }
  target.textContent = text || "";
  target.scrollTop = target.scrollHeight;
}

function closeSource() {
  if (!currentSource) {
    return;
  }
  streamClosedByClient = true;
  currentSource.close();
  currentSource = null;
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
    closeSource();
    setRunning(false);
    return;
  }

  if (name === "job_error") {
    const message = payload.message || "未知错误";
    setStatus("失败");
    logLine(`任务失败：${message}`);
    closeSource();
    setRunning(false);
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
    closeSource();
    setRunning(false);
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
    bindSource(data.job_id);
  } catch (error) {
    setStatus("失败");
    setRunning(false);
    logLine(`任务创建失败：${error.message}`);
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
  closeSource();
  setRunning(false);
  setStatus("已停止显示（后台可能仍在运行）");
  logLine("你已停止当前流式显示。");
});

loadInitial();
