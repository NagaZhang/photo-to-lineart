(() => {
  "use strict";

  const MAX_BYTES = 10 * 1024 * 1024;
  const ALLOWED_EXT = ["jpg", "jpeg", "png", "webp"];

  const $ = (id) => document.getElementById(id);
  const dropzone = $("dropzone");
  const fileInput = $("file-input");
  const uploadPanel = $("upload-panel");
  const resultPanel = $("result-panel");
  const errorBanner = $("error-banner");
  const loading = $("loading");
  const originalImg = $("original-img");
  const resultImg = $("result-img");
  const compareBox = $("compare-box");
  const compareRange = $("compare-range");
  const downloadBtn = $("download-btn");
  const resetBtn = $("reset-btn");
  const sampleBtn = $("sample-btn");

  let objectUrls = [];
  let currentName = "";

  function remember(url) {
    objectUrls.push(url);
    return url;
  }
  function releaseAll() {
    objectUrls.forEach((u) => URL.revokeObjectURL(u));
    objectUrls = [];
  }

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.hidden = false;
  }
  function clearError() {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
  }

  function validate(file) {
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (!ALLOWED_EXT.includes(ext)) {
      return "仅支持 JPG / PNG / WebP 格式的图片";
    }
    if (file.size === 0) return "文件为空，请重新选择";
    if (file.size > MAX_BYTES) return "图片过大，请上传 10MB 以内的文件";
    return null;
  }

  async function handleFile(file) {
    clearError();
    const problem = validate(file);
    if (problem) {
      showError(problem);
      return;
    }
    currentName = file.name.replace(/\.[^.]+$/, "");

    releaseAll();
    resultPanel.hidden = false;
    uploadPanel.hidden = true;
    loading.hidden = false;
    compareBox.style.visibility = "hidden";

    originalImg.src = remember(URL.createObjectURL(file));

    try {
      const form = new FormData();
      form.append("file", file);
      const resp = await fetch("/api/sketch", { method: "POST", body: form });
      if (!resp.ok) {
        let detail = "生成失败，请稍后重试";
        try {
          const data = await resp.json();
          if (data && typeof data.detail === "string") detail = data.detail;
        } catch (_) { /* keep default message */ }
        throw new Error(detail);
      }
      const blob = await resp.blob();
      resultImg.src = remember(URL.createObjectURL(blob));
      await resultImg.decode();
      compareRange.value = 50;
      compareBox.style.setProperty("--pos", "50%");
      loading.hidden = true;
      compareBox.style.visibility = "visible";
    } catch (err) {
      loading.hidden = true;
      resultPanel.hidden = true;
      uploadPanel.hidden = false;
      showError(err.message || "网络异常，请稍后重试");
    }
  }

  // ---- events: click ----
  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) handleFile(fileInput.files[0]);
    fileInput.value = "";
  });

  // ---- events: drag & drop ----
  // Prevent the browser's default "open the file / leave the page" behavior
  // even when the drop happens outside the dropzone.
  ["dragover", "drop"].forEach((evt) =>
    document.addEventListener(evt, (e) => e.preventDefault())
  );
  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // ---- events: paste anywhere ----
  document.addEventListener("paste", (e) => {
    const item = Array.from(e.clipboardData?.items || []).find((i) =>
      i.type.startsWith("image/")
    );
    const pasted = item ? item.getAsFile() : null;
    if (pasted) {
      const subtype = item.type.split("/")[1] || "png";
      const ext = subtype === "jpeg" ? "jpg" : subtype;
      handleFile(new File([pasted], `pasted-image.${ext}`, { type: item.type }));
    }
  });

  // ---- comparison slider ----
  compareRange.addEventListener("input", () => {
    compareBox.style.setProperty("--pos", `${compareRange.value}%`);
  });

  // ---- download ----
  downloadBtn.addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = resultImg.src;
    a.download = `${currentName || "image"}_lineart.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  // ---- reset ----
  resetBtn.addEventListener("click", () => {
    clearError();
    resultPanel.hidden = true;
    uploadPanel.hidden = false;
    releaseAll();
  });

  // ---- built-in sample (NASA public domain photo shipped with the site) ----
  sampleBtn.addEventListener("click", async () => {
    try {
      clearError();
      const resp = await fetch("/sample-astronaut.jpg");
      if (!resp.ok) throw new Error("示例图加载失败");
      const blob = await resp.blob();
      await handleFile(new File([blob], "sample-astronaut.jpg", { type: "image/jpeg" }));
    } catch (err) {
      showError(err.message || "示例图加载失败");
    }
  });
})();
