(() => {
  "use strict";

  const MAX_BYTES = 10 * 1024 * 1024;
  const ALLOWED_EXT = ["jpg", "jpeg", "png", "webp"];
  // Client-side downscale: draw the uploaded image onto a canvas at most this
  // many pixels on the longest side, then re-encode as JPEG. This keeps the
  // upload small and the server's decode/convert buffers tiny — the Render free
  // instance has only 512 MB and large phone photos (4000×3000+) can OOM it.
  const CLIENT_MAX_SIDE = 1536;
  const JPEG_QUALITY = 0.92;

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

  // Decode the file into an Image, draw it onto a canvas (white background for
  // PNG transparency), downscale if needed, and re-encode as JPEG. Returns a
  // File ready for upload. Always re-encodes so the server receives a compact
  // payload regardless of the original format.
  function downscaleForUpload(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        let w = img.naturalWidth;
        let h = img.naturalHeight;
        const scale = CLIENT_MAX_SIDE / Math.max(w, h);
        if (scale < 1) {
          w = Math.max(1, Math.round(w * scale));
          h = Math.max(1, Math.round(h * scale));
        }
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, w, h);
        ctx.drawImage(img, 0, 0, w, h);
        canvas.toBlob((blob) => {
          if (!blob) {
            reject(new Error("图片压缩失败，请换一张图片"));
            return;
          }
          const base = file.name.replace(/\.[^.]+$/, "") || "image";
          resolve(new File([blob], `${base}.jpg`, { type: "image/jpeg" }));
        }, "image/jpeg", JPEG_QUALITY);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("图片解码失败，文件可能已损坏"));
      };
      img.src = url;
    });
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
      // Downscale on the client so the server never sees a multi-megapixel
      // decode spike (the Render free instance OOMs at ~512 MB).
      const uploadFile = await downscaleForUpload(file);
      const form = new FormData();
      form.append("file", uploadFile);
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
