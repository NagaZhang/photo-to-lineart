# 图片速写生成器（Photo → Lineart）

上传一张图片（插画 / 动漫 / 截图为主，也支持真实照片），一键输出**白底黑线、高度还原细节的速写线稿**，支持在线对比预览与 PNG 下载。部署在 Render 上，任何人通过网址即可使用。

> 核心模型全部复用成熟开源项目，**不包含任何自训练权重**；权重在 Docker 构建阶段从 Hugging Face 拉取并导出为 ONNX。

## 技术方案

- 线稿模型：[AnyLine / MTEED](https://huggingface.co/TheMistoAI/MistoLine)（TheMistoAI），基于 [TEED](https://github.com/xavysp/TEED)，通过 [controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)（Fannovel16）加载与导出。评估中它在细节还原（4.4/5）和线稿干净度（4.2/5）上均优于 Anime2Sketch 方案，见 `samples/evaluation.md`
- 推理：构建期 PyTorch → ONNX（opset 17，动态边长）；运行时仅用 [ONNX Runtime](https://onnxruntime.ai/) CPU + OpenCV/Pillow 前后处理，模型文件约 0.2 MB，峰值内存 < 512 MB
- 后端：[FastAPI](https://fastapi.tiangolo.com/)（`POST /api/sketch`，图片只做临时处理、不持久化；Content-Length 预检 + 流式读取硬限 10MB、2500 万像素解压炸弹防护）；前端：原生静态 HTML/CSS/JS，无构建链
- 部署：Docker 多阶段构建（构建镜像含 torch，运行镜像不含）+ [Render](https://render.com) Web Service（免费档）

## 目录结构

```
backend/
  app/main.py        FastAPI 路由（上传校验、限并发、静态托管）
  app/pipeline.py    ONNX Runtime 推理管线（无 torch 依赖）
  export_onnx.py     构建期：下载开源权重并导出 ONNX
  verify_onnx.py     开发期：PyTorch vs ONNX 全管线一致性校验
  tests/test_api.py  接口自动化测试
frontend/            中文静态网页（上传/拖拽/粘贴、对比滑块、下载）
samples/             评估样例图与对照结果（不进入 Docker 镜像）
deploy/Dockerfile    多阶段构建
render.yaml          Render Blueprint
```

## 本地运行

需要 Python 3.13。

```powershell
# 1. 虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 运行期依赖（服务本身不需要 torch）
pip install -r backend/requirements.txt

# 3. 导出模型（需要构建期依赖；中国大陆可先设置 HF_ENDPOINT 镜像）
pip install -r backend/requirements-build.txt
$env:HF_ENDPOINT = "https://hf-mirror.com"
python backend/export_onnx.py --model anyline

# 4. 启动服务
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000 即可使用。

测试：

```powershell
pip install -r backend/requirements-dev.txt
pytest backend/tests -q
```

## 部署到 Render

1. 将本仓库推送到 GitHub
2. Render Dashboard → **New → Blueprint**，选择该仓库（Render 会读取根目录 `render.yaml`）
3. 使用默认的免费实例（Free Web Service）即可；首次构建会在镜像内下载权重并导出 ONNX，需要几分钟
4. 部署完成后 Render 提供 `https://<service-name>.onrender.com` 公网地址，健康检查路径为 `/api/health`

> 免费实例闲置一段时间后会休眠，下次首次访问需等待冷启动；不会产生费用。

## 接口说明

- `POST /api/sketch`：multipart 表单字段 `file`，JPG/PNG/WebP，≤10MB；超过 2048px 的最长边会先等比缩小，解码后超过 2500 万像素的图片返回 400。返回 `image/png`（单通道灰度，尺寸与处理后的输入一致）。大文件 multipart 可能短暂使用系统临时目录（请求结束即删除），服务不做任何持久化存储。
- 可调环境变量（默认值即部署值）：`LINEART_RESOLUTION=704`（检测分辨率）、`LINEART_LEVEL_BLACK=40` / `LINEART_LEVEL_WHITE=200`（白场色阶）、`LINEART_CPU_ARENA=0`、`OMP_NUM_THREADS=2`
- `GET /api/health`：`{"status":"ok"}`

## 开源声明

本项目是对以下开源工作的复用与工程化封装，按各自许可证使用：

- **AnyLine / MistoLine**：TheMistoAI，[TheMistoAI/MistoLine](https://huggingface.co/TheMistoAI/MistoLine)（权重，Apache-2.0）
- **TEED**：[xavysp/TEED](https://github.com/xavysp/TEED)（网络结构，MIT）
- **controlnet_aux**：[Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)（模型加载与前后处理参考实现，Apache-2.0）
- 评估备选：**Anime2Sketch**（[Mukosame/Anime2Sketch](https://github.com/Mukosame/Anime2Sketch)）与 **informative-drawings**（[carolineec/informative-drawings](https://github.com/carolineec/informative-drawings)，MIT）
- [FastAPI](https://github.com/fastapi/fastapi)（MIT）、[Uvicorn](https://github.com/encode/uvicorn)（BSD-3-Clause）、[python-multipart](https://github.com/Kludex/python-multipart)（Apache-2.0）、[ONNX Runtime](https://github.com/microsoft/onnxruntime)（MIT）、[onnx](https://github.com/onnx/onnx)（Apache-2.0，仅构建期）、[huggingface_hub](https://github.com/huggingface/huggingface_hub)（Apache-2.0，仅构建期）、[einops](https://github.com/arogozhnikov/einops)（MIT，controlnet_aux 传递依赖）、[OpenCV](https://github.com/opencv/opencv)（Apache-2.0）、[Pillow](https://github.com/python-pillow/Pillow)（HPND）、[NumPy](https://github.com/numpy/numpy)（BSD-3-Clause）、[PyTorch](https://github.com/pytorch/pytorch)（BSD-3-Clause，仅构建期）
- 内置示例图 `frontend/sample-astronaut.jpg` 源自 scikit-image `data.astronaut()`，原图由 NASA 拍摄，属于公有领域
