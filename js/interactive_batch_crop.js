import { app } from "../../scripts/app.js";

// 全局样式（只注入一次）
if (!window.__interactiveCropStyleInjected) {
    window.__interactiveCropStyleInjected = true;
    const style = document.createElement("style");
    style.textContent = `
        #interactive-crop-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.92); z-index: 10000;
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; font-family: Arial, sans-serif;
            user-select: none; outline: none;
        }
        .crop-toolbar {
            display: flex; align-items: center; gap: 6px;
            margin-bottom: 8px; background: #222; padding: 4px 10px;
            border-radius: 6px; color: white; font-size: 13px; flex-wrap: wrap;
        }
        .crop-toolbar button, .crop-toolbar select, .crop-toolbar input[type="number"] {
            background: #444; color: white; border: none;
            padding: 3px 8px; border-radius: 3px; cursor: pointer;
            font-size: 13px;
        }
        .crop-toolbar button:hover { background: #666; }
        .crop-toolbar button:disabled { opacity: 0.5; }
        #crop-confirm-btn { background: #2a7; }
        #crop-cancel-btn { background: #a33; }
        #crop-reset-btn { background: #555; }
        .crop-canvas-container {
            position: relative; display: inline-block;
            background: #111; border: 1px solid #333;
        }
        #crop-canvas { display: block; }
        .crop-info {
            position: absolute; bottom: 5px; right: 5px;
            background: rgba(0,0,0,0.7); color: #0f0;
            padding: 3px 8px; font-size: 15px; border-radius: 4px;
            pointer-events: none; font-weight: bold;
        }
        .crop-thumbnails {
            display: flex; overflow-x: auto; max-width: 90vw;
            margin-top: 8px; gap: 4px; background: #222;
            padding: 4px; border-radius: 4px;
        }
        .crop-thumbnails img {
            width: 50px; height: 50px; object-fit: contain;
            border: 2px solid transparent; cursor: pointer; border-radius: 2px;
        }
        .crop-thumbnails img.active { border-color: #4af; }
        .crop-path-btns {
            display: flex; gap: 2px; margin-top: 2px;
        }
        .crop-path-btns button {
            height: 24px; padding: 2px 8px; font-size: 12px;
            background: var(--comfy-input-bg, #222);
            color: var(--input-text, white);
            border: 1px solid var(--border-color, #444);
            border-radius: 3px; cursor: pointer; flex:1;
        }
        .size-modal {
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            background: #333; color: white; padding: 20px; border-radius: 8px;
            z-index: 10001; text-align: center;
        }
        .size-modal button { margin: 4px; padding: 8px 16px; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .size-modal button:hover { background: #666; }
    `;
    document.head.appendChild(style);
}

app.registerExtension({
    name: "InteractiveBatchCrop.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "InteractiveBatchCrop") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this);

            // 生成 node_id
            const nodeIdWidget = this.widgets.find(w => w.name === "node_id");
            if (nodeIdWidget && !nodeIdWidget.value) {
                nodeIdWidget.value = Date.now().toString(36) + Math.random().toString(36).substr(2, 6);
            }

            // 统一 path 控件
            const pathWidget = this.widgets.find(w => w.name === "path");
            if (pathWidget) {
                const btnContainer = document.createElement("div");
                btnContainer.className = "crop-path-btns";

                const dirBtn = document.createElement("button");
                dirBtn.textContent = "选择目录";
                dirBtn.onclick = async () => {
                    try {
                        const resp = await fetch("/multi_file_picker/select", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ mode: "directory" })
                        });
                        const data = await resp.json();
                        if (data.path) {
                            pathWidget.value = data.path;
                            app.graph.setDirtyCanvas(true, true);
                        }
                    } catch (e) { alert("失败: " + e); }
                };

                const multiFileBtn = document.createElement("button");
                multiFileBtn.textContent = "多选文件";
                multiFileBtn.onclick = async () => {
                    try {
                        const resp = await fetch("/interactive_crop/select_files", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" }
                        });
                        const data = await resp.json();
                        if (data.error) {
                            alert(data.error);
                        } else if (data.paths) {
                            pathWidget.value = data.paths;
                            app.graph.setDirtyCanvas(true, true);
                        }
                    } catch (e) { alert("选择失败: " + e); }
                };

                const clearParamsBtn = document.createElement("button");
                clearParamsBtn.textContent = "清空参数";
                clearParamsBtn.onclick = () => {
                    const cropParamsWidget = this.widgets.find(w => w.name === "crop_params");
                    if (cropParamsWidget) {
                        cropParamsWidget.value = "";
                    }
                    // 也清空 target_size
                    const targetSizeWidget = this.widgets.find(w => w.name === "target_size");
                    if (targetSizeWidget) targetSizeWidget.value = "";
                    app.graph.setDirtyCanvas(true, true);
                };

                btnContainer.appendChild(dirBtn);
                btnContainer.appendChild(multiFileBtn);
                btnContainer.appendChild(clearParamsBtn);

                const inputEl = pathWidget.inputEl || pathWidget.element?.querySelector("input");
                if (inputEl) {
                    inputEl.style.marginBottom = "2px";
                    inputEl.parentNode.insertBefore(btnContainer, inputEl.nextSibling);
                } else {
                    this.addDOMWidget("path_actions", "custom", btnContainer, { serialize: false });
                }
            }

            // “打开裁剪器”按钮（显示文本为 ✂ 打开裁剪器）
            this.addWidget("button", "open_crop_btn", "✂ 打开裁剪器", () => {
                this.openCropUI();
            }, { serialize: false });

            this._cropParamsWidget = this.widgets.find(w => w.name === "crop_params");
            this._nodeIdWidget = nodeIdWidget;
            this._lastPath = "";                  // 用于检测路径变化
            return result;
        };

        // 打开裁剪器主流程
        nodeType.prototype.openCropUI = async function() {
            const pathWidget = this.widgets.find(w => w.name === "path");
            const imgInput = this.inputs?.find(inp => inp.name === "images");
            const nodeId = this._nodeIdWidget?.value;
            const maxPreviewWidget = this.widgets.find(w => w.name === "max_preview");
            const maxPreview = maxPreviewWidget ? maxPreviewWidget.value : 10;
            const targetSizeWidget = this.widgets.find(w => w.name === "target_size");

            // 检测路径是否变化，若变化则清空 target_size
            const currentPath = pathWidget?.value?.trim() || "";
            if (currentPath !== this._lastPath) {
                this._lastPath = currentPath;
                if (targetSizeWidget) targetSizeWidget.value = "";
            }

            if (imgInput && imgInput.link) {
                if (!nodeId) { alert("节点未正确初始化"); return; }
                const resp = await fetch(`/interactive_crop/get_tensor_preview?node_id=${nodeId}&index=0`);
                if (resp.status === 404) {
                    alert("请先运行一次工作流，再打开裁剪器。");
                    return;
                }
                const data = await resp.json();
                if (data.error) { alert(data.error); return; }
                this.showCropModal(
                    { type: "tensor", nodeId, total: data.total, lastParams: this.getLastParams() },
                    0, data.total
                );
                return;
            }

            if (!currentPath) {
                alert("请连接图像批次或输入路径（文件夹或多选文件）。");
                return;
            }

            // 判断路径类型
            let sourceType = "folder";
            let folderPath = "", filePathsStr = "";
            if (currentPath.includes("|")) {
                sourceType = "files";
                filePathsStr = currentPath;
            } else {
                sourceType = "folder";
                folderPath = currentPath;
            }

            // 获取尺寸分布
            let sizes = null;
            try {
                if (sourceType === "folder") {
                    const resp = await fetch(`/interactive_crop/get_folder_sizes?folder=${encodeURIComponent(folderPath)}`);
                    const data = await resp.json();
                    if (data.error) throw new Error(data.error);
                    sizes = data.sizes;
                } else {
                    const resp = await fetch(`/interactive_crop/get_files_sizes?paths=${encodeURIComponent(filePathsStr)}`);
                    const data = await resp.json();
                    if (data.error) throw new Error(data.error);
                    sizes = data.sizes;
                }
            } catch (e) {
                alert("获取尺寸信息失败: " + e);
                return;
            }

            // 根据尺寸数量决定是否选择
            let selectedSize = targetSizeWidget?.value || "";
            if (sizes.length > 1 && !selectedSize) {
                selectedSize = await this.showSizeSelectionModal(sizes);
                if (selectedSize === null) return; // 取消
                if (targetSizeWidget) targetSizeWidget.value = selectedSize;
            } else if (sizes.length === 1) {
                selectedSize = sizes[0].size;
                if (targetSizeWidget) targetSizeWidget.value = selectedSize;
            } else if (sizes.length === 0) {
                alert("没有找到有效图片。");
                return;
            }

            // 获取符合条件的图片总数，并过滤路径
            let total = 0;
            let filteredPaths = null;
            if (sourceType === "folder") {
                const resp = await fetch(`/interactive_crop/get_folder_preview?folder=${encodeURIComponent(folderPath)}&index=0&size=${selectedSize}`);
                const data = await resp.json();
                if (data.error) { alert(data.error); return; }
                total = data.total;
            } else {
                // 多文件模式：调用过滤接口，只保留匹配尺寸的文件路径
                if (selectedSize) {
                    const resp = await fetch(`/interactive_crop/filter_paths_by_size?paths=${encodeURIComponent(filePathsStr)}&size=${encodeURIComponent(selectedSize)}`);
                    const data = await resp.json();
                    if (data.error) { alert(data.error); return; }
                    filteredPaths = data.paths ? data.paths.split("|").filter(p => p.trim()) : [];
                    total = filteredPaths.length;
                } else {
                    const resp = await fetch(`/interactive_crop/get_files_sizes?paths=${encodeURIComponent(filePathsStr)}`);
                    const data = await resp.json();
                    const sizeObj = data.sizes.find(s => s.size === selectedSize);
                    total = sizeObj ? sizeObj.count : 0;
                }
            }

            if (total === 0) {
                alert("没有找到符合尺寸的图片。");
                return;
            }

            // 根据 maxPreview 生成预览索引
            let indices = [];
            if (total <= maxPreview) {
                indices = Array.from({ length: total }, (_, i) => i);
            } else if (total < maxPreview * 2) {
                const config = await this.showPreviewRangeModal(total, maxPreview);
                if (!config) return;
                for (let i = 0; i < config.count; i++) {
                    indices.push(config.start + i);
                }
            } else {
                const step = Math.ceil(total / maxPreview);
                for (let i = 0; i < total; i += step) {
                    indices.push(i);
                }
            }

            this.showCropModal({
                type: sourceType,
                folderPath: sourceType === "folder" ? folderPath : "",
                paths: sourceType === "files" ? (filteredPaths || filePathsStr.split("|").filter(p => p.trim())) : null,
                sizeFilter: selectedSize,
                previewIndices: indices,
                lastParams: this.getLastParams()
            }, 0, indices.length);
        };

        // 获取上次裁剪参数
        nodeType.prototype.getLastParams = function() {
            if (this._cropParamsWidget && this._cropParamsWidget.value) {
                try { return JSON.parse(this._cropParamsWidget.value); } catch {}
            }
            return null;
        };

        // 尺寸选择模态
        nodeType.prototype.showSizeSelectionModal = function(sizes) {
            return new Promise((resolve) => {
                const modal = document.createElement("div");
                modal.className = "size-modal";
                let html = "<p>检测到多种分辨率，请选择要处理的尺寸：</p>";
                sizes.forEach(s => {
                    html += `<button data-size="${s.size}">${s.size} (${s.count}张)</button><br>`;
                });
                html += '<button id="cancel-size-select" style="background:#a33; margin-top:10px;">取消</button>';
                modal.innerHTML = html;
                document.body.appendChild(modal);

                modal.querySelectorAll("button[data-size]").forEach(btn => {
                    btn.onclick = () => {
                        modal.remove();
                        resolve(btn.dataset.size);
                    };
                });
                modal.querySelector("#cancel-size-select").onclick = () => {
                    modal.remove();
                    resolve(null);
                };
            });
        };

        // 预览范围选择模态
        nodeType.prototype.showPreviewRangeModal = function(total, maxCount) {
            return new Promise((resolve) => {
                const modal = document.createElement("div");
                modal.className = "size-modal";
                modal.innerHTML = `
                    <p>预览范围设置 (总图片: ${total})</p>
                    <label>起始索引: </label><input id="range-start" type="number" value="0" min="0" max="${total-1}" style="width:80px"><br><br>
                    <label>数量: </label><input id="range-count" type="number" value="${maxCount}" min="1" max="${total}" style="width:80px"><br><br>
                    <button id="confirm-range" style="margin-right:10px;">确定</button>
                    <button id="cancel-range" style="background:#a33;">取消</button>
                `;
                document.body.appendChild(modal);

                document.getElementById("confirm-range").onclick = () => {
                    const start = parseInt(document.getElementById("range-start").value);
                    const count = Math.min(parseInt(document.getElementById("range-count").value), total - start);
                    modal.remove();
                    if (isNaN(start) || isNaN(count) || start < 0 || count < 1) resolve(null);
                    else resolve({ start, count });
                };
                document.getElementById("cancel-range").onclick = () => {
                    modal.remove();
                    resolve(null);
                };
            });
        };

        // 核心裁剪界面
        nodeType.prototype.showCropModal = function(sourceInfo, startIdx, displayTotal) {
            let overlay = document.getElementById("interactive-crop-overlay");
            if (overlay) overlay.remove();
            overlay = document.createElement("div");
            overlay.id = "interactive-crop-overlay";
            overlay.setAttribute("tabindex", "-1");
            document.body.appendChild(overlay);
            overlay.focus();

            overlay.innerHTML = `
                <div class="crop-toolbar">
                    <button id="crop-confirm-btn">✅ 确认</button>
                    <button id="crop-cancel-btn">❌ 取消</button>
                    <button id="crop-reset-btn">🔄 重置</button>
                    <span style="margin-left:10px;">填充色:</span>
                    <select id="fill-select">
                        <option value="black">黑</option>
                        <option value="white">白</option>
                        <option value="gray">灰</option>
                        <option value="custom">自定义</option>
                    </select>
                    <input id="custom-color" type="color" value="#000000" style="display:none; width:28px; height:22px; padding:0; border:none; cursor:pointer; vertical-align:middle;">
                    <span id="fill-preview" style="display:none; width:18px; height:18px; border:1px solid #888; vertical-align:middle; border-radius:2px; background:#000;"></span>
                    <label><input type="checkbox" id="keep-ratio"> 保持比例</label>
                    <select id="preset-ratio">
                        <option value="free">自由</option>
                        <option value="1:1">1:1</option>
                        <option value="4:3">4:3</option>
                        <option value="3:2">3:2</option>
                        <option value="16:9">16:9</option>
                        <option value="3:4">3:4</option>
                        <option value="2:3">2:3</option>
                        <option value="9:16">9:16</option>
                    </select>
                    <button id="swap-ratio-btn" title="交换宽高并保持中心">↕</button>
                    <label>W:</label>
                    <input type="number" id="crop-w-input" value="0" min="1" style="width:65px">
                    <label>H:</label>
                    <input type="number" id="crop-h-input" value="0" min="1" style="width:65px">
                    <label><input type="checkbox" id="snap-enable" checked> 吸附</label>
                    <button id="prev-btn">◀</button>
                    <span id="idx-text">${startIdx+1} / ${displayTotal}</span>
                    <button id="next-btn">▶</button>
                </div>
                <div class="crop-canvas-container">
                    <canvas id="crop-canvas"></canvas>
                    <div class="crop-info" id="crop-info">W:0 H:0</div>
                </div>
                <div class="crop-thumbnails" id="thumb-container"></div>
            `;

            const canvas = document.getElementById("crop-canvas");
            const ctx = canvas.getContext("2d");
            const infoDiv = document.getElementById("crop-info");
            canvas.width = Math.min(window.innerWidth * 0.85, 1200);
            canvas.height = Math.min(window.innerHeight * 0.7, 800);

            // 状态
            const state = {
                sourceInfo,
                displayTotal,
                currentIndex: startIdx,
                img: new Image(),
                viewX: 0, viewY: 0, viewScale: 1,
                imgX: 0, imgY: 0,
                cropX: 0, cropY: 0, cropW: 200, cropH: 150,
                aspectRatio: null,
                dragMode: null,
                dragStart: { x: 0, y: 0 },
                dragOrigState: null,
                snapEnabled: true,
                lastParams: sourceInfo.lastParams || null,
                initialized: false            // 标记是否已完成首次设框
            };

            const parseAspect = (str) => {
                if (str === "free") return null;
                const [a, b] = str.split(":").map(Number);
                return a / b;
            };
            const enforceAspect = (w, h, a) => !a ? { w, h } : (w / h > a ? { w: h * a, h } : { w, h: w / a });
            const toWorld = (sx, sy) => ({ x: sx / state.viewScale + state.viewX, y: sy / state.viewScale + state.viewY });
            const toScreen = (wx, wy) => ({ x: (wx - state.viewX) * state.viewScale, y: (wy - state.viewY) * state.viewScale });

            const getSnapThreshold = () => {
                if (!state.img.width) return 10;
                return Math.min(state.img.width, state.img.height) * 0.03;
            };

            const applySnap = (x, y, w, h) => {
                if (!state.snapEnabled || !state.img.width) return { x, y, w, h };
                const imgL = state.imgX, imgR = imgL + state.img.width;
                const imgT = state.imgY, imgB = imgT + state.img.height;
                const t = getSnapThreshold();
                const snap = (val, target) => Math.abs(val - target) < t ? target : val;
                const left = snap(x, imgL);
                const right = snap(x + w, imgR);
                const top = snap(y, imgT);
                const bottom = snap(y + h, imgB);
                return { x: left, y: top, w: right - left, h: bottom - top };
            };

            const draw = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.save();
                ctx.translate(-state.viewX * state.viewScale, -state.viewY * state.viewScale);
                ctx.scale(state.viewScale, state.viewScale);
                if (state.img.width) {
                    ctx.drawImage(state.img, state.imgX, state.imgY, state.img.width, state.img.height);
                }
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 2 / state.viewScale;
                ctx.strokeRect(state.cropX, state.cropY, state.cropW, state.cropH);

                ctx.fillStyle = "rgba(0,0,0,0.5)";
                const INF = 1e6;
                ctx.fillRect(-INF, -INF, 2 * INF, state.cropY + INF);
                ctx.fillRect(-INF, state.cropY + state.cropH, 2 * INF, INF);
                ctx.fillRect(-INF, state.cropY, state.cropX + INF, state.cropH);
                ctx.fillRect(state.cropX + state.cropW, state.cropY, INF, state.cropH);
                ctx.restore();

                infoDiv.textContent = `W:${Math.round(state.cropW)} H:${Math.round(state.cropH)}`;
                document.getElementById("crop-w-input").value = Math.round(state.cropW);
                document.getElementById("crop-h-input").value = Math.round(state.cropH);
            };

            const updateCursor = (e) => {
                const sPos = getCanvasPos(e);
                const wPos = toWorld(sPos.x, sPos.y);
                const margin = 8 / state.viewScale;
                const inCrop = wPos.x >= state.cropX - margin && wPos.x <= state.cropX + state.cropW + margin &&
                               wPos.y >= state.cropY - margin && wPos.y <= state.cropY + state.cropH + margin;
                if (!inCrop) {
                    canvas.style.cursor = "crosshair";
                    return;
                }
                const onLeft = Math.abs(wPos.x - state.cropX) < margin * 2;
                const onRight = Math.abs(wPos.x - (state.cropX + state.cropW)) < margin * 2;
                const onTop = Math.abs(wPos.y - state.cropY) < margin * 2;
                const onBottom = Math.abs(wPos.y - (state.cropY + state.cropH)) < margin * 2;
                if ((onLeft && onTop) || (onRight && onBottom)) canvas.style.cursor = "nwse-resize";
                else if ((onRight && onTop) || (onLeft && onBottom)) canvas.style.cursor = "nesw-resize";
                else if (onLeft || onRight) canvas.style.cursor = "ew-resize";
                else if (onTop || onBottom) canvas.style.cursor = "ns-resize";
                else canvas.style.cursor = "move";
            };

            const getCanvasPos = (e) => {
                const rect = canvas.getBoundingClientRect();
                return { x: e.clientX - rect.left, y: e.clientY - rect.top };
            };

            // --- 鼠标事件 ---
            const onMouseDown = (e) => {
                const sPos = getCanvasPos(e);
                const wPos = toWorld(sPos.x, sPos.y);
                const margin = 8 / state.viewScale;
                const inCrop = wPos.x >= state.cropX - margin && wPos.x <= state.cropX + state.cropW + margin &&
                            wPos.y >= state.cropY - margin && wPos.y <= state.cropY + state.cropH + margin;

                if (e.shiftKey) {
                    state.dragMode = "img";
                    state.dragOrigState = { imgX: state.imgX, imgY: state.imgY };
                } else if (e.ctrlKey) {
                    state.dragMode = "cropMove";
                    state.dragOrigState = { cropX: state.cropX, cropY: state.cropY };
                } else if (inCrop) {
                    const onLeft = Math.abs(wPos.x - state.cropX) < margin * 2;
                    const onRight = Math.abs(wPos.x - (state.cropX + state.cropW)) < margin * 2;
                    const onTop = Math.abs(wPos.y - state.cropY) < margin * 2;
                    const onBottom = Math.abs(wPos.y - (state.cropY + state.cropH)) < margin * 2;
                    if (onLeft && onTop) state.dragMode = "resizeNW";
                    else if (onRight && onTop) state.dragMode = "resizeNE";
                    else if (onLeft && onBottom) state.dragMode = "resizeSW";
                    else if (onRight && onBottom) state.dragMode = "resizeSE";
                    else if (onLeft) state.dragMode = "resizeW";
                    else if (onRight) state.dragMode = "resizeE";
                    else if (onTop) state.dragMode = "resizeN";
                    else if (onBottom) state.dragMode = "resizeS";
                    else state.dragMode = "cropMove";
                    state.dragOrigState = {
                        cropX: state.cropX, cropY: state.cropY,
                        cropW: state.cropW, cropH: state.cropH
                    };
                    // 新增：保存对角锚点，用于保持比例时固定对角
                    if (state.dragMode.startsWith("resize")) {
                        const orig = state.dragOrigState;
                        if (state.dragMode.includes("W")) {
                            orig.anchorX = orig.cropX + orig.cropW; // 右边缘固定
                        } else {
                            orig.anchorX = orig.cropX;               // 左边缘固定
                        }
                        if (state.dragMode.includes("N")) {
                            orig.anchorY = orig.cropY + orig.cropH; // 下边缘固定
                        } else {
                            orig.anchorY = orig.cropY;               // 上边缘固定
                        }
                    }
                } else {
                    state.dragMode = "view";
                    state.dragOrigState = { viewX: state.viewX, viewY: state.viewY };
                }
                state.dragStart = sPos;
                e.preventDefault();
            };


            const onMouseMove = (e) => {
                updateCursor(e);
                if (!state.dragMode) return;
                const sPos = getCanvasPos(e);
                const dx = (sPos.x - state.dragStart.x) / state.viewScale;
                const dy = (sPos.y - state.dragStart.y) / state.viewScale;

                if (state.dragMode === "view") {
                    state.viewX = state.dragOrigState.viewX - dx;
                    state.viewY = state.dragOrigState.viewY - dy;
                } else if (state.dragMode === "img") {
                    state.imgX = state.dragOrigState.imgX + dx;
                    state.imgY = state.dragOrigState.imgY + dy;
                } else if (state.dragMode === "cropMove") {
                    state.cropX = state.dragOrigState.cropX + dx;
                    state.cropY = state.dragOrigState.cropY + dy;
                    if (state.snapEnabled) {
                        // 移动时吸附：仅偏移位置，不改变宽高
                        const imgL = state.imgX, imgR = imgL + state.img.width;
                        const imgT = state.imgY, imgB = imgT + state.img.height;
                        const t = getSnapThreshold();
                        let adjX = 0, adjY = 0;
                        const cropR = state.cropX + state.cropW;
                        const cropB = state.cropY + state.cropH;
                        // 检查左边缘
                        if (Math.abs(state.cropX - imgL) < t) adjX = imgL - state.cropX;
                        // 检查右边缘（仅当左边缘未吸附时）
                        else if (Math.abs(cropR - imgR) < t) adjX = imgR - cropR;
                        // 检查上边缘
                        if (Math.abs(state.cropY - imgT) < t) adjY = imgT - state.cropY;
                        // 检查下边缘（仅当上边缘未吸附时）
                        else if (Math.abs(cropB - imgB) < t) adjY = imgB - cropB;
                        state.cropX += adjX;
                        state.cropY += adjY;
                    }
                } else {
                    // resize 模式
                    let { cropX: origX, cropY: origY, cropW: origW, cropH: origH, anchorX, anchorY } = state.dragOrigState;
                    let newX = origX, newY = origY, newW = origW, newH = origH;
                    if (state.dragMode.includes("W")) {
                        newX = origX + dx;
                        newW = origW - dx;
                    }
                    if (state.dragMode.includes("E")) {
                        newW = origW + dx;
                    }
                    if (state.dragMode.includes("N")) {
                        newY = origY + dy;
                        newH = origH - dy;
                    }
                    if (state.dragMode.includes("S")) {
                        newH = origH + dy;
                    }
                    // 最小尺寸保护
                    if (newW < 1) { newW = 1; newX = origX + origW - 1; }
                    if (newH < 1) { newH = 1; newY = origY + origH - 1; }

                    const keepRatio = document.getElementById("keep-ratio").checked && state.aspectRatio;
                    if (keepRatio) {
                        const isCorner = (state.dragMode.includes("W") || state.dragMode.includes("E")) &&
                                        (state.dragMode.includes("N") || state.dragMode.includes("S"));
                        if (isCorner) {
                            // 角拖动：固定对角，按比例缩放
                            const anchorX = state.dragMode.includes("W") ? origX + origW : origX;
                            const anchorY = state.dragMode.includes("N") ? origY + origH : origY;
                            let cornerX, cornerY;
                            if (state.dragMode.includes("W")) cornerX = origX + dx;
                            else cornerX = origX + origW + dx;
                            if (state.dragMode.includes("N")) cornerY = origY + dy;
                            else cornerY = origY + origH + dy;
                            const absW = Math.abs(cornerX - anchorX);
                            const calcW = Math.max(1, absW);
                            const calcH = calcW / state.aspectRatio;
                            const newX = Math.min(anchorX, cornerX);
                            const newY = anchorY < cornerY ? anchorY : anchorY - calcH;
                            state.cropX = newX;
                            state.cropY = newY;
                            state.cropW = calcW;
                            state.cropH = calcH;
                        } else {
                            // 单边拖动：对边固定，按比例改变另一边
                            if (state.dragMode.includes("W") || state.dragMode.includes("E")) {
                                const fixedTop = state.dragMode.includes("N") ? origY + origH : origY;
                                const calcW = Math.max(1, newW);
                                const calcH = calcW / state.aspectRatio;
                                const calcY = state.dragMode.includes("N") ? fixedTop - calcH : fixedTop;
                                state.cropX = newX;
                                state.cropY = calcY;
                                state.cropW = calcW;
                                state.cropH = calcH;
                            } else if (state.dragMode.includes("N") || state.dragMode.includes("S")) {
                                const fixedLeft = state.dragMode.includes("W") ? origX + origW : origX;
                                const calcH = Math.max(1, newH);
                                const calcW = calcH * state.aspectRatio;
                                const calcX = state.dragMode.includes("W") ? fixedLeft - calcW : fixedLeft;
                                state.cropX = calcX;
                                state.cropY = newY;
                                state.cropW = calcW;
                                state.cropH = calcH;
                            }
                        }
                    } else {
                        // 不保持比例
                        state.cropX = newX;
                        state.cropY = newY;
                        state.cropW = newW;
                        state.cropH = newH;
                    }
                    // 吸附
                    if (state.snapEnabled) {
                        const sn = applySnap(state.cropX, state.cropY, state.cropW, state.cropH);
                        state.cropX = sn.x; state.cropY = sn.y; state.cropW = sn.w; state.cropH = sn.h;
                    }
                    // 如果保持比例，吸附后再次修正
                    if (keepRatio) {
                        const adj = enforceAspect(state.cropW, state.cropH, state.aspectRatio);
                        state.cropW = adj.w;
                        state.cropH = adj.h;
                        // 根据当前拖动模式重新对齐锚点（简单处理：保持中心不动）
                        if (state.dragMode && state.dragMode.startsWith("resize")) {
                            // 保持中心位置不变
                            const cx = state.cropX + state.cropW / 2;
                            const cy = state.cropY + state.cropH / 2;
                            state.cropX = cx - state.cropW / 2;
                            state.cropY = cy - state.cropH / 2;
                        }
                    }
                }
                draw();
            };

            const onMouseUp = () => { state.dragMode = null; };

            canvas.addEventListener("mousedown", onMouseDown);
            canvas.addEventListener("mousemove", onMouseMove);
            canvas.addEventListener("mouseup", onMouseUp);
            canvas.addEventListener("mouseleave", onMouseUp);

            // 滚轮缩放
            canvas.addEventListener("wheel", (e) => {
                e.preventDefault();
                const sPos = getCanvasPos(e);
                const wPos = toWorld(sPos.x, sPos.y);
                const factor = e.deltaY < 0 ? 1.1 : 0.9;
                state.viewScale *= factor;
                state.viewScale = Math.min(10, Math.max(0.05, state.viewScale));
                state.viewX = wPos.x - sPos.x / state.viewScale;
                state.viewY = wPos.y - sPos.y / state.viewScale;
                draw();
            }, { passive: false });

            // --- 工具栏按钮 ---
            const swapRatioBtn = document.getElementById("swap-ratio-btn");
            swapRatioBtn.onclick = () => {
                if (!state.img.width) return;
                const cx = state.cropX + state.cropW / 2;
                const cy = state.cropY + state.cropH / 2;
                if (document.getElementById("keep-ratio").checked && state.aspectRatio) {
                    state.aspectRatio = 1 / state.aspectRatio;
                    const sel = document.getElementById("preset-ratio");
                    const swapMap = { "4:3": "3:4", "3:2": "2:3", "16:9": "9:16", "3:4": "4:3", "2:3": "3:2", "9:16": "16:9" };
                    sel.value = swapMap[sel.value] || sel.value;
                }
                [state.cropW, state.cropH] = [state.cropH, state.cropW];
                state.cropX = cx - state.cropW / 2;
                state.cropY = cy - state.cropH / 2;
                draw();
            };

            const ratioSelect = document.getElementById("preset-ratio");
            const keepRatioCheckbox = document.getElementById("keep-ratio");
            ratioSelect.onchange = () => {
                const val = ratioSelect.value;
                if (val === "free") {
                    state.aspectRatio = null;
                } else {
                    state.aspectRatio = parseAspect(val);
                    // 选择预设比例时自动勾选保持比例
                    if (!keepRatioCheckbox.checked) keepRatioCheckbox.checked = true;
                }
                if (keepRatioCheckbox.checked && state.aspectRatio) {
                    const adj = enforceAspect(state.cropW, state.cropH, state.aspectRatio);
                    state.cropW = adj.w;
                    state.cropH = adj.h;
                }
                draw();
            };
            keepRatioCheckbox.onchange = () => {
                if (keepRatioCheckbox.checked) {
                    // 勾选时：若当前为自由模式，锁定当前裁剪框比例
                    if (ratioSelect.value === "free") {
                        if (state.cropW > 0 && state.cropH > 0) {
                            state.aspectRatio = state.cropW / state.cropH;
                        }
                    } else {
                        state.aspectRatio = parseAspect(ratioSelect.value);
                    }
                } else {
                    // 取消勾选则切回自由模式
                    ratioSelect.value = "free";
                    state.aspectRatio = null;
                }
                draw();
            };

            state.aspectRatio = keepRatioCheckbox.checked
                ? (ratioSelect.value === "free"
                    ? state.cropW / state.cropH
                    : parseAspect(ratioSelect.value))
                : null;

            document.getElementById("crop-w-input").onchange = (e) => {
                let w = Math.max(1, Number(e.target.value));
                if (state.aspectRatio && keepRatioCheckbox.checked) {
                    state.cropW = w;
                    state.cropH = w / state.aspectRatio;
                } else {
                    state.cropW = w;
                }
                draw();
            };
            document.getElementById("crop-h-input").onchange = (e) => {
                let h = Math.max(1, Number(e.target.value));
                if (state.aspectRatio && keepRatioCheckbox.checked) {
                    state.cropH = h;
                    state.cropW = h * state.aspectRatio;
                } else {
                    state.cropH = h;
                }
                draw();
            };

            const fillSelect = document.getElementById("fill-select");
            const customColorInput = document.getElementById("custom-color");
            const fillPreview = document.getElementById("fill-preview");
            
            // 更新预览颜色
            const updateFillPreview = () => {
                const val = fillSelect.value;
                const showCustom = val === "custom";
                customColorInput.style.display = showCustom ? "inline" : "none";
                fillPreview.style.display = showCustom ? "inline" : "none";
                if (showCustom) {
                    fillPreview.style.background = customColorInput.value;
                } else {
                    const mapColor = { black: "#000000", white: "#ffffff", gray: "#888888" };
                    fillPreview.style.background = mapColor[val] || "#000";
                }
                // 始终显示预览
                fillPreview.style.display = "inline";
            };
            customColorInput.oninput = () => {
                fillPreview.style.background = customColorInput.value;
            };
            fillSelect.onchange = updateFillPreview;
            updateFillPreview();  // 初始执行
            state.snapEnabled = (document.getElementById("snap-enable").checked === true);
            document.getElementById("snap-enable").onchange = (e) => { state.snapEnabled = e.target.checked; };

            // 重置
            document.getElementById("crop-reset-btn").onclick = () => {
                if (!state.img.width) return;
                state.cropX = state.imgX;
                state.cropY = state.imgY;
                state.cropW = state.img.width;
                state.cropH = state.img.height;
                if (state.aspectRatio && keepRatioCheckbox.checked) {
                    const adj = enforceAspect(state.cropW, state.cropH, state.aspectRatio);
                    state.cropW = adj.w;
                    state.cropH = adj.h;
                }
                draw();
            };

            // 图片加载与缩略图
            const loadImage = async (previewIdx) => {
                let src;
                const info = state.sourceInfo;
                if (info.type === "tensor") {
                    const resp = await fetch(`/interactive_crop/get_tensor_preview?node_id=${info.nodeId}&index=${previewIdx}`);
                    const data = await resp.json();
                    if (data.error) { alert(data.error); return; }
                    src = data.image;
                } else if (info.type === "files") {
                    const realIdx = info.previewIndices[previewIdx];
                    const filePath = info.paths[realIdx];
                    const resp = await fetch(`/interactive_crop/get_image_by_path?path=${encodeURIComponent(filePath)}`);
                    const data = await resp.json();
                    if (data.error) { alert(data.error); return; }
                    src = data.image;
                } else { // folder
                    const realIdx = info.previewIndices[previewIdx];
                    const sizeFilter = info.sizeFilter || "";
                    const resp = await fetch(`/interactive_crop/get_folder_preview?folder=${encodeURIComponent(info.folderPath)}&index=${realIdx}&size=${sizeFilter}`);
                    const data = await resp.json();
                    if (data.error) { alert(data.error); return; }
                    src = data.image;
                }

                state.img = new Image();
                state.img.onload = () => {
                    state.currentIndex = previewIdx;

                    // 首次加载时确定裁剪框位置
                    if (!state.initialized) {
                        state.initialized = true;
                        if (state.lastParams && state.lastParams.img_x !== undefined) {
                            state.imgX = state.lastParams.img_x;
                            state.imgY = state.lastParams.img_y;
                            state.cropX = state.lastParams.crop_x;
                            state.cropY = state.lastParams.crop_y;
                            state.cropW = state.lastParams.crop_w;
                            state.cropH = state.lastParams.crop_h;
                            state.lastParams = null;

                            // 自动调整视口使裁剪框完全可见
                            const cw = canvas.width, ch = canvas.height;
                            const fitScale = Math.min(cw / state.cropW, ch / state.cropH, 1) * 0.9;
                            state.viewScale = fitScale;
                            state.viewX = state.cropX + state.cropW / 2 - cw / (2 * fitScale);
                            state.viewY = state.cropY + state.cropH / 2 - ch / (2 * fitScale);
                        } else {
                            const cw = canvas.width, ch = canvas.height;
                            const imgW = state.img.width, imgH = state.img.height;
                            const fitScale = Math.min(cw / imgW, ch / imgH, 1) * 0.9;
                            state.viewScale = fitScale;
                            state.viewX = imgW / 2 - cw / (2 * fitScale);
                            state.viewY = imgH / 2 - ch / (2 * fitScale);
                            state.imgX = 0;
                            state.imgY = 0;
                            state.cropX = 0;
                            state.cropY = 0;
                            state.cropW = imgW;
                            state.cropH = imgH;
                            if (state.aspectRatio && keepRatioCheckbox.checked) {
                                const adj = enforceAspect(state.cropW, state.cropH, state.aspectRatio);
                                state.cropW = adj.w;
                                state.cropH = adj.h;
                            }
                        }
                    }
                    // 非首次加载时，裁剪框保持不变，只更新图片绘制
                    draw();
                    updateThumbActive();
                    document.getElementById("idx-text").textContent = `${previewIdx + 1} / ${state.displayTotal}`;
                };
                state.img.src = src;
            };

            const updateThumbActive = () => {
                const thumbs = document.querySelectorAll("#thumb-container img");
                thumbs.forEach((img, i) => img.classList.toggle("active", i === state.currentIndex));
            };

            const loadThumbnails = async () => {
                const container = document.getElementById("thumb-container");
                if (!container) return;
                container.innerHTML = "";
                const info = state.sourceInfo;
                if (info.type === "tensor") return;
                const total = state.displayTotal;
                const indices = info.previewIndices || Array.from({ length: total }, (_, i) => i);
                for (let i = 0; i < indices.length; i++) {
                    const idx = indices[i];
                    let thumbSrc;
                    try {
                        if (info.type === "files") {
                            const path = info.paths[idx];
                            const resp = await fetch(`/interactive_crop/get_image_by_path?path=${encodeURIComponent(path)}`);
                            const data = await resp.json();
                            if (data.error) continue;
                            const img = new Image();
                            img.src = data.image;
                            await new Promise(r => img.onload = r);
                            const cvs = document.createElement("canvas");
                            cvs.width = 50; cvs.height = 50;
                            const ctx2 = cvs.getContext("2d");
                            const scale = Math.min(cvs.width / img.width, cvs.height / img.height);
                            const w = img.width * scale;
                            const h = img.height * scale;
                            const x = (cvs.width - w) / 2;
                            const y = (cvs.height - h) / 2;
                            ctx2.drawImage(img, x, y, w, h);
                            thumbSrc = cvs.toDataURL("image/png");
                        } else {
                            const sizeFilter = info.sizeFilter || "";
                            const resp = await fetch(`/interactive_crop/get_folder_preview?folder=${encodeURIComponent(info.folderPath)}&index=${idx}&size=${sizeFilter}`);
                            const data = await resp.json();
                            if (data.error) continue;
                            const img = new Image();
                            img.src = data.image;
                            await new Promise(r => img.onload = r);
                            const cvs = document.createElement("canvas");
                            cvs.width = 50; cvs.height = 50;
                            const ctx2 = cvs.getContext("2d");
                            const scale = Math.min(cvs.width / img.width, cvs.height / img.height);
                            const w = img.width * scale;
                            const h = img.height * scale;
                            const x = (cvs.width - w) / 2;
                            const y = (cvs.height - h) / 2;
                            ctx2.drawImage(img, x, y, w, h);
                            thumbSrc = cvs.toDataURL("image/png");
                        }
                    } catch { continue; }
                    const thumbImg = document.createElement("img");
                    thumbImg.src = thumbSrc;
                    thumbImg.onclick = () => loadImage(i);
                    container.appendChild(thumbImg);
                }
            };
            loadThumbnails();

            // 左右按钮
            const prevBtn = document.getElementById("prev-btn");
            const nextBtn = document.getElementById("next-btn");
            prevBtn.onclick = () => { if (state.currentIndex > 0) loadImage(state.currentIndex - 1); };
            nextBtn.onclick = () => { if (state.currentIndex < state.displayTotal - 1) loadImage(state.currentIndex + 1); };

            // 键盘左右键（阻止冒泡到画布）
            overlay.addEventListener("keydown", (e) => {
                if (e.key === "ArrowLeft") {
                    prevBtn.click();
                    e.preventDefault();
                    e.stopPropagation();
                } else if (e.key === "ArrowRight") {
                    nextBtn.click();
                    e.preventDefault();
                    e.stopPropagation();
                }
            }, true);

            // 确认/取消
            document.getElementById("crop-confirm-btn").onclick = () => {
                const fillColor = document.getElementById("fill-select").value === "custom" ?
                    document.getElementById("custom-color").value : document.getElementById("fill-select").value;
                const params = {
                    crop_x: state.cropX,
                    crop_y: state.cropY,
                    crop_w: state.cropW,
                    crop_h: state.cropH,
                    img_x: state.imgX,
                    img_y: state.imgY,
                    fill_color: fillColor
                };
                if (this._cropParamsWidget) {
                    this._cropParamsWidget.value = JSON.stringify(params);
                    app.graph.setDirtyCanvas(true, true);
                }
                overlay.remove();
            };
            document.getElementById("crop-cancel-btn").onclick = () => overlay.remove();

            // 启动加载第一张
            loadImage(0);
        };
    }
});
