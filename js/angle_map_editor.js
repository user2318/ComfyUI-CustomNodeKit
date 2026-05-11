import { app } from "../../scripts/app.js";

// ==================== 全局样式 (只注入一次) ====================
if (!window.__angleMapEditorStyleInjected) {
    window.__angleMapEditorStyleInjected = true;
    const style = document.createElement("style");
    style.textContent = `
        #angle-map-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.92); z-index: 10000;
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; font-family: Arial, sans-serif;
            outline: none; user-select: none;
        }
        .angle-map-toolbar {
            display: flex; align-items: center; gap: 8px;
            margin-bottom: 8px; background: #222; padding: 6px 14px;
            border-radius: 6px; color: white; font-size: 14px;
        }
        .angle-map-toolbar button {
            background: #444; color: white; border: none;
            padding: 4px 12px; border-radius: 4px; cursor: pointer;
            font-size: 14px;
        }
        .angle-map-toolbar button:hover { background: #666; }
        #angle-confirm-btn { background: #2a7; }
        #angle-cancel-btn { background: #a33; }
        #angle-reset-btn { background: #555; }
        .angle-canvas-container {
            position: relative; display: inline-block;
            background: #111; border: 1px solid #333; border-radius: 8px;
        }
        #angle-canvas { display: block; }
    `;
    document.head.appendChild(style);
}

// ==================== 共享画布函数 ====================
function showAngleMapCanvas(pointCount, existingAngles, onConfirm) {
    // 移除已有遮罩
    let overlay = document.getElementById("angle-map-overlay");
    if (overlay) overlay.remove();

    overlay = document.createElement("div");
    overlay.id = "angle-map-overlay";
    overlay.setAttribute("tabindex", "-1");
    document.body.appendChild(overlay);
    overlay.focus();

    overlay.innerHTML = `
        <div class="angle-map-toolbar">
            <button id="angle-confirm-btn">✅ 确认</button>
            <button id="angle-cancel-btn">❌ 取消</button>
            <button id="angle-reset-btn">🔄 重置均匀分布</button>
            <span style="margin-left:8px;color:#888;">
                底部=0°(正面) 顶部=±180°(背面) 左=-90° 右=90°
            </span>
            <span id="angle-count-label" style="color:#aaa; margin-left:8px;">点数: ${pointCount}</span>
        </div>
        <div class="angle-canvas-container">
            <canvas id="angle-canvas"></canvas>
        </div>
    `;

    const canvas = document.getElementById("angle-canvas");
    const ctx = canvas.getContext("2d");

    // 椭圆参数
    const cx = 300, cy = 260;
    const rx = 140, ry = 110;
    canvas.width = 600;
    canvas.height = 520;

    // 角度归一化到 (-180, 180]
    const normalizeAngle = (a) => {
        a = a % 360;
        if (a > 180) return a - 360;
        if (a <= -180) return a + 360;
        return a;
    };

    // 初始化点（逆时针均匀分布，0° 在底部）
    let points = [];
    if (existingAngles && existingAngles.length === pointCount) {
        points = existingAngles.map(a => normalizeAngle(a));
    } else {
        for (let i = 0; i < pointCount; i++) {
            points.push(normalizeAngle(i * (360 / pointCount)));
        }
    }

    let dragIdx = -1;

    const angleToPos = (angleDeg) => {
        const rad = (angleDeg * Math.PI) / 180;
        const x = cx + rx * Math.sin(rad);
        const y = cy + ry * Math.cos(rad);
        return { x, y };
    };

    const posToAngle = (px, py) => {
        const dx = px - cx;
        const dy = py - cy;
        const rad = Math.atan2(dx, dy);
        return (rad * 180) / Math.PI;
    };

    const getNearestPointIdx = (px, py) => {
        let minDist = 25;
        let idx = -1;
        for (let i = 0; i < points.length; i++) {
            const pos = angleToPos(points[i]);
            const d = Math.hypot(px - pos.x, py - pos.y);
            if (d < minDist) { minDist = d; idx = i; }
        }
        return idx;
    };

    const draw = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 画椭圆参考
        ctx.strokeStyle = "#444";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        ctx.stroke();

        // 画十字参考线
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(cx - rx - 20, cy);
        ctx.lineTo(cx + rx + 20, cy);
        ctx.moveTo(cx, cy - ry - 20);
        ctx.lineTo(cx, cy + ry + 20);
        ctx.stroke();
        ctx.setLineDash([]);

        // 画方向标签
        ctx.fillStyle = "#666";
        ctx.font = "12px Arial";
        ctx.textAlign = "center";
        ctx.fillText("0° (正面)", cx, cy + ry + 20);
        ctx.fillText("±180° (背面)", cx, cy - ry - 10);
        ctx.fillText("-90°", cx - rx - 25, cy + 4);
        ctx.fillText("+90°", cx + rx + 25, cy + 4);

        // 画连接线
        for (let i = 0; i < points.length; i++) {
            const pos = angleToPos(points[i]);
            ctx.strokeStyle = "rgba(255,255,255,0.15)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
        }

        // 画点
        for (let i = 0; i < points.length; i++) {
            const pos = angleToPos(points[i]);
            const isDragging = (i === dragIdx);

            ctx.beginPath();
            ctx.arc(pos.x, pos.y, isDragging ? 9 : 7, 0, Math.PI * 2);
            ctx.fillStyle = isDragging ? "#ff6600" : "#4af";
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.stroke();

            // 索引标签
            ctx.fillStyle = "#fff";
            ctx.font = "bold 11px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "bottom";
            const labelOffset = isDragging ? 14 : 12;
            ctx.fillText(i.toString(), pos.x, pos.y - labelOffset);
        }
    };

    // 鼠标事件
    const onMouseDown = (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        dragIdx = getNearestPointIdx(mx, my);
        if (dragIdx >= 0) {
            canvas.style.cursor = "grabbing";
            draw();
        }
        e.preventDefault();
    };

    const onMouseMove = (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        if (dragIdx >= 0) {
            const newAngle = posToAngle(mx, my);
            points[dragIdx] = newAngle;
            draw();
        } else {
            const near = getNearestPointIdx(mx, my);
            canvas.style.cursor = near >= 0 ? "grab" : "default";
        }
    };

    const onMouseUp = () => {
        dragIdx = -1;
        canvas.style.cursor = "default";
        draw();
    };

    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("mouseleave", onMouseUp);

    // 确认按钮
    document.getElementById("angle-confirm-btn").onclick = () => {
        const angles = points.map(a => Math.round(a * 10) / 10);
        onConfirm(JSON.stringify(angles));
        overlay.remove();
    };

    // 取消按钮
    document.getElementById("angle-cancel-btn").onclick = () => {
        overlay.remove();
    };

    // 重置按钮（0° 在底部，逆时针均匀分布）
    document.getElementById("angle-reset-btn").onclick = () => {
        for (let i = 0; i < pointCount; i++) {
            points[i] = normalizeAngle(i * (360 / pointCount));
        }
        draw();
    };

    // 键盘操作
    overlay.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            document.getElementById("angle-confirm-btn").click();
        } else if (e.key === "Escape") {
            document.getElementById("angle-cancel-btn").click();
        }
    }, true);

    draw();
}

// ==================== 为 ReferenceImageSelector 注册扩展 ====================
app.registerExtension({
    name: "AngleMapEditor.ReferenceImageSelector",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ReferenceImageSelector") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // 找到 angle_map widget 并添加按钮
            const angleMapWidget = this.widgets.find(w => w.name === "angle_map");
            if (angleMapWidget) {
                const btnContainer = document.createElement("div");
                btnContainer.style.cssText = "display:flex; gap:2px; margin-top:2px;";

                const openBtn = document.createElement("button");
                openBtn.textContent = "🎯 角度映射";
                openBtn.style.cssText =
                    "height:24px; padding:2px 10px; font-size:12px;" +
                    "background:var(--comfy-input-bg,#222); color:var(--input-text,white);" +
                    "border:1px solid var(--border-color,#444); border-radius:3px; cursor:pointer; flex:1;";
                openBtn.onclick = () => { this.openAngleMapUI(); };

                const clearBtn = document.createElement("button");
                clearBtn.textContent = "清空";
                clearBtn.style.cssText =
                    "height:24px; padding:2px 8px; font-size:12px;" +
                    "background:#a33; color:white;" +
                    "border:none; border-radius:3px; cursor:pointer;";
                clearBtn.onclick = () => {
                    angleMapWidget.value = "";
                    app.graph.setDirtyCanvas(true, true);
                };

                btnContainer.appendChild(openBtn);
                btnContainer.appendChild(clearBtn);

                const inputEl = angleMapWidget.inputEl || angleMapWidget.element?.querySelector("input");
                if (inputEl && inputEl.parentNode) {
                    inputEl.style.marginBottom = "2px";
                    inputEl.parentNode.insertBefore(btnContainer, inputEl.nextSibling);
                } else {
                    try {
                        this.addDOMWidget("angle_map_actions", "custom", btnContainer, { serialize: false });
                    } catch (e) {
                        console.debug("[AngleMapEditor] addDOMWidget failed:", e);
                    }
                }
            }

            return result;
        };

        // 尝试从已连接的参考图输入获取张数
        nodeType.prototype._getReferenceImageCount = function () {
            const refInput = this.inputs?.find(inp => inp.name === "reference_images");
            if (refInput && refInput.link !== undefined && refInput.link !== null) {
                const linkId = refInput.link;
                const linkInfo = app.graph.links?.find ? app.graph.links.find(l => {
                    const id = typeof l === 'object' ? l.id : l;
                    return id === linkId;
                }) : null;
                if (linkInfo) {
                    const originId = typeof linkInfo === 'object' ? linkInfo.origin_id : linkInfo[1];
                    const originNode = app.graph.getNodeById(originId);
                    if (originNode && originNode.outputs) {
                        const originSlot = typeof linkInfo === 'object' ? linkInfo.origin_slot : linkInfo[2];
                        const output = originNode.outputs[originSlot];
                        if (output && output.shape && output.shape.length >= 4) {
                            const count = output.shape[0];
                            if (count > 0) return count;
                        }
                    }
                }
            }
            return null;
        };

        // 打开角度映射编辑器
        nodeType.prototype.openAngleMapUI = function () {
            const angleMapWidget = this.widgets.find(w => w.name === "angle_map");

            // 读取当前已有角度映射
            let existingAngles = null;
            if (angleMapWidget && angleMapWidget.value) {
                try { existingAngles = JSON.parse(angleMapWidget.value); } catch {}
            }

            // 多级兜底获取张数
            let pointCount = null;

            // 1. 从已运行的 info 输出解析（运行一次后就有）
            if (this.outputs && this.outputs[1]) {
                const match = String(this.outputs[1]).match(/参考图总数:\s*(\d+)/);
                if (match) pointCount = parseInt(match[1]);
            }
            // 2. 图结构检测兜底
            if (pointCount === null) pointCount = this._getReferenceImageCount();
            // 3. 已有 angle_map 长度兜底
            if (pointCount === null && existingAngles && existingAngles.length > 0) {
                pointCount = existingAngles.length;
            }
            // 4. 最后提示手动输入
            if (pointCount === null) {
                const countStr = prompt("无法自动获取参考图数量，请手动输入:", "2");
                if (!countStr) return;
                pointCount = parseInt(countStr);
                if (isNaN(pointCount) || pointCount < 1) { alert("请输入有效的正整数"); return; }
            }

            showAngleMapCanvas(pointCount, existingAngles, (jsonStr) => {
                if (angleMapWidget) {
                    angleMapWidget.value = jsonStr;
                    app.graph.setDirtyCanvas(true, true);
                }
            });
        };
    }
});

// ==================== 为 WanAnimateToVideoCustom 注册扩展 ====================
app.registerExtension({
    name: "AngleMapEditor.WanAnimateToVideoCustom",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WanAnimateToVideoCustom") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const angleMapWidget = this.widgets.find(w => w.name === "angle_map");
            if (angleMapWidget) {
                const btnContainer = document.createElement("div");
                btnContainer.style.cssText = "display:flex; gap:2px; margin-top:2px;";

                const openBtn = document.createElement("button");
                openBtn.textContent = "🎯 角度映射";
                openBtn.style.cssText =
                    "height:24px; padding:2px 10px; font-size:12px;" +
                    "background:var(--comfy-input-bg,#222); color:var(--input-text,white);" +
                    "border:1px solid var(--border-color,#444); border-radius:3px; cursor:pointer; flex:1;";
                openBtn.onclick = () => { this.openAngleMapUI(); };

                const clearBtn = document.createElement("button");
                clearBtn.textContent = "清空";
                clearBtn.style.cssText =
                    "height:24px; padding:2px 8px; font-size:12px;" +
                    "background:#a33; color:white;" +
                    "border:none; border-radius:3px; cursor:pointer;";
                clearBtn.onclick = () => {
                    angleMapWidget.value = "";
                    app.graph.setDirtyCanvas(true, true);
                };

                btnContainer.appendChild(openBtn);
                btnContainer.appendChild(clearBtn);

                const inputEl = angleMapWidget.inputEl || angleMapWidget.element?.querySelector("input");
                if (inputEl && inputEl.parentNode) {
                    inputEl.style.marginBottom = "2px";
                    inputEl.parentNode.insertBefore(btnContainer, inputEl.nextSibling);
                } else {
                    try {
                        this.addDOMWidget("angle_map_actions", "custom", btnContainer, { serialize: false });
                    } catch (e) {
                        console.debug("[AngleMapEditor] addDOMWidget failed:", e);
                    }
                }
            }

            return result;
        };

        // 打开角度映射编辑器（WanAnimateToVideoCustom 始终手动输入点数）
        nodeType.prototype.openAngleMapUI = function () {
            const angleMapWidget = this.widgets.find(w => w.name === "angle_map");

            // 读取当前已有角度映射
            let existingAngles = null;
            if (angleMapWidget && angleMapWidget.value) {
                try { existingAngles = JSON.parse(angleMapWidget.value); } catch {}
            }

            // 已有 angle_map 时自动检测数量，跳过手动输入
            let pointCount;
            if (existingAngles && Array.isArray(existingAngles) && existingAngles.length > 0) {
                pointCount = existingAngles.length;
            } else {
                const countStr = prompt("参考图张数 (reference_image 批次大小):", "2");
                if (!countStr) return;
                pointCount = parseInt(countStr);
                if (isNaN(pointCount) || pointCount < 1) { alert("请输入有效的正整数"); return; }
            }

            showAngleMapCanvas(pointCount, existingAngles, (jsonStr) => {
                if (angleMapWidget) {
                    angleMapWidget.value = jsonStr;
                    app.graph.setDirtyCanvas(true, true);
                }
            });
        };
    }
});
