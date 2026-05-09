import { app } from "../../scripts/app.js";

// ==================== 全局样式 (只注入一次) ====================
if (!window.__refImgSelectorStyleInjected) {
    window.__refImgSelectorStyleInjected = true;
    const style = document.createElement("style");
    style.textContent = `
        #ref-img-angle-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.92); z-index: 10000;
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; font-family: Arial, sans-serif;
            outline: none; user-select: none;
        }
        .ref-img-toolbar {
            display: flex; align-items: center; gap: 8px;
            margin-bottom: 8px; background: #222; padding: 6px 14px;
            border-radius: 6px; color: white; font-size: 14px;
        }
        .ref-img-toolbar button {
            background: #444; color: white; border: none;
            padding: 4px 12px; border-radius: 4px; cursor: pointer;
            font-size: 14px;
        }
        .ref-img-toolbar button:hover { background: #666; }
        #ref-img-confirm-btn { background: #2a7; }
        #ref-img-cancel-btn { background: #a33; }
        #ref-img-reset-btn { background: #555; }
        .ref-img-canvas-container {
            position: relative; display: inline-block;
            background: #111; border: 1px solid #333; border-radius: 8px;
        }
        #ref-img-canvas { display: block; }
    `;
    document.head.appendChild(style);
}

// ==================== 节点扩展 ====================
app.registerExtension({
    name: "ReferenceImageSelector.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ReferenceImageSelector") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // 找到 angle_map widget
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
                        console.debug("[RefImgSelector] addDOMWidget failed:", e);
                    }
                }
            }

            return result;
        };

        // 尝试从已连接的参考图输入获取图像张数
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
            // 降级：尝试从已有 angle_map 值推断
            const angleMapWidget = this.widgets.find(w => w.name === "angle_map");
            if (angleMapWidget && angleMapWidget.value) {
                try {
                    const arr = JSON.parse(angleMapWidget.value);
                    if (Array.isArray(arr) && arr.length > 0) return arr.length;
                } catch {}
            }
            return null;
        };

        // 打开角度映射编辑器
        nodeType.prototype.openAngleMapUI = function () {
            // 先尝试从已连接的参考图输入自动获取数量
            let pointCount = this._getReferenceImageCount();
            if (pointCount === null) {
                const countStr = prompt("无法自动获取参考图数量，请手动输入:", "2");
                if (!countStr) return;
                pointCount = parseInt(countStr);
                if (isNaN(pointCount) || pointCount < 1) { alert("请输入有效的正整数"); return; }
            }

            // 读取当前已有角度映射
            const angleMapWidget = this.widgets.find(w => w.name === "angle_map");
            let existingAngles = null;
            if (angleMapWidget && angleMapWidget.value) {
                try { existingAngles = JSON.parse(angleMapWidget.value); } catch {}
            }

            this._showAngleMapCanvas(pointCount, existingAngles, angleMapWidget);
        };

        // ==================== 角度映射画布 ====================
        nodeType.prototype._showAngleMapCanvas = function (pointCount, existingAngles, angleMapWidget) {
            // 移除已有遮罩
            let overlay = document.getElementById("ref-img-angle-overlay");
            if (overlay) overlay.remove();

            overlay = document.createElement("div");
            overlay.id = "ref-img-angle-overlay";
            overlay.setAttribute("tabindex", "-1");
            document.body.appendChild(overlay);
            overlay.focus();

            overlay.innerHTML = `
                <div class="ref-img-toolbar">
                    <button id="ref-img-confirm-btn">✅ 确认</button>
                    <button id="ref-img-cancel-btn">❌ 取消</button>
                    <button id="ref-img-reset-btn">🔄 重置均匀分布</button>
                    <span style="margin-left:8px;color:#888;">
                        底部=0°(正面) 顶部=±180°(背面) 左=-90° 右=90°
                    </span>
                    <span id="ref-img-count-label" style="color:#aaa; margin-left:8px;">点数: ${pointCount}</span>
                </div>
                <div class="ref-img-canvas-container">
                    <canvas id="ref-img-canvas"></canvas>
                </div>
            `;

            const canvas = document.getElementById("ref-img-canvas");
            const ctx = canvas.getContext("2d");

            // 椭圆参数
            const cx = 300, cy = 260;
            const rx = 140, ry = 110;
            canvas.width = 600;
            canvas.height = 520;

            // 角度: 底部=0°(正面), 顺时针正, 逆时针负
            let points = [];
            if (existingAngles && existingAngles.length === pointCount) {
                points = existingAngles.map(a => a);
            } else {
                // 默认均匀分布: 从 -180 到 180, 首点靠近 -180
                for (let i = 0; i < pointCount; i++) {
                    const frac = i / pointCount;
                    const angle = frac * 360 - 180;
                    points.push(Math.round(angle * 10) / 10);
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
                let deg = (rad * 180) / Math.PI;
                return deg;
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

                // 椭圆参考
                ctx.strokeStyle = "#444";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
                ctx.stroke();

                // 十字虚线
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

                // 方向标签
                ctx.fillStyle = "#666";
                ctx.font = "12px Arial";
                ctx.textAlign = "center";
                ctx.fillText("0° (正面)", cx, cy + ry + 20);
                ctx.fillText("±180° (背面)", cx, cy - ry - 10);
                ctx.fillText("-90°", cx - rx - 25, cy + 4);
                ctx.fillText("+90°", cx + rx + 25, cy + 4);

                // 连接线
                for (let i = 0; i < points.length; i++) {
                    const pos = angleToPos(points[i]);
                    ctx.strokeStyle = "rgba(255,255,255,0.12)";
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
                    points[dragIdx] = Math.round(newAngle * 10) / 10;
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
            document.getElementById("ref-img-confirm-btn").onclick = () => {
                const jsonStr = JSON.stringify(points.map(a => Math.round(a * 10) / 10));
                if (angleMapWidget) {
                    angleMapWidget.value = jsonStr;
                    app.graph.setDirtyCanvas(true, true);
                }
                overlay.remove();
            };

            // 取消按钮
            document.getElementById("ref-img-cancel-btn").onclick = () => {
                overlay.remove();
            };

            // 重置按钮
            document.getElementById("ref-img-reset-btn").onclick = () => {
                for (let i = 0; i < pointCount; i++) {
                    const frac = i / pointCount;
                    points[i] = Math.round((frac * 360 - 180) * 10) / 10;
                }
                draw();
            };

            // 键盘操作
            overlay.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    document.getElementById("ref-img-confirm-btn").click();
                } else if (e.key === "Escape") {
                    document.getElementById("ref-img-cancel-btn").click();
                }
            }, true);

            draw();
        };
    }
});
