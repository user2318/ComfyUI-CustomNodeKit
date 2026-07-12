import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.IntegerSettingNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "IntegerSettingNode") return;

        // ===== 对齐函数 =====
        const alignValue = (v, startVal, stepVal) => {
            if (stepVal <= 0) return startVal;
            const n = (v - startVal) / stepVal;
            const nFloor = Math.floor(n);
            const nCeil = nFloor + 1;
            const vFloor = startVal + nFloor * stepVal;
            const vCeil = startVal + nCeil * stepVal;
            const diffFloor = Math.abs(v - vFloor);
            const diffCeil = Math.abs(v - vCeil);
            if (diffFloor < diffCeil) return vFloor;
            if (diffCeil < diffFloor) return vCeil;
            if (Math.abs(vFloor - startVal) <= Math.abs(vCeil - startVal)) return vFloor;
            return vCeil;
        };

        // ===== 自定义 Canvas 控件 =====
        const SPINNER_MARGIN = 3;
        const SPINNER_GAP = 4;
        const ARROW_W = 8;
        const INLINE_H = 26;
        const BG_H = 18;
        const LABEL_H = 8;

        class InlineIntegerWidget {
            constructor(node, startWidget, stepWidget, valueWidget) {
                this.type = "custom";
                this.name = "__inline_integer";
                this.options = { serialize: false };
                this.node = node;
                this.startWidget = startWidget;
                this.stepWidget = stepWidget;
                this.valueWidget = valueWidget;

                this.y = 0;
                this.mouseDowned = false;
                this.mouseDownPos = [0, 0];
                this.dragTarget = null;
                this.dragOrigValue = 0;
                this.hoverTarget = null;
                this.repeatTimer = null;
                this.repeatInterval = 150;
                this.repeatAccel = 0;
                this.lastDx = 0;

                this.fields = [
                    { name: "start", label: "Start", get: () => startWidget.value, set: (v) => { startWidget.value = v; } },
                    { name: "step",  label: "Step", get: () => stepWidget.value,  set: (v) => { stepWidget.value = v; } },
                    { name: "value", label: "Value", get: () => valueWidget.value, set: (v) => { valueWidget.value = v; } },
                ];
            }

            getSpinnerRects(width) {
                const totalW = width - SPINNER_MARGIN * 2;
                const eachW = (totalW - SPINNER_GAP * 2) / 3;
                const rects = [];
                for (let i = 0; i < 3; i++) {
                    const x = SPINNER_MARGIN + i * (eachW + SPINNER_GAP);
                    rects.push({ x, w: eachW, centerX: x + eachW / 2 });
                }
                return rects;
            }

            drawSpinner(ctx, rect, y, field, isHover) {
                const val = field.get();
                const text = String(val);
                const label = field.label;

                const topMargin = (INLINE_H - BG_H - LABEL_H) / 2;
                const bgY = y + topMargin;

                ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
                ctx.strokeStyle = isHover ? "#6b9fff" : LiteGraph.WIDGET_OUTLINE_COLOR;
                ctx.lineWidth = isHover ? 1.5 : 1;
                ctx.beginPath();
                ctx.roundRect(rect.x, bgY, rect.w, BG_H, [4]);
                ctx.fill();
                ctx.stroke();

                const midY = bgY + BG_H / 2;
                const arrowH = 8;

                ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                ctx.beginPath();
                ctx.moveTo(rect.x + 4, midY);
                ctx.lineTo(rect.x + 4 + ARROW_W, midY - arrowH / 2);
                ctx.lineTo(rect.x + 4 + ARROW_W, midY + arrowH / 2);
                ctx.closePath();
                ctx.fill();

                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                const fontSize = Math.min(11, rect.w * 0.24);
                ctx.font = `bold ${fontSize}px monospace`;
                ctx.fillText(text, rect.centerX, midY);

                const arrowRightX = rect.x + rect.w - 4 - ARROW_W;
                ctx.beginPath();
                ctx.moveTo(arrowRightX + ARROW_W, midY);
                ctx.lineTo(arrowRightX, midY - arrowH / 2);
                ctx.lineTo(arrowRightX, midY + arrowH / 2);
                ctx.closePath();
                ctx.fill();

                ctx.textAlign = "center";
                ctx.textBaseline = "top";
                ctx.font = `${fontSize}px sans-serif`;
                ctx.fillStyle = "#999";
                ctx.fillText(label, rect.centerX, bgY + BG_H + 2);
            }

            computeSize(width) {
                return [width, INLINE_H];
            }

            draw(ctx, node, width, y, height) {
                this.y = y;
                const rects = this.getSpinnerRects(width);
                for (let i = 0; i < 3; i++) {
                    const isHover = this.hoverTarget && this.hoverTarget.field === this.fields[i].name;
                    this.drawSpinner(ctx, rects[i], y, this.fields[i], isHover);
                }
            }

            getHitTarget(pos, width) {
                const rects = this.getSpinnerRects(width);
                const px = pos[0];
                const localY = pos[1] - this.y;
                const topMargin = (INLINE_H - BG_H - LABEL_H) / 2;
                const bgStartY = topMargin;
                const bgEndY = topMargin + BG_H;

                for (let i = 0; i < 3; i++) {
                    const r = rects[i];
                    if (px >= r.x && px <= r.x + r.w && localY >= bgStartY && localY <= bgEndY) {
                        const field = this.fields[i];
                        if (px < r.x + r.w * 0.22) {
                            return { field: field.name, target: "arrow-down", delta: -1 };
                        }
                        if (px > r.x + r.w * 0.78) {
                            return { field: field.name, target: "arrow-up", delta: 1 };
                        }
                        return { field: field.name, target: "value" };
                    }
                }
                return null;
            }

            getField(name) {
                return this.fields.find(f => f.name === name);
            }

            applyDelta(fieldName, delta) {
                const f = this.getField(fieldName);
                if (!f) return;
                const oldVal = f.get();
                let newVal;

                if (fieldName === "step") {
                    const stepDelta = Math.max(1, Math.floor(Math.abs(oldVal) * 0.1));
                    newVal = Math.max(1, oldVal + delta * stepDelta);
                } else if (fieldName === "start") {
                    newVal = oldVal + delta;
                } else {
                    newVal = oldVal + delta * this.stepWidget.value;
                }

                newVal = Math.max(-99999999, Math.min(99999999, newVal));
                f.set(newVal);
                this.node.graph.setDirtyCanvas(true, true);

                if (this.repeatTimer) {
                    clearTimeout(this.repeatTimer);
                }
                if (delta !== 0) {
                    this.repeatTimer = setTimeout(() => {
                        this.startRepeat(fieldName, delta);
                    }, 300);
                }
            }

            startRepeat(fieldName, delta) {
                if (this.repeatTimer) {
                    clearInterval(this.repeatTimer);
                }
                this.repeatAccel = 0;
                this.repeatTimer = setInterval(() => {
                    this.repeatAccel = Math.min(this.repeatAccel + 1, 10);
                    const speed = 1 + this.repeatAccel * 2;
                    for (let i = 0; i < speed; i++) {
                        this.applyDelta(fieldName, delta);
                    }
                }, this.repeatInterval);
            }

            stopRepeat() {
                if (this.repeatTimer) {
                    clearTimeout(this.repeatTimer);
                    clearInterval(this.repeatTimer);
                    this.repeatTimer = null;
                    this.repeatAccel = 0;
                }
            }

            mouse(event, pos, node) {
                if (event.type === "pointerdown") {
                    const hit = this.getHitTarget(pos, node.size[0]);
                    if (hit) {
                        this.mouseDowned = true;
                        this.mouseDownPos = [...pos];
                        this.dragTarget = hit;
                        this.lastDx = 0;

                        if (hit.target === "value") {
                            const f = this.getField(hit.field);
                            const currentVal = f.get();
                            app.canvas.prompt("输入数值", String(currentVal), (newValStr) => {
                                const parsed = parseInt(newValStr, 10);
                                if (!isNaN(parsed)) {
                                    let newVal = Math.max(-99999999, Math.min(99999999, parsed));
                                    f.set(newVal);
                                    node.graph.setDirtyCanvas(true, true);
                                }
                            }, event);
                        } else {
                            this.applyDelta(hit.field, hit.delta);
                        }
                        return true;
                    }
                }

                if (event.type === "pointermove") {
                    const hit = this.getHitTarget(pos, node.size[0]);
                    const oldHover = this.hoverTarget;
                    if (hit) {
                        this.hoverTarget = { field: hit.field };
                    } else {
                        this.hoverTarget = null;
                    }
                    if (oldHover !== this.hoverTarget) {
                        this.node.graph.setDirtyCanvas(true, true);
                    }

                    if (this.mouseDowned && this.dragTarget && this.dragTarget.target === "value") {
                        const dx = pos[0] - this.mouseDownPos[0];
                        const stepVal = this.stepWidget.value;
                        const minStep = Math.max(1, stepVal);
                        const delta = Math.round(dx / Math.max(4, minStep));
                        if (delta !== this.lastDx) {
                            const f = this.getField(this.dragTarget.field);
                            let newVal = this.dragOrigValue + delta * minStep;
                            newVal = Math.max(-99999999, Math.min(99999999, newVal));
                            f.set(newVal);
                            this.lastDx = delta;
                            this.node.graph.setDirtyCanvas(true, true);
                        }
                        return true;
                    }
                }

                if (event.type === "pointerup") {
                    this.mouseDowned = false;
                    this.dragTarget = null;
                    this.stopRepeat();
                    return true;
                }

                if (event.type === "pointerleave") {
                    this.hoverTarget = null;
                    this.node.graph.setDirtyCanvas(true, true);
                }

                return false;
            }
        }

        // ===== 拦截 onNodeCreated =====
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = origOnNodeCreated?.apply(this, arguments);

            const valueWidget = this.widgets?.find(w => w.name === "value");
            const startWidget = this.widgets?.find(w => w.name === "start");
            const stepWidget = this.widgets?.find(w => w.name === "step");
            if (!valueWidget || !startWidget || !stepWidget) return result;

            startWidget.hidden = true;
            stepWidget.hidden = true;
            valueWidget.hidden = true;

            Object.defineProperty(valueWidget.options, "step", {
                get() { return stepWidget.value; },
                enumerable: true,
                configurable: true
            });
            Object.defineProperty(valueWidget.options, "step2", {
                get() { return stepWidget.value; },
                enumerable: true,
                configurable: true
            });

            const aligned = alignValue(valueWidget.value, startWidget.value, stepWidget.value);
            if (aligned !== valueWidget.value) {
                valueWidget.value = aligned;
            }

            const inlineWidget = new InlineIntegerWidget(this, startWidget, stepWidget, valueWidget);
            this.addCustomWidget(inlineWidget);

            const curSize = this.size;
            if (curSize[0] > 160) {
                this.setSize([140, curSize[1]]);
            }

            return result;
        };

        // ===== onDrawForeground（对齐修正） =====
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx, canvas) {
            origOnDrawForeground?.apply(this, arguments);

            const vw = this.widgets?.find(w => w.name === "value");
            const sw = this.widgets?.find(w => w.name === "start");
            const stw = this.widgets?.find(w => w.name === "step");
            if (!vw || !sw || !stw) return;

            const isAligned = (v, stepVal, startVal) => {
                if (stepVal <= 0) return v === startVal;
                return (v - startVal) % stepVal === 0;
            };

            if (!isAligned(vw.value, stw.value, sw.value)) {
                vw.value = alignValue(vw.value, sw.value, stw.value);
            }
        };
    },
});