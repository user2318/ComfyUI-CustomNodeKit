import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.IntegerSettingNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "IntegerSettingNode") return;

        // 对齐函数：将 v 对齐到距离最近的 start + n*step
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

        // 找到 value 对应的 <input>
        const findValueInput = (nodeEl) => {
            if (!nodeEl) return null;
            const labels = nodeEl.querySelectorAll("label");
            for (const lbl of labels) {
                if (lbl.textContent?.trim() === "value") {
                    const parent = lbl.parentElement;
                    if (parent) return parent.querySelector("input");
                }
            }
            return null;
        };

        // 判断一个值是否已对齐到 start + n*step
        const isAligned = (v, startVal, stepVal) => {
            if (stepVal <= 0) return v === startVal;
            const diff = v - startVal;
            return diff % stepVal === 0;
        };

        // 同步到 Vue 组件的 v-model（原生 setter + input 事件）
        const syncInputToVue = (inputEl, value) => {
            if (!inputEl) return;
            if (String(inputEl.value) === String(value)) return;
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value'
            ).set;
            setter.call(inputEl, String(value));
            inputEl.dispatchEvent(new Event('input', { bubbles: true }));
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = origOnNodeCreated?.apply(this, arguments);

            const valueWidget = this.widgets?.find(w => w.name === "value");
            const startWidget = this.widgets?.find(w => w.name === "start");
            const stepWidget = this.widgets?.find(w => w.name === "step");
            if (!valueWidget || !startWidget || !stepWidget) return result;

            this._ck_valueWidget = valueWidget;
            this._ck_startWidget = startWidget;
            this._ck_stepWidget = stepWidget;
            this._ck_lastStart = startWidget.value;
            this._ck_lastStep = stepWidget.value;

            // ★ 将 value 的 options.step 变为动态 getter
            //   使 <input step> 属性始终跟随 step widget 的当前值
            Object.defineProperty(valueWidget.options, "step", {
                get() { return stepWidget.value; },
                enumerable: true,
                configurable: true
            });

            // ★ 设置 step2，Vue 3 的 useNumberStepCalculation 函数优先读取 step2
            Object.defineProperty(valueWidget.options, "step2", {
                get() { return stepWidget.value; },
                enumerable: true,
                configurable: true
            });

            // 首次对齐
            const aligned = alignValue(valueWidget.value, startWidget.value, stepWidget.value);
            if (aligned !== valueWidget.value) {
                valueWidget.value = aligned;
            }

            // 等 DOM 就绪后同步初始值到 Vue
            requestAnimationFrame(() => {
                if (!this.el) return;
                const inp = findValueInput(this.el);
                if (inp) {
                    syncInputToVue(inp, valueWidget.value);
                }
            });

            return result;
        };

        // ===== onDrawForeground =====
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx, canvas) {
            origOnDrawForeground?.apply(this, arguments);

            const vw = this._ck_valueWidget;
            const sw = this._ck_startWidget;
            const stw = this._ck_stepWidget;
            if (!vw || !sw || !stw) return;

            // ★ 核心修复：Vue 的 useNumberStepCalculation 内部以 start=0 计算 stepIndex，
            //   导致步进按钮使用 1 作为初值。每帧检查 value 是否对齐到当前的 start/step，
            //   若未对齐则立即纠正。
            if (!isAligned(vw.value, sw.value, stw.value)) {
                const aligned = alignValue(vw.value, sw.value, stw.value);
                vw.value = aligned;
            }

            // start/step 变化时对齐 value
            if (sw.value !== this._ck_lastStart || stw.value !== this._ck_lastStep) {
                this._ck_lastStart = sw.value;
                this._ck_lastStep = stw.value;
                const aligned = alignValue(vw.value, sw.value, stw.value);
                if (aligned !== vw.value) {
                    vw.value = aligned;
                }
                const inp = findValueInput(this.el);
                syncInputToVue(inp, vw.value);
            }

            // 每帧同步 value 到 input DOM
            const inp = findValueInput(this.el);
            syncInputToVue(inp, vw.value);
        };
    },
});
