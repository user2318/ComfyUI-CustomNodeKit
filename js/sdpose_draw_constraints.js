import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "SDPoseDrawConstraints.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SDPoseDrawKeypointsV2") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const stickWidthWidget = this.widgets.find(w => w.name === "stick_width");
            const yawThicknessMinWidget = this.widgets.find(w => w.name === "yaw_thickness_min");

            // ---------- stick_width 变化时的约束 ----------
            if (stickWidthWidget) {
                const originalStickCallback = stickWidthWidget.callback;
                stickWidthWidget.callback = (value) => {
                    // 如果下限高于上限，把下限同步为上限
                    if (yawThicknessMinWidget && yawThicknessMinWidget.value > value) {
                        yawThicknessMinWidget.value = value;
                    }
                    // 调用原始回调（如有）
                    if (originalStickCallback) {
                        originalStickCallback.call(this, value);
                    }
                };
            }

            // ---------- yaw_thickness_min 变化时的约束 ----------
            if (yawThicknessMinWidget) {
                const originalMinCallback = yawThicknessMinWidget.callback;
                yawThicknessMinWidget.callback = (value) => {
                    // 下限不能高于上限（stick_width）
                    if (stickWidthWidget && value > stickWidthWidget.value) {
                        yawThicknessMinWidget.value = stickWidthWidget.value;
                    }
                    if (originalMinCallback) {
                        originalMinCallback.call(this, value);
                    }
                };
            }

            return result;
        };
    }
});
