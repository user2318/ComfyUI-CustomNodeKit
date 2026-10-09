// 图像 List 节点前端扩展
// 目前只服务 ImageListSplitOutputs（图像列表分路输出）：
//   后端为了通过 ComfyUI 的 prompt 校验，固定声明了 16 路 IMAGE 输出；
//   这里按 output_count 的值**真实增删输出槽**，使画布上只显示设定的路数（1~16）。
import { app } from "../../scripts/app.js";

const SPLIT_NODE = "ImageListSplitOutputs";
const COUNT_WIDGET = "output_count";
const MIN_OUTPUTS = 1;
const MAX_OUTPUTS = 16;

function getCountWidget(node) {
    return node.widgets?.find((w) => w.name === COUNT_WIDGET);
}

function readCount(widget) {
    let n = Math.round(Number(widget?.value));
    if (!Number.isFinite(n)) n = MIN_OUTPUTS;
    return Math.max(MIN_OUTPUTS, Math.min(MAX_OUTPUTS, n));
}

function syncOutputs(node) {
    if (!node || !node.outputs) return;

    const widget = getCountWidget(node);
    if (!widget) return;

    const target = readCount(widget);
    let changed = false;

    // 回写被 clamp 后的值（防重入，避免与 widget.callback 互相触发）
    if (Number(widget.value) !== target) {
        node.__ilsSyncing = true;
        widget.value = target;
        node.__ilsSyncing = false;
    }

    // 路数变少：从尾部真实删除（先断链，避免残留悬挂连接）
    while (node.outputs.length > target) {
        const last = node.outputs.length - 1;
        if (typeof node.disconnectOutput === "function") node.disconnectOutput(last);
        node.removeOutput(last);
        changed = true;
    }

    // 路数变多：补足槽位
    while (node.outputs.length < target) {
        node.addOutput(`image_${node.outputs.length + 1}`, "IMAGE");
        changed = true;
    }

    // 统一命名，保证与后端 RETURN_NAMES 一致
    node.outputs.forEach((o, i) => {
        const name = `image_${i + 1}`;
        if (o.name !== name) {
            o.name = name;
            changed = true;
        }
        if (o.label && o.label !== name) o.label = name;
    });

    if (changed) {
        // 高度不够则增高（不改变用户设定的宽度）
        const minSize = node.computeSize();
        if (node.size[1] < minSize[1]) node.setSize([node.size[0], minSize[1]]);
        node.graph?.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "ComfyUI.CustomNodeKit.ImageListSplitOutputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== SPLIT_NODE) return;

        // 新建节点：把 16 路默认输出裁剪到 widget 设定的路数
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const widget = getCountWidget(this);
            if (widget) {
                const node = this;
                const origCallback = widget.callback;
                widget.callback = function () {
                    const r = origCallback?.apply(this, arguments);
                    if (!node.__ilsSyncing) syncOutputs(node);
                    return r;
                };
                syncOutputs(node);
            }
            return result;
        };

        // 加载工作流：按保存的 widget 值同步（值一致时不做任何操作，不会误断已有连线）
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            syncOutputs(this);
            return result;
        };

        // 兜底：部分前端版本通过 onWidgetChanged 通知 widget 变化
        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name) {
            const result = onWidgetChanged?.apply(this, arguments);
            if (name === COUNT_WIDGET) syncOutputs(this);
            return result;
        };
    },
});
