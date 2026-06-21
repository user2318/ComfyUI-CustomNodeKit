// custom_nodes/ComfyUI-Custom_Tools/js/multi_file_picker.js
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "PathCollectorNode.UI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PathCollectorNode") return;

        // ======================== 样式注入 ========================
        if (!window.__pathCollectorStyleInjected) {
            window.__pathCollectorStyleInjected = true;
            const style = document.createElement('style');
            style.textContent = `
                .pc-modal-overlay {
                    position: fixed;
                    top: 0; left: 0; right: 0; bottom: 0;
                    background: rgba(0,0,0,0.5);
                    z-index: 10000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .pc-modal {
                    background: #2a2a2a;
                    border: 1px solid #555;
                    border-radius: 8px;
                    width: 600px;
                    max-height: 80vh;
                    display: flex;
                    flex-direction: column;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                    color: #ddd;
                    font-size: 13px;
                }
                .pc-modal-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px 14px;
                    border-bottom: 1px solid #444;
                    font-weight: bold;
                    font-size: 14px;
                }
                .pc-modal-header span {
                    cursor: pointer;
                    color: #888;
                }
                .pc-modal-header span:hover { color: #f66; }
                .pc-modal-body {
                    flex: 1;
                    overflow-y: auto;
                    padding: 8px 14px;
                    min-height: 100px;
                }
                .pc-path-row {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    padding: 4px 6px;
                    border-radius: 4px;
                    margin: 2px 0;
                    cursor: grab;
                    user-select: none;
                    background: #333;
                }
                .pc-path-row.dragging {
                    opacity: 0.5;
                }
                .pc-path-row.drag-over {
                    border-top: 2px solid #4a9eff;
                }
                .pc-path-row.selected {
                    outline: 2px solid #4a9eff;
                    background: #3a3f4a;
                }
                .pc-path-row .pc-drag-handle {
                    cursor: grab;
                    color: #666;
                    font-size: 14px;
                    flex-shrink: 0;
                }
                .pc-path-row .pc-path-text {
                    flex: 1;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    padding: 2px 4px;
                }
                .pc-path-row .pc-path-text.empty {
                    color: #888;
                    font-style: italic;
                }
                .pc-path-row .pc-del-btn {
                    cursor: pointer;
                    color: #888;
                    flex-shrink: 0;
                    padding: 0 4px;
                    font-size: 14px;
                }
                .pc-path-row .pc-del-btn:hover { color: #f66; }
                .pc-modal-footer {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 8px 14px;
                    border-top: 1px solid #444;
                    gap: 6px;
                    flex-wrap: wrap;
                }
                .pc-modal-footer .pc-left-btns,
                .pc-modal-footer .pc-right-btns {
                    display: flex;
                    gap: 6px;
                }
                .pc-btn {
                    background: #444;
                    border: 1px solid #666;
                    color: #ddd;
                    padding: 4px 12px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    white-space: nowrap;
                }
                .pc-btn:hover {
                    background: #555;
                }
                .pc-btn.primary {
                    background: #2a6eb0;
                    border-color: #3a8adf;
                }
                .pc-btn.primary:hover {
                    background: #3a8adf;
                }
                .pc-btn.danger {
                    background: #8a3a3a;
                    border-color: #b04a4a;
                }
                .pc-btn.danger:hover {
                    background: #b04a4a;
                }
                .pc-stats-bar {
                    display: flex;
                    gap: 14px;
                    font-size: 12px;
                    color: #aaa;
                    padding: 4px 14px;
                    flex-shrink: 0;
                }
                .pc-stats-bar span {
                    white-space: nowrap;
                }
                .pc-stats-bar .pc-stat-label {
                    color: #888;
                }
                .pc-path-row .pc-index {
                    color: #666;
                    font-size: 11px;
                    min-width: 26px;
                    text-align: right;
                    flex-shrink: 0;
                    user-select: none;
                }
                .pc-path-row.file-path .pc-path-text {
                    color: #6ecf8a;
                }
                .pc-path-row.folder-path .pc-path-text {
                    color: #cfb96e;
                }
                .pc-path-row.file-path.selected .pc-path-text,
                .pc-path-row.folder-path.selected .pc-path-text {
                    color: #ddd;
                }

                /* ---- 自定义 tooltip ---- */
                .pc-custom-tooltip {
                    position: fixed;
                    z-index: 10001;
                    background: #1a1a1a;
                    border: 1px solid #555;
                    border-radius: 6px;
                    padding: 8px 12px;
                    max-width: 500px;
                    max-height: 200px;
                    overflow-y: auto;
                    word-break: break-all;
                    white-space: pre-wrap;
                    font-size: 12px;
                    line-height: 1.5;
                    color: #e0e0e0;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.6);
                    pointer-events: none;
                    display: none;
                }
                .pc-custom-tooltip.visible {
                    display: block;
                }
            `;
            document.head.appendChild(style);
        }

        // ======================== 管理弹窗类 ========================
        class PathManagerModal {
            constructor(paths, onConfirm) {
                this.paths = paths.slice();   // 当前列表，元素为 { text: string }
                this.onConfirm = onConfirm;
                this.selectedIndex = -1;
                this.dragIndex = -1;
                this._create();
            }

            // 判断路径是否为文件（含扩展名）或文件夹
            _getPathType(text) {
                if (!text || text.trim() === '') return 'empty';
                // 取路径最后一段
                const lastPart = text.replace(/[\\/]$/, '').split(/[\\/]/).pop();
                // 如果含有 . 且不是以 . 开头（排除 .git 这类），认为是文件
                const dotIndex = lastPart.lastIndexOf('.');
                if (dotIndex > 0 && dotIndex < lastPart.length - 1) return 'file';
                return 'folder';
            }

            _create() {
                this.overlay = document.createElement('div');
                this.overlay.className = 'pc-modal-overlay';

                this.modal = document.createElement('div');
                this.modal.className = 'pc-modal';

                // ---- header ----
                const header = document.createElement('div');
                header.className = 'pc-modal-header';
                header.innerHTML = '<span>路径管理</span><span title="关闭" id="pc-close-btn">✕</span>';
                this.modal.appendChild(header);

                // ---- stats bar ----
                this.statsBar = document.createElement('div');
                this.statsBar.className = 'pc-stats-bar';
                this.modal.appendChild(this.statsBar);

                // ---- body (path list) ----
                this.body = document.createElement('div');
                this.body.className = 'pc-modal-body';
                this.modal.appendChild(this.body);

                // ---- footer buttons ----
                const footer = document.createElement('div');
                footer.className = 'pc-modal-footer';

                const leftBtns = document.createElement('div');
                leftBtns.className = 'pc-left-btns';
                const insertFileBtn = this._makeBtn('插入文件', () => this._insertFiles());
                const insertDirBtn = this._makeBtn('插入目录', () => this._insertDir());
                const insertEmptyBtn = this._makeBtn('插入空行', () => this._insertEmptyLine());
                const dedupeBtn = this._makeBtn('去重', () => this._dedupe());
                const clearAllBtn = this._makeBtn('清空', () => this._clearAll(), 'danger');
                leftBtns.append(insertFileBtn, insertDirBtn, insertEmptyBtn, dedupeBtn, clearAllBtn);

                const rightBtns = document.createElement('div');
                rightBtns.className = 'pc-right-btns';
                const okBtn = this._makeBtn('确定', () => this._confirm(), 'primary');
                const cancelBtn = this._makeBtn('取消', () => this._close());
                rightBtns.append(okBtn, cancelBtn);

                footer.append(leftBtns, rightBtns);
                this.modal.appendChild(footer);

                this.overlay.appendChild(this.modal);
                document.body.appendChild(this.overlay);

                // ---- events ----
                header.querySelector('#pc-close-btn').addEventListener('click', () => this._close());
                document.addEventListener('keydown', this._keyHandler = (e) => {
                    if (e.key === 'Escape') this._close();
                });

                // ---- 创建自定义 tooltip 元素 ----
                this.tooltipEl = document.createElement('div');
                this.tooltipEl.className = 'pc-custom-tooltip';
                this.overlay.appendChild(this.tooltipEl);

                this._render();
            }

            _makeBtn(text, onClick, cls = '') {
                const btn = document.createElement('button');
                btn.className = `pc-btn ${cls}`;
                btn.textContent = text;
                btn.addEventListener('click', onClick);
                return btn;
            }

            // ---------- 渲染路径列表 ----------
            _render() {
                this.body.innerHTML = '';

                // 计算统计
                let fileCount = 0, folderCount = 0;
                this.paths.forEach(item => {
                    const type = this._getPathType(item.text);
                    if (type === 'file') fileCount++;
                    else if (type === 'folder') folderCount++;
                });
                // 更新统计栏
                this.statsBar.innerHTML = `
                    <span><span class="pc-stat-label">总计:</span> ${this.paths.length}</span>
                    <span style="color:#6ecf8a"><span class="pc-stat-label">文件:</span> ${fileCount}</span>
                    <span style="color:#cfb96e"><span class="pc-stat-label">文件夹:</span> ${folderCount}</span>
                `;

                this.paths.forEach((item, idx) => {
                    const pathType = this._getPathType(item.text);

                    const row = document.createElement('div');
                    row.className = 'pc-path-row';
                    if (pathType === 'file') row.classList.add('file-path');
                    else if (pathType === 'folder') row.classList.add('folder-path');
                    if (idx === this.selectedIndex) row.classList.add('selected');
                    row.dataset.index = idx;

                    // 索引号（仅显示）
                    const indexSpan = document.createElement('span');
                    indexSpan.className = 'pc-index';
                    indexSpan.textContent = String(idx + 1);
                    row.appendChild(indexSpan);

                    // 拖动手柄
                    const handle = document.createElement('span');
                    handle.className = 'pc-drag-handle';
                    handle.textContent = '⣿';
                    row.appendChild(handle);

                    // 路径文本
                    const textSpan = document.createElement('span');
                    textSpan.className = 'pc-path-text';
                    if (pathType === 'empty') {
                        textSpan.classList.add('empty');
                        textSpan.textContent = '(空行)';
                    } else {
                        textSpan.textContent = item.text;
                    }
                    row.appendChild(textSpan);

                    // 删除按钮
                    const delBtn = document.createElement('span');
                    delBtn.className = 'pc-del-btn';
                    delBtn.textContent = '✕';
                    delBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        this.paths.splice(idx, 1);
                        if (this.selectedIndex >= this.paths.length) {
                            this.selectedIndex = this.paths.length - 1;
                        }
                        this._render();
                    });
                    row.appendChild(delBtn);

                    // 点击选择
                    row.addEventListener('click', () => {
                        this.selectedIndex = idx;
                        this._render();
                    });

                    // ---------- 自定义 tooltip 悬浮显示 ----------
                    // 只在路径内容不为空且文本被截断时显示自定义 tooltip
                    row.addEventListener('mouseenter', (e) => {
                        if (pathType === 'empty') return;
                        // 判断文本是否被截断：scrollWidth > clientWidth
                        if (textSpan.scrollWidth > textSpan.clientWidth) {
                            this.tooltipEl.textContent = item.text;
                            this.tooltipEl.classList.add('visible');
                            this._positionTooltip(e);
                        }
                    });
                    row.addEventListener('mousemove', (e) => {
                        if (this.tooltipEl.classList.contains('visible')) {
                            this._positionTooltip(e);
                        }
                    });
                    row.addEventListener('mouseleave', () => {
                        this.tooltipEl.classList.remove('visible');
                        this.tooltipEl.textContent = '';
                    });

                    // ---------- 拖拽排序（原生 Drag & Drop） ----------
                    row.draggable = true;
                    row.addEventListener('dragstart', (e) => {
                        this.dragIndex = idx;
                        row.classList.add('dragging');
                        e.dataTransfer.effectAllowed = 'move';
                        e.dataTransfer.setData('text/plain', '');
                    });
                    row.addEventListener('dragend', () => {
                        row.classList.remove('dragging');
                        this._clearDragOver();
                    });
                    row.addEventListener('dragover', (e) => {
                        e.preventDefault();
                        e.dataTransfer.dropEffect = 'move';
                        this._clearDragOver();
                        row.classList.add('drag-over');
                    });
                    row.addEventListener('dragleave', () => {
                        row.classList.remove('drag-over');
                    });
                    row.addEventListener('drop', (e) => {
                        e.preventDefault();
                        row.classList.remove('drag-over');
                        if (this.dragIndex !== -1 && this.dragIndex !== idx) {
                            const [moved] = this.paths.splice(this.dragIndex, 1);
                            // 计算插入位置：拖拽到下方
                            const targetIdx = idx > this.dragIndex ? idx - 1 : idx;
                            this.paths.splice(targetIdx, 0, moved);
                            this.selectedIndex = targetIdx;
                        }
                        this.dragIndex = -1;
                        this._render();
                    });

                    this.body.appendChild(row);
                });

                // 动态高度控制
                const rowCount = this.paths.length;
                if (rowCount === 0) {
                    this.body.style.maxHeight = 'none';
                } else if (rowCount <= 15) {
                    // 少于等于15条：自适应高度，不设 maxHeight
                    this.body.style.maxHeight = 'none';
                } else {
                    // 超过15条：固定高度，出现滚动条
                    this.body.style.maxHeight = `${15 * 34 + 16}px`;
                }

                // 如果没有行，显示提示
                if (this.paths.length === 0) {
                    const emptyMsg = document.createElement('div');
                    emptyMsg.style.cssText = 'padding: 20px; text-align: center; color: #666;';
                    emptyMsg.textContent = '暂无路径，请点击下方按钮添加';
                    this.body.appendChild(emptyMsg);
                }
            }

            // ---------- 定位 tooltip ----------
            _positionTooltip(e) {
                const tooltip = this.tooltipEl;
                if (!tooltip) return;

                const padding = 12; // 鼠标与 tooltip 的间距
                let left = e.clientX + padding;
                let top = e.clientY + padding;

                // 获取 tooltip 尺寸（每次需要重新获取，因为内容可能变化）
                const rect = tooltip.getBoundingClientRect();
                const tooltipW = rect.width;
                const tooltipH = rect.height;
                const viewW = window.innerWidth;
                const viewH = window.innerHeight;

                // 水平溢出处理：如果右边超出视口，则显示在鼠标左侧
                if (left + tooltipW > viewW - 10) {
                    left = e.clientX - tooltipW - padding;
                }
                // 垂直溢出处理：如果底部超出视口，则显示在鼠标上方
                if (top + tooltipH > viewH - 10) {
                    top = e.clientY - tooltipH - padding;
                }
                // 防止溢出到左侧或顶部（鼠标太靠边缘时）
                if (left < 5) left = 5;
                if (top < 5) top = 5;

                tooltip.style.left = left + 'px';
                tooltip.style.top = top + 'px';
            }

            _clearDragOver() {
                this.body.querySelectorAll('.drag-over').forEach(el => el.classList.remove('drag-over'));
            }

            // ---------- 插入逻辑 ----------
            _insertAt(index, items) {
                // items: 字符串数组
                const insertIdx = (index >= 0 && index < this.paths.length) ? index + 1 : this.paths.length;
                const newItems = items.map(text => ({ text }));
                this.paths.splice(insertIdx, 0, ...newItems);
                this.selectedIndex = insertIdx + newItems.length - 1;
                this._render();
            }

            async _insertFiles() {
                try {
                    const resp = await fetch('/multi_file_picker/select', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ mode: 'file', multi: true })
                    });
                    const result = await resp.json();
                    if (result.error) {
                        alert('选择文件失败：' + result.error);
                        return;
                    }
                    const paths = result.paths || [];
                    if (paths.length > 0) {
                        this._insertAt(this.selectedIndex, paths);
                    }
                } catch (err) {
                    console.error('选择文件失败:', err);
                    alert('选择文件失败，请查看控制台了解详情。');
                }
            }

            async _insertDir() {
                try {
                    const resp = await fetch('/multi_file_picker/select', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ mode: 'directory', multi: true })
                    });
                    const result = await resp.json();
                    if (result.error) {
                        alert('选择目录失败：' + result.error);
                        return;
                    }
                    const paths = result.paths || [];
                    if (paths.length > 0) {
                        this._insertAt(this.selectedIndex, paths);
                    }
                } catch (err) {
                    console.error('选择目录失败:', err);
                    alert('选择目录失败，请查看控制台了解详情。');
                }
            }

            _insertEmptyLine() {
                this._insertAt(this.selectedIndex, ['']);
            }

            // ---------- 去重 ----------
            _dedupe() {
                const before = this.paths.length;
                const seen = new Set();
                this.paths = this.paths.filter(item => {
                    const text = (item.text || '').trim();
                    if (text === '') return true; // 空行不参与去重
                    if (seen.has(text)) return false;
                    seen.add(text);
                    return true;
                });
                if (this.selectedIndex >= this.paths.length) {
                    this.selectedIndex = this.paths.length - 1;
                }
                const removed = before - this.paths.length;
                console.log(`[路径管理] 去重完成：移除 ${removed} 条重复路径`);
                this._render();
            }

            // ---------- 清空 ----------
            _clearAll() {
                this.paths = [];
                this.selectedIndex = -1;
                this._render();
            }

            // ---------- 确认 / 关闭 ----------
            _confirm() {
                const texts = this.paths.map(p => p.text);
                this.onConfirm(texts);
                this._close();
            }

            _close() {
                if (this._keyHandler) {
                    document.removeEventListener('keydown', this._keyHandler);
                }
                if (this.overlay && this.overlay.parentNode) {
                    this.overlay.parentNode.removeChild(this.overlay);
                }
            }
        }

        // ======================== 节点 UI 改造 ========================
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this);

            const pathsWidget = this.widgets.find(w => w.name === 'paths');
            if (!pathsWidget) return result;

            // ---------- 打开管理弹窗 ----------
            const openManager = () => {
                const existingText = pathsWidget.value || '';
                const trimmed = existingText.trim();
                const items = trimmed ? trimmed.split('\n').map(t => ({ text: t })) : [];
                new PathManagerModal(items, (finalTexts) => {
                    pathsWidget.value = finalTexts.join('\n');
                    app.graph.setDirtyCanvas(true, true);
                });
            };

            // ---------- 创建并排按钮 ----------
            const btnContainer = document.createElement('div');
            btnContainer.style.cssText = 'display:flex;gap:2px;height:fit-content;';

            const mgrBtn = document.createElement('button');
            mgrBtn.textContent = '路径管理';
            mgrBtn.style.cssText = 'flex:1;height:24px;padding:2px 8px;font-size:12px;background:var(--comfy-input-bg,#222);color:var(--input-text,white);border:1px solid var(--border-color,#444);border-radius:3px;cursor:pointer;';
            mgrBtn.onclick = openManager;

            const clearBtn = document.createElement('button');
            clearBtn.textContent = '清空';
            clearBtn.style.cssText = 'flex:1;height:24px;padding:2px 8px;font-size:12px;background:var(--comfy-input-bg,#222);color:var(--input-text,white);border:1px solid var(--border-color,#444);border-radius:3px;cursor:pointer;';
            clearBtn.onclick = () => {
                pathsWidget.value = '';
                app.graph.setDirtyCanvas(true, true);
            };

            btnContainer.appendChild(mgrBtn);
            btnContainer.appendChild(clearBtn);

            // ---------- textarea 样式 + 将按钮包裹进 widget ----------
            const styleTextarea = () => {
                const textarea = pathsWidget.inputEl || pathsWidget.element?.querySelector('textarea');
                if (!textarea) {
                    setTimeout(styleTextarea, 50);
                    return;
                }
                textarea.style.cssText = 'width:100%;box-sizing:border-box;min-height:50px;resize:none;background:var(--comfy-input-bg,#222);color:var(--input-text,white);border:1px solid var(--border-color,#444);border-radius:3px;outline:none;';
                textarea.placeholder = '每行一个路径，或点击按钮选择';
                textarea.addEventListener('dblclick', openManager);

                // 将 textarea + 按钮包裹在同一个容器中（使按钮高度计入节点尺寸）
                const wrapper = document.createElement('div');
                wrapper.style.cssText = 'display:flex;flex-direction:column;gap:2px;height:100%;width:100%;box-sizing:border-box;';
                textarea.parentNode.insertBefore(wrapper, textarea);
                wrapper.appendChild(textarea);
                wrapper.appendChild(btnContainer);

                // 用 ResizeObserver 监听节点 DOM 的尺寸变化，动态调整 textarea 高度
                // 当用户拉伸节点时，节点 DOM 的 offsetHeight 一定会变化，这是最可靠的信号
                const ro = new ResizeObserver(() => {
                    // 节点 DOM 元素是 this.el（LiteGraph 为节点创建的 div）
                    const nodeEl = this.el;
                    if (!nodeEl || !nodeEl.isConnected) return;
                    // 精确计算 wrapper 顶部相对于节点顶部的偏移量（标题栏 + 上方 padding，动态适应各种主题）
                    const wrapperTop = wrapper.getBoundingClientRect().top;
                    const nodeTop = nodeEl.getBoundingClientRect().top;
                    const offsetTop = wrapperTop - nodeTop;
                    const btnH = btnContainer.offsetHeight;
                    const nodeH = nodeEl.offsetHeight;
                    textarea.style.height = Math.max(50, nodeH - offsetTop - btnH - 2) + 'px';
                });
                ro.observe(this.el);

                // 保持引用以便 onRemoved 中清理
                if (this._pathCollectorResizeObserver) this._pathCollectorResizeObserver.disconnect();
                this._pathCollectorResizeObserver = ro;

                // ---- 自动恢复按钮：当 widget 被重建导致 wrapper 被销毁时，自动重建按钮 ----
                if (this._pathCollectorMutationObserver) this._pathCollectorMutationObserver.disconnect();
                const mo = new MutationObserver((mutations) => {
                    for (const m of mutations) {
                        for (const node of m.removedNodes) {
                            if (node === wrapper || (node.nodeType === 1 && node.contains(wrapper))) {
                                mo.disconnect();
                                requestAnimationFrame(() => setTimeout(styleTextarea, 0));
                                return;
                            }
                        }
                    }
                });
                mo.observe(wrapper.parentNode, { childList: true });
                this._pathCollectorMutationObserver = mo;

                // 设置 widget 的 computeSize，保证节点最小高度合理
                pathsWidget.computeSize = function(width) {
                    const textareaMinHeight = 50;   // textarea min-height
                    const gap = 2;                   // wrapper gap
                    const btnHeight = 26;             // 按钮占用高度 (24px content + border/padding)
                    return [width, textareaMinHeight + gap + btnHeight];
                };
            };
            requestAnimationFrame(() => setTimeout(styleTextarea, 0));

            // 在节点被移除时清理所有 Observer，防止内存泄漏
            const origOnRemoved = this.onRemoved;
            this.onRemoved = function() {
                this._pathCollectorMutationObserver?.disconnect();
                this._pathCollectorResizeObserver?.disconnect();
                return origOnRemoved?.call(this);
            };

            return result;
        };
    }
});