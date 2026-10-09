"""context_handler 的「包装链」—— 多个节点共用同一个 handler 时的**唯一包装入口**。

为什么需要它（S1 地基）
----------------------
`context_window_anchor.py` 早期直接在实例上遮蔽方法：

    handler.execute = execute            # 闭包里套住原始方法

「只有一个节点包装 handler」时这样没问题，但第二个节点（例如后面的滚动/接力节点）也这么做会坏：

  · **覆盖**：后安装者的赋值直接顶掉先安装者 => 谁在图里靠后谁生效，另一个**静默失效**；
  · **嵌套**：先安装者保存的「原始方法」可能指向**已被包装的版本**
    => 再次安装会把包装层套进包装层里，调用深度随重装次数增长。

本模块把包装收拢成一条显式的链（存在 `handler._handler_wrap_chain` 上）：

    install(handler, "ContextWindowAnchor", {
        "get_context_windows": make_get_context_windows,   # factory(next_call) -> wrapper
        "get_resized_cond":    make_get_resized_cond,
        "execute":             make_execute,
    })

约定（与既有实现严格一致，保证「迁移后行为不变」）
------------------------------------------------
1. 包装器是**普通函数**（不带 self）—— 与 `handler.xxx = fn` 的既有语义一致：
   实例属性上的函数**不会**被描述符协议绑定成方法，因此签名必须与调用方完全一致
   （ComfyUI 侧全部是位置传参调用，见 `samplers.py:212` / `context_windows.py:579,687`）。
2. 链顺序 = 安装顺序；**后安装 = 更外层**（先被调用，再转交内层）。
   只有一个包装器时两种顺序不可区分 => 迁移不影响现有行为。
3. 同一 `owner` 重复安装 = **原地更新**（保持它在链中的位置不变），不叠加层数；
   重建时使用工厂里**当次捕获的配置** => 重装即生效（旧实现靠 `_cwa_originals` 达到同样效果）。
   「保持位置」是有意的：重装不应该把一个节点的包装器挪到链尾，
   否则会**静默改变**它与其他节点包装器的相对顺序。
4. 每个方法的「真原始」只在**首次**安装时解析一次，且优先从**类**上取
   （`type(handler).<name>`）=> 不受 `copy.copy(handler)`（浅拷贝）与历史实例属性的污染。
5. `copy.copy(handler)` 后，新实例上的链会**重新绑定到该实例并重放包装**
   => 浅拷贝出来的 handler 与原件行为一致（这是 `apply_anchor` 的关键前提，
   否则包装会被写到原件上、副本静默失效）。
6. 某个方法被卸载干净后，实例属性会被**删除**，回到类方法（= 真正「没装过」）。

没有任何节点使用本模块时零开销。
"""

from __future__ import annotations

import logging

# 链挂在 handler 实例上的属性名
_CHAIN_ATTR = "_handler_wrap_chain"

# handler 上允许被上下文化逻辑包装的方法（仅作为调用方的参考清单）
WRAPPABLE_METHODS = ("get_context_windows", "get_resized_cond", "execute")


def _resolve_original(handler, name: str):
    """取方法的「真原始」可调用：优先类属性（绑定到 handler），退回实例属性。

    用类属性是为了绕开两类污染：
      · `copy.copy(handler)` 之后实例 `__dict__` 里可能已经有包装函数；
      · 旧实现留在实例上的 `handler.xxx = wrapper`。
    找不到时返回 None（调用方按「该方法不可包装」处理）。
    """
    cls = type(handler)
    fn = getattr(cls, name, None)
    if fn is not None:
        try:
            return fn.__get__(handler, cls)      # 绑定成 bound method（保留 self）
        except Exception:
            return fn
    return getattr(handler, name, None)


class HandlerWrapChain:
    """挂在 handler 实例上的包装链（`handler._handler_wrap_chain`）。"""

    def __init__(self, handler):
        self.handler = handler
        self.originals: dict = {}      # method -> 真原始可调用
        self.entries: list = []        # [(owner, {method: factory})]，按安装顺序（内 -> 外）

    # ---------------- 安装 / 卸载 ----------------

    def install(self, owner: str, methods: dict) -> list:
        """注册（或原地替换）owner 的包装器；返回**实际安装成功**的方法名列表。"""
        owner = str(owner)
        staged: dict = {}
        applied: list = []
        for name, factory in methods.items():
            if not callable(factory):
                raise TypeError("[HandlerWrap] %s.%s: factory 必须是可调用对象" % (owner, name))
            if name not in self.originals:
                orig = _resolve_original(self.handler, name)
                if orig is None:
                    logging.debug("[HandlerWrap] handler 上找不到 %s，跳过（owner=%s）", name, owner)
                    continue
                self.originals[name] = orig
            staged[name] = factory
            applied.append(name)
        if not applied:
            return []
        # 原地更新：同 owner 已存在时**保持它在链中的位置**（重装不应改变相对顺序）
        idx = None
        for i, (o, _m) in enumerate(self.entries):
            if o == owner:
                idx = i
                break
        if idx is None:
            self.entries.append((owner, staged))
        else:
            self.entries[idx] = (owner, staged)
        for name in staged:
            self._rebuild(name)
        return applied

    def uninstall(self, owner: str) -> bool:
        """摘掉 owner 的包装器并重建其余包装；返回是否真的摘到了。"""
        owner = str(owner)
        before = len(self.entries)
        self.entries = [(o, m) for (o, m) in self.entries if o != owner]
        if len(self.entries) == before:
            return False
        for name in list(self.originals):
            self._rebuild(name)
        return True

    # ---------------- 查询 ----------------

    def has_owner(self, owner: str) -> bool:
        return any(o == str(owner) for o, _ in self.entries)

    def owners(self) -> list:
        return [o for o, _ in self.entries]

    def describe(self) -> str:
        """可读的链描述：左 = 最内层（最靠近真原始），右 = 最外层。"""
        if not self.entries:
            return "（空）"
        return " -> ".join("%s[%s]" % (o, ",".join(sorted(m))) for o, m in self.entries)

    # ---------------- 内部 ----------------

    def _rebuild(self, name: str) -> None:
        """按链顺序重建 `handler.<name>`：原始(最内) -> entries[0] -> ... -> entries[-1](最外)。"""
        orig = self.originals.get(name)
        if orig is None:
            return
        composed = orig
        wrapped = False
        for _owner, methods in self.entries:
            factory = methods.get(name)
            if factory is None:
                continue
            composed = factory(composed)
            wrapped = True
        if wrapped:
            setattr(self.handler, name, composed)
        else:
            # 没有任何包装：删掉实例属性，回到类方法（= 真正「没装过」）
            try:
                delattr(self.handler, name)
            except AttributeError:
                pass


def _is_chain(obj) -> bool:
    """鸭子类型判断：不依赖 `isinstance` —— 万一 handler_wrap 被加载成两份模块
    （例如同时存在顶层导入与包内相对导入），`isinstance` 会失配并让已有链被误判为「不存在」，
    进而丢掉已注册的包装器。这里只要求结构性兼容。"""
    return obj is not None and hasattr(obj, "handler") and hasattr(obj, "originals") and hasattr(obj, "entries")


def chain_of(handler, create: bool = False):
    """取 handler 上的链；`create=True` 时不存在则新建。

    浅拷贝安全：`copy.copy(handler)` 会把链对象一并复制（同一引用），此时链里记录的
    `handler` 仍是**原件** => 必须为副本新建一条链（继承 originals / entries 后重放），
    否则 `_rebuild` 会把属性写到原件上、副本静默失效。
    """
    chain = getattr(handler, _CHAIN_ATTR, None)
    if _is_chain(chain) and chain.handler is handler:
        return chain
    if not create:
        return None
    new = HandlerWrapChain(handler)
    if _is_chain(chain):
        # 继承条目顺序，但 originals **必须重新绑定到本实例** ——
        # 否则副本上的包装会去调原件的 bound method（读原件的实例属性）。
        new.entries = [(o, dict(m)) for o, m in chain.entries]
        for name in list(chain.originals):
            orig = _resolve_original(handler, name)
            if orig is not None:
                new.originals[name] = orig
    setattr(handler, _CHAIN_ATTR, new)
    # 把继承来的包装链真正装到本实例上（首次安装时 entries 为空 => 空操作）
    for name in list(new.originals):
        new._rebuild(name)
    return new


def install(handler, owner: str, methods: dict) -> list:
    """给 handler 安装（或替换）owner 的包装器；返回实际安装的方法名列表。"""
    return chain_of(handler, True).install(owner, methods)


def uninstall(handler, owner: str) -> bool:
    """摘掉 owner 的包装器（实例属性会随之还原/删除）；返回是否真的摘到了。"""
    chain = chain_of(handler, False)
    return bool(chain and chain.uninstall(owner))


def has_owner(handler, owner: str) -> bool:
    chain = chain_of(handler, False)
    return bool(chain and chain.has_owner(owner))


def owners_of(handler) -> list:
    chain = chain_of(handler, False)
    return chain.owners() if chain else []


def describe(handler) -> str:
    chain = chain_of(handler, False)
    return chain.describe() if chain else "（未安装）"
