# install.py
# ComfyUI 启动时自动执行，确保 grounding_dino_nodes.py 的依赖已安装。
# 健壮性优化：加入重试机制、超时控制、更详细的错误信息。

import subprocess
import sys
import importlib
import time

_REQUIRED_PACKAGES = [
    ("groundingdino-py", "groundingdino"),
    ("transformers", "transformers"),
]

MAX_RETRIES = 3
RETRY_DELAY = 2  # 重试间隔（秒）
INSTALL_TIMEOUT = 120  # 单次安装超时（秒）


def _install_package(pip_name):
    """安装指定 pip 包，支持重试和超时，返回 (success, message)"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[ComfyUI-CustomNodeKit] 正在安装依赖: {pip_name} (第{attempt}次尝试) ...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name, "-q"],
                capture_output=True,
                text=True,
                timeout=INSTALL_TIMEOUT,
            )
            if result.returncode == 0:
                return True, f"{pip_name} 安装成功"
            else:
                error_msg = result.stderr.strip()[-300:] if result.stderr else "未知错误"
                if attempt < MAX_RETRIES:
                    print(f"[ComfyUI-CustomNodeKit] {pip_name} 安装失败 (attempt {attempt}): {error_msg}")
                    print(f"[ComfyUI-CustomNodeKit] {RETRY_DELAY} 秒后重试...")
                    time.sleep(RETRY_DELAY)
                else:
                    return False, error_msg
        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRIES:
                print(f"[ComfyUI-CustomNodeKit] {pip_name} 安装超时 (attempt {attempt})，{RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
            else:
                return False, f"安装超时（{INSTALL_TIMEOUT}秒）"
        except Exception as e:
            return False, str(e)
    return False, "已达到最大重试次数"


def main():
    all_success = True
    for pip_name, import_name in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            print(f"[ComfyUI-CustomNodeKit] 依赖已就绪: {import_name}")
            continue
        except ImportError:
            pass  # 未安装，需要安装

        success, message = _install_package(pip_name)
        if success:
            print(f"[ComfyUI-CustomNodeKit] {message}")
        else:
            all_success = False
            print(f"[ComfyUI-CustomNodeKit] 自动安装 {pip_name} 失败，请手动执行:")
            print(f'  "{sys.executable}" -m pip install {pip_name}')
            print(f"  错误信息: {message}")

    if all_success:
        print("[ComfyUI-CustomNodeKit] 所有依赖安装完成")
    else:
        print("[ComfyUI-CustomNodeKit] 部分依赖安装失败，请查看上方错误信息手动处理")


if __name__ == "__main__":
    main()