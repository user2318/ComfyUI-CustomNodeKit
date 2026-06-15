# install.py
# ComfyUI 启动时自动执行，确保所有必要依赖已安装。
# 包括：pip 包 + tkinter 自动修复（针对嵌入式 Python）

import subprocess
import sys
import importlib
import time
import os
import zipfile
import urllib.request
import urllib.error
import shutil
import tempfile

# ============================================================
# pip 包依赖
# ============================================================
_REQUIRED_PACKAGES = [
    ("groundingdino-py", "groundingdino"),
    ("transformers", "transformers"),
]

MAX_RETRIES = 3
RETRY_DELAY = 2
INSTALL_TIMEOUT = 120


def _install_package(pip_name: str):
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


def _ensure_pip_packages():
    """确保 pip 包已安装"""
    all_success = True
    for pip_name, import_name in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            print(f"[ComfyUI-CustomNodeKit] 依赖已就绪: {import_name}")
            continue
        except ImportError:
            pass

        success, message = _install_package(pip_name)
        if success:
            print(f"[ComfyUI-CustomNodeKit] {message}")
        else:
            all_success = False
            print(f"[ComfyUI-CustomNodeKit] 自动安装 {pip_name} 失败，请手动执行:")
            print(f'  "{sys.executable}" -m pip install {pip_name}')
            print(f"  错误信息: {message}")

    if all_success:
        print("[ComfyUI-CustomNodeKit] 所有 pip 依赖安装完成")
    else:
        print("[ComfyUI-CustomNodeKit] 部分 pip 依赖安装失败，请查看上方错误信息手动处理")


# ============================================================
# tkinter 自动修复
# ============================================================

def _get_tkinter_status() -> str:
    """
    检测 tkinter 状态，返回:
    - "ok" : 正常工作
    - "missing" : 缺少模块或 DLL
    """
    try:
        import tkinter as tk
        # 进一步验证能否创建 Tk 实例（某些嵌入式环境有模块但缺 DLL）
        root = tk.Tk()
        root.destroy()
        return "ok"
    except ImportError:
        return "missing"
    except Exception as e:
        # TclError 等说明有模块但缺 tcl/tk DLL
        return "missing"


def _get_python_base_dir() -> str:
    """返回 Python 安装根目录（与 python.exe 同级）"""
    return os.path.dirname(os.path.abspath(sys.executable))


def _get_python_version_tuple() -> tuple:
    """返回 (major, minor, patch)"""
    return (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)


def _download_nuget_package(version: str, dest_dir: str):
    """
    从 nuget.org 下载指定版本的 Python nuget 包并解压到 dest_dir。
    返回解压后 tools 目录的路径，失败返回 None。
    """
    url = f"https://www.nuget.org/api/v2/package/python/{version}"
    nupkg_path = os.path.join(dest_dir, f"python.{version}.nupkg")

    print(f"[ComfyUI-CustomNodeKit] 正在从 NuGet 下载 Python {version} 的 tkinter 组件...")
    print(f"[ComfyUI-CustomNodeKit] 下载地址: {url}")

    try:
        # 下载 .nupkg (本质是个 zip 文件)
        req = urllib.request.Request(url, headers={
            "User-Agent": "ComfyUI-CustomNodeKit/1.0"
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192
            with open(nupkg_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        print(f"\r[ComfyUI-CustomNodeKit] 下载进度: {percent}% ({downloaded // 1024}KB / {total_size // 1024}KB)", end="")
        print()

        # 解压到临时目录
        extract_to = os.path.join(dest_dir, "nuget_extract")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(nupkg_path, "r") as zf:
            zf.extractall(extract_to)

        tools_dir = os.path.join(extract_to, "tools")
        if os.path.isdir(tools_dir):
            return tools_dir
        else:
            print(f"[ComfyUI-CustomNodeKit] 错误: NuGet 包中没有 tools 目录")
            return None

    except urllib.error.URLError as e:
        print(f"[ComfyUI-CustomNodeKit] 下载失败 (网络错误): {e}")
        return None
    except Exception as e:
        print(f"[ComfyUI-CustomNodeKit] 下载/解压失败: {e}")
        return None


def _install_tkinter_from_nuget():
    """
    从 NuGet 下载 Python 包并提取 tkinter 所需的文件到当前 Python 目录。
    
    需要复制的文件/目录:
      1. tools/DLLs/_tkinter.pyd  → python/DLLs/
      2. tools/DLLs/tcl*.dll       → python/DLLs/
      3. tools/DLLs/tk*.dll        → python/DLLs/
      4. tools/Lib/tkinter/        → python/Lib/tkinter/
      5. tools/tcl/                → python/tcl/
    """
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    base_dir = _get_python_base_dir()

    # 目标路径
    dlls_dir = os.path.join(base_dir, "DLLs")
    lib_tkinter_dir = os.path.join(base_dir, "Lib", "tkinter")
    tcl_dir = os.path.join(base_dir, "tcl")

    # 使用临时目录
    with tempfile.TemporaryDirectory(prefix="tkinter_fix_") as tmpdir:
        print(f"[ComfyUI-CustomNodeKit] 使用临时目录: {tmpdir}")

        tools_dir = _download_nuget_package(version, tmpdir)
        if not tools_dir:
            return False

        # 检查源文件是否存在
        src_dlls = os.path.join(tools_dir, "DLLs")
        src_lib = os.path.join(tools_dir, "Lib")
        src_tcl = os.path.join(tools_dir, "tcl")

        if not os.path.isdir(src_dlls):
            print(f"[ComfyUI-CustomNodeKit] 错误: 未找到 DLLs 目录: {src_dlls}")
            return False

        # 1. 复制 DLL 文件
        needed_dlls = ["_tkinter.pyd"]
        print(f"[ComfyUI-CustomNodeKit] 正在复制 DLL 文件...")
        for fname in os.listdir(src_dlls):
            flower = fname.lower()
            if flower.startswith("tcl") or flower.startswith("tk"):
                if fname.endswith(".dll") or fname.endswith(".pyd"):
                    needed_dlls.append(fname)

        # 去重
        needed_dlls = list(set(needed_dlls))

        copied_dlls = 0
        for fname in needed_dlls:
            src = os.path.join(src_dlls, fname)
            if os.path.isfile(src):
                dst = os.path.join(dlls_dir, fname)
                try:
                    shutil.copy2(src, dst)
                    print(f"[ComfyUI-CustomNodeKit]   + {fname}")
                    copied_dlls += 1
                except Exception as e:
                    print(f"[ComfyUI-CustomNodeKit]   ! 复制 {fname} 失败: {e}")

        if copied_dlls == 0:
            print(f"[ComfyUI-CustomNodeKit] 错误: 未复制任何 DLL 文件")
            return False

        # 2. 复制 tkinter 模块目录
        src_tkinter = os.path.join(src_lib, "tkinter")
        if os.path.isdir(src_tkinter):
            print(f"[ComfyUI-CustomNodeKit] 正在复制 tkinter 模块...")
            if os.path.isdir(lib_tkinter_dir):
                # 备份旧目录（如果存在）
                backup_dir = lib_tkinter_dir + "_bak"
                try:
                    if os.path.isdir(backup_dir):
                        shutil.rmtree(backup_dir)
                    os.rename(lib_tkinter_dir, backup_dir)
                    print(f"[ComfyUI-CustomNodeKit]   已将旧 tkinter 目录备份为: {backup_dir}")
                except Exception as e:
                    print(f"[ComfyUI-CustomNodeKit]   ! 备份旧目录失败: {e}")

            try:
                shutil.copytree(src_tkinter, lib_tkinter_dir, symlinks=False, ignore_dangling_symlinks=True)
                print(f"[ComfyUI-CustomNodeKit]   + tkinter/ 目录已复制")
            except Exception as e:
                print(f"[ComfyUI-CustomNodeKit]   ! 复制 tkinter 目录失败: {e}")
                return False
        else:
            print(f"[ComfyUI-CustomNodeKit] 警告: NuGet 包中无 tkinter 模块目录，可能无需复制")

        # 3. 复制 tcl 目录
        if os.path.isdir(src_tcl):
            print(f"[ComfyUI-CustomNodeKit] 正在复制 tcl 目录...")
            # 合并 tcl 内容（不删除现有内容，只补充缺失的）
            for item in os.listdir(src_tcl):
                src_item = os.path.join(src_tcl, item)
                dst_item = os.path.join(tcl_dir, item)
                try:
                    if os.path.isdir(src_item):
                        if not os.path.isdir(dst_item):
                            shutil.copytree(src_item, dst_item, symlinks=False, ignore_dangling_symlinks=True)
                            print(f"[ComfyUI-CustomNodeKit]   + tcl/{item}/")
                    else:
                        if not os.path.isfile(dst_item):
                            shutil.copy2(src_item, dst_item)
                            print(f"[ComfyUI-CustomNodeKit]   + tcl/{item}")
                except Exception as e:
                    print(f"[ComfyUI-CustomNodeKit]   ! 复制 tcl/{item} 失败: {e}")
        else:
            print(f"[ComfyUI-CustomNodeKit] 警告: NuGet 包中无 tcl 目录")

        # 验证
        print(f"[ComfyUI-CustomNodeKit] 正在验证 tkinter 修复结果...")
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import tkinter; print('ok')"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(f"[ComfyUI-CustomNodeKit] ✓ tkinter 修复成功！")
                return True
            else:
                print(f"[ComfyUI-CustomNodeKit] ✗ tkinter 仍然无法导入: {result.stderr}")
                return False
        except Exception as e:
            print(f"[ComfyUI-CustomNodeKit] ✗ 验证失败: {e}")
            return False


def _ensure_tkinter():
    """确保 tkinter 可用，如果不可用则尝试自动修复"""
    status = _get_tkinter_status()
    if status == "ok":
        print("[ComfyUI-CustomNodeKit] tkinter 已就绪")
        return True

    print("[ComfyUI-CustomNodeKit] ⚠ 检测到 tkinter 缺失")
    print(f"[ComfyUI-CustomNodeKit] Python 版本: {sys.version}")
    print(f"[ComfyUI-CustomNodeKit] Python 路径: {sys.executable}")

    # 检测是否是嵌入式 Python（通常没有完整 tcl/tk）
    base_dir = _get_python_base_dir()
    is_embedded = not os.path.isdir(os.path.join(base_dir, "tcl", "tk8.6"))

    if is_embedded:
        print("[ComfyUI-CustomNodeKit] 检测到嵌入式 Python 环境，将自动下载并安装 tkinter 组件...")
    else:
        print("[ComfyUI-CustomNodeKit] 检测到已有 tcl/tk 目录但导入失败，尝试修复...")

    success = _install_tkinter_from_nuget()
    if success:
        print("[ComfyUI-CustomNodeKit] ✅ tkinter 自动修复完成！")
        print("[ComfyUI-CustomNodeKit] 部分节点（交互式批量裁剪等）的文件选择功能现已可用。")
        return True
    else:
        print("[ComfyUI-CustomNodeKit] ❌ tkinter 自动修复失败，请尝试手动解决：")
        print(f"[ComfyUI-CustomNodeKit]   1. 下载完整版 Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} 安装包")
        print(f"[ComfyUI-CustomNodeKit]   2. 复制以下文件到 {base_dir}\\DLLs\\ 目录：")
        print(f"      - _tkinter.pyd")
        print(f"      - tcl86t.dll")
        print(f"      - tk86t.dll")
        print(f"   3. 复制 Python 安装目录中的 Lib\\tkinter\\ 文件夹到 {base_dir}\\Lib\\")
        print(f"   4. 复制 Python 安装目录中的 tcl\\ 文件夹到 {base_dir}\\")
        print(f"   5. 重启 ComfyUI")
        return False


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("  ComfyUI-CustomNodeKit - 依赖检查")
    print("=" * 60)

    # 1. 确保 pip 包
    _ensure_pip_packages()

    # 2. 确保 tkinter
    _ensure_tkinter()

    print("=" * 60)
    print("  ComfyUI-CustomNodeKit - 依赖检查完成")
    print("=" * 60)


# 当直接在 __init__.py 中 import 时也会自动执行
# 如果不想在 import 时运行，可以改为手动调用 run_install()
def run_install():
    main()

if __name__ == "__main__":
    main()
