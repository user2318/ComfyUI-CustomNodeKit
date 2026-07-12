import math


class IntegerSettingNode:
    """自定义整数设置节点（单行版）
    
    功能：
    - 设置起始值 start 和步进值 step
    - 输入数值自动对齐到 start + step*n
    - 支持负整数
    - 单行布局：Start、Step、Value 水平排列
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {
                    "default": 0,
                    "min": -99999999,
                    "max": 99999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "等差数列的起始值。The starting value of the arithmetic sequence.",
                }),
                "step": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 99999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "等差数列的步进值（正数）。The step value of the arithmetic sequence (positive).",
                }),
                "value": ("INT", {
                    "default": 0,
                    "min": -99999999,
                    "max": 99999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "输入数值，会自动对齐到最接近的 start + step*n。Input value, automatically aligned to the nearest start + step*n.",
                }),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("aligned_value",)
    OUTPUT_NODE = False
    FUNCTION = "apply"
    CATEGORY = "CustomNodes/Utils"
    DESCRIPTION = "将输入数值对齐到指定的等差数列（start + step*n）。Aligns the input value to the specified arithmetic sequence (start + step*n)."

    def apply(self, start, step, value):
        aligned = self._align_to_step(start, step, value)
        return (aligned,)

    def _align_to_step(self, start, step, value):
        """将数值对齐到最接近的 start + step*n"""
        if step <= 0:
            return start

        n = (value - start) / step
        n_floor = int(math.floor(n))
        n_ceil = n_floor + 1

        val_floor = start + n_floor * step
        val_ceil = start + n_ceil * step

        diff_floor = abs(value - val_floor)
        diff_ceil = abs(value - val_ceil)

        if diff_floor < diff_ceil:
            return val_floor
        elif diff_ceil < diff_floor:
            return val_ceil
        else:
            if abs(val_floor - start) <= abs(val_ceil - start):
                return val_floor
            else:
                return val_ceil


NODE_CLASS_MAPPINGS = {
    "IntegerSettingNode": IntegerSettingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntegerSettingNode": "Integer Setting (整数设置)",
}