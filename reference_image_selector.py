import torch
import json
import math


class ReferenceImageUtils:
    """可复用的参考图选择/排序/拼接纯逻辑
    
    供 ReferenceImageSelector 和 ContextWindows 动态前缀共用。
    所有方法均为类方法，无需实例化。
    """
    
    _ANGLE_CACHE = {}
    
    @classmethod
    def _angular_distance(cls, a, b):
        """计算两个角度 (-180~180) 之间的最短弧距离 (0~180)"""
        diff = (a - b) % 360
        if diff > 180:
            diff = 360 - diff
        return diff
    
    @classmethod
    def _angle_in_range(cls, angle, min_a, max_a):
        """判断角度是否在环形区间 [min_a, max_a] 内 (带环绕处理)"""
        span = (max_a - min_a) % 360
        offset = (angle - min_a) % 360
        return offset <= span or abs(span - 360) < 1e-9
    
    @classmethod
    def _flatten_yaw(cls, yaw_angles):
        """将各种格式的 yaw_angles 展平为一维列表"""
        if isinstance(yaw_angles, (int, float)):
            return [float(yaw_angles)]
        if isinstance(yaw_angles, list):
            return [float(v) for v in yaw_angles]
        if isinstance(yaw_angles, torch.Tensor):
            flat = yaw_angles.flatten().tolist()
            return [float(v) for v in flat]
        return []
    
    @classmethod
    def _filter_by_range(cls, angles, yaw_min, yaw_max):
        """筛选能包住 [yaw_min, yaw_max] 的参考图索引 (带环形环绕)
        区间内的所有参考图 + 左右各最邻近一张
        """
        candidates = set()
        
        # 1. 区间内的参考图
        for i, a in enumerate(angles):
            if cls._angle_in_range(a, yaw_min, yaw_max):
                candidates.add(i)
        
        # 2. 左侧最近邻
        left_idx = None
        left_best = float('inf')
        for i, a in enumerate(angles):
            if not cls._angle_in_range(a, yaw_min, yaw_max):
                dist = cls._angular_distance(a, yaw_min)
                offset = (yaw_min - a) % 360
                if offset <= 180 and dist < left_best:
                    left_best = dist
                    left_idx = i
        if left_idx is not None:
            candidates.add(left_idx)
        
        # 3. 右侧最近邻
        right_idx = None
        right_best = float('inf')
        for i, a in enumerate(angles):
            if not cls._angle_in_range(a, yaw_min, yaw_max):
                dist = cls._angular_distance(a, yaw_max)
                offset = (a - yaw_max) % 360
                if offset <= 180 and dist < right_best:
                    right_best = dist
                    right_idx = i
        if right_idx is not None:
            candidates.add(right_idx)
        
        return sorted(candidates)
    
    @classmethod
    def _find_main_reference(cls, candidate_indices, angles, yaw_list):
        """找覆盖帧数最多的参考图作为主参考图"""
        counts = {idx: 0 for idx in candidate_indices}
        for yaw in yaw_list:
            best_idx = min(candidate_indices, key=lambda i: cls._angular_distance(angles[i], yaw))
            counts[best_idx] += 1
        return max(counts, key=counts.get)
    
    @classmethod
    def _build_background_block(cls, background_images):
        """将每张背景图复制4份, 拼接成块; 无效输入返回 None"""
        if background_images is None or background_images.shape[0] == 0:
            return None
        parts = []
        for i in range(background_images.shape[0]):
            parts.append(background_images[i:i+1].repeat(4, 1, 1, 1))
        return torch.cat(parts, dim=0)
    
    @classmethod
    def _build_batch_flat(cls, total_count, reference_images, info_lines, background_images=None):
        """全量 1+4n 输出: 第0张×1 + 前N-1张×4 + (可选背景图块) + 最后一张×4"""
        if total_count == 0:
            info_lines.append("背景图: 无")
            return (reference_images.clone(), info_lines)
        if total_count == 1:
            info_lines.append("背景图: 无")
            return (reference_images.clone(), info_lines)
        
        parts = [reference_images[0:1]]
        # 前 total_count-1 张（除最后一张）每张×4
        for i in range(1, total_count - 1):
            parts.append(reference_images[i:i+1].repeat(4, 1, 1, 1))
        # 背景图插入到倒数第二张和最后一张之间
        bg_block = cls._build_background_block(background_images)
        if bg_block is not None:
            parts.append(bg_block)
            info_lines.append(f"背景图使用: 有 ({background_images.shape[0]} 张 ×4 拼接)")
        else:
            info_lines.append("背景图: 无")
        # 最后一张×4
        parts.append(reference_images[total_count-1:total_count-1+1].repeat(4, 1, 1, 1))
        
        result = torch.cat(parts, dim=0)
        info_lines.append(f"输出图像张数: {result.shape[0]}")
        return (result, info_lines)
    
    @classmethod
    def _build_batch_ordered(cls, ordered_indices, reference_images, info_lines, background_images=None):
        """按排序索引构建批次: 第0张×1 + 中间每张×4 + (可选背景图块) + 最后一张×4"""
        if len(ordered_indices) == 0:
            return reference_images[0:1].clone()
        
        parts = [reference_images[ordered_indices[0]:ordered_indices[0]+1].clone()]
        
        # 中间元素（索引 1 ~ N-2）每张×4
        for idx in ordered_indices[1:-1]:
            parts.append(reference_images[idx:idx+1].repeat(4, 1, 1, 1))
        
        # 背景图插入到中间元素和最后一个元素之间
        bg_block = cls._build_background_block(background_images)
        if bg_block is not None:
            parts.append(bg_block)
            info_lines.append(f"背景图使用: 有 ({background_images.shape[0]} 张 ×4 拼接)")
        else:
            info_lines.append("背景图: 无")
        
        # 最后一张×4
        parts.append(reference_images[ordered_indices[-1]:ordered_indices[-1]+1].repeat(4, 1, 1, 1))
        
        return torch.cat(parts, dim=0)
    
    @classmethod
    def select_and_order(cls, reference_images, angle_map_list, yaw_list,
                         select_references=True, allow_switch_main=True,
                         background_images=None):
        """核心选择+排序+拼接逻辑
        
        Args:
            reference_images: (N, H, W, C) 原始参考图批次
            angle_map_list: list[float] 每张参考图对应的角度
            yaw_list: list[float] 目标偏航角序列
            select_references: 是否筛选（False=全部使用仅排序）
            allow_switch_main: 是否允许更换主参考图
            background_images: 可选的背景图批次
        
        Returns:
            (image_batch: torch.Tensor, info_lines: list[str])
        """
        info_lines = []
        total_ref_count = reference_images.shape[0]
        
        yaw_min = min(yaw_list)
        yaw_max = max(yaw_list)
        
        info_lines.append(
            f"片段偏航角范围: [{yaw_min:.1f}, {yaw_max:.1f}], 帧数: {len(yaw_list)}"
        )
        
        # 筛选候选参考图
        if select_references:
            candidate_indices = cls._filter_by_range(angle_map_list, yaw_min, yaw_max)
            
            if len(candidate_indices) == 0:
                info_lines.append("无参考图角度在偏航角范围内")
                return cls._build_batch_flat(
                    total_ref_count, reference_images, info_lines, background_images
                )
            
            candidate_angles = [angle_map_list[i] for i in candidate_indices]
            info_lines.append(f"候选参考图索引: {candidate_indices}, 角度: {candidate_angles}")
        else:
            candidate_indices = list(range(total_ref_count))
            candidate_angles = [angle_map_list[i] for i in candidate_indices]
            info_lines.append(f"仅排序模式: 使用全部 {total_ref_count} 张参考图")
        
        # 仅1张候选
        if len(candidate_indices) == 1:
            solo_idx = candidate_indices[0]
            info_lines.append(f"仅1张候选参考图(索引{solo_idx}), 输出1张")
            solo_img = reference_images[solo_idx:solo_idx+1].clone()
            bg_block = cls._build_background_block(background_images)
            if bg_block is None:
                info_lines.append("背景图: 无")
                return (solo_img, info_lines)
            main_tail = solo_img.repeat(4, 1, 1, 1)
            result = torch.cat([solo_img, bg_block, main_tail], dim=0)
            info_lines.append(f"背景图使用: 有 ({background_images.shape[0]} 张 ×4 拼接)")
            info_lines.append(f"输出图像张数: {result.shape[0]}")
            return (result, info_lines)
        
        # 确定主参考图
        main_index = cls._find_main_reference(candidate_indices, angle_map_list, yaw_list)
        
        if not allow_switch_main:
            if 0 in candidate_indices:
                main_index = 0
                info_lines.append("主参考图固定为第一张 (不允许更换)")
            else:
                # 方案A: 强制将第一张纳入候选集
                info_lines.append("固定第一张: 索引0不在候选集内, 强制加入候选集")
                candidate_indices.insert(0, 0)
                main_index = 0
        
        info_lines.append(f"主参考图索引: {main_index}, 角度: {angle_map_list[main_index]:.1f}°")
        
        # 排序辅助参考图
        first_frame_yaw = yaw_list[0]
        aux_indices = [idx for idx in candidate_indices if idx != main_index]
        aux_indices.sort(
            key=lambda i: cls._angular_distance(angle_map_list[i], first_frame_yaw),
            reverse=True
        )
        
        info_lines.append(f"首帧偏航角: {first_frame_yaw:.1f}°")
        for i, idx in enumerate(aux_indices):
            diff = cls._angular_distance(angle_map_list[idx], first_frame_yaw)
            info_lines.append(
                f"  辅助参考图[{i}]: 索引{idx}, 角度{angle_map_list[idx]:.1f}°, 偏差{diff:.1f}°"
            )
        
        # 边界处理
        closest_to_first = min(
            candidate_indices,
            key=lambda i: cls._angular_distance(angle_map_list[i], first_frame_yaw)
        )
        if closest_to_first == main_index:
            info_lines.append("主参考图最贴合首帧偏航角, 在末尾追加主参考图副本")
            aux_indices.append(main_index)
        
        # 构建输出批次
        ordered_indices = [main_index] + aux_indices
        info_lines.append(f"最终排序索引: {ordered_indices}")
        
        selected_tensor = cls._build_batch_ordered(
            ordered_indices, reference_images, info_lines, background_images
        )
        info_lines.append(f"输出图像张数: {selected_tensor.shape[0]}")
        
        return (selected_tensor, info_lines)


class ReferenceImageSelector:
    """参考图选择器节点
    
    输入:
    - reference_images: 参考图批次 (IMAGE, N张不同视角)
    - angle_map: JSON格式的角度映射字符串, 如 "[-90, -45, 0, 45, 90]"
    - yaw_angles: (可选) 偏航角数组 (FLOAT, 每帧一个值)
    - select_references: (可选) 是否挑选参考图；True=筛选+排序, False=仅排序(所有参考图都使用)
    
    输出:
    - selected_images: 排序后的原始参考图（不做1+4n，内部1+4n交给主节点处理）
    - raw_reference_images: 未排序的原始参考图（用于 ContextWindows 动态前缀，因为内部会重新筛选排序）
    - info: 调试/状态信息
    - reference_angle_map: 验证后的角度映射 JSON 字符串 (给 ContextWindows 动态前缀)
    """
    
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE", {"tooltip": "参考图批次，用于按角度筛选和排序。Reference image batch for angle-based filtering and sorting."}),
                "angle_map": ("STRING", {"default": "", "multiline": False, "tooltip": "角度映射 JSON，指定每张参考图对应的角度或范围。Angle map JSON specifying the angle or range for each reference image."}),
            },
            "optional": {
                "yaw_angles": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "tooltip": "目标偏航角，用于按角度筛选最匹配的参考图。Target yaw angle for filtering the best matching reference image."}),
                "background_images": ("IMAGE", {"tooltip": "背景图批次，将直接拼接到选中参考图之后。Background image batch, appended directly after the selected references."}),
                "select_references": ("BOOLEAN", {"default": True, "label_on": "筛选+排序", "label_off": "仅排序", "tooltip": "True=按角度筛选并排序；False=仅排序不筛选。True=filter and sort by angle; False=sort only without filtering."}),
                "allow_switch_main": ("BOOLEAN", {"default": True, "label_on": "允许更换", "label_off": "固定第一张", "tooltip": "True=允许交换第一张主参考图；False=固定第一张不变。True=allow swapping the first main reference image; False=keep the first image fixed."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("selected_images", "raw_reference_images", "info", "reference_angle_map")
    FUNCTION = "select"
    CATEGORY = "CustomNodes/SDPose"
    DESCRIPTION = "通过角度映射对参考图进行筛选和排序，并支持拼接背景图。Filter and sort reference images by angle map, with optional background image concatenation."

    @staticmethod
    def _append_background(images, background_images, info_lines):
        """将 background_images 拼接到 images 末尾（如果存在）"""
        if background_images is not None and background_images.shape[0] > 0:
            parts = [images]
            for i in range(background_images.shape[0]):
                parts.append(background_images[i:i+1].clone())
            info_lines.append(f"背景图使用: 有 ({background_images.shape[0]} 张，原始图片直接插入)")
            return torch.cat(parts, dim=0)
        info_lines.append("背景图: 无")
        return images

    def select(self, reference_images, angle_map, yaw_angles=None, background_images=None, select_references=True, allow_switch_main=True):
        info_lines = []
        total_ref_count = reference_images.shape[0]
        info_lines.append(f"参考图总数: {total_ref_count}")
        info_lines.append(f"挑选模式: {'筛选+排序' if select_references else '仅排序(全部使用)'}")

        # ---- 解析 angle_map ----
        angle_map_list = self._parse_angle_map(angle_map, info_lines)
        validated_angle_map = json.dumps(angle_map_list) if angle_map_list is not None else ""
        raw_images = reference_images.clone()

        # ==================== 0. 空张量守卫 ====================
        if total_ref_count == 0:
            info_lines.append("参考图为空")
            result = self._append_background(raw_images, background_images, info_lines)
            info = "\n".join(info_lines)
            return (result, result, info, validated_angle_map)

        # ==================== 1. yaw_angles 未接入 → 直接输出原始参考图（含背景图） ====================
        if yaw_angles is None:
            info_lines.append("yaw_angles 未接入, 直接输出全部参考图")
            result = self._append_background(raw_images, background_images, info_lines)
            info = "\n".join(info_lines)
            return (result, raw_images, info, validated_angle_map)

        # ==================== 2. angle_map 无效 → 直接输出原始参考图（含背景图） ====================
        if angle_map_list is None:
            info_lines.append("angle_map 无效, 直接输出全部参考图")
            result = self._append_background(raw_images, background_images, info_lines)
            info = "\n".join(info_lines)
            return (result, raw_images, info, validated_angle_map)

        map_count = len(angle_map_list)
        if map_count != total_ref_count:
            info_lines.append(
                f"angle_map 数量({map_count})与参考图数量({total_ref_count})不匹配, 直接输出全部参考图"
            )
            result = self._append_background(raw_images, background_images, info_lines)
            info = "\n".join(info_lines)
            return (result, raw_images, info, validated_angle_map)

        info_lines.append(f"角度映射: {angle_map_list}")

        # ==================== 3. 处理 yaw_angles 输入 ====================
        yaw_list = ReferenceImageUtils._flatten_yaw(yaw_angles)

        if len(yaw_list) <= 1:
            if len(yaw_list) == 0:
                info_lines.append("偏航角数据为空, 直接输出全部参考图")
            else:
                info_lines.append("偏航角数据不足(仅1帧), 视为无效输入, 直接输出全部参考图")
            result = self._append_background(raw_images, background_images, info_lines)
            info = "\n".join(info_lines)
            return (result, raw_images, info, validated_angle_map)

        # ==================== 4-9. 选择排序核心逻辑 ====================
        candidate_indices = ReferenceImageUtils._filter_by_range(angle_map_list, min(yaw_list), max(yaw_list)) if select_references else list(range(total_ref_count))
        if len(candidate_indices) == 0:
            info_lines.append("无参考图角度在偏航角范围内, 直接输出全部参考图")
            result = self._append_background(raw_images, background_images, info_lines)
            info = "\n".join(info_lines)
            return (result, raw_images, info, validated_angle_map)

        candidate_angles = [angle_map_list[i] for i in candidate_indices]
        info_lines.append(f"候选参考图索引: {candidate_indices}, 角度: {candidate_angles}")

        # 确定主参考图
        main_index = ReferenceImageUtils._find_main_reference(candidate_indices, angle_map_list, yaw_list)

        if not allow_switch_main:
            if 0 in candidate_indices:
                main_index = 0
                info_lines.append("主参考图固定为第一张 (不允许更换)")
            else:
                # 方案A: 强制将第一张纳入候选集
                info_lines.append("固定第一张: 索引0不在候选集内, 强制加入候选集")
                candidate_indices.insert(0, 0)
                main_index = 0

        info_lines.append(f"主参考图索引: {main_index}, 角度: {angle_map_list[main_index]:.1f}°")

        first_frame_yaw = yaw_list[0]
        aux_indices = [idx for idx in candidate_indices if idx != main_index]
        aux_indices.sort(key=lambda i: ReferenceImageUtils._angular_distance(angle_map_list[i], first_frame_yaw), reverse=True)

        closest_to_first = min(candidate_indices, key=lambda i: ReferenceImageUtils._angular_distance(angle_map_list[i], first_frame_yaw))
        if closest_to_first == main_index:
            aux_indices.append(main_index)

        ordered_indices = [main_index] + aux_indices
        info_lines.append(f"最终排序索引: {ordered_indices}")

        # 构建输出：主参考图 + 前N-1张辅助参考图（原始图片）+ 背景图块 + 最后一张辅助参考图（原始图片）
        output_parts = [reference_images[main_index:main_index+1].clone()]
        
        # 辅助参考图分为前部(除最后一张)和尾部(最后一张)
        if len(aux_indices) > 0:
            # 前部：前 len(aux_indices)-1 张辅助参考图
            for idx in aux_indices[:-1]:
                output_parts.append(reference_images[idx:idx+1].clone())
            # 背景图插入到前部和最后一张之间
            if background_images is not None and background_images.shape[0] > 0:
                for i in range(background_images.shape[0]):
                    output_parts.append(background_images[i:i+1].clone())
                info_lines.append(f"背景图使用: 有 ({background_images.shape[0]} 张，原始图片直接插入)")
            else:
                info_lines.append("背景图: 无")
            # 尾部：最后一张辅助参考图
            output_parts.append(reference_images[aux_indices[-1]:aux_indices[-1]+1].clone())
        else:
            # 没有辅助参考图：只有主参考图，背景图后面补主参考图副本
            if background_images is not None and background_images.shape[0] > 0:
                for i in range(background_images.shape[0]):
                    output_parts.append(background_images[i:i+1].clone())
                info_lines.append(f"背景图使用: 有 ({background_images.shape[0]} 张，原始图片直接插入)")
                output_parts.append(reference_images[main_index:main_index+1].clone())
                info_lines.append("辅助参考图: 无, 在背景图后追加主参考图副本")
            else:
                info_lines.append("背景图: 无")
        
        selected_images = torch.cat(output_parts, dim=0)
        info_lines.append(f"输出图像张数: {selected_images.shape[0]}")

        info = "\n".join(info_lines)
        return (selected_images, raw_images, info, validated_angle_map)

    # ==================== 辅助方法（委托给 ReferenceImageUtils） ====================

    def _parse_angle_map(self, angle_map, info_lines):
        """解析 angle_map JSON 字符串, 失败返回 None"""
        if not angle_map or not angle_map.strip():
            info_lines.append("angle_map 为空")
            return None
        try:
            result = json.loads(angle_map)
            if not isinstance(result, list):
                raise ValueError("angle_map 不是数组格式")
            return [float(v) for v in result]
        except Exception as e:
            info_lines.append(f"angle_map 解析失败 ({e})")
            return None

    def _flatten_yaw(self, yaw_angles):
        """委托: 展平 yaw_angles"""
        return ReferenceImageUtils._flatten_yaw(yaw_angles)

    def _filter_by_range(self, angles, yaw_min, yaw_max):
        """委托: 按偏航角范围筛选"""
        return ReferenceImageUtils._filter_by_range(angles, yaw_min, yaw_max)

    def _find_main_reference(self, candidate_indices, angles, yaw_list):
        """委托: 找主参考图"""
        return ReferenceImageUtils._find_main_reference(candidate_indices, angles, yaw_list)

    def _build_batch_flat(self, total_count, reference_images, info_lines, background_images=None):
        """委托: 全量 1+4n 输出"""
        return ReferenceImageUtils._build_batch_flat(total_count, reference_images, info_lines, background_images)

    def _build_batch_ordered(self, ordered_indices, reference_images, info_lines, background_images=None):
        """委托: 按排序索引构建批次"""
        return ReferenceImageUtils._build_batch_ordered(ordered_indices, reference_images, info_lines, background_images)

    def _angular_distance(self, a, b):
        """委托: 角度弧距"""
        return ReferenceImageUtils._angular_distance(a, b)

    def _angle_in_range(self, angle, min_a, max_a):
        """委托: 环形区间判断"""
        return ReferenceImageUtils._angle_in_range(angle, min_a, max_a)

    def _build_background_block(self, background_images):
        """委托: 背景图块构建"""
        return ReferenceImageUtils._build_background_block(background_images)


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "ReferenceImageSelector": ReferenceImageSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReferenceImageSelector": "Reference Image Selector (参考图选择器)",
}