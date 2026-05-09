import torch
import json


class ReferenceImageSelector:
    """参考图选择器节点
    
    输入:
    - reference_images: 参考图批次 (IMAGE, N张不同视角)
    - angle_map: JSON格式的角度映射字符串, 如 "[-90, -45, 0, 45, 90]"
    - yaw_angles: (可选) 偏航角数组 (FLOAT, 每帧一个值)
    
    输出:
    - selected_images: 按规则拼接后的图像批次
    - info: 调试/状态信息
    """
    
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",),
                "angle_map": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "yaw_angles": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("selected_images", "info")
    FUNCTION = "select"
    CATEGORY = "CustomNodes/SDPose"

    def select(self, reference_images, angle_map, yaw_angles=None):
        info_lines = []
        total_ref_count = reference_images.shape[0]
        info_lines.append(f"参考图总数: {total_ref_count}")

        # ==================== 0. 空张量守卫 ====================
        if total_ref_count == 0:
            info_lines.append("参考图为空")
            return (reference_images.clone(), "\n".join(info_lines))

        # ==================== 1. yaw_angles 未接入 → 全量 1+4n ====================
        if yaw_angles is None:
            info_lines.append("yaw_angles 未接入, 直接输出全部参考图 (1+4n)")
            return self._build_batch_flat(total_ref_count, reference_images, info_lines)

        # ==================== 2. angle_map 无效 → 全量 1+4n ====================
        angle_map_list = self._parse_angle_map(angle_map, info_lines)
        if angle_map_list is None:
            info_lines.append("angle_map 无效, 直接输出全部参考图 (1+4n)")
            return self._build_batch_flat(total_ref_count, reference_images, info_lines)

        map_count = len(angle_map_list)
        if map_count != total_ref_count:
            info_lines.append(
                f"angle_map 数量({map_count})与参考图数量({total_ref_count})不匹配, 直接输出全部参考图 (1+4n)"
            )
            return self._build_batch_flat(total_ref_count, reference_images, info_lines)

        info_lines.append(f"角度映射: {angle_map_list}")

        # ==================== 3. 处理 yaw_angles 输入 ====================
        yaw_list = self._flatten_yaw(yaw_angles)

        if len(yaw_list) == 0:
            # yaw_angles 为空数组
            info_lines.append("偏航角数据为空, 直接输出全部参考图 (1+4n)")
            return self._build_batch_flat(total_ref_count, reference_images, info_lines)

        info_lines.append(
            f"片段偏航角范围: [{min(yaw_list):.1f}, {max(yaw_list):.1f}], 帧数: {len(yaw_list)}"
        )

        # ==================== 4. 筛选覆盖偏航角范围的参考图 ====================
        yaw_min = min(yaw_list)
        yaw_max = max(yaw_list)

        candidate_indices = self._filter_by_range(angle_map_list, yaw_min, yaw_max)

        if len(candidate_indices) == 0:
            info_lines.append("无参考图角度在偏航角范围内, 直接输出全部参考图 (1+4n)")
            return self._build_batch_flat(total_ref_count, reference_images, info_lines)

        candidate_angles = [angle_map_list[i] for i in candidate_indices]
        info_lines.append(f"候选参考图索引: {candidate_indices}, 角度: {candidate_angles}")

        # ==================== 5. 仅 1 张候选 → 出 1 张 ====================
        if len(candidate_indices) == 1:
            solo_idx = candidate_indices[0]
            info_lines.append(f"仅1张候选参考图(索引{solo_idx}), 输出1张")
            return (reference_images[solo_idx:solo_idx+1].clone(), "\n".join(info_lines))

        # ==================== 6. 确定主参考图 (覆盖帧数最多) ====================
        main_index = self._find_main_reference(candidate_indices, angle_map_list, yaw_list)
        info_lines.append(f"主参考图索引: {main_index}, 角度: {angle_map_list[main_index]:.1f}°")

        # ==================== 7. 排序辅助参考图 ====================
        first_frame_yaw = yaw_list[0]
        aux_indices = [idx for idx in candidate_indices if idx != main_index]
        # 按与首帧偏航角偏差从大到小排列 (最后一张最贴合首帧)
        aux_indices.sort(key=lambda i: abs(angle_map_list[i] - first_frame_yaw), reverse=True)

        info_lines.append(f"首帧偏航角: {first_frame_yaw:.1f}°")
        for i, idx in enumerate(aux_indices):
            diff = abs(angle_map_list[idx] - first_frame_yaw)
            info_lines.append(
                f"  辅助参考图[{i}]: 索引{idx}, 角度{angle_map_list[idx]:.1f}°, 偏差{diff:.1f}°"
            )

        # ==================== 8. 边界处理 ====================
        # 若主参考图角度最接近首帧偏航角, 追加主参考图副本
        closest_to_first = min(
            candidate_indices, key=lambda i: abs(angle_map_list[i] - first_frame_yaw)
        )
        if closest_to_first == main_index:
            info_lines.append("主参考图最贴合首帧偏航角, 在末尾追加主参考图副本")
            aux_indices.append(main_index)

        # ==================== 9. 构建输出批次 ====================
        ordered_indices = [main_index] + aux_indices
        info_lines.append(f"最终排序索引: {ordered_indices}")

        selected_tensor = self._build_batch_ordered(ordered_indices, reference_images)
        info_lines.append(f"输出图像张数: {selected_tensor.shape[0]}")

        info = "\n".join(info_lines)
        return (selected_tensor, info)

    # ==================== 辅助方法 ====================

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
        """将各种格式的 yaw_angles 展平为一维列表"""
        if isinstance(yaw_angles, (int, float)):
            return [float(yaw_angles)]
        if isinstance(yaw_angles, list):
            return [float(v) for v in yaw_angles]
        if isinstance(yaw_angles, torch.Tensor):
            flat = yaw_angles.flatten().tolist()
            return [float(v) for v in flat]
        return []

    def _filter_by_range(self, angles, yaw_min, yaw_max):
        """筛选能包住 [yaw_min, yaw_max] 的参考图索引
        区间内的所有参考图 + 左右各最邻近一张
        """
        candidates = set()

        # 1. 区间内的参考图
        for i, a in enumerate(angles):
            if yaw_min <= a <= yaw_max:
                candidates.add(i)

        # 2. 左侧最近邻 (最接近 yaw_min 且 < yaw_min)
        left_idx = None
        left_best = float('inf')
        for i, a in enumerate(angles):
            if a < yaw_min:
                dist = yaw_min - a
                if dist < left_best:
                    left_best = dist
                    left_idx = i
        if left_idx is not None:
            candidates.add(left_idx)

        # 3. 右侧最近邻 (最接近 yaw_max 且 > yaw_max)
        right_idx = None
        right_best = float('inf')
        for i, a in enumerate(angles):
            if a > yaw_max:
                dist = a - yaw_max
                if dist < right_best:
                    right_best = dist
                    right_idx = i
        if right_idx is not None:
            candidates.add(right_idx)

        return sorted(candidates)

    def _find_main_reference(self, candidate_indices, angles, yaw_list):
        """找覆盖帧数最多的参考图作为主参考图"""
        counts = {idx: 0 for idx in candidate_indices}
        for yaw in yaw_list:
            best_idx = min(candidate_indices, key=lambda i: abs(angles[i] - yaw))
            counts[best_idx] += 1
        return max(counts, key=counts.get)

    def _build_batch_flat(self, total_count, reference_images, info_lines):
        """全量 1+4n 输出: 第0张×1 + 其余每张×4"""
        if total_count == 0:
            return (reference_images.clone(), "\n".join(info_lines))
        if total_count == 1:
            return (reference_images.clone(), "\n".join(info_lines))

        parts = [reference_images[0:1]]
        for i in range(1, total_count):
            parts.append(reference_images[i:i+1].repeat(4, 1, 1, 1))

        info_lines.append(f"输出图像张数: {1 + 4 * (total_count - 1)}")
        return (torch.cat(parts, dim=0), "\n".join(info_lines))

    def _build_batch_ordered(self, ordered_indices, reference_images):
        """按排序索引构建批次: 第0张×1 + 其余每张×4"""
        if len(ordered_indices) == 0:
            return reference_images[0:1].clone()

        parts = [reference_images[ordered_indices[0]:ordered_indices[0]+1].clone()]

        for idx in ordered_indices[1:]:
            parts.append(reference_images[idx:idx+1].repeat(4, 1, 1, 1))

        return torch.cat(parts, dim=0)


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "ReferenceImageSelector": ReferenceImageSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReferenceImageSelector": "Reference Image Selector (参考图选择器)",
}
