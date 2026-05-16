import torch
import json


class ReferenceImageSelector:
    """参考图选择器节点
    
    输入:
    - reference_images: 参考图批次 (IMAGE, N张不同视角)
    - angle_map: JSON格式的角度映射字符串, 如 "[-90, -45, 0, 45, 90]"
    - yaw_angles: (可选) 偏航角数组 (FLOAT, 每帧一个值)
    - select_references: (可选) 是否挑选参考图；True=筛选+排序, False=仅排序(所有参考图都使用)
    
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
                "background_images": ("IMAGE",),
                "select_references": ("BOOLEAN", {"default": True, "label_on": "筛选+排序", "label_off": "仅排序"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("selected_images", "info")
    FUNCTION = "select"
    CATEGORY = "CustomNodes/SDPose"

    def select(self, reference_images, angle_map, yaw_angles=None, background_images=None, select_references=True):
        info_lines = []
        total_ref_count = reference_images.shape[0]
        info_lines.append(f"参考图总数: {total_ref_count}")
        info_lines.append(f"挑选模式: {'筛选+排序' if select_references else '仅排序(全部使用)'}")

        # ==================== 0. 空张量守卫 ====================
        if total_ref_count == 0:
            info_lines.append("参考图为空")
            info_lines.append("背景图: 无")
            return (reference_images.clone(), "\n".join(info_lines))

        # ==================== 1. yaw_angles 未接入 → 全量 1+4n ====================
        if yaw_angles is None:
            info_lines.append("yaw_angles 未接入, 直接输出全部参考图 (1+4n)")
            return self._build_batch_flat(total_ref_count, reference_images, info_lines, background_images)

        # ==================== 2. angle_map 无效 → 全量 1+4n ====================
        angle_map_list = self._parse_angle_map(angle_map, info_lines)
        if angle_map_list is None:
            info_lines.append("angle_map 无效, 直接输出全部参考图 (1+4n)")
            return self._build_batch_flat(total_ref_count, reference_images, info_lines, background_images)

        map_count = len(angle_map_list)
        if map_count != total_ref_count:
            info_lines.append(
                f"angle_map 数量({map_count})与参考图数量({total_ref_count})不匹配, 直接输出全部参考图 (1+4n)"
            )
            return self._build_batch_flat(total_ref_count, reference_images, info_lines, background_images)

        info_lines.append(f"角度映射: {angle_map_list}")

        # ==================== 3. 处理 yaw_angles 输入 ====================
        yaw_list = self._flatten_yaw(yaw_angles)

        if len(yaw_list) == 0:
            # yaw_angles 为空数组
            info_lines.append("偏航角数据为空, 直接输出全部参考图 (1+4n)")
            return self._build_batch_flat(total_ref_count, reference_images, info_lines, background_images)

        info_lines.append(
            f"片段偏航角范围: [{min(yaw_list):.1f}, {max(yaw_list):.1f}], 帧数: {len(yaw_list)}"
        )

        # ==================== 4. 筛选候选参考图 ====================
        yaw_min = min(yaw_list)
        yaw_max = max(yaw_list)

        if select_references:
            # 筛选模式：选出覆盖偏航角范围的参考图
            candidate_indices = self._filter_by_range(angle_map_list, yaw_min, yaw_max)

            if len(candidate_indices) == 0:
                info_lines.append("无参考图角度在偏航角范围内, 直接输出全部参考图 (1+4n)")
                return self._build_batch_flat(total_ref_count, reference_images, info_lines, background_images)

            candidate_angles = [angle_map_list[i] for i in candidate_indices]
            info_lines.append(f"候选参考图索引: {candidate_indices}, 角度: {candidate_angles}")
        else:
            # 仅排序模式：全部参考图都作为候选
            candidate_indices = list(range(total_ref_count))
            candidate_angles = [angle_map_list[i] for i in candidate_indices]
            info_lines.append(f"仅排序模式: 使用全部 {total_ref_count} 张参考图")

        # ==================== 5. 仅 1 张候选 → 出 1 张 (或含背景图) ====================
        if len(candidate_indices) == 1:
            solo_idx = candidate_indices[0]
            info_lines.append(f"仅1张候选参考图(索引{solo_idx}), 输出1张")
            solo_img = reference_images[solo_idx:solo_idx+1].clone()
            bg_block = self._build_background_block(background_images)
            if bg_block is None:
                info_lines.append("背景图: 无")
                return (solo_img, "\n".join(info_lines))
            # 主参考图 + 背景图块 + 主参考图×4
            main_tail = solo_img.repeat(4, 1, 1, 1)
            result = torch.cat([solo_img, bg_block, main_tail], dim=0)
            info_lines.append(f"背景图使用: 有 ({background_images.shape[0]} 张 ×4 拼接)")
            info_lines.append(f"输出图像张数: {result.shape[0]}")
            return (result, "\n".join(info_lines))

        # ==================== 6. 确定主参考图 (覆盖帧数最多) ====================
        main_index = self._find_main_reference(candidate_indices, angle_map_list, yaw_list)
        info_lines.append(f"主参考图索引: {main_index}, 角度: {angle_map_list[main_index]:.1f}°")

        # ==================== 7. 排序辅助参考图 ====================
        first_frame_yaw = yaw_list[0]
        aux_indices = [idx for idx in candidate_indices if idx != main_index]
        # 按与首帧偏航角偏差从大到小排列 (最后一张最贴合首帧)
        aux_indices.sort(key=lambda i: self._angular_distance(angle_map_list[i], first_frame_yaw), reverse=True)

        info_lines.append(f"首帧偏航角: {first_frame_yaw:.1f}°")
        for i, idx in enumerate(aux_indices):
            diff = self._angular_distance(angle_map_list[idx], first_frame_yaw)
            info_lines.append(
                f"  辅助参考图[{i}]: 索引{idx}, 角度{angle_map_list[idx]:.1f}°, 偏差{diff:.1f}°"
            )

        # ==================== 8. 边界处理 ====================
        # 若主参考图角度最接近首帧偏航角, 追加主参考图副本
        closest_to_first = min(
            candidate_indices, key=lambda i: self._angular_distance(angle_map_list[i], first_frame_yaw)
        )
        if closest_to_first == main_index:
            info_lines.append("主参考图最贴合首帧偏航角, 在末尾追加主参考图副本")
            aux_indices.append(main_index)

        # ==================== 9. 构建输出批次 ====================
        ordered_indices = [main_index] + aux_indices
        info_lines.append(f"最终排序索引: {ordered_indices}")

        selected_tensor = self._build_batch_ordered(ordered_indices, reference_images, info_lines, background_images)
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
        """筛选能包住 [yaw_min, yaw_max] 的参考图索引 (带环形环绕)
        区间内的所有参考图 + 左右各最邻近一张
        """
        candidates = set()

        # 1. 区间内的参考图 (使用环形区间判断)
        for i, a in enumerate(angles):
            if self._angle_in_range(a, yaw_min, yaw_max):
                candidates.add(i)

        # 2. 左侧最近邻 (最接近 yaw_min 且不在区间内)
        left_idx = None
        left_best = float('inf')
        for i, a in enumerate(angles):
            if not self._angle_in_range(a, yaw_min, yaw_max):
                dist = self._angular_distance(a, yaw_min)
                offset = (yaw_min - a) % 360
                if offset <= 180 and dist < left_best:
                    left_best = dist
                    left_idx = i
        if left_idx is not None:
            candidates.add(left_idx)

        # 3. 右侧最近邻 (最接近 yaw_max 且不在区间内)
        right_idx = None
        right_best = float('inf')
        for i, a in enumerate(angles):
            if not self._angle_in_range(a, yaw_min, yaw_max):
                dist = self._angular_distance(a, yaw_max)
                offset = (a - yaw_max) % 360
                if offset <= 180 and dist < right_best:
                    right_best = dist
                    right_idx = i
        if right_idx is not None:
            candidates.add(right_idx)

        return sorted(candidates)

    def _find_main_reference(self, candidate_indices, angles, yaw_list):
        """找覆盖帧数最多的参考图作为主参考图"""
        counts = {idx: 0 for idx in candidate_indices}
        for yaw in yaw_list:
            best_idx = min(candidate_indices, key=lambda i: self._angular_distance(angles[i], yaw))
            counts[best_idx] += 1
        return max(counts, key=counts.get)

    def _build_batch_flat(self, total_count, reference_images, info_lines, background_images=None):
        """全量 1+4n 输出: 第0张×1 + (可选背景图块) + 其余每张×4"""
        if total_count == 0:
            info_lines.append("背景图: 无")
            return (reference_images.clone(), "\n".join(info_lines))
        if total_count == 1:
            info_lines.append("背景图: 无")
            return (reference_images.clone(), "\n".join(info_lines))

        parts = [reference_images[0:1]]
        # 先检查是否需要插入背景图块
        bg_block = self._build_background_block(background_images)
        if bg_block is not None:
            parts.append(bg_block)
            info_lines.append(f"背景图使用: 有 ({background_images.shape[0]} 张 ×4 拼接)")
        else:
            info_lines.append("背景图: 无")
        for i in range(1, total_count):
            parts.append(reference_images[i:i+1].repeat(4, 1, 1, 1))

        result = torch.cat(parts, dim=0)
        info_lines.append(f"输出图像张数: {result.shape[0]}")
        return (result, "\n".join(info_lines))

    def _build_batch_ordered(self, ordered_indices, reference_images, info_lines, background_images=None):
        """按排序索引构建批次: 第0张×1 + (可选背景图块) + 其余每张×4"""
        if len(ordered_indices) == 0:
            return reference_images[0:1].clone()

        parts = [reference_images[ordered_indices[0]:ordered_indices[0]+1].clone()]
        # 在主参考图和辅助参考图之间插入背景图块
        bg_block = self._build_background_block(background_images)
        if bg_block is not None:
            parts.append(bg_block)
            info_lines.append(f"背景图使用: 有 ({background_images.shape[0]} 张 ×4 拼接)")
        else:
            info_lines.append("背景图: 无")

        for idx in ordered_indices[1:]:
            parts.append(reference_images[idx:idx+1].repeat(4, 1, 1, 1))

        return torch.cat(parts, dim=0)

    def _angular_distance(self, a, b):
        """计算两个角度 (-180~180) 之间的最短弧距离 (0~180)"""
        diff = (a - b) % 360
        if diff > 180:
            diff = 360 - diff
        return diff

    def _angle_in_range(self, angle, min_a, max_a):
        """判断角度是否在环形区间 [min_a, max_a] 内 (带环绕处理)"""
        span = (max_a - min_a) % 360
        offset = (angle - min_a) % 360
        return offset <= span or abs(span - 360) < 1e-9

    def _build_background_block(self, background_images):
        """将每张背景图复制4份, 拼接成块; 无效输入返回 None"""
        if background_images is None or background_images.shape[0] == 0:
            return None
        parts = []
        for i in range(background_images.shape[0]):
            parts.append(background_images[i:i+1].repeat(4, 1, 1, 1))
        return torch.cat(parts, dim=0)


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "ReferenceImageSelector": ReferenceImageSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReferenceImageSelector": "Reference Image Selector (参考图选择器)",
}