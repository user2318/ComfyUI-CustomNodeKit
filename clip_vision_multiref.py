"""CLIP Vision Multi-Reference Switch Node.

Preprocesses CLIP vision output for multi-reference support in Wan SCAIL models.
When enabled, flattens batch dimension [N, 257, 1280] into token dimension [1, N*257, 1280],
so all reference images pass through the model's MLPProj instead of only the first one.

Place between CLIPVisionEncode and WanSCAILToVideoMultiRef / WanAnimateToVideoCustom.
"""

import copy
import torch
import logging

from comfy_api.latest import io


class CLIPVisionMultiRefSwitch(io.ComfyNode):
    """Switch to enable/disable multi-image CLIP feature concatenation.
    
    When enabled, all images in a batch CLIP vision output contribute to conditioning
    by concatenating their patch tokens in the token dimension.
    When disabled, the original CLIP vision output is passed through unchanged
    (default behavior: only the first image's features are used by the model).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPVisionMultiRefSwitch",
            category="conditioning/video_models/scail",
            display_name="CLIP Vision Multi-Ref Switch",
            is_experimental=True,
            inputs=[
                io.ClipVisionOutput.Input(
                    "clip_vision_output",
                    tooltip="CLIP vision output from CLIPVisionEncode. Supports batch (N images).",
                ),
                io.Boolean.Input(
                    "enabled",
                    default=False,
                    tooltip="Enable multi-image CLIP feature concatenation. "
                            "When ON, all N images in the batch contribute. "
                            "When OFF, passes through unchanged (only 1st image used).",
                ),
            ],
            outputs=[
                io.ClipVisionOutput.Output(
                    display_name="clip_vision_output",
                    tooltip="Processed CLIP vision output. If enabled, penultimate_hidden_states "
                            "is reshaped from [N, 257, 1280] to [1, N*257, 1280].",
                ),
            ],
        )

    @classmethod
    def execute(cls, clip_vision_output, enabled=False) -> io.NodeOutput:
        if clip_vision_output is None:
            return io.NodeOutput(None)

        if not enabled:
            # Pass through unchanged
            return io.NodeOutput(clip_vision_output)

        # Check if multi-image concatenation is needed
        hs = clip_vision_output.penultimate_hidden_states
        if hs is None or hs.shape[0] <= 1:
            # Single image or no penultimate states, no processing needed
            return io.NodeOutput(clip_vision_output)

        # Create a new Output object with reshaped penultimate_hidden_states
        b, t, d = hs.shape
        reshaped_hs = hs.reshape(1, b * t, d)
        logging.info(
            "[CLIPVisionMultiRef] Reshaped penultimate_hidden_states: [%d, %d, %d] -> [1, %d, %d]",
            b, t, d, b * t, d,
        )

        # Create output preserving all original attributes
        out = type(clip_vision_output)()
        for attr_name in dir(clip_vision_output):
            if not attr_name.startswith('_') and attr_name != 'penultimate_hidden_states':
                try:
                    val = getattr(clip_vision_output, attr_name)
                    if not callable(val):
                        setattr(out, attr_name, val)
                except Exception:
                    pass
        out.penultimate_hidden_states = reshaped_hs

        return io.NodeOutput(out)


NODE_CLASS_MAPPINGS = {
    "CLIPVisionMultiRefSwitch": CLIPVisionMultiRefSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPVisionMultiRefSwitch": "CLIP Vision Multi-Ref Switch",
}