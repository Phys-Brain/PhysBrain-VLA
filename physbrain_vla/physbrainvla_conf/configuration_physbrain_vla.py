from typing import Literal

# 输入给 RightVLM 的 user prompt template
# 要求包含 {inst} 和 {state} 两个占位符且只能出现一次
DEFUALT_RIGHT_VLM_USER_PROMPT_TEMPLATE = (
    "The images are the robot observation.\n"
    "Instruction: {inst}\n"
    "Predict the next action for the robot based on the observation, instruction and robot state.\n"
    "State: {state}"
).strip()

# 不包含 state 的 RightVLM user prompt template
RIGHT_VLM_USER_PROMPT_TEMPLATE_NO_STATE = (
    "The images are the robot observation.\n"
    "Instruction: {inst}\n"
    "Robot Type: {robot_type}\n"
    "Predict the next action for the robot based on the observation, instruction and robot type.\n"
).strip()


# LeftVLM system prompt
DEFUALT_LEFT_VLM_SYSTEM_PROMPT = (
    "You are a helpful robot brain that can understand images and texts.\n"
    "You will be provided with multiple observation images and an instruction. Take action to execute the instruction."
).strip()

# RightVLM system prompt
DEFAULT_RIGHT_VLM_SYSTEM_PROMPT = (
    "You are a helpful robot brain that can understand images, texts, and robot states.\n"
    "You will be provided with multiple observation images, an instruction, and the robot state. Take action to execute the instruction."
).strip()


# Fixed input dimension for state encoder
# 对于输入中不足 20 维的 state，会通过 padding 补齐；超过 20 维的 state 会被截断
# 然后会根据 state_mask 来掩盖掉多余的部分
FIXED_STATE_DIM = 20


class DoubleVLAConfig:
    
    state_special_token = "<|propri|>"  # state_tensor 的占位符（与 Wall-X 的做法相同），在处理 inputs 时，会将 state_encoder 得到的 state_tensor 替换到该位置
    action_query_token = "<|action|>"  # action_query_token，输入给 RightVLM 并将对应的 hidden states 作为给 AE 做动作预测的 condition
    
    def __init__(
        self,
        mot_attention_mode: Literal["unidirectional", "bidirectional"],
        state_style: Literal["state_encoder", "text_prompt", "none"],
        max_num_embodiments: int,
        state_encoder_hidden_size: int,
        freeze_left_vlm: bool,
        right_vlm_prompt_template: str,
        hidden_states_selected_layer: int,
        task_template: Literal["default", "none"],
        rotation_mot: Literal["0", "1"] = "0",
        use_action_query_token: bool = False,
        action_query_token_num: int | str = 8,
        train_linear_attention: bool = False,
    ):
        # Param: motion_attention_mode
        #   - "unidirectional": RightVLM 可以 attn 到 LeftVLM，但 LeftVLM 不能 attn 到 RightVLM（应当是这个）
        #   - "bidirectional": LeftVLM 和 RightVLM 互相 attn（用于做消融）
        self.mot_attention_mode = mot_attention_mode
        
        # Param: max_num_embodiments
        #   最大的 robot embodiments 数量，遵从 Isaac GR00T 的设计，这里默认为 32
        #   该参数会用来初始化 `embodiment_conditioned_mlp.py` 中的相关模块，并根据预先配置的 embodiment_id 来选择不同的权重
        self.max_num_embodiments = max_num_embodiments
        
        # Param: state_style
        #   - "state_encoder": 使用 state encoder 来编码 state，参考 https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d6/gr00t_n1d6.py
        #   - "none": 不使用 state 信息
        #   - "text_prompt": 使用文本 prompt 来编码 state，参考 pi05
        #   目前仅支持 `state_encoder` 和 `none` 的方式
        self.state_style = state_style
        
        # Param: state_encoder_hidden_size
        #   state encoder 的 hidden size，默认为 1024
        self.state_encoder_hidden_size = state_encoder_hidden_size
        
        # Param: freeze_left_vlm
        #   是否冻结 LeftVLM 的参数。默认 True，且应当为 True，除非做消融
        self.freeze_left_vlm = freeze_left_vlm
        
        # Param: right_vlm_prompt_template
        self.right_vlm_prompt_template = right_vlm_prompt_template
        
        # Pram: hidden_states_selected_layer
        #   从 RightVLM 中选取哪个层的 hidden states 作为动作预测的 condition
        self.hidden_states_selected_layer = hidden_states_selected_layer
        
        self.task_template = task_template
        
        # Param: rotation_mot
        #   - "0": 每一层都进行 Joint Attention（默认行为）
        #   - "1": 只在奇数层（1,3,5...）进行 Joint Attention，偶数层（0,2,4...）不进行 Joint Attention
        self.rotation_mot = rotation_mot
        
        
        # Param: action_query_token_num
        #   如果使用 <|action|> 作为动作查询 token，则可以指定使用多少个连续的该 token 作为查询
        self.action_query_token_num = int(action_query_token_num)

        # Param: train_linear_attention
        #   仅对 Qwen3.5 有效。
        #   - False（默认）：GDN（Gated Delta Net）线性注意力层在 no_grad 下运行，参数不更新。
        #     原因：fla Triton kernel backward 在部分 GPU 上不稳定，且纯 PyTorch fallback 显存开销极大。
        #   - True：GDN 层允许梯度流动并参与训练，代价是显存占用显著增加且可能触发 Triton 编译问题。
        self.train_linear_attention = bool(train_linear_attention)

    @classmethod
    def from_dict(cls, config_dict) -> "DoubleVLAConfig":
        return DoubleVLAConfig(
            mot_attention_mode=config_dict.get("mot_attention_mode", "unidirectional"),
            max_num_embodiments=config_dict.get("max_num_embodiments", 32),
            state_style=config_dict.get("state_style", "state_encoder"),
            state_encoder_hidden_size=config_dict.get("state_encoder_hidden_size", 1024),
            freeze_left_vlm=config_dict.get("freeze_left_vlm", True) != "none",
            right_vlm_prompt_template=config_dict.get("right_vlm_prompt_template", DEFUALT_RIGHT_VLM_USER_PROMPT_TEMPLATE),
            hidden_states_selected_layer=config_dict.get("hidden_states_selected_layer", -1),
            task_template=config_dict.get("task_template", "default"),
            rotation_mot=config_dict.get("rotation_mot", "0"),
            action_query_token_num=config_dict.get("action_query_token_num", 8),
            train_linear_attention=config_dict.get("train_linear_attention", False),
        )
    
    def to_dict(self) -> dict:
        return {
            "mot_attention_mode": self.mot_attention_mode,
            "max_num_embodiments": self.max_num_embodiments,
            "state_style": self.state_style,
            "state_encoder_hidden_size": self.state_encoder_hidden_size,
            "freeze_left_vlm": self.freeze_left_vlm,
            "right_vlm_prompt_template": self.right_vlm_prompt_template,
            "hidden_states_selected_layer": self.hidden_states_selected_layer,
            "task_template": self.task_template,
            "rotation_mot": self.rotation_mot,
            "action_query_token_num": self.action_query_token_num,
            "train_linear_attention": self.train_linear_attention,
        }