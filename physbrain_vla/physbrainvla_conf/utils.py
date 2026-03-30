from starVLA.model.modules.vlm.QWen2_5 import _QWen_VL_Interface
from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface

from starVLA.model.framework.double_vla.configuration_double_vla import DoubleVLAConfig, FIXED_STATE_DIM
from starVLA.model.framework.double_vla.embodiment_conditioned_mlp import CategorySpecificMLP


def freeze_qwen_vl(qwen_interface: _QWen3_VL_Interface | _QWen_VL_Interface):
    """
    冻结 Qwen VL 模型的参数
    """
    qwen_interface.model.eval()
    for param in qwen_interface.model.parameters():
        param.requires_grad = False


def create_state_encoder(
    output_dim: int,
    vla_config: DoubleVLAConfig,
) -> CategorySpecificMLP | None:
    """
    根据 config 创建一个 state_encoder，用于编码 Robot 本体的状态信息。
    固定输入维度为 20（通过 padding/truncation 适配不同本体的 state_dim），输出是 VLM 的 hidden size。
    
    Args:
        output_dim: VLM 的 hidden_size
        vla_config: PhysBrainVLA 配置
    """
    if vla_config.state_style != "state_encoder":
        return None
    
    return CategorySpecificMLP(
        num_categories=vla_config.max_num_embodiments,
        input_dim=FIXED_STATE_DIM,  # 固定使用 20 维作为 state_encoder 的输入维度
        hidden_dim=vla_config.state_encoder_hidden_size,
        output_dim=output_dim,
    )


embodied_id_to_robot_type_cache = {}


def get_robot_type(embodied_id: int) -> str:
    """
    根据 embodiment_id 获取 robot type 字符串
    """
    global embodied_id_to_robot_type_cache
    if not embodied_id_to_robot_type_cache:
        from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EMBODIMENT_TAG_MAPPING, ROBOT_TYPE_TO_EMBODIMENT_TAG

        for robot_type, embodiment_tag in ROBOT_TYPE_TO_EMBODIMENT_TAG.items():
            embodied_id_to_robot_type_cache[EMBODIMENT_TAG_MAPPING[embodiment_tag.value]] = robot_type

    return embodied_id_to_robot_type_cache.get(embodied_id, "unknown_robot")