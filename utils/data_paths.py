
import os.path as osp

# 数据集根目录（按实际部署环境修改）
DATA_ROOT = "/code_gj"

# 参与训练的数据集目录（每个目录直接含 imagesTr/labelsTr 等）
img_datas = [
    osp.join(DATA_ROOT, "xijing_dataset_sam3d"),
    # 如需联合训练另外两个中心，取消注释并确认服务器上目录名一致：
    # osp.join(DATA_ROOT, "tangdu_dataset_sam3d"),
    # osp.join(DATA_ROOT, "neijiang_dataset_sam3d"),
]
