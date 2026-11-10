'''
数据集路径配置（SAM-Med3D 训练唯一入口）。

说明：
- img_datas 列表中每个目录必须【直接包含】以下子目录：
    imagesTr/ labelsTr/                      （训练，必需）
    imagesVal/ labelsVal/ imagesTs/ labelsTs/（验证/测试，可选）
- 图像与标签文件名必须 base 一致（xxx.nii.gz），
  DataLoader 用 label_path.replace("labels","images") 推导图像路径。
- 数据已由 standardize_sam3d.py 离线重采样到 1.5x1.5x1.5mm。

Linux 服务器上的数据集根目录。
'''
import os.path as osp

# 数据集根目录（按实际部署环境修改）
DATA_ROOT = "/huangdong/code_gj"

# 参与训练的数据集目录（每个目录直接含 imagesTr/labelsTr 等）
img_datas = [
    osp.join(DATA_ROOT, "xijing_dataset_sam3d"),
    # 如需联合训练另外两个中心，取消注释并确认服务器上目录名一致：
    # osp.join(DATA_ROOT, "tangdu_dataset_sam3d"),
    # osp.join(DATA_ROOT, "neijiang_dataset_sam3d"),
]

# ---------------- 医生勾画 GTV 目录（临床 GTV->CTV 推理入口） ----------------
# 唯一的服务器标准路径；gtv_to_ctv_infer.py / val_on_gtv.sh 均以此为准。
JIAOTONG_DOCTOR_GTV_ROOT = (
    "/huangdong/SAM-Med3D/doctor_annotations/rectum_jiaotong_flat")

# 本地调试镜像：仅当上面标准路径在当前机器未挂载时，按顺序回退；
# 部署到服务器后标准路径存在，会自动优先使用标准路径。
JIAOTONG_DOCTOR_GTV_LOCAL_MIRRORS = (
    r"E:\Data_gj\gouhua_11\rectum_jiaotong_flat",
    r"E:\Data_gj\Code_gj\SAM-Med3D-main\doctor_annoation\rectum_jiaotong_flat",
)
