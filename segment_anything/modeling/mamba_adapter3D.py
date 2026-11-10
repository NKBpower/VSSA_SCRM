"""
Mamba 风格轻量适配器驱动的 SAM-Med3D 高效参数微调模块
========================================================

对应创新点一：Mamba 风格轻量适配器驱动的基础模型高效微调范式
  - 冻结 SAM-Med3D 图像/提示/掩码解码器主体权重，仅在编码器各阶段旁路注入轻量适配器；
  - SS3D 适配器：线性复杂度 O(N) 沿轴向/六方向传播状态，聚合"肿瘤—周围组织"长程上下文（C1）；
  - 轴向/六方向扫描的跨层状态传播建立层间连续性约束（C4）；
  - 方向解耦卷积适配器：(1,3,3) 层内分支与 (3,1,1) 跨层分支拆分建模；
    数据已统一重采样为 1.5mm 各向同性体素，两分支物理感受野等大，
    拆分目的为轴向解耦（轴状面/层间统计特性不同）与参数效率（C6 工程约束）；
  - 适配器以逐层可学习残差门控 gate_i 旁路相加 F=frozen+gate_i·adapter,
    gate 由 ImageEncoderViT3DWithAdapter.all_gate 统一持有(每层 1 个标量,
    初值 0,训练初期严格恒等冻结骨干,随后逐层"苏醒");二阶段粗到精另有
    独立的融合 α(见 refinement_decoder3D.py)。

设计哲学承袭 MambaLiteUNet：以线性复杂度状态空间建模替代重型注意力，
将医学影像的任务定制归纳偏置直接编码进适配器。

依赖说明：
  - 优先使用 mamba_ssm（CUDA 加速），未安装时自动回退至纯 PyTorch 实现 SelectiveSSM；
  - 纯 PyTorch 实现适用于短序列（L ≤ 64），SAM-Med3D 的 token 序列长度通常为 8–16，完全适用。
"""

import math
from typing import Optional, Tuple, Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. SSM 后端选择（优先 mamba_ssm，回退纯 PyTorch 实现）
# ============================================================

try:
    from mamba_ssm import Mamba as _CudaMamba
    _USE_CUDA_MAMBA = True
except ImportError:
    _USE_CUDA_MAMBA = False

# 适配器旁路残差门控的初值:每个编码层一个可学习标量 gate_i,
# 前向 F = F_frozen + gate_i * adapter(x)。初值 0 使训练起步严格恒等
# 冻结的 SAM-Med3D 骨干(随机初始化的适配器不会注入噪声);gate 收到的
# 梯度 ∂L/∂gate = ∂L/∂F · adapter(x) 非零,从第一步即开始学习,
# 长大之后适配器权重才随之收梯度——与二阶段精修 α 同构的渐进苏醒设计。
# 裸标量(不套 sigmoid/softplus):允许门控自由增减、可为负,不人为设界。
ADAPTER_GATE_INIT = 0.0

# SS3D 扫描方向消融的三种合法模式:
#   bi_separate —— 6 个独立 SSM 核(3 轴 × 前/后向各一套,2026-09-25 前默认);
#   bi_shared   —— 3 个核,前/后向共享同一组权重(VMamba K=1 式翻转复用,
#                  选择性参数 B/C/Δ 数据相关,翻转后自动重算),SSM 核参数减半;
#   uni         —— 3 个核仅前向单向扫描(因果序列式,轴向起点存在盲区,
#                  作为"3D 医学体数据无需双向/长程扫描"的对照)。
SCAN_MODES = ('bi_separate', 'bi_shared', 'uni')


class SelectiveSSM(nn.Module):
    """纯 PyTorch 选择性状态空间模型（Mamba 核心实现）。

    实现完整的 Mamba 前向流程：输入投影 → 因果深度卷积 → SiLU →
    选择性参数生成（Δ, B, C） → ZOH 离散化 → 顺序状态扫描 → 门控 → 输出投影。

    适用于短序列（L ≤ 64）。SAM-Med3D 的 token 序列长度通常为 8（128/16）或 16（256/16），
    顺序扫描开销极小。更长序列时建议安装 mamba_ssm 以使用 CUDA 并行扫描。

    Args:
        d_model: 模型维度（输入/输出通道数）。
        d_state: 状态空间维度（默认 16）。
        d_conv: 因果卷积核大小（默认 4）。
        expand: 内部扩展系数（默认 2，d_inner = expand * d_model）。
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model

        # 输入投影：x → (x_db, z) 门控对
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 因果深度卷积（分组卷积，groups=d_inner）
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM 参数投影：x_db → (Δ, B, C)
        # dt_rank 取 d_inner（简化版，省去单独的 dt_proj）
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)

        # A（对数参数化，保证 A = -exp(A_log) < 0，维持系统稳定）
        # 初始化：A_log = log(Uniform(1, 10))，使 A ∈ (-10, -1)
        A_init = torch.rand(self.d_inner, d_state) * 9.0 + 1.0
        self.A_log = nn.Parameter(torch.log(A_init))

        # D（跳跃连接，沿通道可学习）
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.x_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        if self.conv1d.weight is not None:
            nn.init.kaiming_normal_(self.conv1d.weight, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            y: (B, L, d_model)
        """
        # 状态空间递推对精度敏感:AMP 下 fp16 逐步累加易丢失尾数、
        # 越过 65504 上限后产出 inf→NaN。强制 fp32 计算递推,输出再转回
        # 原精度(仅纯 PyTorch 路径;CUDA mamba_ssm 内核不受此影响)。
        input_dtype = x.dtype
        x = x.float()
        B, L, _ = x.shape
        d_inner, d_state = self.d_inner, self.d_state

        # ---- 输入投影 → (x_db, z) ----
        xz = self.in_proj(x)                       # (B, L, 2*d_inner)
        x_db, z = xz.chunk(2, dim=-1)              # 各 (B, L, d_inner)

        # ---- 因果深度卷积 ----
        x_db = x_db.transpose(1, 2)                # (B, d_inner, L)
        x_db = self.conv1d(x_db)[:, :, :L]          # 因果截断 → (B, d_inner, L)
        x_db = x_db.transpose(1, 2)                 # (B, L, d_inner)
        x_db = F.silu(x_db)

        # ---- 选择性参数 Δ, B, C ----
        ssm_params = self.x_proj(x_db)             # (B, L, d_inner + 2*d_state)
        dt, B_param, C_param = ssm_params.split(
            [d_inner, d_state, d_state], dim=-1
        )
        dt = F.softplus(dt)                        # Δ > 0, (B, L, d_inner)

        # ---- 离散化（ZOH 近似：A_bar = exp(Δ·A), B_bar ≈ Δ·B）----
        A = -torch.exp(self.A_log)                 # (d_inner, d_state), A < 0
        # dA: (B, L, d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dB: (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)

        # ---- 顺序状态扫描 ----
        h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_db[:, t].unsqueeze(-1)
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(-1)   # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                 # (B, L, d_inner)

        # ---- 跳跃连接 + 门控 ----
        y = y + x_db * self.D
        y = y * F.silu(z)

        # ---- 输出投影 ----
        y = self.out_proj(y)                        # (B, L, d_model)
        return y.to(input_dtype)


def make_ssm(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> nn.Module:
    """创建 SSM 层：优先使用 CUDA 加速的 mamba_ssm，否则使用纯 PyTorch 实现。

    选择策略：
      - mamba_ssm 已安装且 CUDA 可用 → 使用 CUDA 加速的 Mamba 内核；
      - 否则 → 使用纯 PyTorch SelectiveSSM（适用于短序列 L ≤ 64）。
    """
    if _USE_CUDA_MAMBA and torch.cuda.is_available():
        return _CudaMamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
    return SelectiveSSM(d_model, d_state, d_conv, expand)


# ============================================================
# 2. SS3D：三维多方向选择性状态空间扫描块
# ============================================================

class SS3D(nn.Module):
    """三维选择性状态空间块（SS3D），通过多方向轴向扫描建模三维长程依赖。

    设计要点：
      - 沿 D（层厚）、H、W 三个轴向扫描,方向数由 scan_mode 控制:
        * 'bi_separate':前向/后向各用一个独立 SSM 核,共 6 个核(旧默认);
        * 'bi_shared':每轴仅 1 个核,翻转序列复用同一核扫反向(VMamba 式,
          选择性参数随输入重算,共享成立),共 3 个核、SSM 参数减半;
        * 'uni':每轴仅前向扫描,共 3 个核(单向对照,轴向起点有盲区);
      - 多轴/多方向结果合并由 scan_merge 控制:
        * 'mean'(新默认):按实际参与的方向项数取平均(双向 6 项、单向 3 项),
          使三种 scan_mode 的输出尺度可比,避免方向越多注入幅度越大;
        * 'sum':裸求和(2026-09-25 前旧 checkpoint 的历史行为,
          仅验证侧为旧 ckpt 自动恢复,新训练一律 mean);
      - 瓶颈结构（ratio 压缩通道）保证极小参数量；
      - 线性复杂度 O(N)，规避三维全局自注意力 O(N²) 的不可行性；
      - 输入/输出格式：[B, D, H, W, C]（通道在后，与 SAM-Med3D Block3D 一致）。

    各方向语义（双向时）：
      D 轴前向/后向：沿切片方向传播状态，建立跨层连续性（C4）；
      H 轴前向/后向：沿矢状方向传播状态，聚合层内长程上下文（C1）；
      W 轴前向/后向：沿冠状方向传播状态，聚合层内长程上下文（C1）。

    Args:
        dim: 输入/输出通道数。
        d_state: SSM 状态维度。
        d_conv: SSM 因果卷积核大小。
        expand: SSM 内部扩展系数。
        ratio: 瓶颈压缩比（inner_dim = dim // ratio）。
        scan_mode: 扫描模式,见 SCAN_MODES。
        scan_merge: 多分支合并方式,'mean'(默认)或 'sum'(旧 ckpt 兼容)。
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ratio: int = 8,
        scan_mode: str = 'bi_separate',
        scan_merge: str = 'mean',
    ):
        super().__init__()
        if scan_mode not in SCAN_MODES:
            raise ValueError(
                f"scan_mode 只能是 {SCAN_MODES},收到: {scan_mode!r}")
        if scan_merge not in ('mean', 'sum'):
            raise ValueError(
                f"scan_merge 只能是 'mean' 或 'sum',收到: {scan_merge!r}")
        self.dim = dim
        self.scan_mode = scan_mode
        self.scan_merge = scan_merge
        inner_dim = max(dim // ratio, 8)
        self.inner_dim = inner_dim

        # 瓶颈投影 + 归一化
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, inner_dim)
        self.proj_out = nn.Linear(inner_dim, dim)

        # SSM 核分配:
        #  bi_separate → 6 个独立核(旧 state_dict 键名 ssm_{d,h,w}_{fwd,bwd});
        #  bi_shared/uni → 每轴 1 个核(键名 ssm_{d,h,w})。
        #  两种键名空间不相交,错配会被验证侧形状预检/missing 键审计拦截。
        ssm = lambda: make_ssm(inner_dim, d_state, d_conv, expand)
        if scan_mode == 'bi_separate':
            self.ssm_d_fwd = ssm()  # D 轴前向
            self.ssm_d_bwd = ssm()  # D 轴后向(独立核)
            self.ssm_h_fwd = ssm()  # H 轴前向
            self.ssm_h_bwd = ssm()  # H 轴后向(独立核)
            self.ssm_w_fwd = ssm()  # W 轴前向
            self.ssm_w_bwd = ssm()  # W 轴后向(独立核)
            self._bidirectional = True
        else:
            self.ssm_d = ssm()      # D 轴核(双向时翻转复用)
            self.ssm_h = ssm()      # H 轴核
            self.ssm_w = ssm()      # W 轴核
            self._bidirectional = (scan_mode == 'bi_shared')

        # 实际参与合并的方向项数:双向 6(3 轴 × 2)、单向 3
        self.n_scan_terms = 6 if self._bidirectional else 3

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        if self.proj_in.bias is not None:
            nn.init.zeros_(self.proj_in.bias)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    @staticmethod
    def _scan_axis(
        x: torch.Tensor,
        ssm_fwd: nn.Module,
        fwd_perm: Tuple[int, ...],
        bwd_perm: Tuple[int, ...],
        L: int,
        C_inner: int,
        ssm_bwd: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """沿单一轴向扫描。

        置换策略：将扫描轴移至位置 3（紧邻通道维 C 的前一维），
        再用 reshape(-1, L, C) 直接展平所有非扫描维度为 batch。

        Args:
            x: [B, D, H, W, C_inner]
            ssm_fwd: 前向 SSM 模块。
            fwd_perm: 将扫描轴移至位置 3（C 前一维）的置换。
            bwd_perm: fwd_perm 的逆置换。
            L: 扫描轴长度。
            C_inner: 通道数。
            ssm_bwd: 后向 SSM 模块;None=仅前向(uni 单向);
                与 ssm_fwd 为同一对象时=共享核双向(bi_shared,翻转复用);
                为独立对象时=独立核双向(bi_separate)。
        Returns:
            y: [B, D, H, W, C_inner](可能为 1 项或 2 项之和,未归一化)
        """
        # 置换：将扫描轴移至位置 3（C 前一维）→ [B, d1, d2, L, C]
        x_perm = x.permute(*fwd_perm).contiguous()
        # 展平非扫描维度 → [B * d1 * d2, L, C]
        x_flat = x_perm.reshape(-1, L, C_inner)

        # 前向扫描
        y = ssm_fwd(x_flat)
        # 后向扫描（翻转序列 → SSM → 翻转回来);
        # 共享核模式下 ssm_bwd is ssm_fwd,同一组权重处理两个方向,
        # B/C/Δ 由翻转后的输入重新投影,无需为反向单设参数(VMamba K=1)
        if ssm_bwd is not None:
            y = y + ssm_bwd(x_flat.flip(dims=[1])).flip(dims=[1])

        # 还原为 permute 后的 5D 形状，再逆置换
        y = y.reshape(x_perm.shape)
        y = y.permute(*bwd_perm).contiguous()
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, H, W, C]
        Returns:
            适配器增量 delta（不包含自身残差,由 MambaAdapterBlock 直接旁路相加）
        """
        B, D, H, W, C = x.shape
        C_inner = self.inner_dim

        # 归一化 + 瓶颈投影
        x_norm = self.norm(x)
        x_in = self.proj_in(x_norm)  # [B, D, H, W, C_inner]

        # 按 scan_mode 组装各轴的前/后向模块(None=不扫反向)
        if self.scan_mode == 'bi_separate':
            axes = (
                (self.ssm_d_fwd, self.ssm_d_bwd,
                 (0, 2, 3, 1, 4), (0, 3, 1, 2, 4), D),
                (self.ssm_h_fwd, self.ssm_h_bwd,
                 (0, 1, 3, 2, 4), (0, 1, 3, 2, 4), H),
                (self.ssm_w_fwd, self.ssm_w_bwd,
                 (0, 1, 2, 3, 4), (0, 1, 2, 3, 4), W),
            )
        else:
            ssm_bwd = self.ssm_d if self._bidirectional else None
            axes = (
                (self.ssm_d, ssm_bwd,
                 (0, 2, 3, 1, 4), (0, 3, 1, 2, 4), D),
                (self.ssm_h,
                 self.ssm_h if self._bidirectional else None,
                 (0, 1, 3, 2, 4), (0, 1, 3, 2, 4), H),
                (self.ssm_w,
                 self.ssm_w if self._bidirectional else None,
                 (0, 1, 2, 3, 4), (0, 1, 2, 3, 4), W),
            )

        # ---- 多方向扫描并累加(未归一化) ----
        y = None
        for ssm_fwd, ssm_b, fwd_perm, bwd_perm, Lax in axes:
            y_ax = self._scan_axis(
                x_in, ssm_fwd, fwd_perm, bwd_perm, Lax, C_inner, ssm_b)
            y = y_ax if y is None else y + y_ax

        # ---- 合并:mean=按方向项数平均(三模式同尺度);sum=旧裸求和 ----
        if self.scan_merge == 'mean':
            y = y / self.n_scan_terms

        # 瓶颈投影回原通道(残差缩放在 MambaAdapterBlock 由逐层 gate 完成)
        delta = self.proj_out(y)
        return delta


# ============================================================
# 3. 各向异性卷积适配器
# ============================================================

class AnisotropicConvAdapter3D(nn.Module):
    """方向解耦卷积适配器：轴向分解的双分支深度可分离卷积。

    数据已统一重采样为 1.5×1.5×1.5mm 各向同性体素，两分支物理感受野
    等大（1.5mm 输入网格上均为 4.5mm），不再依赖物理层厚各向异性：
      - 层内分支：kernel (1,3,3) 捕获 H-W 轴状面局部细节；
      - 跨层分支：kernel (3,1,1) 建模 D（切片）方向层间局部关联。
    以轴向解耦 + 深度可分离卷积节省参数（18 vs (3,3,3) 的 27 个权重/通道），
    将轴状面/层间方向解耦归纳偏置编码进适配器。

    Args:
        dim: 输入/输出通道数。
        ratio: 瓶颈压缩比。
    """

    def __init__(self, dim: int, ratio: int = 8):
        super().__init__()
        self.dim = dim
        inner_dim = max(dim // ratio, 8)

        # 通道后 → 通道前的转换在 forward 中用 permute 完成
        self.norm = nn.LayerNorm(dim)

        # 瓶颈降维
        self.proj_in = nn.Conv3d(dim, inner_dim, kernel_size=1, bias=False)

        # 层内分支：(1,3,3) 核，捕获 H-W 轴状面局部细节
        self.in_plane = nn.Conv3d(
            inner_dim, inner_dim,
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            groups=inner_dim,  # 深度可分离
        )

        # 跨层分支：(3,1,1) 核，建模 D 方向层间局部关联
        self.cross_plane = nn.Conv3d(
            inner_dim, inner_dim,
            kernel_size=(3, 1, 1), padding=(1, 0, 0),
            groups=inner_dim,  # 深度可分离
        )

        # 逐点卷积融合
        self.fuse = nn.Conv3d(inner_dim * 2, inner_dim, kernel_size=1, bias=False)

        # 瓶颈升维
        self.proj_out = nn.Conv3d(inner_dim, dim, kernel_size=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in [self.proj_in, self.in_plane, self.cross_plane, self.fuse, self.proj_out]:
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, H, W, C]（通道在后）
        Returns:
            delta: [B, D, H, W, C]（适配器增量）
        """
        # 通道后 → 通道前 [B, C, D, H, W]
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # 瓶颈降维
        x = self.proj_in(x)  # [B, inner, D, H, W]

        # 双分支方向解耦卷积
        y_in = self.in_plane(x)       # 层内细节 (1,3,3)
        y_cross = self.cross_plane(x)  # 跨层关联 (3,1,1)
        y = torch.cat([y_in, y_cross], dim=1)  # [B, 2*inner, D, H, W]
        y = F.gelu(y)
        y = self.fuse(y)  # [B, inner, D, H, W]

        # 瓶颈升维
        delta = self.proj_out(y)  # [B, C, D, H, W]

        # 通道前 → 通道后 [B, D, H, W, C](残差缩放在 MambaAdapterBlock 由逐层 gate 完成)
        delta = delta.permute(0, 2, 3, 4, 1).contiguous()
        return delta


# ============================================================
# 4. Mamba 适配器块（冻结 Block3D + 旁路适配器直接残差相加）
# ============================================================

class MambaAdapterBlock(nn.Module):
    """冻结 Block3D + 旁路适配器(逐层可学习 gate 残差)。

    前向流程（旁路注入设计）::
        F_frozen = frozen_block(x)              # 冻结骨干（梯度可穿越至前序适配器）
        F_adapter = adapter(x)                  # 旁路适配器，参数可训练
        F_out = F_frozen + gate_i * F_adapter   # gate_i: 本层共享的可学习标量

    gate_i 不在这里注册(由 ImageEncoderViT3DWithAdapter.all_gate 统一持有,
    避免同一 Parameter 在多个子模块路径下重复注册),仅以 object.__setattr__
    保存共享引用;state_dict 中的规范键名为 image_encoder.all_gate.<i>。

    Args:
        frozen_block: 冻结的 SAM-Med3D Block3D 实例。
        adapter: 适配器模块（SS3D 或 AnisotropicConvAdapter3D）;传 None 表示
            A-bl（无适配器纯冻结骨干消融 Baseline）,退化为恒等旁路。
        gate: 本层残差门控标量(nn.Parameter, shape [1], 初值 0)。
    """

    def __init__(self, frozen_block: nn.Module, adapter: nn.Module,
                 gate: nn.Parameter):
        super().__init__()
        self.frozen_block = frozen_block
        self.adapter = adapter
        # 绕过 nn.Module.__setattr__:gate 已在父编码器 all_gate 中注册,
        # 直接 self.gate = gate 会把同一 Parameter 重复登记到本模块
        object.__setattr__(self, 'gate', gate)

        # 冻结原始 Block3D 参数（梯度仍可穿越，仅不更新本块权重）
        for p in self.frozen_block.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, H, W, C]
        Returns:
            [B, D, H, W, C]
        """
        # 冻结骨干前向（梯度可穿越至前序适配器，本块权重不更新）
        F_frozen = self.frozen_block(x)
        # 无适配器 Baseline：恒等旁路，仅输出冻结骨干特征
        if self.adapter is None:
            return F_frozen
        # 旁路适配器直接残差相加（无门控）
        return F_frozen + self.adapter(x)


# ============================================================
# 5. 注入适配器的图像编码器
# ============================================================

class ImageEncoderViT3DWithAdapter(nn.Module):
    """冻结 SAM-Med3D 图像编码器 + 渐进式适配器注入。

    渐进式预算分配（Progressive Adapter Budget Allocation）::
      - 浅层（0 ~ mamba_start-1）：各向异性卷积适配器（低成本局部细节/边缘）；
      - 深层（mamba_start ~ depth-1）：SS3D Mamba 适配器（长程语义，感受野随深度扩张）。
    将有限可训练参数"用在刀刃上"，在维持极低参数量开销的同时最大化领域适配收益。

    所有原始 SAM-Med3D 参数冻结,仅适配器与逐层 gate 参与训练:
      F_i = frozen_block_i(x) + gate_i * adapter_i(x),i = 0..depth-1
    gate 全部初值 0(起步严格恒等冻结骨干),由独立优化器参数组以较大 lr、
    weight_decay=0 训练(苏醒速度是该设计的关键,不可与适配器权重同 lr)。
    二阶段粗到精另有独立融合 α。

    Args:
        frozen_encoder: 预构建的 ImageEncoderViT3D（权重已加载）。
        embed_dim: 编码器嵌入维度。
        mamba_start: Mamba 适配器起始层索引（之前的层用卷积适配器）。
        ratio: 适配器瓶颈压缩比。
        d_state: SS3D 状态空间维度。
        d_conv: SS3D 因果卷积核大小。
        expand: SS3D 内部扩展系数。
        scan_mode: SS3D 扫描模式('bi_separate' 6 独立核/'bi_shared' 3 核
            翻转复用/'uni' 3 核仅前向),消融用,见 SCAN_MODES。
        scan_merge: 多分支合并('mean' 按方向项数平均,新训练默认;
            'sum' 仅旧 checkpoint 验证时恢复)。
        use_adapter: 是否注入旁路适配器。False 为"无适配器纯冻结骨干"消融
            Baseline(A-bl):每层挂 adapter=None,退化为恒等旁路、不注册
            all_gate,参数量/键结构完全退化为原始冻结 Vit;True(默认)为
            常规 SS3D/卷积适配器注入。
    """

    def __init__(
        self,
        frozen_encoder: nn.Module,
        embed_dim: int,
        mamba_start: int = 6,
        ratio: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        scan_mode: str = 'bi_separate',
        scan_merge: str = 'mean',
        use_adapter: bool = True,
    ):
        super().__init__()

        # 冻结组件（共享引用，不复制权重）
        self.patch_embed = frozen_encoder.patch_embed
        self.pos_embed = frozen_encoder.pos_embed
        self.neck = frozen_encoder.neck
        self.img_size = frozen_encoder.img_size

        depth = len(frozen_encoder.blocks)
        self.mamba_start = mamba_start
        # 记录扫描结构(epoch 日志/验证侧识别用;随 ckpt args 持久化)
        self.scan_mode = scan_mode
        self.scan_merge = scan_merge

        # 逐层可学习残差门控:每层 1 个标量,统一在编码器上注册,
        # state_dict 键名 image_encoder.all_gate.0 .. all_gate.<depth-1>
        # 无适配器 Baseline 不注册(all_gate/适配器权重都不存在,完全退化为冻结 Vit)
        self.all_gate = nn.ParameterList([
            nn.Parameter(torch.full((1,), ADAPTER_GATE_INIT))
            for _ in range(depth)
        ]) if use_adapter else None

        # 渐进式适配器注入(旁路 × 逐层 gate 残差)
        self.blocks = nn.ModuleList()
        for i, blk in enumerate(frozen_encoder.blocks):
            if not use_adapter:
                # 无适配器 Baseline：挂 None 适配器（恒等旁路）
                adapter = None
            elif i < mamba_start:
                # 浅层：各向异性卷积适配器
                adapter = AnisotropicConvAdapter3D(embed_dim, ratio=ratio)
            else:
                # 深层：SS3D Mamba 适配器
                adapter = SS3D(
                    embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    ratio=ratio,
                    scan_mode=scan_mode,
                    scan_merge=scan_merge,
                )
            self.blocks.append(
                MambaAdapterBlock(blk, adapter, None if self.all_gate is None else self.all_gate[i]))

        # 冻结非适配器参数
        self._freeze_base()

    def _freeze_base(self):
        """冻结 patch_embed / pos_embed / neck 的所有参数。"""
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        for p in self.neck.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]（与原始 ImageEncoderViT3D 一致）
        Returns:
            [B, out_chans, D', H', W']（与原始编码器输出一致）
        """
        # Patch 嵌入 → [B, D', H', W', C]
        x = self.patch_embed(x)
        # 位置编码
        if self.pos_embed is not None:
            x = x + self.pos_embed
        # 逐块前向（冻结 Block3D + 旁路适配器）
        for blk in self.blocks:
            x = blk(x)
        # Neck：[B, D', H', W', C] → [B, C, D', H', W'] → Conv → [B, out_chans, D', H', W']
        x = self.neck(x.permute(0, 4, 1, 2, 3))
        return x


# ============================================================
# 6. 构建函数
# ============================================================

def build_sam3D_with_mamba_adapter(
    model_type: str = 'vit_b_ori',
    checkpoint: Optional[str] = None,
    mamba_start: Optional[int] = None,
    ratio: int = 8,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    scan_mode: str = 'bi_separate',
    scan_merge: str = 'mean',
    freeze_prompt_encoder: bool = True,
    freeze_mask_decoder: bool = True,
    use_refinement: bool = False,
    low_res_size: int = 32,
    use_pyramid: bool = False,
    use_refine_alpha: bool = True,
    refine_type: str = 'sam',
    use_adapter: bool = True,
):
    """构建带 Mamba 风格适配器的 SAM-Med3D 模型。

    流程：
      1. 构建基础 SAM-Med3D（加载预训练权重）；
      2. 用 ImageEncoderViT3DWithAdapter 替换图像编码器（注入适配器）;
         当 use_adapter=False 时退化为"无适配器纯冻结骨干"(A-bl 消融 Baseline),
         all_gate 与适配器权重均不构建,参数量/键结构与原始冻结 Vit 完全一致;
      3. 按 freeze_* 标志冻结 prompt_encoder / mask_decoder(均可独立选择);
      4. 可选挂载两阶段粗到精解码器 RefinementDecoder3D(创新点二),
         粗解码器是否冻结由 freeze_mask_decoder 单独决定(默认解冻、与精修头联合训练);
      5. 返回 Sam3D。

    Args:
        model_type: SAM-Med3D 模型类型 ('vit_b_ori' / 'vit_b' / 'vit_l' / 'vit_h')。
        checkpoint: 预训练权重路径（可选）。
        mamba_start: Mamba 适配器起始层索引。None 时自动取 depth//2。
        ratio: 适配器瓶颈压缩比。
        d_state: SS3D 状态空间维度。
        d_conv: SS3D 因果卷积核大小。
        expand: SS3D 内部扩展系数。
        scan_mode: SS3D 多方向扫描消融:'bi_separate'(6 独立核,旧默认)、
            'bi_shared'(3 核翻转复用,SSM 参数减半)、'uni'(3 核仅前向)。
        scan_merge: 多分支合并口径,'mean'(按方向项数平均,新训练默认)或
            'sum'(2026-09-25 前裸求和,仅验证旧 ckpt 时由验证脚本自动传)。
        freeze_prompt_encoder: 是否冻结提示编码器。
        freeze_mask_decoder: 是否冻结掩码解码器(独立于 use_refinement;
            默认 False,即粗解码器可与精修头联合训练)。
        use_refinement: 是否挂载创新点二的粗到精精修解码器。
        low_res_size: 粗/精修掩码 low-res 空间尺寸(128^3 输入为 32)。
        use_pyramid: 精修解码器是否启用 U 形多尺度金字塔(仅 use_refinement=True 时生效,
            消融实验用;新结构需重新训练,旧 checkpoint 不兼容金字塔权重)。
        use_refine_alpha: 精修解码器融合 α 是否启用(默认 True)。False 为消融:
            α 恒为 1(refined=coarse+delta,固定单位残差,保留粗掩码通路);
            仅 use_refinement=True 时有效。(编码器适配器另有逐层 gate,
            见 image_encoder.all_gate,不存在 use_adapter_alpha 开关)
        refine_type: 二阶段结构(仅 use_refinement=True 时生效):
            'sam'(默认,主线)——二阶段 PE/Decoder 与一阶段结构同构
            (RefinementPromptEncoder3D/RefinementMaskDecoder3D,~7.58M,
            从头随机训练,粗掩码为唯一二阶段 prompt);
            'light'——旧纯卷积轻量版(RefinementPromptEncoder +
            RefinementDecoder3D,~73k,use_pyramid 仅对该类型有效)。

    Returns:
        Sam3D 实例(冻结/可训练范围由 freeze_* 参数决定)。
    """
    # 延迟导入以避免循环依赖
    from ..build_sam3D import sam_model_registry3D
    from .sam3D import Sam3D
    from .refinement_decoder3D import (
        RefinementDecoder3D,
        RefinementPromptEncoder,
        RefinementPromptEncoder3D,
        RefinementMaskDecoder3D,
    )

    # 1. 构建基础 SAM-Med3D
    base_sam = sam_model_registry3D[model_type](checkpoint=checkpoint)

    # 2. 提取编码器配置
    frozen_encoder = base_sam.image_encoder
    depth = len(frozen_encoder.blocks)
    embed_dim = frozen_encoder.blocks[0].attn.qkv.in_features

    # 自动确定 Mamba 起始层（默认前半卷积、后半 Mamba）
    if mamba_start is None:
        mamba_start = depth // 2

    # 3. 构建适配器注入的编码器
    adapter_encoder = ImageEncoderViT3DWithAdapter(
        frozen_encoder=frozen_encoder,
        embed_dim=embed_dim,
        mamba_start=mamba_start,
        ratio=ratio,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        scan_mode=scan_mode,
        scan_merge=scan_merge,
        use_adapter=use_adapter,
    )

    # 4. 替换编码器
    base_sam.image_encoder = adapter_encoder

    # 5. 冻结提示编码器与掩码解码器(完全由对应 flag 控制,
    #    use_refinement 不再强制冻结粗解码器:可做"冻结粗头"与"联合解冻粗头"两种配置)
    if freeze_prompt_encoder:
        for p in base_sam.prompt_encoder.parameters():
            p.requires_grad = False
    if freeze_mask_decoder:
        for p in base_sam.mask_decoder.parameters():
            p.requires_grad = False

    # 6. 挂载创新点二:两阶段粗到精解码器 + 二阶段提示编码器
    if use_refinement:
        if refine_type not in ('sam', 'light'):
            raise ValueError(
                f"refine_type 只能是 'sam' 或 'light',收到: {refine_type!r}")
        # neck 输出通道 = prompt_embed_dim(384),与 mask_decoder.transformer_dim 一致
        encoder_out_ch = base_sam.mask_decoder.transformer_dim

        if refine_type == 'sam':
            # SAM 同构版:二阶段 PE/Decoder 结构与一阶段完全一致,从头随机初始化
            # (不拷贝一阶段权重,学习独立的 coarse→refine 优化映射)
            stage1_pe = base_sam.prompt_encoder
            base_sam.refinement_prompt_encoder = RefinementPromptEncoder3D(
                embed_dim=stage1_pe.embed_dim,
                image_embedding_size=stage1_pe.image_embedding_size,
                input_image_size=stage1_pe.input_image_size,
                mask_in_chans=16,  # 与 vit_b_ori 注册表中一阶段配置一致
            )
            base_sam.refinement_decoder = RefinementMaskDecoder3D(
                transformer_dim=encoder_out_ch,
                num_multimask_outputs=3,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                use_alpha=use_refine_alpha,
            )
        else:
            # 旧轻量版:纯 3D 卷积,粗掩码特征 + 编码器特征融合
            prompt_dim = 64  # RefinementPromptEncoder 与 RefinementDecoder3D 共享
            base_sam.refinement_prompt_encoder = RefinementPromptEncoder(
                encoder_embed_dim=encoder_out_ch,
                prompt_dim=prompt_dim,
            )
            base_sam.refinement_decoder = RefinementDecoder3D(
                encoder_embed_dim=encoder_out_ch,
                low_res_size=low_res_size,
                prompt_dim=prompt_dim,
                use_pyramid=use_pyramid,
                use_alpha=use_refine_alpha,
            )
        # 标记二阶段结构类型,供训练/推理选择调用接口
        base_sam.refine_type = refine_type

    return base_sam


# ============================================================
# 7. 工具函数：参数统计
# ============================================================

def count_parameters(model: nn.Module) -> dict:
    """统计模型的可训练/冻结参数量。

    Returns:
        dict: {
            'total': 总参数量,
            'trainable': 可训练参数量,
            'frozen': 冻结参数量,
            'trainable_pct': 可训练占比(%),
            'trainable_by_type': 按模块类型分类的可训练参数量,
        }
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # 按模块类型分类
    trainable_by_type = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 提取模块类型关键字
        parts = name.split('.')
        module_type = 'unknown'
        for part in parts:
            if 'adapter' in part.lower() or 'alpha' in part.lower() or 'gamma' in part.lower():
                module_type = part
                break
        trainable_by_type[module_type] = trainable_by_type.get(module_type, 0) + p.numel()

    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_pct': 100.0 * trainable / total if total > 0 else 0.0,
        'trainable_by_type': trainable_by_type,
    }


def print_param_stats(model: nn.Module):
    """打印模型参数统计信息。"""
    stats = count_parameters(model)
    print("=" * 60)
    print("SAM-Med3D + Mamba 适配器 参数统计")
    print("=" * 60)
    print(f"总参数量:       {stats['total']:>15,}")
    print(f"可训练参数:     {stats['trainable']:>15,}")
    print(f"冻结参数:       {stats['frozen']:>15,}")
    print(f"可训练占比:     {stats['trainable_pct']:>14.4f} %")
    print("-" * 60)
    print("可训练参数按模块分类:")
    for k, v in sorted(stats['trainable_by_type'].items(), key=lambda x: -x[1]):
        print(f"  {k:<30s} {v:>15,}")
    print("=" * 60)
