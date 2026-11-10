# -*- coding: utf-8 -*-
# DFCNet 论文口径的分割指标批处理（Expert Systems With Applications 2026, §4.3）
# 全部基于 google-deepmind 的 surface-distance 库：
#   距离类（越低越好 ↓）：ASD（平均表面距离）、HD95（95% 豪斯多夫距离），单位 mm
#   重叠类（越高越好 ↑，百分制）：
#       VD  = Volumetric Dice  体积 Dice（即常规 DSC）
#       SD  = Surface Dice     表面 Dice（容差内表面点 F1，即 NSD）
#       SO  = Surface Overlap  表面重叠率（双向表面重叠的均值）
#       Pre = Precision、Rec = Recall
# 论文用 2D 像素（voxel distance）报告 ASD/HD95；本脚本为 3D NIfTI，
# 按 GT header 的物理 spacing 计算，距离单位为 mm，更具物理意义。

import argparse
import datetime
import glob
import json
import os
from collections import defaultdict

import nibabel as nib
import numpy as np
from surface_distance import (compute_average_surface_distance,
                              compute_dice_coefficient,
                              compute_robust_hausdorff,
                              compute_surface_dice_at_tolerance,
                              compute_surface_distances,
                              compute_surface_overlap_at_tolerance)
from tqdm import tqdm

# 指标展示元信息：key -> (论文缩写, 全称, 单位, 方向)
METRIC_META = [
    ('vd',   'VD',   'Volumetric Dice',    '% (↑)', 'high'),
    ('sd',   'SD',   'Surface Dice',       '% (↑)', 'high'),
    ('so',   'SO',   'Surface Overlap',    '% (↑)', 'high'),
    ('pre',  'Pre',  'Precision',          '% (↑)', 'high'),
    ('rec',  'Rec',  'Recall',             '% (↑)', 'high'),
    ('asd',  'ASD',  'Avg Surface Dist',   'mm (↓)', 'low'),
    ('hd95', 'HD95', '95% Hausdorff Dist', 'mm (↓)', 'low'),
]
METRIC_KEYS = [m[0] for m in METRIC_META]


def compute_confusion(gt: np.ndarray, pred: np.ndarray):
    gt = gt > 0
    pred = pred > 0
    tp = float(np.sum(gt & pred))
    fp = float(np.sum(pred & ~gt))
    fn = float(np.sum(gt & ~pred))
    return tp, fp, fn


def compute_metrics_for_class(gt_mask, pred_mask, spacing, tolerance_mm):
    """单类别的 DFCNet 口径指标。百分制指标取 0-100；距离指标单位 mm。"""
    m = {k: np.nan for k in METRIC_KEYS}
    gt_empty = not bool(np.any(gt_mask))
    pred_empty = not bool(np.any(pred_mask))

    tp, fp, fn = compute_confusion(gt_mask, pred_mask)

    # ---- VD：体积 Dice（deepmind 原函数，空-空记 1.0）----
    m['vd'] = 100.0 * float(compute_dice_coefficient(gt_mask, pred_mask))

    # ---- Pre / Rec（百分制）----
    m['pre'] = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else (np.nan if fn == 0 else 0.0)
    m['rec'] = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else (np.nan if fp == 0 else 0.0)

    # ---- 表面类指标（需两侧均非空）----
    if not gt_empty and not pred_empty:
        sd_dist = compute_surface_distances(gt_mask, pred_mask, spacing)

        # ASD：双向平均表面距离的均值（对称 ASD），mm
        asd_pred2gt, asd_gt2pred = compute_average_surface_distance(sd_dist)
        m['asd'] = float(0.5 * (asd_pred2gt + asd_gt2pred))

        # HD95：95% 鲁棒豪斯多夫距离，mm
        m['hd95'] = float(compute_robust_hausdorff(sd_dist, 95.0))

        # SD：表面 Dice（容差内表面点 F1）
        m['sd'] = 100.0 * float(
            compute_surface_dice_at_tolerance(sd_dist, tolerance_mm))

        # SO：表面重叠率。deepmind 返回 (pred侧, gt侧) 两个方向重叠比例，
        # 这里取对称均值作为论文 SO，同时保留两个方向值便于核对。
        ov_pred, ov_gt = compute_surface_overlap_at_tolerance(sd_dist, tolerance_mm)
        m['so'] = 100.0 * float(0.5 * (ov_pred + ov_gt))
        m['so_gt'] = 100.0 * float(ov_gt)
        m['so_pred'] = 100.0 * float(ov_pred)
    elif gt_empty and pred_empty:
        # 双空：表面指标无定义；VD 已记 100
        pass
    else:
        # 一侧为空：表面重叠/距离无定义；SD/SO 记 0，距离记 NaN
        m['sd'] = 0.0
        m['so'] = 0.0
        m['so_gt'] = 0.0
        m['so_pred'] = 0.0

    return m


def build_result(gt_data, pred_data, classes, tolerance_mm):
    spacing = tuple(float(s) for s in gt_data.header.get_zooms()[:3])
    gt_arr = np.asarray(gt_data.get_fdata())
    pred_arr = np.asarray(pred_data.get_fdata())

    if classes is None:
        classes = sorted(np.unique(gt_arr).tolist())
        if 0 in classes:
            classes.remove(0)
    if not classes:
        return {}, spacing

    results = defaultdict(dict)
    for c in classes:
        gt_mask = (gt_arr == c).astype(bool)
        pred_mask = (pred_arr == c).astype(bool)
        try:
            results[str(c)] = compute_metrics_for_class(
                gt_mask, pred_mask, spacing, tolerance_mm)
        except Exception as e:
            print(f"Warning: 类别 {c} 计算失败: {e}")
            results[str(c)] = {k: np.nan for k in
                               ('vd', 'sd', 'so', 'pre', 'rec', 'asd', 'hd95')}
    return dict(results), spacing


def _fmt(values):
    vals = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def main():
    parser = argparse.ArgumentParser(
        description='DFCNet 口径分割指标批处理 (VD/SD/SO/Pre/Rec↑, ASD/HD95↓)。')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Ground truth NIfTI 目录 (.nii.gz)。')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='预测 NIfTI 目录 (.nii.gz)。')
    parser.add_argument('--output_json', type=str, default=None,
                        help='(可选) 保存详细结果的 JSON 文件。')
    parser.add_argument('--classes', type=int, nargs='*', default=None,
                        help='(可选) 指定类别，如 --classes 1；默认取 GT 的非零类别。')
    parser.add_argument('--tolerance', type=float, default=1.0,
                        help='表面 Dice/Overlap 的物理容差，单位 mm（默认 1.0）。'
                             'DFCNet 2D 实验用 1 像素；3D 建议按体素尺度设 1-2mm。')
    args = parser.parse_args()

    if not os.path.isdir(args.gt_dir):
        print(f"Error: GT 目录不存在 {args.gt_dir}")
        raise SystemExit(1)
    if not os.path.isdir(args.pred_dir):
        print(f"Error: 预测目录不存在 {args.pred_dir}")
        raise SystemExit(1)

    prediction_files = sorted(glob.glob(os.path.join(args.pred_dir, '*.nii.gz')))
    if not prediction_files:
        print(f"预测目录中未找到 .nii.gz 文件: {args.pred_dir}")
        return

    all_file_results = {}
    aggregated = defaultdict(lambda: defaultdict(list))
    errors = []
    missing_gt_count = 0
    spacing_seen = set()

    print(f"找到 {len(prediction_files)} 个预测文件；表面容差 = {args.tolerance} mm")
    for pred_file in tqdm(prediction_files, desc="Computing metrics", unit="file"):
        filename = os.path.basename(pred_file)
        gt_file = os.path.join(args.gt_dir, filename)
        if not os.path.exists(gt_file):
            msg = f"Error: 找不到对应 GT {gt_file}"
            print(msg)
            errors.append(msg)
            missing_gt_count += 1
            continue
        try:
            gt_nii = nib.load(gt_file)
            pred_nii = nib.load(pred_file)
            per_class, spacing = build_result(gt_nii, pred_nii,
                                              args.classes, args.tolerance)
            spacing_seen.add(spacing)
            if not per_class:
                print(f"Warning: {filename} 无有效结果")
                continue
            all_file_results[filename] = per_class
            for cls, metric_dict in per_class.items():
                for metric, val in metric_dict.items():
                    aggregated[cls][metric].append(val)
        except Exception as e:
            msg = f"Error: 处理 {filename}: {str(e)}"
            print(msg)
            errors.append(msg)

    if not all_file_results:
        print("没有成功计算任何文件。")
        return

    print(f"\n数据 spacing（mm）取值: {sorted(spacing_seen)}")

    # ---- 类别级汇总（按论文指标顺序展示，带方向箭头）----
    per_class_summary = {}
    print("\n====== Per-Class Results (DFCNet 口径) ======")
    for cls in sorted(aggregated.keys()):
        print(f"Class {cls}:")
        per_class_summary[cls] = {}
        for key, abbr, full, unit, _direction in METRIC_META:
            ms = _fmt(aggregated[cls].get(key, []))
            if ms:
                mean, std = ms
                print(f"  {abbr:4s} {full:22s} {unit:8s}: {mean:7.3f} ± {std:6.3f}")
                per_class_summary[cls][key] = {'mean': mean, 'std': std}
            else:
                print(f"  {abbr:4s} {full:22s} {unit:8s}: N/A")
                per_class_summary[cls][key] = {'mean': None, 'std': None}
    print("=" * 50)

    # ---- 总体汇总 ----
    print("\n======== Overall Results (Mean ± Std) ========")
    overall_summary = {}
    for key, abbr, full, unit, _direction in METRIC_META:
        all_vals = [v for cls in aggregated.values() for v in cls.get(key, [])]
        ms = _fmt(all_vals)
        if ms:
            mean, std = ms
            print(f"  {abbr:4s} {unit:8s}: {mean:7.3f} ± {std:6.3f}")
            overall_summary[key] = {'mean': mean, 'std': std}
        else:
            print(f"  {abbr:4s} {unit:8s}: N/A")
            overall_summary[key] = {'mean': None, 'std': None}
    print("=" * 50)
    print("说明：VD/SD/SO/Pre/Rec 为百分制、越高越好；ASD/HD95 单位 mm、越低越好。")

    if args.output_json:
        data = {
            "metadata": {
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "metric_convention": "DFCNet (ESWA 2026, §4.3), deepmind/surface-distance",
                "surface_tolerance_mm": args.tolerance,
                "spacing_mm_seen": [list(s) for s in sorted(spacing_seen)],
                "gt_directory": args.gt_dir,
                "pred_directory": args.pred_dir,
                "specified_classes": args.classes,
                "processed_files_count": len(all_file_results),
                "skipped_missing_gt_count": missing_gt_count,
                "errors_count": len(errors),
                "errors": errors,
            },
            "per_file_results": all_file_results,
            "per_class_summary": per_class_summary,
            "overall_summary": overall_summary,
        }
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\n详细结果已保存到 {args.output_json}")


if __name__ == "__main__":
    main()
