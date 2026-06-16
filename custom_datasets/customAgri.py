import cv2
import numpy as np
from pathlib import Path
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS, DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@TRANSFORMS.register_module()
class LoadCustomRaster(BaseTransform):
    """
    第一步加载：只负责读取、BGR转RGB、Resize、标签映射。
    注意：为保证后续的色彩增强不出错，这里保持图片为 uint8 格式，不在这一步归一化。
    """
    def __init__(self, img_size: int = 512):
        self.img_size = img_size

    def transform(self, results: dict) -> dict:
        img_path = results['img_path']
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # BGR -> RGB，此时保持为默认的 0-255 uint8 格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_path = results['seg_map_path']
        ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)

        ori_h, ori_w = img.shape[:2]

        if self.img_size != ori_h or self.img_size != ori_w:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            ann = cv2.resize(ann, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 标签映射: 255 转为 1
        gt_seg_map = ann.astype(np.int64)
        gt_seg_map[gt_seg_map == 255] = 1

        results["img"] = img
        results["gt_seg_map"] = gt_seg_map
        results["img_shape"] = (self.img_size, self.img_size)
        results["ori_shape"] = (self.img_size, self.img_size)
        results["seg_fields"] = results.get("seg_fields", []) + ["gt_seg_map"]

        return results


@TRANSFORMS.register_module()
class CustomRandomRotate90(BaseTransform):
    """强数据增强：随机 90/180/270 度旋转"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.rand() >= self.prob:
            return results
        
        k = np.random.randint(1, 4)
        results["img"] = np.ascontiguousarray(np.rot90(results["img"], k))
        
        for key in results.get("seg_fields", []):
            results[key] = np.ascontiguousarray(np.rot90(results[key], k))
            
        return results


@TRANSFORMS.register_module()
class CustomNormalize(BaseTransform):
    """
    直接内置你的全局均值和方差，放在数据加载器的最后一步直接计算。
    """
    def __init__(self):
        # 把统计好的常量直接放在加载器中
        self.mean = np.array([72.4085, 89.7399, 69.6123], dtype=np.float32)
        self.std  = np.array([32.8544, 23.9954, 23.1234], dtype=np.float32)

    def transform(self, results: dict) -> dict:
        # 转为 float32 并直接计算归一化
        img = results["img"].astype(np.float32)
        results["img"] = (img - self.mean) / self.std
        
        # 备注：如果你希望“不使用全局统计量，对每张单独的图片求它自己的均值方差（实例归一化）”
        # 请注释掉上面两行，改为：
        # img = results["img"].astype(np.float32)
        # results["img"] = (img - img.mean(axis=(0, 1))) / (img.std(axis=(0, 1)) + 1e-8)
        
        return results


@DATASETS.register_module()
class CustomAgriDataset(BaseSegDataset):
    """接管底层数据解析逻辑的数据集"""

    METAINFO = dict(classes=['background', 'cropland'], palette=[[0, 0, 0], [34, 139, 34]])

    def __init__(self, data_root: str, split: str = 'train', pipeline=None, **kwargs):
        self._custom_root = Path(data_root)
        self._split = split
        
        super().__init__(
            data_root=data_root,
            ann_file="",
            img_suffix='.png',
            seg_map_suffix='_mask_seg.png',
            pipeline=pipeline,
            **kwargs)

    def load_data_list(self) -> list:
        txt_path = self._custom_root / f"{self._split}.txt"
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        img_dir = self._custom_root / "img_dir"
        ann_dir = self._custom_root / "ann_dir"

        samples = []
        for basename in lines:
            img_path = img_dir / f"{basename}{self.img_suffix}"
            ann_path = ann_dir / f"{basename}{self.seg_map_suffix}"
            
            if img_path.exists() and ann_path.exists():
                samples.append({
                    'img_path': str(img_path),
                    'seg_map_path': str(ann_path),
                    'label_map': self.label_map,
                    'reduce_zero_label': self.reduce_zero_label,
                    'seg_fields': []
                })
        return samples