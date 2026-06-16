import cv2
import numpy as np
from pathlib import Path
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS, DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@TRANSFORMS.register_module()
class LoadCustomRaster(BaseTransform):
    """
    完全自定义的加载器：读取 -> BGR转RGB -> 调整大小 -> 归一化 -> 标签映射
    """
    def __init__(self, img_size: int = 512, mean=None, std=None):
        self.img_size = img_size
        # 如果传入了 mean 和 std，则在 CPU 数据集加载阶段进行归一化
        self.mean = np.array(mean, dtype=np.float32) if mean else None
        self.std = np.array(std, dtype=np.float32) if std else None

    def transform(self, results: dict) -> dict:
        # 1. 图像读取与通道转换
        img_path = results['img_path']
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        # 2. 标注读取 (单通道灰度图)
        ann_path = results['seg_map_path']
        ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)

        ori_h, ori_w = img.shape[:2]

        # 3. 尺寸调整 (Resize)
        if self.img_size != ori_h or self.img_size != ori_w:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            ann = cv2.resize(ann, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 4. 数据归一化 (直接使用给定的均值和方差)
        if self.mean is not None and self.std is not None:
            img = (img - self.mean) / self.std

        # 5. 标签映射: 255 转为 1
        gt_seg_map = ann.astype(np.int64)
        gt_seg_map[gt_seg_map == 255] = 1

        # 6. 组装结果
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


@DATASETS.register_module()
class CustomAgriDataset(BaseSegDataset):
    """接管底层数据解析逻辑的数据集"""

    METAINFO = dict(classes=['background', 'cropland'], palette=[[0, 0, 0], [34, 139, 34]])

    def __init__(self, data_root: str, split: str = 'train', pipeline=None, **kwargs):
        self._custom_root = Path(data_root)
        self._split = split  # 类似于 pastis 的 split 控制
        
        super().__init__(
            data_root=data_root,
            ann_file="",  # 留空，因为我们要手写 load_data_list 来接管
            img_suffix='.png',
            seg_map_suffix='_mask_seg.png',
            pipeline=pipeline,
            **kwargs)

    def load_data_list(self) -> list:
        """
        这就是完美平替 _build_data_list 的核心函数。
        根据传入的 split，自动找到对应的 txt 并组装所有的路径。
        """
        # 如果你测试集的 txt 叫做 test10_txt.txt，请确保 config 里传入的 split 名字能对上
        txt_path = self._custom_root / f"{self._split}.txt"
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        img_dir = self._custom_root / "img_dir"
        ann_dir = self._custom_root / "ann_dir"

        samples = []
        for basename in lines:
            img_path = img_dir / f"{basename}{self.img_suffix}"
            ann_path = ann_dir / f"{basename}{self.seg_map_suffix}"
            
            # 安全检查
            if img_path.exists() and ann_path.exists():
                samples.append({
                    'img_path': str(img_path),
                    'seg_map_path': str(ann_path),
                    # 这两个是 mmsegmentation 内部组装必需的字段
                    'label_map': self.label_map,
                    'reduce_zero_label': self.reduce_zero_label,
                    'seg_fields': []
                })
        return samples