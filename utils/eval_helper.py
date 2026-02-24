import glob
import logging
import os

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.ndimage import label
from bisect import bisect


def dump(save_dir, outputs,input):
    filenames = input["filename"] # ['cookie/train/good/xyz/172.tiff', ...]
    filenames_xyz = input["filename_xyz"]
    batch_size = len(filenames)
    preds = outputs["pred"].cpu().numpy()  # B x 1 x H x W   torch.Size([24, 1, 224, 224])
    masks = input["mask"].cpu().numpy()  # B x 1 x H x W    torch.Size([24, 1, 224, 224]) 
    heights = input["height"].cpu().numpy() #
    widths = input["width"].cpu().numpy()
    clsnames = input["clsname"]
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        file_dir_xyz, filename_xyz = os.path.split(filenames_xyz[i])
        _, subname = os.path.split(file_dir)
        _xyz, subname_xyz = os.path.split(file_dir_xyz)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename_xyz = "{}_{}_{}".format(clsnames[i], subname_xyz, filename_xyz)
        filename, _ = os.path.splitext(filename)
        filename_xyz, _xyz = os.path.splitext(filename_xyz)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file,
            filename=filenames[i],
            filename_xyz=filenames_xyz[i],
            pred=preds[i],
            mask=masks[i],
            height=heights[i],
            width=widths[i],
            clsname=clsnames[i],
        )


def merge_together(save_dir):#/home/admin1/2Tsdb/lkf/uniform-3dad/IUF-master/experiments/MVTec_3DAD/9_1_only_RGB/result_eval_temp
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    # print("*************************",npz_file_list)
    fileinfos = []
    preds = []
    masks = []
    # print("npz_file_list",len(npz_file_list)) # 231
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "filename_xyz": str(npz["filename_xyz"]),
                "height": npz["height"],
                "width": npz["width"],
                "clsname": str(npz["clsname"]),
            }
        )
        preds.append(npz["pred"])
        masks.append(npz["mask"]) # "mask" shape--- torch.Size([24, 1, 224, 224]), 
    preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W  N表示样本数量 
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    # print("preds",preds.shape)
    # print("mask",masks.shape)
    return fileinfos, preds, masks


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        # masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int)  # 原来(N, )
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(int)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1)  # (N, )


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        )  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        # print("PerPixelAUC masks",self.masks.shape)
        # print("PerPixelAUC preds",self.preds.shape)
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


# class EvalAUPRO:
#     def __init__(self, data_meta):
#         """
#         初始化 AUPRO 评估类。
#         :param data_meta: 包含预测结果和真实标签的数据对象。
#         """
#         self.preds = data_meta.preds  # 预测结果 (N, H, W)
#         self.masks = data_meta.masks  # 真实标签 (N, H, W)

#     def eval_auc(self):
#         """
#         计算 AUPRO。
#         :return: AUPRO 值。
#         """
#         # 将预测结果和真实标签展平
#         preds_flat = np.concatenate([pred.flatten() for pred in self.preds], axis=0)
#         masks_flat = np.concatenate([mask.flatten() for mask in self.masks], axis=0)

#         # 初始化区域重叠统计
#         region_overlaps = []
#         region_fprs = []

#         # 遍历不同的阈值
#         thresholds = np.linspace(preds_flat.min(), preds_flat.max(), 100)
#         for threshold in thresholds:
#             # 根据阈值生成二值化预测
#             binary_preds = (preds_flat >= threshold).astype(int)

#             # 计算每个区域的召回率和 FPR
#             recall, fpr = self._calculate_region_overlap(binary_preds, masks_flat)
#             region_overlaps.append(recall)
#             region_fprs.append(fpr)

#         # 计算 AUPRO
#         aupro = metrics.auc(region_fprs, region_overlaps)
#         return aupro

#     def _calculate_region_overlap(self, binary_preds, masks_flat):
#         """
#         计算每个区域的召回率和 FPR。
#         :param binary_preds: 二值化预测结果 (1D 数组)。
#         :param masks_flat: 真实标签 (1D 数组)。
#         :return: 召回率和 FPR。
#         """
#         # 标记真实标签中的区域
#         labeled_masks, num_regions = label(masks_flat)

#         # 初始化统计变量
#         total_recall = 0
#         total_fpr = 0

#         # 遍历每个区域
#         for region_id in range(1, num_regions + 1):
#             region_mask = (labeled_masks == region_id)
#             region_size = np.sum(region_mask)

#             # 计算当前区域的召回率
#             true_positives = np.sum(binary_preds[region_mask] == 1)
#             recall = true_positives / region_size if region_size > 0 else 0

#             # 计算当前区域的 FPR
#             false_positives = np.sum(binary_preds[~region_mask] == 1)
#             fpr = false_positives / np.sum(~region_mask) if np.sum(~region_mask) > 0 else 0

#             total_recall += recall
#             total_fpr += fpr

#         # 计算平均召回率和 FPR
#         avg_recall = total_recall / num_regions if num_regions > 0 else 0
#         avg_fpr = total_fpr / num_regions if num_regions > 0 else 0

#         return avg_recall, avg_fpr


class EvalAUPRO:
    def __init__(self, data_meta):
        """
        初始化类。
        :param data_meta: 包含预测结果和真实标签的数据对象。
        """
        # 将预测结果和真实标签展平并拼接
        # self.preds = np.concatenate(
        #     [pred.flatten() for pred in data_meta.preds], axis=0
        # )
        # self.masks = np.concatenate(
        #     [mask.flatten() for mask in data_meta.masks], axis=0
        # )
        # self.masks[self.masks > 0] = 1
        # self.preds = self.preds.reshape(-1, 224, 224)  # 恢复为 (N, 224, 224)
        # self.masks = self.masks.reshape(-1, 224, 224)  # 恢复为 (N, 224, 224)
        self.preds =data_meta.preds
        self.masks = data_meta.masks

    def eval_auc(self, integration_limit=0.3, num_thresholds=100):
        """
        计算 AUPRO（PRO 曲线下的面积）。
        :param integration_limit: 积分上限（默认 0.3）。
        :param num_thresholds: 使用的阈值数量。
        :return: AUPRO 值。
        """
        # 计算 PRO 曲线
        # print(self.preds.shape) #(28, 224, 224)
        # print(self.masks.shape) #(28, 224, 224)
        pro_curve = compute_pro(anomaly_maps=self.preds, ground_truth_maps=self.masks, num_thresholds=num_thresholds)
        au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=integration_limit)
        au_pro /= integration_limit
        return au_pro

def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):

    # Fetch sorted anomaly scores.
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)

    # Select equidistant thresholds.
    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)

    fprs = [1.0]
    pros = [1.0]
    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]

        # Compute the false positive rate for this threshold.
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)

        # Compute the PRO value for this threshold.
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)

        fprs.append(fpr)
        pros.append(pro)

    # Return (FPR/PRO) pairs in increasing FPR order.
    fprs = fprs[::-1]
    pros = pros[::-1]

    return fprs, pros

def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    assert len(anomaly_maps) == len(ground_truth_maps)

    # Initialize ground truth components and scores of potential fp pixels.
    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)
    # print(anomaly_scores_ok_pixels.shape) # (3276800,)---50*256*256

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)
    # print(structure) # [[1,1,1],[1,1,1],[1,1,1]]
    # print(structure.shape) # (3, 3)

    # Collect anomaly scores within each ground truth region and for all potential fp pixels.
    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):

        # Compute the connected components in the ground truth map.
        # print(gt_map.shape) # (256, 256)
        labeled, n_components = label(gt_map, structure)
        # print("labeled:", n_components)
        # print("Number of components:", n_components)

        # Store all potential fp scores.
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index:ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
        ok_index += num_ok_pixels

        # Fetch anomaly scores within each GT component.
        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(GroundTruthComponent(component_scores))

    # Sort all potential false positive scores.
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()
    # print(len(anomaly_scores_ok_pixels))
    # print(len(ground_truth_components))

    return ground_truth_components, anomaly_scores_ok_pixels

class GroundTruthComponent:
    """
    Stores sorted anomaly scores of a single ground truth component.
    Used to efficiently compute the region overlap for many increasing thresholds.
    """

    def __init__(self, anomaly_scores):
        """
        Initialize the module.

        Args:
            anomaly_scores: List of all anomaly scores within the ground truth
                            component as numpy array.
        """
        # Keep a sorted list of all anomaly scores within the component.
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()

        # Pointer to the anomaly score where the current threshold divides the component into OK / NOK pixels.
        self.index = 0

        # The last evaluated threshold.
        self.last_threshold = None

    def compute_overlap(self, threshold):
        """
        Compute the region overlap for a specific threshold.
        Thresholds must be passed in increasing order.

        Args:
            threshold: Threshold to compute the region overlap.

        Returns:
            Region overlap for the specified threshold.
        """
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold

        # Increase the index until it points to an anomaly score that is just above the specified threshold.
        while (self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold):
            self.index += 1

        # Compute the fraction of component pixels that are correctly segmented as anomalous.
        return 1.0 - self.index / len(self.anomaly_scores)

def trapezoid(x, y, x_max=None):

    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print(
            """WARNING: Not all x and y values passed to trapezoid are finite. Will continue with only the finite values.""")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between the last x[ins-1] and x_max. Since we do not
            # know the exact value of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction

eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
    "aupro": EvalAUPRO
}


def performances(fileinfos, preds, masks, config): #性能评估
    #fileinfos:[{'filename': 'potato/test/hole/rgb/011.png', 'height': array(800), 'width': array(800), 'clsname': 'potato'},...]
    
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos]) #['bagel',cable,...10个]
    for clsname in clsnames: #循环10次
        preds_cls = []
        masks_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...]) #如果pred 的形状是 (H, W)，则 pred[None, ...] 的形状变为 (1, H, W)。
                masks_cls.append(mask[None, ...])
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W  N*224*224
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
        data_meta = EvalDataMeta(preds_cls, masks_cls)

        # auc
        if config.get("auc", None):
            for metric in config.auc: #循环评估config.auc的指标
                evalname = metric["name"]
                kwargs = metric.get("kwargs", {})
                eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
                auc = eval_method.eval_auc()
                ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc

    if config.get("auc", None):
        for metric in config.auc:
            evalname = metric["name"]
            evalvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_auc = np.mean(np.array(evalvalues))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc

    return ret_metrics


def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]

    # auc
    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        logger.info(f"\n{record}")
