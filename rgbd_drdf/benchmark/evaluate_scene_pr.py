import os
import os.path as osp
import pickle as pkl
from dataclasses import asdict, dataclass, fields

import numpy as np
import trimesh
from filelock import FileLock
from loguru import logger
from tqdm import tqdm

from rgbd_drdf.benchmark.pr_utils import utils
from rgbd_drdf.config import benchmark_config
from rgbd_drdf.utils import parse_args


## This only works for one instance.
class EvaluateInstance:
    def __init__(
        self, cfg, pkl_file_path, pr_point_count=10000, thresholds=[0.05, 0.1, 0.25]
    ):
        ## sample the fixed set of points from the gt mesh
        ## sample a fixed set of points from the predicted point cloud
        self.gt_eval_dir = osp.join(
            cfg.EVAL_DIR, "raw", cfg.EVAL_GT_OUTPUTS, cfg.EVAL_SPLIT
        )
        self.pkl_path = pkl_file_path
        self.uuid_name = osp.basename(self.pkl_path)

        self.gt_uuid_pkl = osp.join(
            self.gt_eval_dir, f"{self.uuid_name.replace('.jpg','')}"
        )

        self.pr_point_count = pr_point_count
        self.rng_state = np.random.RandomState([ord(c) for c in self.uuid_name])
        self.thresholds = thresholds

        return

    def evaluate(
        self,
    ):
        with open(self.pkl_path, "rb") as f:
            pkl_data = pkl.load(f)

        if not osp.exists(self.gt_uuid_pkl):
            return None

        try:
            with open(self.gt_uuid_pkl, "rb") as f:
                gt_pkl_data = pkl.load(f)
        except:
            logger.info("Error reading GT pkl data")
            return None

        self.mesh_pts = self.sample_points_mesh(gt_pkl_data["gt_mesh"])
        self.pred_pts = self.sample_points_pcl(pkl_data["pcl"])
        self.gt_mesh = gt_pkl_data["gt_mesh"]
        eval_result = self.calculate_pr(
            gt_mesh=self.gt_mesh, gt_points=self.mesh_pts, pred_points=self.pred_pts
        )
        return eval_result

    def calculate_pr(self, gt_mesh, gt_points, pred_points):

        thresholds = self.thresholds
        precision = []
        recall = []
        fscore = []
        pred_distance2mesh = utils.distance_p2m(pred_points, gt_mesh)

        gt_dist2pred_points = utils.distance_p2p(
            points_src=gt_points, points_tgt=pred_points
        )

        for tx, threshold in enumerate(thresholds):
            # self.calculate_prAT(mesh=gt_mesh, gt_points=gt_points,
            #                     pred_points=pred_points,
            #                     threshold=threshold)
            precisionTx = (pred_distance2mesh < threshold).mean()
            recallTx = (gt_dist2pred_points < threshold).mean()
            fscoreTx = 2 * (precisionTx * recallTx) / (precisionTx + recallTx + 1e-4)
            precision.append(precisionTx)
            recall.append(recallTx)
            fscore.append(fscoreTx)

        precision = np.stack(precision)
        recall = np.stack(recall)
        fscore = np.stack(fscore)

        eval_result = Struct(
            precision=precision,
            recall=recall,
            fscore=fscore,
            thresholds=self.thresholds,
            gt_points=gt_points,
            pred_points=pred_points,
            pred_dist2mesh=pred_distance2mesh,
            gt_dist2pred_points=gt_dist2pred_points,
        )

        return eval_result

    def sample_points_mesh(
        self,
        mesh,
    ):
        np.random.set_state(self.rng_state.get_state())
        sampled_pts, _ = trimesh.sample.sample_surface(mesh, count=self.pr_point_count)
        return sampled_pts
        ## sample mesh points

    def sample_points_pcl(self, pcl):
        np.random.set_state(self.rng_state.get_state())

        sampled_pt_index = np.random.choice(len(pcl), size=self.pr_point_count)
        sampled_pts = pcl[sampled_pt_index]
        return sampled_pts
        ## sample pcl points


class Struct:
    def __init__(self, **kwargs):
        self.struct_keys = []
        for key, val in kwargs.items():
            setattr(self, key, val)
            self.struct_keys.append(key)

    def keys(
        self,
    ):
        return self.struct_keys


@dataclass
class EvalResult:
    precision = None
    recall = None
    fscore = None
    threshods = None
    gt_points = None
    pred_dist2mesh = None
    gt_dist2pred_points = None


## This will carry out the complete evaluation
class EvaluationRepository:
    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg
        if cfg.TEST_EPOCH_NUMBER >= 0:
            raw_dir = osp.join(
                cfg.EVAL_DIR,
                "raw",
                cfg.MODEL_NAME,
                f"{cfg.EVAL_SPLIT}_{cfg.TEST_EPOCH_NUMBER}",
            )
        else:
            raw_dir = osp.join(cfg.EVAL_DIR, "raw", cfg.MODEL_NAME, cfg.EVAL_SPLIT)
        # raw_dir = osp.join(cfg.EVAL_DIR, "raw", cfg.MODEL_NAME, cfg.EVAL_SPLIT)
        self.uuid_pkls = [
            f.path for f in os.scandir(raw_dir) if f.path.endswith(".pkl")
        ]

        self.result_dir = osp.join(
            cfg.EVAL_DIR,
            "result",
        )
        os.makedirs(self.result_dir, exist_ok=True)
        self.scene_pr_thresholds = cfg.SCENE_PR_THRESHOLDS
        return

    def evaluate(self, save_to_disk=True):
        eval_result_lst = []
        count = 0
        cfg = self.cfg
        for ix, uuid_pkl in tqdm(enumerate(self.uuid_pkls)):
            # logger.info(f"{ix} , {osp.basename(uuid_pkl)} ")
            eval_result = EvaluateInstance(
                cfg, uuid_pkl, thresholds=cfg.SCENE_PR_THRESHOLDS
            ).evaluate()
            # if ix > 100:
            #     break
            if eval_result is not None:
                eval_result_lst.append(eval_result)
                count += 1

        precision = []
        recall = []
        fscore = []
        for eval_result in eval_result_lst:
            precision.append(eval_result.precision)
            recall.append(eval_result.recall)
            fscore.append(eval_result.fscore)

        precision = np.stack(precision)
        recall = np.stack(recall)
        fscore = np.stack(fscore)

        precision = np.mean(precision, axis=0)
        recall = np.mean(recall, axis=0)
        fscore = np.mean(fscore, axis=0)
        combined_result = Struct(precision=precision, recall=recall, fscore=fscore)

        logger.info("coupute PR for all")

        eval_result_dict = {}
        eval_result_dict["result"] = eval_result_lst
        eval_result_dict["aggregrate"] = combined_result
        if cfg.TEST_EPOCH_NUMBER >= 0:
            model_name = f"{self.cfg.MODEL_NAME}_epoch{cfg.TEST_EPOCH_NUMBER}"
        else:
            model_name = self.cfg.MODEL_NAME
        save_file = osp.join(self.result_dir, f"{model_name}_{cfg.EVAL_SPLIT}.pkl")

        with open(save_file, "wb") as f:
            pkl.dump(eval_result_dict, f)

        threshold_str = utils.rowData2latex(np.array(self.scene_pr_thresholds))
        precision_str = utils.rowData2latex(precision * 100)
        recall_str = utils.rowData2latex(recall * 100)
        fscore_str = utils.rowData2latex(fscore * 100)

        logger.info(f"Model Name {model_name}")
        logger.info(f"Model Nick Name {cfg.NICK_NAME}")

        logger.info(f"Threshold {threshold_str}")
        logger.info(f"Precision  {precision_str} ")
        logger.info(f"Recall  {recall_str} ")
        logger.info(f"Fscore  {fscore_str} ")

        results_filep = "results.pkl"
        dataframe = {}
        dataframe["scene_pr"] = {
            "num_iter": count,
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
            "threshold": np.array(self.scene_pr_thresholds),
        }

        if True:
            lock = FileLock(f"{results_filep}.lock")
            try:
                with lock:
                    logger.info(f"Locking file {results_filep}")
                    if osp.exists(results_filep):
                        with open(results_filep, "rb") as f:
                            results_dict = pkl.load(f)
                    else:
                        results_dict = {}

                    if model_name not in results_dict.keys():
                        results_dict[model_name] = {}

                    if "dataframe" not in results_dict[model_name].keys():
                        results_dict[model_name]["dataframe"] = {}

                    for key in dataframe.keys():
                        results_dict[model_name]["dataframe"][key] = dataframe[key]

                    with open(results_filep, "wb") as f:
                        pkl.dump(results_dict, f)
            finally:
                lock.release()

        ### Write results to a common file.

        return eval_result_lst, combined_result


def evaluate_instances(cfg):
    evaluation_repo = EvaluationRepository(cfg)
    eval_result_lst, combined_result = evaluation_repo.evaluate()

    return


if __name__ == "__main__":
    cfg = benchmark_config.get_cfg_defaults()
    cmd_args = parse_args.parse_args()
    if cmd_args.cfg_file is not None:
        cfg.merge_from_file(cmd_args.cfg_file)
    if cmd_args.set_cfgs is not None:
        cfg.merge_from_list(cmd_args.set_cfgs)
    evaluate_instances(cfg)
