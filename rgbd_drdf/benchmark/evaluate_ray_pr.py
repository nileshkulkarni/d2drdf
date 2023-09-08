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


class SceneRayPR:
    """
    Precision: For every predicted-point find the closest ground-truth  point and record it's distance.
    Recall:  For every ground truth point find the closest predicted point.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.results = {}
        self.name = "Precision Recall Evaluator"
        self.results["precision_distances"] = [
            np.zeros((0,))
        ]  ## array of precision_distances
        self.results["recall_distances"] = [np.zeros((0,))]
        self.results["precision"] = []
        self.results["recall"] = []
        self.results["pred_rays"] = []
        self.results["gt_rays"] = []
        n_intersections = kwargs["n_intersections"]
        self.thresholds = kwargs["thresholds"]
        if type(self.thresholds) == list:
            self.thresholds = np.array(self.thresholds)
        self.precision = np.zeros((n_intersections, len(self.thresholds)))
        self.recall = np.zeros((n_intersections, len(self.thresholds)))
        # (N,)
        return

    def compute_precision(self, pred_ray, gt_ray):
        if len(pred_ray) > 0:
            if len(gt_ray) == 0:
                gt_ray = pred_ray * 0.0 + 100
            precision_distances = np.abs(pred_ray[:, None] - gt_ray[None, :])
            precision_signs = np.sign(pred_ray[:, None] - gt_ray[None, :])
            precision_distances = np.min(precision_distances, axis=1)
            self.results["precision_distances"].append(precision_distances)

        return

    def get_precision(
        self,
    ):
        precision_distances = np.concatenate(self.results["precision_distances"])
        precision = precision_distances[:, None] < self.thresholds[None, :]
        n_instances = np.nansum(precision[:, 0] * 0 + 1)
        precision = np.nanmean(precision, axis=0)
        return precision, n_instances

    def get_recall(
        self,
    ):
        # pdb.set_trace()
        recall_distances = np.concatenate(self.results["recall_distances"])
        recall = recall_distances[:, None] < self.thresholds[None, :]
        n_instances = np.nansum(recall[:, 0] * 0 + 1)
        recall = np.nanmean(recall, axis=0)
        return recall, n_instances

    def compute_recall(self, pred_ray, gt_ray):
        if len(gt_ray) > 0:
            gt_ray[(np.abs(gt_ray) < 0.01)] = np.nan
            if len(pred_ray) == 0:
                pred_ray = gt_ray * 0 + 100
            recall_distances = np.abs(gt_ray[:, None] - pred_ray[None, :])
            recall_distances = np.min(recall_distances, axis=1)
            self.results["recall_distances"].append(recall_distances)
        return

    def single_ray_pr(self, pred_ray, gt_ray, **kwargs):
        # if len(pred_ray) > 4:
        #     choice = np.random.choice(len(pred_ray), 4)
        #     pred_ray = pred_ray[choice]

        self.results["pred_rays"].append(pred_ray)
        self.results["gt_rays"].append(gt_ray)
        self.compute_precision(pred_ray, gt_ray)
        self.compute_recall(pred_ray, gt_ray)
        return

    def get_pr_and_fscore(
        self,
    ):
        precision, p_instances = self.get_precision()
        recall, r_instances = self.get_recall()
        fscore = np.divide(2 * precision * recall, precision + recall)
        return precision, recall, fscore


## This only works for one instance.
class EvaluateInstance:
    def __init__(
        self, pkl_file_path, gt_file_path, evaluator, thresholds=[0.05, 0.1, 0.25]
    ):
        ## sample the fixed set of points from the gt mesh
        ## sample a fixed set of points from the predicted point cloud
        self.pkl_path = pkl_file_path
        self.gt_file_path = gt_file_path
        self.uuid_name = osp.basename(self.pkl_path)
        # self.pr_point_count = pr_point_count
        self.rng_state = np.random.RandomState([ord(c) for c in self.uuid_name])
        self.thresholds = thresholds
        self.evaluator = evaluator
        return

    @staticmethod
    def clean_zeros(intersection):
        int_valid = np.where(intersection > 1e-4)
        intersection = intersection[int_valid]
        return intersection

    def evaluate(
        self,
        occluded=False,
    ):

        uuid = osp.basename(self.pkl_path)
        with open(self.pkl_path, "rb") as f:
            pkl_data = pkl.load(f)

        try:
            with open(self.gt_file_path, "rb") as f:
                gt_pkl_data = pkl.load(f)
        except:
            logger.info("Error reading GT pkl data")
            return None

        raw_pred_intersections = pkl_data["raw_intersections"].reshape(-1, 5)
        raw_gt_intersections = gt_pkl_data["raw_intersections"].reshape(-1, 5)

        valid_pred_mask_inds = raw_pred_intersections > 0
        valid_gt_mask_inds = raw_gt_intersections > 0
        rng_state = np.random.RandomState([ord(c) for c in uuid])
        ray_inds = rng_state.choice(len(raw_pred_intersections), 2000)

        for ray_ind in ray_inds:
            pred_ray = raw_pred_intersections[ray_ind]
            gt_ray = raw_gt_intersections[ray_ind]

            pred_ray = self.clean_zeros(pred_ray)
            gt_ray = self.clean_zeros(gt_ray)
            if occluded:
                pred_ray = pred_ray[1:]
                gt_ray = gt_ray[1:]

            self.evaluator.single_ray_pr(pred_ray=pred_ray, gt_ray=gt_ray)

        # precision, recall, fscore = self.evaluator_int.get_pr_and_fscore()
        # self.mesh_pts = self.sample_points_mesh(pkl_data["gt_mesh"])
        # self.pred_pts = self.sample_points_pcl(pkl_data["pcl"])
        # self.gt_mesh = pkl_data["gt_mesh"]

        # eval_result = self.calculate_pr(
        #     gt_mesh=self.gt_mesh, gt_points=self.mesh_pts, pred_points=self.pred_pts
        # )
        return

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
    def __init__(self, cfg):
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
        self.uuid_pkls = [
            f.path for f in os.scandir(raw_dir) if f.path.endswith(".pkl")
        ]
        self.result_dir = osp.join(
            cfg.EVAL_DIR,
            "result",
        )
        os.makedirs(self.result_dir, exist_ok=True)
        self.scene_pr_thresholds = cfg.SCENE_PR_THRESHOLDS
        self.gt_eval_dir = osp.join(
            cfg.EVAL_DIR, "raw", cfg.EVAL_GT_OUTPUTS, cfg.EVAL_SPLIT
        )
        return

    def evaluate(
        self,
        occluded,
        save_to_disk=True,
    ):
        eval_result_lst = []
        ray_pr_thresholds = cfg.RAY_PR_THRESHOLDS
        self.evaluator = SceneRayPR(n_intersections=5, thresholds=ray_pr_thresholds)
        count = 0
        for ix, uuid_pkl in tqdm(enumerate(self.uuid_pkls)):
            base_uuid_name = osp.basename(uuid_pkl)
            # print(f"{ix} , {base_uuid_name} ")

            gt_uuid_pkl = osp.join(
                self.gt_eval_dir, f"{base_uuid_name.replace('.jpg','')}"
            )
            if not osp.exists(gt_uuid_pkl):
                continue
            EvaluateInstance(
                uuid_pkl,
                gt_uuid_pkl,
                thresholds=ray_pr_thresholds,
                evaluator=self.evaluator,
            ).evaluate(occluded=occluded)
            count += 1

        print(f"coupute Ray PR for all occ is {occluded}")
        precision, recall, fscore = self.evaluator.get_pr_and_fscore()

        combined_result = {"precision": precision, "recall": recall, "fscore": fscore}
        eval_result_dict = {}
        eval_result_dict["aggregrate"] = combined_result
        eval_result_dict["thresholds"] = ray_pr_thresholds
        # cfg.MODEL_NAME, cfg.EVAL_SPLIT
        suffix = ""
        if occluded:
            suffix = "_occ"
        eval_dir = "ray_pr" + suffix

        if cfg.TEST_EPOCH_NUMBER >= 0:
            model_name = f"{self.cfg.MODEL_NAME}_epoch{cfg.TEST_EPOCH_NUMBER}"
        else:
            model_name = self.cfg.MODEL_NAME


        threshold_str = utils.rowData2latex(np.array(ray_pr_thresholds))
        precision_str = utils.rowData2latex(precision * 100)
        recall_str = utils.rowData2latex(recall * 100)
        fscore_str = utils.rowData2latex(fscore * 100)

        print(f"Threshold {threshold_str}")
        print(f"Precision  {precision_str} ")
        print(f"Recall  {recall_str} ")
        print(f"Fscore  {fscore_str} ")

        results_filep = "results.pkl"

        dataframe = {}

        eval_str = "ray_pr"
        if occluded:
            eval_str = "ray_pr_occluded"
        dataframe[eval_str] = {
            "num_iter": count,
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
            "threshold": np.array(ray_pr_thresholds),
        }

        if True:
            lock = FileLock(f"{results_filep}.lock")
            try:
                with lock:
                    print(f"Locking file {results_filep}")
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

        return eval_result_lst, combined_result


def evaluate_instances(cfg):
    evaluation_repo = EvaluationRepository(cfg)
    eval_result_lst, combined_result = evaluation_repo.evaluate(occluded=True)
    eval_result_lst, combined_result = evaluation_repo.evaluate(occluded=False)
    return


if __name__ == "__main__":
    cfg = benchmark_config.get_cfg_defaults()
    cmd_args = parse_args.parse_args()
    if cmd_args.cfg_file is not None:
        cfg.merge_from_file(cmd_args.cfg_file)
    if cmd_args.set_cfgs is not None:
        cfg.merge_from_list(cmd_args.set_cfgs)
    evaluate_instances(cfg)
