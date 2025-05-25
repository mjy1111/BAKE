import collections
import json
from pprint import pprint
from typing import List, Optional

import numpy as np
from scipy.stats import hmean
from pathlib import Path
# from util.globals import *


def main(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    # dir_name 可能是一个具体路径，也可能是 "results/..." 下的目录
    base_dir = Path(dir_name) if abs_path else Path(f"results/{dir_name}")

    for run_dir in base_dir.iterdir():
        # 如果指定了 runs 且都不匹配，则跳过
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("*case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))

        print(f"[{run_dir}] total case files: {len(files)}")
        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")
                continue

            case_id = data.get("case_id", -1)
            if first_n_cases is not None and case_id >= first_n_cases:
                break

            # 记录执行时长
            if "time" in data:
                cur_sum["time"].append(data["time"])

            # 只关心 pre / post 两个阶段
            for prefix in ["pre", "post"]:
                if prefix not in data:
                    continue

                # —— 关键改动：新结构下，所有提示的 probs/correct 都在 reverse_evaluation 里 ——
                reverse_eval = data[prefix].get("reverse_evaluation", {})

                # （A）概率度量：对于 new 应该 < true 的
                #     对应 old 脚本里的 ["rewrite_prompts_probs", "paraphrase_prompts_probs"]
                for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                    if key not in reverse_eval:
                        continue
                    if not reverse_eval[key]:  # 空列表
                        continue

                    sum_key_discrete = f"{prefix}_{key.split('_prompts')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_prompts')[0]}_diff"

                    # 离散：true > new 的比例
                    # 连续：exp(-new) - exp(-true) 的均值
                    cur_sum[sum_key_discrete].append(
                        np.mean([x["target_true"] > x["target_new"] for x in reverse_eval[key]])
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean([np.exp(-x["target_new"]) - np.exp(-x["target_true"]) for x in reverse_eval[key]])
                    )

                # （B）reverse_qa_prompts_probs
                #     你原先的逻辑：先判断 data["pre"][key][0] 里面是否 true > new，再决定是否继续
                if "reverse_qa_prompts_probs" in reverse_eval:
                    key = "reverse_qa_prompts_probs"
                    if reverse_eval[key]:
                        # 过滤
                        y = reverse_eval[key][0]
                        if y["target_true"] > y["target_new"]:
                            # 如果 pre 阶段就发现 target_true > target_new，则不继续？
                            # 原脚本写的是 'continue'。这里也照搬
                            pass
                        else:
                            sum_key_discrete = f"{prefix}_reverse_qa_success"
                            sum_key_cont = f"{prefix}_reverse_qa_diff"

                            cur_sum[sum_key_discrete].append(
                                np.mean([x["target_true"] > x["target_new"] for x in reverse_eval[key]])
                            )
                            cur_sum[sum_key_cont].append(
                                np.mean([np.exp(-x["target_new"]) - np.exp(-x["target_true"]) for x in reverse_eval[key]])
                            )

                # （C）reverse_judge_prompts_probs
                if "reverse_judge_prompts_probs" in reverse_eval:
                    key = "reverse_judge_prompts_probs"
                    if reverse_eval[key]:
                        # filter：
                        x0 = reverse_eval[key][0]
                        if x0["target_true"] > x0["target_new"]:
                            # 原脚本是直接 continue
                            pass
                        else:
                            sum_key_discrete = f"{prefix}_reverse_judge_success"
                            sum_key_cont = f"{prefix}_reverse_judge_diff"

                            # 注意：旧逻辑里是对 every x in data[prefix][key] 做单独 append，
                            #       这里可以简化成一次性均值
                            cur_sum[sum_key_discrete].append(
                                np.mean([x["target_true"] > x["target_new"] for x in reverse_eval[key]])
                            )
                            cur_sum[sum_key_cont].append(
                                np.mean([np.exp(-x["target_new"]) - np.exp(-x["target_true"]) for x in reverse_eval[key]])
                            )

                # （D）Probability metrics for which true < new (neighborhood)
                #     对应 old 脚本的 key = "neighborhood_prompts_probs"
                #     这时 true 应该 < new
                neigh_key = "neighborhood_prompts_probs"
                if neigh_key in reverse_eval and reverse_eval[neigh_key]:
                    sum_key_discrete = f"{prefix}_neighborhood_success"
                    sum_key_cont = f"{prefix}_neighborhood_diff"
                    cur_sum[sum_key_discrete].append(
                        np.mean([x["target_true"] < x["target_new"] for x in reverse_eval[neigh_key]])
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean([np.exp(-x["target_true"]) - np.exp(-x["target_new"]) for x in reverse_eval[neigh_key]])
                    )

                # （E）Accuracy-based evaluation metrics
                #     原脚本里：for key in ["rewrite", "paraphrase", "neighborhood"]
                #     现在都存储在 reverse_evaluation 下：f"{key}_prompts_correct"
                for name in ["rewrite", "paraphrase", "neighborhood"]:
                    correctness_key = f"{name}_prompts_correct"  # e.g. "rewrite_prompts_correct"
                    if correctness_key in reverse_eval:
                        # 这里先做个平均
                        sum_key = f"{prefix}_{name}_acc"
                        cur_sum[sum_key].append(np.mean(reverse_eval[correctness_key]))

                # （F）其它可直接平均的度量(ngram_entropy / reference_score / essence_score等)
                #     如果这些还在 data[prefix] 顶层，就跟原脚本保持不变
                for met_key in ["ngram_entropy", "reference_score", "essence_score"]:
                    if met_key in data[prefix]:
                        cur_sum[f"{prefix}_{met_key}"].append(data[prefix][met_key])

        # 若当前 run 没产生任何统计，则跳过
        if len(cur_sum) == 0:
            continue

        # 记录有多少case
        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }
        uncompressed.append(dict(cur_sum, **metadata))

        # 计算平均 & 标准差
        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}

        # 按原逻辑：把除了 essence_score / time 以外的指标都 *100
        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

        # 额外生成一个 score (hmean) 指标
        for prefix in ["pre", "post"]:
            # 例：("pre_rewrite_success", "pre_paraphrase_success", "pre_neighborhood_success")
            # 取它们的调和平均
            for k_efficacy, k_generalization, k_specificity in [
                (
                    f"{prefix}_rewrite_success",
                    f"{prefix}_paraphrase_success",
                    f"{prefix}_neighborhood_success",
                ),
            ]:
                if all(k in cur_sum for k in [k_efficacy, k_generalization, k_specificity]):
                    hmean_list = [
                        cur_sum[k_efficacy][0],
                        cur_sum[k_generalization][0],
                        cur_sum[k_specificity][0],
                    ]
                    cur_sum[f"{prefix}_score"] = (hmean(hmean_list), np.nan)
                    break

        cur_sum.update(metadata)
        pprint(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
             "Useful for comparing different in-progress runs on the same slice of data.",
    )
    args = parser.parse_args()

    main(
        args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
    )
