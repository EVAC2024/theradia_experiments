from lib import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import torch
import speechbrain as sb

"""
Utilities Module

This module contains utility functions and classes for data preprocessing, target label management, metric calculation,
and experiment fusion used in various machine learning tasks. 

Functions:
    - add_target_label_class: Adds binary target labels based on a specified emotion.
    - add_target_label_class2: Adds binary target labels using a threshold.
    - add_target_label: Adds continuous or binary target labels based on summarization methods.
    - add_dims_summary_annots2: Adds dimensional summaries based on various methods.
    - setup_partition: Sets up a dataset partition for regression tasks.
    - setup_partition_class: Sets up a dataset partition for classification tasks.
    - equalise_dicts: Equalizes the number of samples for each class in a dictionary.
    - UAR: Calculates Unweighted Average Recall.
    - CCC: Calculates Concordance Correlation Coefficient.
    - MAE: Calculates Mean Absolute Error.
    - calculate_metric_on_csv: Calculates metrics (MAE, Pearson correlation, CCC, R2) from a CSV file.
    - calculate_metric_on_csv_uar: Calculates UAR from a CSV file.
    - calc_unique_tars_percentage: Calculates the percentage of unique target values.
    - calculate_uars: Calculates UARs for all experiments in a directory.
    - fusion_linreg_uar: Performs linear regression fusion and calculates UAR.
    - widen_summary_uar: Organizes summary based on specified columns.
    - apply_df_filter: Applies filters to a DataFrame based on metadata.
    - fusion_linreg: Performs linear regression fusion for dimensional predictions.
    - fusion_UAR2: Performs linear regression fusion for UAR.
    - calculate_maes: Calculates MAEs for all experiments in a directory.
    - widen_summary: Organizes summary based on specified columns.
    - print_progress_bar: Prints a progress bar to the terminal.

"""


def add_target_label_class(data, target_label="frustrated"):
    for k, datum in data.items():
        tars_array = np.array(datum["annots"]["labels"])
        for i, row in enumerate(tars_array):
            if row[1] == target_label:
                idx = i
                break
        threshold = 2
        counter = 0
        for i in range(2, 8):
            if tars_array[idx, i].astype(float) > 0:
                counter += 1
        target = 1 if counter > threshold else 0
        datum["target"] = target

def add_target_label_class2(data, target_label="frustrated", threshold=2):
    for k, datum in data.items():
        tars_array = datum["annots"]["labels"]
        idx = None
        counter = 0
        for i, annot in enumerate(tars_array):
            if annot['categories_names'] == target_label:
                idx = i
                break
        if idx is not None:
            user_scores = [int(annot[user]) for user in tars_array[idx] if user.startswith('user_')]
            counter = sum(1 for score in user_scores if score > 0)
            target = 1 if counter > threshold else 0
        else:
            target = 0
        datum["target"] = target

def setup_partition_class(data, target_label="frustrated", seed=0):
    data_c = data.copy()
    add_target_label_class2(data_c, target_label)
    data_c = equalise_dicts(data_c, "target", seed=seed)
    output_keys = ["ID", "video_path", "wav_path", "trs", "trs_google", "target"]
    data_sb = sb.dataio.dataset.DynamicItemDataset(data_c, output_keys=output_keys)
    return data_sb

def add_target_label(data, target_label="frustrated", gs_method="m"):
    dims_list = ["arousal", "novelty", "goal conduciveness", "intrinsic pleasantness", "coping"]
    labels_list = ['annoyed', 'anxious', 'confident', 'desperate', 'frustrated', 'happy', 'interested', 'relaxed', 'satisfied', 'surprised']
    for k, datum in data.items():
        if target_label in labels_list:
            annots_type = "labels"
        elif target_label in dims_list:
            annots_type = "dimension_summaries"
        else:
            continue
        tars_array = datum["annots"][annots_type]
        target_values = []
        for annot in tars_array:
            if annot.get('categories_names', '') == target_label or annot.get('dimensions', '') == target_label:
                user_scores = [float(annot[user]) for user in datum["annots"][annots_type][0] if user.startswith('user_')]
                target_values.extend(user_scores)
        if not target_values:
            print(f"No matching annotations found for {target_label} in {k}.")
            continue
        if gs_method == "m":
            target = float(np.mean(target_values))
        elif gs_method == "mnz":
            non_zero_values = [v for v in target_values if v != 0]
            target = float(np.mean(non_zero_values)) if non_zero_values else 0
        elif gs_method == "max":
            target = float(np.max(target_values))
        elif gs_method == "m+v":
            target = np.array([np.mean(target_values), np.std(target_values)])
        elif gs_method == "all":
            target = np.array(target_values)
        datum["target2"] = target

def add_dims_summary_annots2(data, gs_method="m"):
    tar_dims = ["arousal", "novelty", "goal conduciveness", "intrinsic pleasantness", "coping"]
    for k, datum in data.items():
        targets = []
        if "dimension_summaries" not in datum["annots"]:
            print(f"Dimension summaries not found for {k}.")
            continue
        dim_annotations = datum["annots"]["dimension_summaries"]
        for tar_dim in tar_dims:
            dim_values = [annot for annot in dim_annotations if annot['dimensions'] == tar_dim]
            if not dim_values:
                print(f"No data for {tar_dim} in {k}.")
                continue
            scores = [float(value[user]) for value in dim_values for user in value if user.startswith('user_')]
            if gs_method == "m":
                target = np.mean(scores)
            elif gs_method == "mnz":
                non_zero_scores = [score for score in scores if score != 0]
                target = np.mean(non_zero_scores) if non_zero_scores else 0
            elif gs_method == "max":
                target = np.max(scores)
            elif gs_method == "m+v":
                mean = np.mean(scores)
                std_dev = np.std(scores)
                target = np.array([mean, std_dev])
            elif gs_method == "all":
                target = np.array(scores)
            targets.append(target)
        datum["target"] = np.array(targets)

def setup_partition(data, target_label="frustrated", seed=0, gs_method="m"):
    data_c = data.copy()
    add_dims_summary_annots2(data_c, gs_method)
    output_keys = ["ID", "video_path", "wav_path", "trs", "target"]
    data_sb = sb.dataio.dataset.DynamicItemDataset(data_c, output_keys=output_keys)
    return data_sb

def equalise_dicts(dicts, equaliseKey: str, seed=0) -> dict:
    np.random.seed(seed)
    equalised_dicts = {}
    for key in dicts:
        equalised_dicts[key] = dicts[key].copy()
    dataIndexed = {}
    for key in dicts:
        targetData = dicts[key]
        IdxKey = targetData[equaliseKey]
        if IdxKey not in dataIndexed:
            dataIndexed[IdxKey] = []
        dataIndexed[IdxKey].append(targetData)
    maxNums = max(len(dataIndexed[key]) for key in dataIndexed)
    for key in dataIndexed:
        updateSize = maxNums - len(dataIndexed[key])
        updateIds = np.random.choice(list(range(len(dataIndexed[key]))), size=updateSize, replace=True)
        for k, updateId in enumerate(updateIds):
            equalised_dicts[f"replicated_{key}_{k}"] = dataIndexed[key][updateId]
    return equalised_dicts

# Metrics

def UAR(tars, outs):
    tarsSet = list(set(tars))
    corrects = {i: 0 for i in tarsSet}
    totals = {i: 0 for i in tarsSet}
    for i, out in enumerate(outs):
        tar = tars[i]
        totals[tar] += 1
        if out == tar:
            corrects[tar] += 1
    uar = sum(corrects[i] / totals[i] for i in tarsSet) / len(tarsSet)
    return uar

def CCC(array1, array2):
    mean_gt = np.mean(array2, 0)
    mean_pred = np.mean(array1, 0)
    var_gt = np.var(array2, 0)
    var_pred = np.var(array1, 0)
    v_pred = array1 - mean_pred
    v_gt = array2 - mean_gt
    denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
    cov = np.mean(v_pred * v_gt)
    numerator = 2 * cov
    if denominator == 0:
        return 1.0 if list(array1) == list(array2) else 0.0
    ccc = numerator / denominator
    return ccc

def MAE(array1, array2):
    return np.mean(np.abs(array1 - array2))

def calculate_metric_on_csv(csv_path, out_row="output", tar_row="target"):
    df = pd.read_csv(csv_path)
    df_filtered = df
    outs = df_filtered[out_row].to_numpy()
    tars = df_filtered[tar_row].to_numpy()
    if type(tars[0]) == str:
        tars = np.array([float(tar.replace("[", "").replace("]", "")) for tar in tars])
    mae = MAE(tars, outs)
    pc = pearsonr(tars, outs)[0]
    ccc = CCC(tars, outs)
    r2 = r2_score(tars, outs)
    return mae, pc, ccc, r2

def calculate_metric_on_csv_uar(csv_path, out_row="output", tar_row="target"):
    df = pd.read_csv(csv_path)
    outs = df[out_row].to_numpy()
    tars = df[tar_row].to_numpy()
    return UAR(tars, outs)

def calc_unique_tars_percentage(csv_path, id_row="ID", tar_row="target"):
    df = pd.read_csv(csv_path)
    id_uniques = []
    unique_tars = []
    for index, row in df.iterrows():
        if row[id_row] not in id_uniques:
            id_uniques.append(row[id_row])
            unique_tars.append(row[tar_row])
    return sum(unique_tars)

def calculate_uars(exp_dir, outs_name, uars_path):
    outs_files = glob.glob(f"{exp_dir}/**/{outs_name}", recursive=False)
    df_data = {"Experiment": [], "Result": [], "Ones": []}
    for outs_file in outs_files:
        file_name = os.path.split(os.path.split(outs_file)[0])[1]
        result = calculate_metric_on_csv_uar(outs_file)
        perc = calc_unique_tars_percentage(outs_file)
        df_data["Experiment"].append(file_name)
        df_data["Result"].append(result)
        df_data["Ones"].append(perc)
    df = pd.DataFrame(df_data)
    df.to_csv(uars_path, index=False)

# Fusion and Summary Functions

def fusion_linreg_uar(exp_dir, save_dir, feats, labels, model="linear-hyper0", preds_train="outputs_train.csv", preds_dev="outputs_dev.csv", preds_test="outputs.csv", col_out="output", col_tar="target"):
    new_df_data = {"Experiment": [], "test_uar": []}
    for feat in feats:
        new_df_data[f"coef_{feat}"] = []
    for label in labels:
        outs_train_feats = []
        outs_dev_feats = []
        outs_test_feats = []
        tars_train_feats = []
        tars_dev_feats = []
        tars_test_feats = []
        for feat in feats:
            preds_train_path = os.path.join(exp_dir, f"{label}-{feat}-{model}", preds_train)
            preds_dev_path = os.path.join(exp_dir, f"{label}-{feat}-{model}", preds_dev)
            preds_test_path = os.path.join(exp_dir, f"{label}-{feat}-{model}", preds_test)
            if not os.path.exists(preds_test_path):
                continue
            df_train = pd.read_csv(preds_train_path)
            df_dev = pd.read_csv(preds_dev_path)
            df_test = pd.read_csv(preds_test_path)
            outs_train = df_train[col_out].to_numpy()
            outs_dev = df_dev[col_out].to_numpy()
            outs_test = df_test[col_out].to_numpy()
            tars_train = df_train[col_tar].to_numpy()
            tars_dev = df_dev[col_tar].to_numpy()
            tars_test = df_test[col_tar].to_numpy()
            outs_train_feats.append(outs_train)
            outs_dev_feats.append(outs_dev)
            outs_test_feats.append(outs_test)
            tars_train_feats.append(tars_train)
            tars_dev_feats.append(tars_dev)
            tars_test_feats.append(tars_test)
        if len(outs_train_feats) == 0:
            continue
        outs_train_feats = np.array(outs_train_feats).transpose() / 100
        outs_dev_feats = np.array(outs_dev_feats).transpose() / 100
        outs_test_feats = np.array(outs_test_feats).transpose() / 100
        tars_train_feats = tars_train_feats[0]
        tars_dev_feats = tars_dev_feats[0]
        tars_test_feats = tars_test_feats[0]
        reg = LinearRegression().fit(np.concatenate([outs_train_feats, outs_dev_feats]), np.concatenate([tars_train_feats, tars_dev_feats]))
        coefs = reg.coef_ / sum(reg.coef_)
        preds = reg.predict(outs_test_feats)
        preds = [round(pred) for pred in preds]
        uar = UAR(tars_test_feats, preds)
        exp_str = "+".join(feats) + f"-{label}"
        new_df_data["Experiment"].append(exp_str)
        new_df_data["test_uar"].append(uar)
        for f, feat in enumerate(feats):
            new_df_data[f"coef_{feat}"].append(coefs[f])
    df = pd.DataFrame(new_df_data)
    df.to_csv(save_dir, index=False)

def widen_summary_uar(summ_dir, save_dir, name_col="Experiment", tar_col="Result", sep="-", row_name="emotion", rows=[], cols=[]):
    df = pd.read_csv(summ_dir)
    new_cols = [row_name] + cols
    new_df_data = {col: [] for col in new_cols}
    for col in cols:
        for row in rows:
            df_str_0 = f'{row}'
            df_str = f'{row}{sep}{col}'
            df_str = df_str.replace("+", "\+")
            df_filtered = df[df[name_col].str.contains(df_str)]
            val = np.mean(df_filtered[tar_col].to_numpy())
            if df_str_0 not in new_df_data[row_name]:
                new_df_data[row_name].append(df_str_0)
            new_df_data[col].append(val)
    new_df = pd.DataFrame(new_df_data)
    new_df.to_csv(save_dir, index=False)

def apply_df_filter(df_in, filter_data="none", subject_meta_df=None):
    df = df_in.copy()
    if filter_data == "none":
        return df
    elif "mci" in filter_data:
        df = df[df["ID"].str.startswith("M")]
    elif "senior" in filter_data:
        df = df[df["ID"].str.startswith("A")]
    if "female" in filter_data:
        subject_meta_df = subject_meta_df.dropna()
        rule = subject_meta_df["gender"].str.startswith("homme")
        subjects = subject_meta_df[rule]["id"].to_list()
        drop_ids = [df_idx for idx in subjects for df_idx, sequence in zip(df.index.tolist(), df["ID"].to_list()) if idx in sequence]
        df = df.drop(index=drop_ids)
    elif "male" in filter_data:
        subject_meta_df = subject_meta_df.dropna()
        rule = subject_meta_df["gender"].str.startswith("femme")
        subjects = subject_meta_df[rule]["id"].to_list()
        drop_ids = [df_idx for idx in subjects for df_idx, sequence in zip(df.index.tolist(), df["ID"].to_list()) if idx in sequence]
        df = df.drop(index=drop_ids)
    return df

def fusion_linreg(exp_dir, save_dir, feats, labels, model="mlp-hyper0", preds_train="outputs_train.csv", preds_dev="outputs_dev.csv", preds_test="outputs.csv", cols_out=[f"output_{i}" for i in range(6)], cols_tar=[f"target_{i}" for i in range(6)], filter_data="none", subject_meta_df=None):
    new_df_data = {"Experiment": [], "test_ccc": [], "dev_ccc": []}
    for feat in feats:
        for col in cols_out:
            new_df_data[f"coef_{feat}_{col}"] = []
    for col in cols_out:
        new_df_data[f"intercept_{col}"] = []
    for label in labels:
        outs_train_feats = []
        outs_dev_feats = []
        outs_test_feats = []
        tars_train_feats = []
        tars_dev_feats = []
        tars_test_feats = []
        for feat in feats:
            preds_train_path = os.path.join(exp_dir, f"{label}-{feat}-{model}", preds_train)
            preds_dev_path = os.path.join(exp_dir, f"{label}-{feat}-{model}", preds_dev)
            preds_test_path = os.path.join(exp_dir, f"{label}-{feat}-{model}", preds_test)
            if not os.path.exists(preds_test_path):
                continue
            df_train = pd.read_csv(preds_train_path)
            df_dev = pd.read_csv(preds_dev_path)
            df_test = pd.read_csv(preds_test_path)
            df_dev = apply_df_filter(df_dev, filter_data=filter_data, subject_meta_df=subject_meta_df)
            df_test = apply_df_filter(df_test, filter_data=filter_data, subject_meta_df=subject_meta_df)
            outs_train = []
            outs_dev = []
            outs_test = []
            for col_out in cols_out:
                outs_train.append(df_train[col_out].to_list())
                outs_dev.append(df_dev[col_out].to_list())
                outs_test.append(df_test[col_out].to_list())
            outs_train = np.stack(outs_train, 1)
            outs_dev = np.stack(outs_dev, 1)
            outs_test = np.stack(outs_test, 1)
            tars_train = []
            tars_dev = []
            tars_test = []
            for col_tar in cols_tar:
                tars_train.append(df_train[col_tar].to_list())
                tars_dev.append(df_dev[col_tar].to_list())
                tars_test.append(df_test[col_tar].to_list())
            tars_train = np.stack(tars_train, 1)
            tars_dev = np.stack(tars_dev, 1)
            tars_test = np.stack(tars_test, 1)
            outs_train_feats.append(outs_train)
            outs_dev_feats.append(outs_dev)
            outs_test_feats.append(outs_test)
            tars_train_feats.append(tars_train)
            tars_dev_feats.append(tars_dev)
            tars_test_feats.append(tars_test)
        if len(outs_train_feats) == 0:
            continue
        outs_train_feats = np.concatenate(outs_train_feats, 1)
        outs_dev_feats = np.concatenate(outs_dev_feats, 1)
        outs_test_feats = np.concatenate(outs_test_feats, 1)
        tars_train_feats = tars_train_feats[0]
        tars_dev_feats = tars_dev_feats[0]
        tars_test_feats = tars_test_feats[0]
        train_data_feats = np.concatenate((outs_train_feats, outs_dev_feats), 0)
        train_data_tars = np.concatenate((tars_train_feats, tars_dev_feats), 0)
        reg = LinearRegression().fit(outs_train_feats, tars_train_feats)
        coefs = reg.coef_
        preds = reg.predict(outs_test_feats)
        preds_d = reg.predict(outs_dev_feats)
        ccc = CCC(np.mean(tars_test_feats, 1), np.mean(preds, 1))
        ccc_dev = CCC(np.mean(tars_dev_feats, 1), np.mean(preds_d, 1))
        exp_str = "+".join(feats) + f"-{label}"
        new_df_data["Experiment"].append(exp_str)
        new_df_data["test_ccc"].append(ccc)
        new_df_data["dev_ccc"].append(ccc_dev)
        total_coef = 0
        counter = 0
        for f, feat in enumerate(feats):
            for c, col in enumerate(cols_out):
                total_coef += coefs[c, counter]
                counter += 1
        counter = 0
        for f, feat in enumerate(feats):
            for c, col in enumerate(cols_out):
                new_df_data[f"coef_{feat}_{col}"].append(coefs[c, counter])
                counter += 1
        for c, col in enumerate(cols_out):
            new_df_data[f"intercept_{col}"].append(reg.intercept_[c])
    df = pd.DataFrame(new_df_data)
    df.to_csv(save_dir, index=False)

def fusion_UAR2(exp_dir, save_dir, feats, labels, model="linear-hyper0", preds_name="outputs.csv"):
    new_df_data = {"Experiment": [], "test_uar": []}
    for label in labels:
        all_outs = []
        all_tars = []
        for feat in feats:
            preds_path = os.path.join(exp_dir, f"{label}-{feat}-{model}", preds_name)
            if not os.path.exists(preds_path):
                continue
            df = pd.read_csv(preds_path)
            preds = df["output"].to_numpy()
            all_outs.append(preds)
            tars = df["target"].to_numpy()
            all_tars.append(tars)
        if len(all_tars) == 0:
            continue
        all_outs = np.mean(all_outs, 0)
        all_tars = np.mean(all_tars, 0)
        all_outs = [round(out) for out in all_outs]
        uar = "nan" if sum(all_tars) == 0 else UAR(all_tars, all_outs)
        exp_str = "+".join(feats) + f"-{label}"
        new_df_data["Experiment"].append(exp_str)
        new_df_data["test_uar"].append(uar)
    df = pd.DataFrame(new_df_data)
    df.to_csv(save_dir, index=False)

def calculate_maes(exp_dir, outs_name, uars_path, out_row="output", tar_row="target"):
    outs_files = glob.glob(f"{exp_dir}/**/{outs_name}", recursive=False)
    df_data = {"Experiment": [], "MAE": [], "PC": [], "CCC": [], "R2": []}
    for outs_file in outs_files:
        file_name = os.path.split(os.path.split(outs_file)[0])[1]
        mae, pc, ccc, r2 = calculate_metric_on_csv(outs_file, out_row=out_row, tar_row=tar_row)
        df_data["Experiment"].append(file_name)
        df_data["MAE"].append(mae)
        df_data["PC"].append(pc)
        df_data["CCC"].append(ccc)
        df_data["R2"].append(r2)
    df = pd.DataFrame(df_data)
    df.to_csv(uars_path, index=False)

def widen_summary(summ_dir, save_dir, name_col="Experiment", tar_col="Result", sep="-", row_name="emotion", rows=[], cols=[]):
    df = pd.read_csv(summ_dir)
    new_cols = [row_name] + cols
    new_df_data = {col: [] for col in new_cols}
    for col in cols:
        for row in rows:
            df_str_0 = f'{row}'
            df_str = f'{row}{sep}{col}'
            df_str = df_str.replace("+", "\+")
            df_filtered = df[df[name_col].str.contains(df_str)]
            val = np.mean(df_filtered[tar_col].to_numpy())
            if df_str_0 not in new_df_data[row_name]:
                new_df_data[row_name].append(df_str_0)
            new_df_data[col].append(val)
    new_df = pd.DataFrame(new_df_data)
    new_df.to_csv(save_dir, index=False)

# Progress Bar

def print_progress_bar(iteration: int, total: int, prefix='', suffix='', decimals=1, length="fit", fill='█') -> None:
    if length == "fit":
        rows, columns = os.popen('stty size', 'r').read().split()
        length = int(columns) // 2
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()
