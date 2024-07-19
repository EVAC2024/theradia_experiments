from data_loader import *
from exp_manager import *
from feat_manager import *
from lib import *
from models import *
from utils import *
import os
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


# Define paths 
base_dir = "/data/"
out_dir = '/data/baseline_exp/'
private_dataset_base = os.path.join(base_dir, "processed_theradia_data/private_dataset/")
public_dataset_base = os.path.join(base_dir, "processed_theradia_data/public_dataset/")

partitions_csv_path = os.path.join(out_dir, "partitions.csv")
key_file = os.path.join(out_dir, "partitions2.csv")

audio_source = 'farfield'
json_name = "data_" + audio_source + "_mono.json"
save_dir = os.path.join(out_dir, "exp_labels_reg" + audio_source)

data_path = os.path.join(save_dir, "data")
results_path = os.path.join(save_dir, "results")

# Initialize the data loader
loader = TheradiaDataLoader(partitions_csv_path, public_dataset_base, private_dataset_base, save_dir, audio_source, json_name)
loader.load_from_json()

# Load the partitions
partitions = loader.load_partitions()
train_dict = partitions.get('train', {})
dev_dict = partitions.get('dev', {})
test_dict = partitions.get('test', {})

# Continuous Dimensions Regression Section
def exp_main_cont_reg(train_data, dev_data, test_data, feat_titles, feat_params, feat_funcs, model_title, model_params, hyper_title, hyper_params, target_label, gs_method):
    exp_name = "data_" + audio_source + "_dim_cont_reg"
    exp_id = f"{exp_name}/{target_label}-{feat_titles['w2v2_xlsr']}-{feat_titles['CLIP_csv']}-{feat_titles['bert_sentiment_google']}-{model_title}-{hyper_title}-{gs_method}"
    outs_file = "outputs.csv"
    exp_dir = os.path.join(results_path, exp_id)
    outs_path = os.path.join(exp_dir, outs_file)
    if os.path.exists(outs_path): return
    print(exp_id)
    torch.manual_seed(hyper_params["seed"])  # important for reproducibility

    # Initialize feature extractors for each modality
    feat_extractor_audio = Feature_extracter(os.path.join(data_path, "Features"), key_file, feat_titles["w2v2_xlsr"], True, **feat_params["w2v2_xlsr"], feat_func=feat_funcs["w2v2_xlsr"])
    feat_extractor_video = Feature_extracter(os.path.join(data_path, "Features"), key_file, feat_titles["CLIP_csv"], True, **feat_params["CLIP_csv"], feat_func=feat_funcs["CLIP_csv"])
    feat_extractor_text = Feature_extracter(os.path.join(data_path, "Features"), key_file, feat_titles["bert_sentiment_google"], True, **feat_params["bert_sentiment_google"], feat_func=feat_funcs["bert_sentiment_google"])

    # Get feature sizes for each modality
    feat_size_audio = feat_extractor_audio.get_feat_size()
    feat_size_video = feat_extractor_video.get_feat_size()
    feat_size_text = feat_extractor_text.get_feat_size()
    print("sizes: ", feat_size_audio, feat_size_video, feat_size_text)

    # Initialize the fusion model with separate feature sizes
    fusion_model = myGRU_mul(feat_size_audio, feat_size_video, feat_size_text, **model_params["params"])
    compute_cost = CEWrapper()  # or MSEWrapper()
    exp = Experimenter_dim_c(exp_dir=exp_dir)
    exp.set_modules(compute_cost=compute_cost)
    exp.set_modules(audio_model=feat_extractor_audio)  # Set audio model
    exp.set_modules(video_model=feat_extractor_video)  # Set video model
    exp.set_modules(text_model=feat_extractor_text)  # Set text model
    exp.set_modules_recoverable(fusion_model=fusion_model)  # Set fusion model

    exp.init_hparams(**hyper_params)
    exp.set_brain_class(exp1Brain_mul)  # Ensure this initializes the brain

    # Only set trs_tar if the brain is properly initialized
    if "trs_tar" in feat_params["bert_sentiment_google"]:
        exp.brain.trs_tar = feat_params["bert_sentiment_google"]["trs_tar"]

    data_train_sb = setup_partition(train_data, target_label=target_label, gs_method=gs_method, seed=hyper_params["seed"])
    data_dev_sb = setup_partition(dev_data, target_label=target_label, gs_method=gs_method, seed=hyper_params["seed"])
    data_test_sb = setup_partition(test_data, target_label=target_label, gs_method=gs_method, seed=hyper_params["seed"])

    exp.fit_brain(data_train_sb, data_dev_sb)
    loss = exp.evaluate_brain(data_test_sb)
    outputs, targets = exp.save_outputs(data_test_sb, outs_path)

def exp_cont_reg_loops(train_data, dev_data, test_data, device="cuda:0"):
    seed = 0
    feats = {
        "w2v2_xlsr": {"source": "voidful/wav2vec2-xlsr-multilingual-56", "device": device, "freeze": True},
        "bert_sentiment_google": {"source": "nlptown/bert-base-multilingual-uncased-sentiment", "device": device, "freeze": True, "trs_tar": "trs"},
        "CLIP_csv": {"csv_dir": "/data/safaa/experiements_theradia/Data/Theradia/Features/clip", "feats_size": 512},
    }
    feat_funcs = {
        "w2v2_xlsr": lambda x: x.squeeze(0),
        "bert_sentiment_google": lambda x: x,
        "CLIP_csv": lambda x: x,
    }
    models = {
        "gru": {"model": myGRU_mul, "params": {"hidden_size": 2302, "output_size": 5}},
    }
    gs_list = ["m"]
    hypers = {
        "hyper0": {
            "device": device,
            "max_epoch": 50,
            "batch_size": 1,
            "grad_acc": 10,
            "lr": 0.0001,
            "seed": seed,
            "limit_to_stop": 5,
            "limit_warmup": 5
        },
    }
    target_labels = ["arousal"]

    for target_label in target_labels:
        for model_title, model_params in models.items():
            for hyper_title, hyper_params in hypers.items():
                for gs_method in gs_list:
                    if gs_method == "m+v":
                        model_params["params"]["output_size"] = 2
                    elif gs_method == "all":
                        model_params["params"]["output_size"] = 6
                    else:
                        model_params["params"]["output_size"] = 5
                    exp_main_cont_reg(train_data, dev_data, test_data, {"w2v2_xlsr": "w2v2_xlsr", "CLIP_csv": "CLIP_csv", "bert_sentiment_google": "bert_sentiment_google"}, feats, feat_funcs, model_title, model_params, hyper_title, hyper_params, target_label, gs_method)

exp_cont_reg_loops(train_dict, dev_dict, test_dict)

# Dimensions Regression Section
def exp_main_dim_reg(train_data, dev_data, test_data, feat_title, feat_params, feat_funcs, model_title, model_params, hyper_title, hyper_params, target_label, gs_method):
    exp_name = "data_" + audio_source + "_dim_sum_reg"
    exp_id = f"{exp_name}/{target_label}-{feat_title}-{model_title}-{hyper_title}-{gs_method}"
    outs_file = "outputs.csv"
    exp_dir = os.path.join(results_path, exp_id)
    outs_path = os.path.join(exp_dir, outs_file)
    if os.path.exists(outs_path): return
    print(exp_id)
    torch.manual_seed(hyper_params["seed"])  # important for reproducibility

    feat_extractor = Feature_extracter(os.path.join(data_path, "Features"), key_file, feat_title, **feat_params, feat_func=feat_funcs[feat_title])
    feat_size = feat_extractor.get_feat_size()
    if "hidden_size" in list(model_params["params"].keys()):
        model_params["params"]["hidden_size"] = feat_size // 2
    main_model = model_params["model"](feat_size, **model_params["params"])
    compute_cost = MSEWrapper()  # CEWrapper()
    exp = Experimenter_dim(exp_dir=exp_dir)
    exp.set_modules(compute_cost=compute_cost)
    exp.set_modules(compute_features=feat_extractor)  # it should go under recoverables if u want fine-tuning
    exp.set_modules_recoverable(main_model=main_model)

    exp.init_hparams(**hyper_params)
    exp.set_brain_class(exp1Brain)
    if "trs_tar" in feat_params:
        exp.brain.trs_tar = feat_params["trs_tar"]

    data_train_sb = setup_partition(train_data, target_label=target_label, gs_method=gs_method, seed=hyper_params["seed"])
    data_dev_sb = setup_partition(dev_data, target_label=target_label, gs_method=gs_method, seed=hyper_params["seed"])
    data_test_sb = setup_partition(test_data, target_label=target_label, gs_method=gs_method, seed=hyper_params["seed"])
    exp.fit_brain(data_train_sb, data_dev_sb)
    loss = exp.evaluate_brain(data_test_sb)
    outputs, targets = exp.save_outputs(data_test_sb, outs_path)
    outputs, targets = exp.save_outputs(data_train_sb, os.path.join(exp_dir, "outputs_train.csv"))
    outputs, targets = exp.save_outputs(data_dev_sb, os.path.join(exp_dir, "outputs_dev.csv"))

def exp_dim_reg_loops(train_data, dev_data, test_data, device="cuda:0"):
    seed = 0
    feats = {
        "CLIP_csv": {"csv_dir": "/data/safaa/experiements_theradia/Data/Theradia/Features/clip", "feats_size": 512},
    }
    feat_funcs = {
        "CLIP_csv": lambda x: x[:, 2:],
    }
    models = {
        "mlp": {"model": myMLP, "params": {"output_size": 5, "hidden_size": "feat_size/2"}},
    }
    gs_list = ["m"]
    hypers = {
        "hyper0": {"device": device, "max_epoch": 50, "batch_size": 1, "grad_acc": 10, "lr": 0.0001, "seed": seed, "limit_to_stop": 5, "limit_warmup": 5},
    }
    target_labels = ["arousal", "novelty", "goal conduciveness", "intrinsic pleasantness", "coping"]

    for target_label in target_labels:
        for feat_title, feat_params in feats.items():
            for model_title, model_params in models.items():
                for hyper_title, hyper_params in hypers.items():
                    for gs_method in gs_list:
                        if gs_method == "m+v":  # in case prediction is based on mean and variation of annotations
                            model_params["params"]["output_size"] = 2
                        elif gs_method == "all":
                            model_params["params"]["output_size"] = 6
                        else:
                            model_params["params"]["output_size"] = 5
                        exp_main_dim_reg(train_data, dev_data, test_data, feat_title, feat_params, feat_funcs, model_title, model_params, hyper_title, hyper_params, target_label, gs_method)

exp_dim_reg_loops(train_dict, dev_dict, test_dict)

# Labels Classification Section
def exp_main_labels_class(train_data, dev_data, test_data, feat_title, feat_params, feat_funcs, model_title, model_params, hyper_title, hyper_params, target_label):
    exp_name = "data_" + audio_source + "_labels_class_MLP"
    exp_id = f"{exp_name}/{target_label}-{feat_title}-{model_title}-{hyper_title}"
    outs_file = "outputs.csv"
    exp_dir = os.path.join(results_path, exp_id)
    outs_path = os.path.join(exp_dir, outs_file)
    if os.path.exists(outs_path): return
    print(exp_dir)
    torch.manual_seed(hyper_params["seed"])  # important for reproducibility

    feat_extractor = Feature_extracter(os.path.join(data_path, "Features"), key_file, feat_title, **feat_params, feat_func=feat_funcs[feat_title])
    feat_size = feat_extractor.get_feat_size()
    if "hidden_size" in list(model_params["params"].keys()):
        model_params["params"]["hidden_size"] = feat_size // 2
    main_model = model_params["model"](feat_size, **model_params["params"])
    compute_cost = CEWrapper()
    exp = Experimenter(exp_dir=exp_dir)
    exp.set_modules(compute_cost=compute_cost)
    exp.set_modules(compute_features=feat_extractor)  # it should go under recoverables if u want fine-tuning
    exp.set_modules_recoverable(main_model=main_model)

    exp.init_hparams(**hyper_params)
    exp.set_brain_class(exp1Brain)
    if "trs_tar" in feat_params:
        exp.brain.trs_tar = feat_params["trs_tar"]

    data_train_sb = setup_partition_class(train_data, target_label=target_label, seed=hyper_params["seed"])
    data_dev_sb = setup_partition_class(dev_data, target_label=target_label, seed=hyper_params["seed"])
    data_test_sb = setup_partition_class(test_data, target_label=target_label, seed=hyper_params["seed"])
    exp.fit_brain(data_train_sb, data_dev_sb)
    loss = exp.evaluate_brain(data_test_sb)
    outputs, targets = exp.save_outputs(data_train_sb, os.path.join(exp_dir, "outputs_train.csv"))
    outputs, targets = exp.save_outputs(data_dev_sb, os.path.join(exp_dir, "outputs_dev.csv"))
    outputs, targets = exp.save_outputs(data_test_sb, outs_path)
    uar = UAR(outputs, targets)
    print(f"{exp_id} UAR:", uar)

def exp_labels_class_loops(train_data, dev_data, test_data, device="cuda:0"):
    seed = 0
    feats = {
        "tfidf": {"corpus": [datum["trs"] for datum in list(train_data.values())], "seed": seed, "trs_tar": "trs"},
        "tfidf_google": {"corpus": [datum["trs_google"] for datum in list(train_data.values())], "seed": seed, "trs_tar": "trs_google"},
        "mfb": {"n_mels": 80},
        "mfb_normed": {"n_mels": 80},
        "fau": {"fau_dir": "/data/safaa/experiements_theradia/Data/Theradia/Features/fau"},
        "w2v2_xlsr": {"source": "voidful/wav2vec2-xlsr-multilingual-56", "device": device, "freeze": True},
        "bert_sentiment": {"source": "nlptown/bert-base-multilingual-uncased-sentiment", "device": device, "freeze": True, "trs_tar": "trs"},
        "bert_sentiment_google": {"source": "nlptown/bert-base-multilingual-uncased-sentiment", "device": device, "freeze": True, "trs_tar": "trs_google"},
        "CLIP_csv": {"csv_dir": "/data/safaa/experiements_theradia/Data/Theradia/Features/clip", "feats_size": 512},
    }
    csv_mean_path = save_dir + "/data/Features/mfb_mean.csv"
    csv_std_path = save_dir + "/data/Features/mfb_std.csv"
    mean = pd.read_csv(csv_mean_path).to_numpy().squeeze(1)
    std = pd.read_csv(csv_std_path).to_numpy().squeeze(1)
    feat_funcs = {
        "tfidf": lambda x: x,
        "tfidf_google": lambda x: x,
        "mfb": lambda x: x.squeeze(0),
        "mfb_normed": lambda x: (x.squeeze(0) - mean) / std,
        "fau": lambda x: x.squeeze(0),
        "w2v2_xlsr": lambda x: x.squeeze(0),
        "bert_sentiment": lambda x: x,
        "bert_sentiment_google": lambda x: x,
        "CLIP_csv": lambda x: x[:, 2:],
    }
    models = {
        "mlp": {"model": myMLP, "params": {"output_size": 2, "hidden_size": "feat_size/2"}},
    }
    hypers = {
        "hyper0": {"device": device, "max_epoch": 50, "batch_size": 10, "grad_acc": 10, "lr": 0.001, "seed": seed, "limit_to_stop": 5, "limit_warmup": 5},
    }
    target_labels = ['annoyed', 'anxious', 'confident', 'desperate', 'frustrated', 'happy', 'interested', 'relaxed', 'satisfied', 'surprised']

    for target_label in target_labels:
        for feat_title, feat_params in feats.items():
            for model_title, model_params in models.items():
                for hyper_title, hyper_params in hypers.items():
                    exp_main_labels_class(train_data, dev_data, test_data, feat_title, feat_params, feat_funcs, model_title, model_params, hyper_title, hyper_params, target_label)

exp_labels_class_loops(train_dict, dev_dict, test_dict)
