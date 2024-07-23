from lib import *
from utils import *

"""
Experiment Manager Module

The module is designed to handle general experiment  setup, including initializing hyperparameters, setting up optimizers, managing modules, and checkpointing. 
It also includes specialised classes for handling different types of experiments.

Classes:
    - Experimenter: Handles general experiment setup and management, including hyperparameters, optimizer, modules, and checkpoints.
    - Experimentor: Manages common tasks across different experiments like logging and saving/loading results.
    - expBrain: Extends the SpeechBrain Brain class to implement the model's forward pass, objective computation, and training procedures.
    - exp1Brain: A specialized version of expBrain for specific experiment setups, particularly involving speech recognition.
    - exp1Brain_mul: A specialized version of expBrain for multimodal experiments involving audio, video, and text features.
    - Experimenter_lab: A specialized version of Experimenter for experiments predicting mean and standard deviation.
    - Experimenter_dim: A specialized version of Experimenter for experiments handling dimension-based data.
    - Experimenter_dim_c: A specialized version of Experimenter for experiments handling continuous dimension-based data.

"""

class Experimenter():
    def __init__(self, exp_dir="./Results"):
        self.file_dir = os.path.dirname(os.path.abspath("__file__"))
        self.exp_dir = exp_dir
        self.checkpointer = self.init_checkpointer()
        self.hparams = {}
        self.init_hparams()
        self.set_optimizer()
        self.modules = {}
        self.brain = None
        
    def init_checkpointer(self):
        return sb.utils.checkpoints.Checkpointer(
            checkpoints_dir=os.path.join(self.exp_dir, "checkpoints"),
            recoverables={}
        )
            
    def init_hparams(self, max_epoch=50, batch_size=1, device="cpu", grad_acc=1, seed=0, limit_to_stop=5, limit_warmup=5, lr=0.001):
        self.hparams["epoch_counter"] = sb.utils.epoch_loop.EpochCounterWithStopper(
            limit=max_epoch,
            limit_to_stop=limit_to_stop,
            limit_warmup=limit_warmup,
            direction="min",
        )
        self.checkpointer.add_recoverable("epoch_counter", self.hparams["epoch_counter"])

        self.hparams["dataloader_options"] = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 2,  # 2 on linux but 0 works on windows
            "drop_last": False,
        }
        
        self.hparams["train_logger"] = sb.utils.train_logger.FileTrainLogger(
            save_file=os.path.join(self.exp_dir, "train_log.txt")
        )
        
        self.device = device
        self.hparams["run_opts"] = {
            "seed": seed,
            "device": device,
            "ckpt_interval_minutes": 0,
            "grad_accumulation_factor": grad_acc,
        }
        
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if "cuda" in device:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        self.set_optimizer(lr=lr)
        
    def set_hparams(self, **kwargs):
        for k, w in kwargs.items():
            self.hparams[k] = w
            
    def set_optimizer(self, lr=0.001):
        self.opt_class = lambda x: torch.optim.Adam(x, lr)
        
    def set_modules(self, **kwargs):
        for k, w in kwargs.items():
            self.modules.update({k:w})
    
    def set_modules_recoverable(self, **kwargs):
        for k, w in kwargs.items():
            self.modules.update({k:w})
            self.checkpointer.add_recoverable(k, w)
            
    def set_brain_class(self, brain):
        self.brain = brain(
            modules=self.modules,
            opt_class=self.opt_class,
            hparams=self.hparams,
            checkpointer=self.checkpointer,
            run_opts=self.hparams["run_opts"],
        )
        
    def fit_brain(self, data_train_dynamic, data_dev_dynamic):
        self.save_data_dynamic(data_train_dynamic, "train")
        self.save_data_dynamic(data_dev_dynamic, "dev")
        time0 = time.time() # Calculating the training time
        self.brain.fit(
            epoch_counter=self.hparams["epoch_counter"],
            train_set=data_train_dynamic,
            valid_set=data_dev_dynamic,
            train_loader_kwargs=self.hparams["dataloader_options"],
            valid_loader_kwargs=self.hparams["dataloader_options"],
        )
        time1 = time.time() # Calculating the training time
        save_dir = os.path.join(self.exp_dir, "train_time.txt")
        with open(save_dir, "w") as text_file: # Saving the training time to a file
            text_file.write("{}".format(str(timedelta(seconds=time1-time0))))
    
    def evaluate_brain(self, data_test_dynamic):
        self.save_data_dynamic(data_test_dynamic, "test")
        loss = self.brain.evaluate(
            test_set=data_test_dynamic,
            min_key="loss",
            test_loader_kwargs=self.hparams["dataloader_options"],
        )
        return loss
    
    def load_best_model(self):
        self.brain.on_evaluate_start(min_key="loss")
        
    def save_data_dynamic(self, data_dynamic, part):
        save_dir = os.path.join(self.exp_dir, f"data_{part}.csv")
        data = []
        for i, item in enumerate(data_dynamic):
            datum = {}
            item_keys = list(item.keys())
            for key in item_keys:
                datum[key] = item[key]
            data.append(datum)
        df = pd.DataFrame(data=data)
        df.to_csv(save_dir, index=False)
    
    def save_outputs(self, data_dynamic, save_dir):
        if os.path.exists(save_dir):
            df = pd.read_csv(save_dir)
            outputs = df["output"].tolist()
            targets = df["target"].tolist()
        else:
            data = []
            outputs = []
            targets = []
            for i, item in enumerate(data_dynamic):
                print_progress_bar(i + 1, len(data_dynamic), prefix='Calculating outputs of the model:', suffix='complete', length=50)
                datum = {}
                item_keys = list(item.keys())
                for key in item_keys: # to save all items except for input and target
                    datum[key] = item[key]
                self.update_datum_for_item(datum, item)
                data.append(datum)
                outputs.append(datum["output"])
                if "target" in list(datum.keys()): targets.append(datum["target"])
            df = pd.DataFrame(data=data)
            df.to_csv(save_dir, index=False)
        return outputs, targets
    
    def update_datum_for_item(self, datum, item):
        # we want to save outputs one by one in different cells
        output = self.calc_item_output(item)
        probs = torch.softmax(output, -1)
        probs = probs.view(-1).cpu().detach().numpy().tolist()
        datum["output"] = np.argmax(probs)
        for i, out in enumerate(probs):
            datum[f"output_{i}"] = round(probs[i] * 100, 3)
    
    def calc_item_output(self, item):
        fake_batch = sb.dataio.batch.PaddedBatch([item])
        output = self.brain.compute_forward(fake_batch, None)
        return output
        
    def save_module_jit(self, inputs_fake, module_key="main_model"):
        with torch.no_grad(): # the tracing with jit is needed to save a model without the code
            traced_cell = torch.jit.trace(self.modules[module_key], inputs_fake)
        save_path = os.path.join(self.exp_dir, f"{module_key}.pth")
        torch.jit.save(traced_cell, save_path)
    
    def save_module(self, module_key="main_model"):
        with torch.no_grad():
            save_path = os.path.join(self.exp_dir, f"{module_key}.pth")
            torch.save(self.modules[module_key], save_path)


class Experimentor():
    '''
    This class handles common tasks across different experiments, such as storing different parameters, saving and loading results, etc.
    '''
    def __init__(self, ID, save_dir, seed=0):
        self.ID = ID
        self.save_dir = os.path.join(save_dir, self.ID)
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.logger = {"ID": ID, "seed": seed}
        self.logger_path = os.path.join(self.save_dir, "logs.json")
        
    def save_logger(self):
        '''Saves the logger dict in json format to the logger_path.
        '''
        with open(self.logger_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.logger, json_file)
            
    def load_logger(self):
        '''Loads the logger from the logger_path in numpy dict.
        '''
        with open(self.logger_path, 'r', encoding='utf-8') as json_file:
            self.logger = json.load(json_file)
    
    def save_data(self, data, save_path):
        with open(save_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file)
            
    def load_data(self, save_path):
        with open(save_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data


class expBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on computing features and performing the main model.
        """
        batch = batch.to(self.device)
        try:
            inputs, lens = batch.input
        except:
            inputs = batch.input
        outputs = self.compute_outputs(inputs)
        return outputs
    
    def compute_outputs(self, inputs):
        feats = self.modules.compute_features(inputs)
        outputs = self.modules.main_model(feats)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss.
        """
        target = batch.target
        loss = self.modules.compute_cost(predictions, target)
        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        (loss / self.hparams.run_opts["grad_accumulation_factor"]).backward()

        if self.step % self.hparams.run_opts["grad_accumulation_factor"] == 0:
            self.check_gradients()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        pass

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stats = {"loss": stage_loss}

        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            if self.hparams.epoch_counter.should_stop(current=epoch, current_metric=stage_loss):
                self.hparams.epoch_counter.current = self.hparams.epoch_counter.limit

            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


class exp1Brain(expBrain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        outputs = self.compute_outputs(batch)
        return outputs

    def compute_outputs(self, batch):
        trs = batch.trs
        try:
            if self.trs_tar == "trs_google":
                trs = batch.trs_google
        except:
            pass
        feats = self.modules.compute_features(ID=batch.ID, wav_path=batch.wav_path, trs=trs).to(self.device).float()
        outputs = self.modules.main_model(feats)
        return outputs


class exp1Brain_mul(expBrain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        outputs = self.compute_outputs(batch)
        return outputs

    def compute_outputs(self, batch):
        trs = batch.trs
        try:
            if self.trs_tar == "trs_google":
                trs = batch.trs_google
        except:
            pass
        
        audio_feats = self.modules.audio_model(ID=batch.ID, wav_path=batch.wav_path, target_len=batch.target_len, times=batch.times, len_lst=batch.id_to_length).to(self.device).float()
        video_feats = self.modules.video_model(ID=batch.ID, wav_path=batch.wav_path, target_len=batch.target_len, times=batch.times, len_lst=batch.id_to_length).to(self.device).float()
        text_feats = self.modules.text_model(ID=batch.ID, trs=trs, target_len=batch.target_len).to(self.device).float()

        combined_feats = torch.cat((audio_feats, video_feats, text_feats), dim=-1)
        outputs = self.modules.fusion_model(combined_feats)
        return outputs


class Experimenter_lab(Experimenter):
    def update_datum_for_item(self, datum, item):
        outs = self.calc_item_output(item)
        outputs = outs.view(-1).cpu().detach().numpy().tolist()
        datum["output"] = outputs[0]

        if len(outputs) == 2:  # if prediction was for mean+std
            datum["output_var"] = outputs[1]
            datum["target_var"] = list(datum["target"])[1]
            datum["target"] = list(datum["target"])[0]
        if len(outputs) > 2:
            targets = list(datum["target"])
            datum["output"] = np.mean(outputs)
            datum["target"] = np.mean(targets)
            for o in range(len(outputs)):
                datum[f"output_{o}"] = outputs[o]
                datum[f"target_{o}"] = targets[o]


class Experimenter_dim(Experimenter):
    def update_datum_for_item(self, datum, item):
        outs_dim = self.calc_item_output(item)
        outputs = outs_dim.view(-1).cpu().detach().numpy().tolist()
        targets = list(datum["target"])

        for o in range(len(outputs)):
            datum["output"] = outputs
            datum["target"] = targets
            for o in range(len(outputs)):
                datum[f"output_{o}"] = outputs[o]
                datum[f"target_{o}"] = targets[o]


class Experimenter_dim_c(Experimenter):
    def update_datum_for_item(self, datum, item):
        outs_dim = self.calc_item_output(item)
        outputs = outs_dim.squeeze(0).cpu().detach().numpy()
        targets = np.array(datum["target"])
        for o in range(5):
            datum[f"output_{o}"] = outputs[:, o]
            datum[f"target_{o}"] = targets[:, o]
