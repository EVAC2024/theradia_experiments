from lib import *


"""
Feature Manager Module

This module combines functionalities from two previously separate feature manager files, ensuring a unified
and efficient process for extracting and managing various types of features from audio, video, and textual data.
The module supports continuous dimensions regression and handles various feature extraction methods such as 
Mel-Filter Bank (MFB), BERT, TF-IDF, and CSV features.

Classes:
    - Feature_extracter: A class for extracting audio, video, and textual features, with support for continuous 
      dimensions and saving/loading features to avoid re-processing.

Main Features:
    - Initialization and management of different feature extractors.
    - Extraction of features from audio, video, and text data.
    - Handling and processing of continuous and non-continuous features.
    - Resampling, interpolation, and padding of features for consistency.

"""


class Feature_extracter(nn.Module):
    """A class for extracting audio, video, and textual features.
    Also, for saving and loading the features to avoid re-processing when possible.
    
    Attributes
    ----------
    feat_title : str
        The title of the feature to be extracted, e.g. "mfb" for Mel-filter bank on audio, or "roberta-large" for the large RoBERTa model on text.
    save_dir : str
        The path referring to the folder containing video files of theradia data.
    feat_func: function
        The function that will be applied on top of feature extraction. For example, to turn torch tensors to numpy arrays: `feat_func = lambda x: x.numpy()`.
    **feat_params: dict
        The keyword arguments related to the feature extractor that is to be used. e.g. for "mfb" features, one can pass the argument `n_mels=80`.
        
    Methods
    -------
    init_feat()
        Initialises the feature extractor.
    get_MFB()
        Extracts Mel-Filter Bank (MFB) features. Activated when feat_title="mfb".
    extract(x)
        Extracts features based on the given input x.
        
    """
    
    def __init__(self,
                 save_dir,
                 key_file,
                 feat_title,
                 continuous=False,
                 feat_func=lambda a: a,
                 **feat_params):
        super().__init__()
        self.save_dir = save_dir
        self.key_file = key_file
        self.feat_title = feat_title
        self.feat_params = feat_params
        self.feat_func = feat_func
        self.continuous = continuous
        self.fau = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r',
                    'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

        if self.continuous:
            self.feats_dir = os.path.join(self.save_dir, 'continuous', self.feat_title)
            if not os.path.exists(self.feats_dir): os.makedirs(self.feats_dir)
        else:
            self.feats_dir = os.path.join(self.save_dir, self.feat_title)
            if not os.path.exists(self.feats_dir): os.makedirs(self.feats_dir)

        self.init_feat()

    def init_feat(self):
        '''Initialises the feature extractor.
        '''
        if "mfb" in self.feat_title:
            self.feat_extractor = self.get_MFB()
        elif self.feat_title == "fau":
            pass
        elif "tfidf" in self.feat_title:
            self.feat_extractor = self.get_tfidf()
        elif "w2v2" in self.feat_title.lower():
            pass
        elif "bert" in self.feat_title.lower():
            self.feat_extractor = self.get_BERT()
        elif "csv" in self.feat_title:
            pass
        else:
            print("Warning: the feature_title is not implemented.")

    def chunks(self, l, n):
        for i in range(0, n):
            yield l[i::n]

    def resample(self, arr, newLength):
        return np.array([np.mean(chunk) for chunk in self.chunks(arr, newLength)])

    def resample2d(self, arr, newLength):
        return np.array([np.mean(chunk, axis=0) for chunk in self.chunks(arr, newLength)])

    def interp_feat(self, x, new_length):
        X = np.zeros((new_length, x.shape[1]))
        for col in range(x.shape[1]):
            a = x[:, col]
            if len(a) == 1:
                X[:, col] = a * new_length
            else:
                old_indices = np.arange(0, len(a))
                new_indices = np.linspace(0, len(a) - 1, new_length)
                spl = UnivariateSpline(old_indices, a, k=1, s=0)
                new_col = spl(new_indices)
                X[:, col] = new_col
        return X        

    def get_tfidf(self):
        '''Extractor function for extracting Term Frequency - Inverse Document Frequency (TF-IDF) features. Activated when feat_title="tfidf".
        '''
        random.seed(self.feat_params["seed"])
        np.random.seed(self.feat_params["seed"])
        self.tfidf_vectorizer = TfidfVectorizer()
        _ = self.tfidf_vectorizer.fit_transform(self.feat_params["corpus"])
        def extractor(x):
            return self.tfidf_vectorizer.transform(x).toarray()
        return extractor
    
    def get_MFB(self):
        '''Extractor function for extracting Mel-Filter Bank (MFB) features. Activated when feat_title="mfb".
        '''
        MFB = Fbank(n_mels=self.feat_params["n_mels"])
        return MFB
    
    def get_w2v2(self):
        from huggingface_wav2vec import HuggingFaceWav2Vec2
        save_path = os.path.join(self.save_dir, "HuggingFace", self.feat_params["source"])
        W2V2 = HuggingFaceWav2Vec2(source=self.feat_params["source"], 
                                   save_path=save_path, 
                                   output_norm=False, 
                                   freeze=self.feat_params["freeze"], 
                                   freeze_feature_extractor=self.feat_params["freeze"])
        return W2V2.to(self.feat_params["device"])
    
    def get_BERT(self):
        from huggingface_LMs import HuggingFaceBERT
        save_path = os.path.join(self.save_dir, "HuggingFace", self.feat_params["source"])
        bert = HuggingFaceBERT(source=self.feat_params["source"], save_path=save_path, output_norm=False, freeze=self.feat_params["freeze"])
        return bert
    
    def extract(self, x):
        '''Extract feature based on given input
        This is the most basic method for extracting features that is used across all other related methods in this class
        '''
        feats = self.feat_extractor(x)
        feats = self.feat_func(feats)
        return feats
    
    def extract_audio_feats(self, ID_list=[], wav_paths=[], override=False):
        '''Extracts audio featues from a list of IDs and wav_paths
        Should pass feat_func = lambda x: x.squeeze(0).numpy() in __init__
        '''
        for i, (ID, wav_path) in enumerate(zip(ID_list, wav_paths)):
            print_progress_bar(i + 1, len(ID_list), prefix=f'Extracting {self.feat_title} features:', suffix='completed', length=50)
            save_path = os.path.join(self.feats_dir, f"{ID}.csv")
            if os.path.exists(save_path) and (not override): continue
            waves = sb.dataio.dataio.read_audio(wav_path)
            feats = self.extract(waves.unsqueeze(0))
            df = pd.DataFrame(feats)
            df.to_csv(save_path, index=None)
            
    def get_audio_feat(self, ID="", wav_path="", n="", override=False, save_feats=True):
        '''Extracts audio featue based on an ID and a wav_path.
        The ID helps to store the feature and thus avoid re-calculating.
        '''
        save_path = os.path.join(self.feats_dir, f"{ID}.csv")

        if self.continuous == False:
            if "w2v2" in self.feat_title.lower():
                if not self.feat_params["freeze"]: override=True
            if os.path.exists(save_path) and (not override): 
                feats = pd.read_csv(save_path, index_col=None).to_numpy()
                feats = torch.tensor(feats)
                if "w2v2" in self.feat_title.lower(): 
                    feats = feats.to(self.feat_params["device"])
            else:
                waves = sb.dataio.dataio.read_audio(wav_path).unsqueeze(0)
                if waves.shape[1] == 0:
                    wav_path = wav_path.replace("closetalk", "farfield")
                    print('replaced ID', ID)
                    waves = sb.dataio.dataio.read_audio(wav_path).unsqueeze(0)
                if "w2v2" in self.feat_title.lower():
                    waves = waves.to(self.feat_params["device"])
                feats = self.extract(waves)
                if save_feats:
                    df = pd.DataFrame(feats.cpu().detach())
                    df.to_csv(save_path, index=None)
            return feats

        else:
            save_path = os.path.join(self.feats_dir, f"{ID}.csv")
            new_save_path = save_path.replace("/continuous", "")
            
            if os.path.exists(save_path):
                df_data = pd.read_csv(save_path, sep=',')
                X = df_data.iloc[:, 1:].values
            else:
                df_data = pd.read_csv(new_save_path, sep=',')
                x = df_data.values
                X = self.resample2d(x, n)
                X[np.isnan(X)] = 0.
                pd.DataFrame(X).to_csv(save_path, sep=',')
            return torch.tensor(X)
    
    def get_bert_feat(self, ID_list=[], trs_list=[], n_list=[], override=False, save_feats=True):
        '''Extracts bert featue based on an ID and transcription list.
        The ID helps to store the feature and thus avoid re-calculating.
        '''
        if self.continuous == False:
            all_feats = []
            if not self.feat_params["freeze"]: override=True
            paths_exist = True
            for ID in ID_list:
                save_path = os.path.join(self.feats_dir, f"{ID}.csv")
                if not os.path.exists(save_path): paths_exist = False
            if paths_exist and (not override):
                for ID in ID_list:
                    save_path = os.path.join(self.feats_dir, f"{ID}.csv")
                    feats = pd.read_csv(save_path, index_col=None).to_numpy()
                    feats = torch.tensor(feats)
                    feats = feats.to(self.feat_params["device"])
                    all_feats.append(feats)
                feats = pad_sequence(all_feats, batch_first=True, padding_value=0.0)
            else:
                feats = self.extract(trs_list)
                if save_feats:
                    for i, ID in enumerate(ID_list):
                        save_path = os.path.join(self.feats_dir, f"{ID}.csv")
                        df = pd.DataFrame(feats.cpu().detach()[i])
                        df = df[(df.T != 0).any()]  # remove padded zeros
                        df.to_csv(save_path, index=None)
            return feats
        else:
            all_feats = []
            for ID, n in zip(ID_list, n_list):
                save_path = os.path.join(self.feats_dir, f"{ID}.csv")
                new_save_path = save_path.replace("/continuous", "")
                if os.path.exists(save_path):
                    df_data = pd.read_csv(save_path, sep=',')
                    X = df_data.iloc[:, 1:].values
                    feats = torch.tensor(X)
                    all_feats.append(feats)
                else:
                    df_data = pd.read_csv(os.path.join(new_save_path), sep=',')
                    x = df_data.values
                    X = self.interp_feat(x, n)
                    df_txt = pd.DataFrame(X)
                    df_txt.to_csv(os.path.join(save_path), sep=',')
                    feats = torch.tensor(X)
                    all_feats.append(feats)
            feats = pad_sequence(all_feats, batch_first=True, padding_value=0.0)
            return feats

    def get_fau_feat(self, ID="", times=[], video_path="", override=False, save_feats=True):
        '''Get a fau feature 
        '''
        df = pd.read_csv(self.key_file)
        newID = df[df['new_key'] == ID]['key'].values[0]
        save_path = os.path.join(self.feat_params["fau_dir"], f"{newID}.csv")

        if os.path.exists(save_path) and (not override): 
            df = pd.read_csv(save_path, index_col=None)
            if self.continuous == False:
                feats = df.to_numpy()
            else:
                idxs = times
                df['timestamp'] = df['timestamp'].apply(lambda x: round(float(x), 2))
                df['timestamp'] = df['timestamp'].apply(lambda x: round(float(x), 1))
                df_filtered = df[pd.Series(list(df.timestamp), index=df.index).isin(idxs)]
                df_filtered2 = df_filtered.drop_duplicates(subset='timestamp', keep="last")
                df_data = df_filtered2.drop(['frame', 'face_id', 'timestamp', 'confidence', 'success'], axis=1)
                df_data = df_data[self.fau].values
                if len(idxs) != df_data.shape[0]:
                    padding_length = len(idxs) - df_data.shape[0]
                    padding = np.zeros((padding_length, df_data.shape[1]))
                    df_data = np.vstack([df_data, padding])
                feats = df_data
        else:
            print("FAU extraction is not implemented.")
        return feats

    def get_csv_feat(self, IDs=[], n_list2=[], override=False, save_feats=True):
        '''Get features from a csv file
        '''
        feats_all = []
        n_list_file = "len_df.csv"
        df = pd.read_csv(self.key_file)

        if os.path.isfile(n_list_file):
            n_list_df = pd.read_csv(n_list_file)
            n_list = dict(zip(n_list_df['ID'], n_list_df['Length']))
        else:
            raise FileNotFoundError(f"{n_list_file} not found. Please ensure the file exists.")

        for ID in IDs:
            newID = df[df['new_key'] == ID]['key'].values[0]
            csv_dir = os.path.join(self.feat_params["csv_dir"], "**", f"{newID}.csv")
            csv_path = glob.glob(csv_dir, recursive=True)[0]
            feats = pd.read_csv(csv_path, sep=',').iloc[:, 2:].values
            n = n_list.get(ID)
            if n is None:
                n = n_list2[0].get(ID)

            if len(feats) > n:
                feats = feats[:n]  # Truncate features to length n
            elif len(feats) < n:
                pad_width = ((0, n - len(feats)), (0, 0))  # Padding only for the sequence length
                feats = np.pad(feats, pad_width, mode='constant', constant_values=0)
            
            feats = self.feat_func(feats)
            feats = torch.tensor(feats)
            feats_all.append(feats)
        
        feats_all = pad_sequence(feats_all, batch_first=True, padding_value=0.0)
        return feats_all

    def get_feat_size(self):
        '''Get the size of the feature 
        '''
        if "mfb" in self.feat_title:
            return self.feat_params["n_mels"]
        elif self.feat_title == "fau":
            return 17
        elif "tfidf" in self.feat_title:
            feats = self.extract([""])[0]
            return len(feats)
        elif "w2v2" in self.feat_title.lower():
            return 1024
        elif "bert" in self.feat_title.lower():
            feats = self.extract([""])
            return feats.size()[-1]
        elif "csv" in self.feat_title:
            return self.feat_params["feats_size"]
        
    def forward(self, **params_dict):
        '''Get a feature based on a params_dict, which contains necessary info for extracting a feature based on feature_title
        '''
        if "mfb" in self.feat_title or "w2v2" in self.feat_title.lower():
            feats = []
            for ID, wav_path, seq_len in zip(params_dict["ID"], params_dict["wav_path"], params_dict["target_len"]):
                feat = self.get_audio_feat(ID, wav_path, seq_len)
                feats.append(feat)
            feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        elif self.feat_title == "fau":
            feats = []
            for ID, times in zip(params_dict["ID"], params_dict["times"]):
                feat = self.get_fau_feat(ID, times)
                feat = torch.tensor(feat)
                feats.append(feat)
            feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        elif "tfidf" in self.feat_title:
            feats = self.extract(params_dict["trs"])
            feats = torch.tensor(feats).unsqueeze(1)  # tfidf does not have sequence
        elif "bert" in self.feat_title.lower():
            feats = self.get_bert_feat(params_dict["ID"], params_dict["trs"], params_dict["target_len"])
        elif "csv" in self.feat_title.lower():
            feats = self.get_csv_feat(params_dict["ID"], params_dict["len_lst"])
        return feats
