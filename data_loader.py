import os
import sys
import random
import pickle
import json
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import speech_recognition 
from tqdm import tqdm

"""
Theradia Data Loader Module

This module provides the `TheradiaDataLoader` class for loading, preprocessing, and managing the Theradia dataset. 
The dataset includes various data types such as audio, video, and annotations, and this class facilitates their 
efficient loading and preprocessing.

Classes:
    - TheradiaDataLoader: A class for loading and preprocessing Theradia data, saving/loading them in JSON format, 
      and handling various operations related to data indexing, annotation filtering, and transcription processing.

Main Features:
    - Initialization with paths to partitions CSV, public and private datasets, and save directory.
    - Indexing of data paths for video, annotations, and transcriptions.
    - Loading and saving data in JSON format for easy access and manipulation.
    - Partitioning of data into training, development, and test sets.
    - Filtering of annotations to include only specified users.
    - Automatic Speech Recognition (ASR) transcription using Google ASR.
"""


class TheradiaDataLoader:
    """
    This class can load and preprocess Theradia data, and save/load them in JSON format.
    """
    def __init__(self, partitions_csv_path, public_dataset_base, private_dataset_base, save_dir, audio_source, json_name="data.json"):
        """
        Initializes the data loader with paths to the partitions CSV, the public and private datasets,
        and the directory to save processed data.
        """
        self.partitions_csv_path = partitions_csv_path
        self.public_dataset_base = public_dataset_base
        self.private_dataset_base = private_dataset_base
        self.save_dir = save_dir
        self.audio_source = audio_source
        self.json_path = os.path.join(save_dir, json_name)
        self.data = {}
        self.partition_info = self.load_partition_info()

        self.dims_list = ["labels", "dimension_summaries", "dimension_arousal", "dimension_novelty", 
                          "dimension_goal_conduciveness", "dimension_intrinsic_pleasantness", "dimension_coping"]

        self.labels_list = ['annoyed', 'anxious', 'confident', 'desperate', 'frustrated', 'happy', 'interested',
                            'relaxed', 'satisfied', 'surprised']

        self.users_list = ['user_17', 'user_18', 'user_19', 'user_20', 'user_21', 'user_22']

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def load_partition_info(self):
        """
        Loads the partition information from a CSV file.
        """
        return pd.read_csv(self.partitions_csv_path)

    def index_data(self):
        """
        Constructs the paths for video, annotations, and transcriptions based on IDs and partitions.
        """
        for index, row in self.partition_info.iterrows():
            ID, partition = row['key'], row['partition']
            subject_id, session_id = ID.split('_')[0], ID.split('_')[1]
            base_path = self.determine_base_path(partition)
            subject_folder = f"{subject_id[0]}-subjects/{subject_id}"
            subject_folder2 = f"processed-audio-{subject_id[0]}-subjects/{subject_id}" ## path to resampled audios (44->16)

            session_folder = f"{subject_id}_seance_{session_id}"
            subject_path = os.path.join(base_path, subject_folder)
            subject_path2 = os.path.join(base_path, subject_folder2)
 
            correct_seance_folder = self.find_correct_seance_folder(subject_path, session_id)
            full_path = os.path.join(subject_path, correct_seance_folder)
            full_path2 = os.path.join(subject_path2, correct_seance_folder)
            
            video_path = os.path.join(full_path, 'Video', 'video_segments_farfield')
            annots_path = os.path.join(full_path, 'Annotations')
            transcripts_logs_path = os.path.join(full_path, 'Transcripts and Logs')
            trs_path = glob.glob(os.path.join(transcripts_logs_path, '*audio_farfield_trs.csv'))

            trs_df = pd.read_csv(trs_path[0])
            trs = trs_df[trs_df['segment_id'] == ID]['content'].values[0]

            if self.audio_source == 'closetalk':
                wav_path = os.path.join(full_path2, 'Audio', 'audio_segments_closetalk')
            elif self.audio_source == 'farfield':
                wav_path = os.path.join(full_path2, 'Audio', 'audio_segments_farfield')
            else:
                print('Invalid audio source:', self.audio_source)

            datum = {
                "ID": ID,
                "video_path": os.path.join(video_path, f"{ID}.mp4"),
                "wav_path": os.path.join(wav_path, f"{ID}.wav"),
                "annots_path": {
                    "labels": os.path.join(annots_path, f"{ID}_labels.csv"),
                    "dimension_arousal": os.path.join(annots_path, f"{ID}_dimension_arousal.csv"),
                    "dimension_novelty": os.path.join(annots_path, f"{ID}_dimension_novelty.csv"),
                    "dimension_goal_conduciveness": os.path.join(annots_path, f"{ID}_dimension_goal_conduciveness.csv"),
                    "dimension_intrinsic_pleasantness": os.path.join(annots_path, f"{ID}_dimension_intrinsic_pleasantness.csv"),
                    "dimension_coping": os.path.join(annots_path, f"{ID}_dimension_coping.csv"),
                    "dimension_summaries": os.path.join(annots_path, f"{ID}_dimension_summaries.csv"),
                },
                "trs_path": trs_path,
                "trs": trs
            }

            self.data[ID] = datum

    def determine_base_path(self, partition):
        """
        Determines the base path for data based on partition.
        """
        if partition in ['train', 'dev']:
            return self.private_dataset_base

        elif partition in ['test']:
            return self.private_dataset_base

    def find_correct_seance_folder(self, subject_path, session_id):
        """
        Finds the correct seance folder based on the session_id.
        """
        try:
            seance_subfolders = os.listdir(subject_path)
            for folder in seance_subfolders:
                f = folder.split('_')[1]
                if session_id in f:
                    return folder
        except FileNotFoundError:
            print(f"Subject path not found: {subject_path}")
        return None

    def save_to_json(self):
        '''Saves the data in json format to the json_path.
        '''
        with open(self.json_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.data, json_file)

    def load_from_json(self):
        '''Loads the data from the json_path to python dict.
        '''
        with open(self.json_path, 'r', encoding='utf-8') as json_file:
            self.data = json.load(json_file)

    def load_partitions(self):
        """
        Organize the data into train, dev, test dicts based on the partition information
        in self.partition_info.
        Returns a dictionary with partition names as keys and dicts of corresponding data as values.
        """
        partitioned_data = {'train': {}, 'dev': {}, 'test': {}}

        # Iterate over each row in the partition_info DataFrame
        for _, row in self.partition_info.iterrows():
            key, part = row['key'], row['partition']

            if key in self.data:
                partitioned_data[part][key] = self.data[key]
            else:
                print(f"Key {key} not found in loaded data.")

        return partitioned_data

    def add_annotations_for_users(self):
        '''Only keeping annotations of the annotators that are within the list of users
        '''
        for i, (k, datum) in enumerate(self.data.items()):
            datum.setdefault('annots', {})

            for annot_type, annot_path in datum["annots_path"].items():
                filtered_df = None
                if annot_path.endswith("labels.csv"):
                    filtered_df = self.filter_labels(annot_path)
                elif annot_path.endswith("dimension_summaries.csv"):
                    filtered_df = self.filter_dimensions(annot_path)
                elif any(annot_path.endswith(f"dimension_{dim}.csv") for dim in ["arousal", "novelty", "goal_conduciveness", "intrinsic_pleasantness", "coping"]):
                    filtered_df = self.filter_dimensions_c(annot_path)
                else:
                    print(f"Unknown annotation type for {annot_path}")
                    continue

                if filtered_df is not None:
                    datum["annots"][annot_type] = filtered_df.to_dict(orient='records')
                else:
                    print(f"Filtering resulted in None for {annot_path}")

    def filter_labels(self, file_path):
        '''Filter label annotations to include only the specified users and retain the emotion names.
        '''
        df = pd.read_csv(file_path)
        valid_columns = ["categories_names"] + self.users_list
        return df[valid_columns].dropna()

    def filter_dimensions(self, file_path):
        '''Filter dimension annotations to include only the specified users and retain the dimension names.
        '''
        df = pd.read_csv(file_path)
        valid_columns = ["dimensions"] + self.users_list
        return df[valid_columns].dropna()

    def filter_dimensions_c(self, file_path):
        '''Filter dimension annotations to include only the specified users and retain the dimension names.
           Skips files that are not found.
        '''
        try:
            df = pd.read_csv(file_path)
            base_column = 'times' if 'times' in df.columns else 'dimensions'
            valid_columns = [base_column] + [col for col in df.columns if col.startswith('user_') and col in self.users_list]
            return df[valid_columns].dropna()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None

    def get_ASR_trans_google(self, audio_path, lang="fr-FR"):
        r = speech_recognition.Recognizer()
        transcription = ""
        with speech_recognition.AudioFile(audio_path) as source:
            audio = r.record(source)
            try:
                transcription = r.recognize_google(audio, language=lang)  # "en-US" "fr-FR"
            except speech_recognition.UnknownValueError:
                transcription = ""
            except speech_recognition.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
        return transcription

    def add_google_ASR_transcriptions(self):
        """
        Adds Google ASR transcriptions to the data dictionary for audio files, with a progress bar.
        """
        for item_id in tqdm(self.data, desc="Processing transcriptions", unit="transcription"):
            item_data = self.data[item_id]
            audio_path = item_data.get('wav_path')
            if audio_path and os.path.exists(audio_path):
                item_data['trs_google'] = self.get_ASR_trans_google(audio_path)
            else:
                item_data['trs_google'] = "Audio file not found."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]
