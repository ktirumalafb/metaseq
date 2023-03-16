# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import pandas as pd
import json

from metaseq.file_io import PathManager

from metaseq.data import data_utils
from . import BaseWrapperDataset

logger = logging.getLogger(__name__)

class FilterDataset(BaseWrapperDataset):
    """Filters dataset by excluding examples that are easy/hard to learn.
    
    Hard/easy to learn are defined by a metric between 0.0 and 1.0, with
    0.0 meaning easy to learn, and 1.0 meaning hard to learn.
    During initialization, this class expects:
    - `frac_data`: how much of the original dataset we want to keep 
    during training. It calculates the closest `frac_data` we can keep
    (keep the hard examples, throw away easy examples, 
    as per https://arxiv.org/abs/2206.14486.
    - `metric_file`: path to jsonl file, where each line should be
        {
            "name": dataset_name, 
            "index": index in dataset jsonl file, 
            "metric": metric value
        }
    where `metric` should be between 0.0 and 1.0 as described above
    """

    def __init__(self, dataset, frac_data, metric_data, dataset_name_to_index, random_include_examples_back=None):
        super().__init__(dataset)
        assert 0.0 <= frac_data <= 1.0

        if random_include_examples_back is not None:
            assert 0.0 <= random_include_examples_back <= 1.0
            
        self.frac_data = frac_data
        self.concat_dataset = dataset
        self.metric_data = metric_data

        # We only ever include stuff that is in metric_data. If our actual training set is a superset - we don't care.
        limit = int(np.ceil(len(self.metric_data) * self.frac_data))

        self.metric_data.sort_values('metric', inplace=True, ascending=False)

        if random_include_examples_back is not None:
            # randomly include `random_include_examples_back` of the other examples back
            logger.info(f"Randomly throwing back {random_include_examples_back} of examples")
            sampled_df = self.metric_data[limit:].sample(frac=random_include_examples_back)

        self.metric_data = self.metric_data[:limit]

        if random_include_examples_back is not None:
            # add the sampled df bac
            self.metric_data = pd.concat([self.metric_data, sampled_df])
    
        # If there are a subset of data points in the csv file, then just train on those data points
        # otherwise, take the limit defined by `frac_data`
        self.length = len(self.metric_data)

        self.dataset_name_to_index = dataset_name_to_index

    @staticmethod
    def retrieve_metric_df(metric_file, cur_shard_str, multidataset_prune_list=None):

        # Allow for templated metric file paths so that we don't load all data. For example:
        # --use-data-pruning-metrics-filepath "/checkpoint/danielsimig/c4_v2/pruning_metrics/avg_ppl_improvement_125m_350m_<SHARD>.jsonl"
        metric_file = metric_file.replace("<SHARD>", cur_shard_str)
        assert PathManager.isfile(metric_file), "Error! Provided `metric_file` is not a valid file"
        assert metric_file.endswith(".jsonl") or metric_file.endswith(".csv"), "Error! `metric_file` must be a `jsonl` file"

        # We use a single file to store metrics for all shards / datasets, but our
        # indexing logic depends on the index having data points only from the currently processed shard.
        logger.info(f"Will filter metric file for shard {cur_shard_str}")

        if metric_file.endswith(".jsonl"):
            lines = []
            scanned_lines = 0
            with open(metric_file, "r") as f:
                logger.info(f"Reading metric file: {metric_file}")
                for line in f:
                    json_parsed = json.loads(line[:-1])
                    if cur_shard_str in json_parsed["name"]:
                        lines.append(json_parsed)
                    scanned_lines += 1
                    if scanned_lines % 1_000_000 == 0:
                        logger.info(f"  Scanned {scanned_lines} lines")

            logger.info(f"Done processing metric file. {scanned_lines} lines scanned, {len(lines)} kept for this shard.")
            df = pd.DataFrame(lines)

        elif metric_file.endswith(".csv"):
            df = pd.read_csv(metric_file)
            original_length = len(df)
            df = df[df['name'].str.contains(cur_shard_str, regex=False)]  # Not tested
            logger.info(f"Filtering by shard str: {original_length} lines scanned, {len(df)} kept for this shard.")

        
        # Now if the multidataset option is turned on, modify the metric file so that only rows with names in multidataset_prune_list
        # are in the metric file. Otherwise, there will be rows in self.metric_data that are not in `dataset_name_to_index` and it will throw an error
        if multidataset_prune_list:
            multidataset_prune_list = multidataset_prune_list.strip().split(",")
            original_length = len(df)
            # df.apply(lambda x: x['name'].replace(f"{cur_shard_str}/", "") in multidataset_prune_list, axis=1, reduce=True)
            # df = df[df['name'].isin(multidataset_prune_list)]  # Not tested
            df = df[df.apply(lambda x: x['name'].replace(f"{cur_shard_str}/", "") in multidataset_prune_list, axis=1)]
            logger.info(f"Filtering by dataset names in multidataset_prune_list: {original_length} lines scanned, {len(df)} kept for this shard.")

        

        return df

    def __getitem__(self, index):
        assert 0 <= index < self.length

        metadata = self.metric_data.iloc[index]
        dataset_name = str(metadata["name"])
        sample_idx = int(metadata["index"])

        assert dataset_name in self.dataset_name_to_index, f"Error: dataset path {dataset_name} not in dataset_index. Keys: {list(self.dataset_name_to_index.keys())}"
        dataset_index = self.dataset_name_to_index[str(metadata["name"])]

        
        return self.concat_dataset.datasets[dataset_index][sample_idx]

    def __len__(self):
        return self.length