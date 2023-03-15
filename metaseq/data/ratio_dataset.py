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

class RatioDataset(BaseWrapperDataset):
    """
    This class takes in a `dataset` and a `ratio` and returns 
    """

    def __init__(self, dataset, ratio):
        super().__init__(dataset)
        self.dataset = dataset
        self.ratio = ratio
        original_length = len(self.dataset)
        self.length = int(original_length * ratio)

    def __getitem__(self, index):
        assert 0 <= index < self.length
        return self.dataset[index]

    def __len__(self):
        return self.length