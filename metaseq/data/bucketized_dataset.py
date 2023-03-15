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

class BucketizedDataset(BaseWrapperDataset):
    """
    This just returns a dataset that returns values within a specified index range of the source dataset
    """

    def __init__(self, dataset, bottom_limit, top_limit):
        super().__init__(dataset)
            
        self.dataset = dataset
        self.top_limit = top_limit
        self.bottom_limit = bottom_limit

        self.length = self.top_limit - self.bottom_limit

    def __getitem__(self, index):
        assert 0 <= index < self.length
        return self.dataset[bottom_limit + index]

    def __len__(self):
        return self.length