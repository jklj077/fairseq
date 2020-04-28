# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import time
from collections import OrderedDict
from typing import Dict, Optional

try:
    import torch

    def type_as(a, b):
        if torch.is_tensor(a) and torch.is_tensor(b):
            return a.to(b)
        else:
            return a


except ImportError:
    torch = None

    def type_as(a, b):
        return a


try:
    import numpy as np
except ImportError:
    np = None


class Meter(object):
    """Base class for Meters."""

    def __init__(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def reset(self):
        raise NotImplementedError

    @property
    def smoothed_value(self) -> float:
        """Smoothed value used for logging."""
        raise NotImplementedError


def safe_round(number, ndigits):
    if hasattr(number, "__round__"):
        return round(number, ndigits)
    elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
        return safe_round(number.item(), ndigits)
    elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
        return safe_round(number.item(), ndigits)
    else:
        return numbe


class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def __init__(self, round: Optional[int] = None):
        self.round = round
        self.reset()

    def reset(self):
        self.val = None  # most recent update
        self.sum = 0  # sum from all updates
        self.count = 0  # total n from all updates

    def update(self, val, n=1):
        if val is not None:
            self.val = val
            if n > 0:
                self.sum = type_as(self.sum, val) + (val * n)
                self.count = type_as(self.count, n) + n

    def state_dict(self):
        return {
            "val": self.val,
            "sum": self.sum,
            "count": self.count,
            "round": self.round,
        }

    def load_state_dict(self, state_dict):
        self.val = state_dict["val"]
        self.sum = state_dict["sum"]
        self.count = state_dict["count"]
        self.round = state_dict.get("round", None)

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else self.val

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val


class TimeMeter(Meter):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init: int = 0, n: int = 0, round: Optional[int] = None):
        self.round = round
        self.reset(init, n)

    def reset(self, init=0, n=0):
        self.init = init
        self.start = time.perf_counter()
        self.n = n
        self.i = 0

    def update(self, val=1):
        self.n = type_as(self.n, val) + val
        self.i += 1

    def state_dict(self):
        return {"init": self.elapsed_time, "n": self.n, "round": self.round}

    def load_state_dict(self, state_dict):
        if "start" in state_dict:
            # backwards compatibility for old state_dicts
            self.reset(init=state_dict["init"])
        else:
            self.reset(init=state_dict["init"], n=state_dict["n"])
            self.round = state_dict.get("round", None)

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.perf_counter() - self.start)

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val


class StopwatchMeter(Meter):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self, round: Optional[int] = None):
        self.round = round
        self.sum = 0
        self.n = 0
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.perf_counter() - self.start_time
            self.sum = self.sum + delta
            self.n = type_as(self.n, n) + n

    def reset(self):
        self.sum = 0  # cumulative time during which stopwatch was active
        self.n = 0  # total n across all start/stop
        self.start()

    def state_dict(self):
        return {"sum": self.sum, "n": self.n, "round": self.round}

    def load_state_dict(self, state_dict):
        self.sum = state_dict["sum"]
        self.n = state_dict["n"]
        self.start_time = None
        self.round = state_dict.get("round", None)

    @property
    def avg(self):
        return self.sum / self.n if self.n > 0 else self.sum

    @property
    def elapsed_time(self):
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    @property
    def smoothed_value(self) -> float:
        val = self.avg if self.sum > 0 else self.elapsed_time
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val
