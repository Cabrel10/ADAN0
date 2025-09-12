#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Handles data access for the trading environment."""
import pandas as pd
from ..common.utils import get_logger

logger = get_logger()

class DataHandler:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.current_step = 0

    def reset(self):
        """Resets the data handler to the beginning of the dataset."""
        self.current_step = 0

    def get_current_observation(self) -> pd.Series:
        """Returns the observation for the current step."""
        return self.data.iloc[self.current_step]

    def get_current_prices(self, assets: list[str]) -> dict[str, float]:
        """Returns the current prices for all assets."""
        prices = {}
        for asset in assets:
            close_col = f"{asset}_close_1m" # Assuming 1m close is the reference price
            if close_col in self.data.columns:
                prices[asset] = self.data.iloc[self.current_step][close_col]
        return prices

    def step(self) -> bool:
        """Moves to the next step in the data."""
        self.current_step += 1
        return self.current_step < len(self.data)
