"""Module containing spend models, that map spend --> imp

This model can be configured to allow spend in some channels to influence
impressions in others. E.g. upper funnel --> lower funnel.
"""
from typing import Dict

from jax import numpy as jnp
import numpyro

from lightweight_mmm import media_transforms

def spend_model(
    spend_data: jnp.ndarray,
    media_data: jnp.ndarray,
    mappings: Dict,
):
    