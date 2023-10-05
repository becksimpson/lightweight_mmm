# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Media transformations for accounting for lagging or media effects."""

import functools
from typing import Union

import jax
import jax.numpy as jnp

MAX_DEGREES_SEASONALITY = 4
ADSTOCK_LIMIT = 0.97
AD_EFFECT_RETENTION_LIMIT = 0.999 #0.999


@functools.partial(jax.jit, static_argnums=[0, 1, 3])
def calculate_seasonality(
    number_periods: int,
    degrees: int,
    gamma_seasonality: Union[int, float, jnp.ndarray],
    frequency: int = 52,
) -> jnp.ndarray:
  """Calculates cyclic variation seasonality using Fourier terms.

  For detailed info check:
    https://en.wikipedia.org/wiki/Seasonality#Modeling

  Args:
    number_periods: Number of seasonal periods in the data. Eg. for 1 year of
      seasonal data it will be 52, for 3 years of the same kind 156.
    degrees: Number of degrees to use. Must be greater or equal than 1.
    gamma_seasonality: Factor to multiply to each degree calculation. Shape must
      be aligned with the number of degrees.
    frequency: Frequency of the seasonality being computed. By default is 52 for
      weekly data (52 weeks in a year).

  Returns:
    An array with the seasonality values.
  """

  seasonality_range = jnp.expand_dims(a=jnp.arange(number_periods), axis=-1)
  degrees_range = jnp.arange(1, degrees+1)
  inner_value = seasonality_range * 2 * jnp.pi * degrees_range / frequency
  season_matrix_sin = jnp.sin(inner_value)
  season_matrix_cos = jnp.cos(inner_value)
  season_matrix = jnp.concatenate([
      jnp.expand_dims(a=season_matrix_sin, axis=-1),
      jnp.expand_dims(a=season_matrix_cos, axis=-1)
  ],
                                  axis=-1)
  return (season_matrix * gamma_seasonality).sum(axis=2).sum(axis=1)


@jax.jit
def adstock(data: jnp.ndarray,
            lag_weight: float,
            adstock_normalise: bool = True) -> jnp.ndarray:
  """Calculates the adstock value of a given array.

  To learn more about advertising lag:
  https://en.wikipedia.org/wiki/Advertising_adstock

  Args:
    data: Input array.
    lag_weight: lag_weight effect of the adstock function. Default is 0.9.
    normalise: Whether to normalise the output value. This normalization will
      divide the output values by (1 / (1 - lag_weight)).

  Returns:
    The adstock output of the input array.
  """
  #lag_weight = jnp.clip(lag_weight, None, RETENTION_LIMIT)
  lag_weight = lag_weight * ADSTOCK_LIMIT

  def adstock_internal(prev_adstock: jnp.ndarray,
                       data: jnp.ndarray,
                       lag_weight: float = lag_weight) -> jnp.ndarray:
    adstock_value = prev_adstock * lag_weight + data
    return adstock_value, adstock_value# jax-ndarray

  _, adstock_values = jax.lax.scan(
      f=adstock_internal, init=data[0, ...], xs=data[1:, ...])
  adstock_values = jnp.concatenate([jnp.array([data[0, ...]]), adstock_values])
  
  norms =  (1. / (1 - lag_weight.reshape(1, -1))) / (1. / (1 - lag_weight.reshape(1, -1) ** (1 + jnp.arange(0, data.shape[0])).reshape(-1, 1)))

  return jax.lax.cond(
      adstock_normalise,
      lambda adstock_values: adstock_values / norms,
      #lambda adstock_values: adstock_values / (1. / (1 - lag_weight)),
      lambda adstock_values: adstock_values,
      operand=adstock_values)

@functools.partial(jax.jit, static_argnames=('logistic_normalise', ))
def logistic_saturation(
  data: jnp.ndarray,
  saturation: jnp.ndarray,
  logistic_normalise: bool = False,
) -> jnp.ndarray:
  """Calculates the logistic saturation function for a given array of values.

  Simpler (less parameters) than the hill function.

  Args:
    data: Input data.
    saturation: Controls the saturation, higher slope, stronger saturation
  """
  # TODO: Used with scaled beta
  saturation = 1.0 + saturation * 3.0
  log_data = (1.0 - jnp.exp(-saturation * data)) / (1.0 + jnp.exp(-saturation * data))
  one_norm = (1.0 - jnp.exp(-saturation)) / (1.0 + jnp.exp(-saturation))
  
  return jax.lax.cond(
    logistic_normalise,
    lambda log_data: log_data / one_norm,
    lambda log_data: log_data,
    operand=log_data
  )


@functools.partial(jax.jit, static_argnames=("hill_normalise",))
def hill_constrained(data: jnp.ndarray,
         half_max_effective_concentration_constrained: jnp.ndarray,
         slope_constrained: jnp.ndarray,
         hill_normalise: bool = False,
  ) -> jnp.ndarray:
  """Calculates the hill function for a given array of values.

  Refer to the following link for detailed information on this equation:
    https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)

  Args:
    data: Input data.
    half_max_effective_concentration: ec50 value for the hill function.
    slope: Slope of the hill function.

  Returns:
    The hill values for the respective input data.
  """
  # Range 0.3 --> 1.0
  half_max_effective_concentration = half_max_effective_concentration_constrained * 0.7 + 0.3 #
  # Range 0.5 --> 3.0
  slope = slope_constrained * 2.5 + 0.5

  save_transform = apply_exponent_safe(
      data=data / half_max_effective_concentration, exponent=-slope)
  hill_media = jnp.where(save_transform == 0, x=0, y=1. / (1 + save_transform))

  return jax.lax.cond(
    hill_normalise,
    lambda hill_media: hill_media * (2 * half_max_effective_concentration),
    lambda hill_media: hill_media,
    operand=hill_media
  )

#@jax.jit
@functools.partial(jax.jit, static_argnames=("hill_normalise",))
def hill(data: jnp.ndarray,
         half_max_effective_concentration: jnp.ndarray,
         slope: jnp.ndarray,
         hill_normalise: bool = False,
  ) -> jnp.ndarray:
  """Calculates the hill function for a given array of values.

  Refer to the following link for detailed information on this equation:
    https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)

  Args:
    data: Input data.
    half_max_effective_concentration: ec50 value for the hill function.
    slope: Slope of the hill function.

  Returns:
    The hill values for the respective input data.
  """
  save_transform = apply_exponent_safe(
      data=data / half_max_effective_concentration, exponent=-slope)
  hill_media = jnp.where(save_transform == 0, x=0, y=1. / (1 + save_transform))

  return jax.lax.cond(
    hill_normalise,
    lambda hill_media: hill_media * (2 * half_max_effective_concentration),
    lambda hill_media: hill_media,
    operand=hill_media
  )

@functools.partial(jax.vmap, in_axes=(1, 1, None), out_axes=1)#, None , None , None
def _carryover_convolve(data: jnp.ndarray,
                        weights: jnp.ndarray,
                        number_lags: int, 
                        ) -> jnp.ndarray:
  """Applies the convolution between the data and the weights for the carryover.

  Args:
    data: Input data.
    weights: Window weights for the carryover.
    number_lags: Number of lags the window has.

  Returns:
    The result values from convolving the data and the weights with padding.
  """
  window = jnp.concatenate([jnp.zeros(number_lags - 1), weights])
  return jnp.convolve(data, window, mode='same') / weights.sum()

#Staticed out as used in jnp.arange (fails with dynamic)
@functools.partial(jax.jit, static_argnames=("number_lags",))
def carryover(data: jnp.ndarray,
              ad_effect_retention_rate: jnp.ndarray,
              peak_effect_delay: jnp.ndarray,
              number_lags: int = 30,
              ) -> jnp.ndarray:
  """Calculates media carryover.

  More details about this function can be found in:
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46001.pdf

  Args:
    data: Input data. It is expected that data has either 2 dimensions for
      national models and 3 for geo models.
    ad_effect_retention_rate: Retention rate of the advertisement effect.
      Default is 0.5.
    peak_effect_delay: Delay of the peak effect in the carryover function.
      Default is 1.
    number_lags: Number of lags to include in the carryover calculation. Default
      is 13.

  Returns:
    The carryover values for the given data with the given parameters.
  """
  #number_lags = 60
  ad_effect_retention_rate = ad_effect_retention_rate * AD_EFFECT_RETENTION_LIMIT
  lags_arange = jnp.expand_dims(jnp.arange(number_lags, dtype=jnp.float32),
                                axis=-1)
  weights = ad_effect_retention_rate**((lags_arange - peak_effect_delay)**2)

  zeros = jnp.zeros((number_lags - 1, data.shape[1]))
  window = jnp.concatenate([zeros, weights])

  return jnp.concatenate([
      jax.numpy.convolve(
        data[:, i],
        window[:, i],
        mode='same',
      ).reshape(-1, 1)
    for i in range(data.shape[1])
  ], axis=1) / weights.sum(axis=0).reshape(1, -1)

# Faster, but jax.scipy.fftconvolve is not accessible in python 3.7
@functools.partial(jax.jit, static_argnames=('number_lags',))
def carryover_310(
  data: jnp.ndarray,
  ad_effect_retention_rate: jnp.ndarray,
  peak_effect_delay: jnp.ndarray,
  number_lags: int = 100) -> jnp.ndarray:

  #ad_effect_retention_rate = jnp.clip(ad_effect_retention_rate, None, RETENTION_LIMIT)
  ad_effect_retention_rate = ad_effect_retention_rate * AD_EFFECT_RETENTION_LIMIT

  lags_arange = jnp.expand_dims(jnp.arange(number_lags, dtype=jnp.float32),
                              axis=-1)
  weights = ad_effect_retention_rate**((lags_arange - peak_effect_delay)**2)
  #weights = weights / weights.sum(axis=0)
  window = jnp.concatenate([jnp.zeros((number_lags - 1, data.shape[1])), weights])
  # Create accurate normaliser
  # ws = jax.scipy.signal.fftconvolve(
  #   weights,
  #   jnp.ones(data.shape),
  #   axes=0
  # )[:data.shape[0]]
  return jax.scipy.signal.fftconvolve(
      data,
      window,
      mode='same',
      axes=0
  ).clip(min=0.0) / weights.sum(axis=0).reshape(1, -1) #ws

@jax.jit
def apply_exponent_safe(
    data: jnp.ndarray,
    exponent: jnp.ndarray,
    ) -> jnp.ndarray:
  """Applies an exponent to given data in a gradient safe way.

  More info on the double jnp.where can be found:
  https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf

  Args:
    data: Input data to use.
    exponent: Exponent required for the operations.

  Returns:
    The result of the exponent operation with the inputs provided.
  """
  exponent_safe = jnp.where(condition=(data == 0), x=1, y=data) ** exponent
  return jnp.where(condition=(data == 0), x=0, y=exponent_safe)
