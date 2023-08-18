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
RETENTION_LIMIT = 0.99


#@functools.partial(jax.jit, static_argnums=[0, 1])
def calculate_seasonality_ensemble(
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
  degrees_range = jnp.arange(1, degrees + 1) # MAX_DEGREES_SEASONALITY
  inner_value = seasonality_range * 2 * jnp.pi * degrees_range / frequency
  season_matrix_sin = jnp.sin(inner_value)
  season_matrix_cos = jnp.cos(inner_value)
  season_matrix = jnp.concatenate([
      jnp.expand_dims(a=season_matrix_sin, axis=-1),
      jnp.expand_dims(a=season_matrix_cos, axis=-1)
  ],
                                  axis=-1)
  
  # gamma_seasonality = jax.lax.dynamic_slice(
  #   gamma_seasonality,
  #   (0, 0), (2, degrees)
  # )
  # gamma_seasonality = jax.lax.dynamic_slice(
  #   gamma_seasonality,
  #   (0, 0), (1, degrees)
  # )
  waves = (season_matrix * gamma_seasonality).sum(axis=2)

  # return jnp.where(
  #   jnp.repeat(
  #     (jnp.arange(waves.shape[1]) < degrees).reshape(-1, 1),
  #     number_periods,
  #     axis=1
  #   ),
  #   waves.T,
  #   jnp.zeros((MAX_DEGREES_SEASONALITY, number_periods))
  # ).T.sum(axis=1)

  return waves.sum(axis=1)

  return jax.lax.dynamic_slice(
    waves,
    (0, 0), (waves.shape[0], degrees)
  ).sum(axis=1)
  
  # return jnp.where(
  #   jnp.arange(waves.shape[1]) <= degrees,
  #   arr,
  #   0
  # ).sum(axis=1)
  # .sum(axis=1)


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


#@functools.partial(jax.jit, static_argnames=("adstock_normalise",))
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
  lag_weight = lag_weight * RETENTION_LIMIT

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
  log_data = (1.0 - jnp.exp(-saturation * data)) / (1.0 + jnp.exp(-saturation * data))
  one_norm = (1.0 - jnp.exp(-saturation)) / (1.0 + jnp.exp(-saturation))
  
  return jax.lax.cond(
    logistic_normalise,
    lambda log_data: log_data / one_norm,
    lambda log_data: log_data,
    operand=log_data
  )
  #return d /  ((1 - jnp.exp(-saturation )) / (1 + jnp.exp(-saturation)))#/ d.sum(axis=0) * data.sum(axis=0)


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
  slope = slope_constrained * 2.7 + 0.3

  # slope = jnp.clip(slope, a_min=0.5, a_max=3.0)
  # half_max_effective_concentration = jnp.clip(half_max_effective_concentration, a_min=0.3, a_max=1.0)
  #hill_normalise = True
  save_transform = apply_exponent_safe(
      data=data / half_max_effective_concentration, exponent=-slope)
  hill_media = jnp.where(save_transform == 0, x=0, y=1. / (1 + save_transform))

  #return hill_media #* (2 * half_max_effective_concentration)
  # Normalisation keeps linear scaling at half_max_effective_concentration point
  # if hill_normalise:
  #   return hill_media * (2 * half_max_effective_concentration)
  # else:
  #   return hill_media
  #return hill_media
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
  # slope = jnp.clip(slope, a_min=0.5, a_max=3.0)
  # half_max_effective_concentration = jnp.clip(half_max_effective_concentration, a_min=0.3, a_max=1.0)
  #hill_normalise = True
  save_transform = apply_exponent_safe(
      data=data / half_max_effective_concentration, exponent=-slope)
  hill_media = jnp.where(save_transform == 0, x=0, y=1. / (1 + save_transform))

  #return hill_media #* (2 * half_max_effective_concentration)
  # Normalisation keeps linear scaling at half_max_effective_concentration point
  # if hill_normalise:
  #   return hill_media * (2 * half_max_effective_concentration)
  # else:
  #   return hill_media
  #return hill_media
  return jax.lax.cond(
    hill_normalise,
    lambda hill_media: hill_media * (2 * half_max_effective_concentration),
    lambda hill_media: hill_media,
    operand=hill_media
  )

#@functools.partial(jax.jit, static_argnums=(2,))
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
  #number_lags = 60
  window = jnp.concatenate([jnp.zeros(number_lags - 1), weights])
  #return jax.scipy.signal.convolve(data, window, mode="same") / weights.sum()
  return jnp.convolve(data, window, mode='same') / weights.sum()
  #return jnp.convolve(data, window, mode='same') / weights.sum()


# @functools.partial(jax.jit, static_argnames=("number_lags",))
# def carryover(data: jnp.ndarray,
#               ad_effect_retention_rate: jnp.ndarray,
#               peak_effect_delay: jnp.ndarray,
#               number_lags: int = 30) -> jnp.ndarray:
#   """Calculates media carryover.

#   More details about this function can be found in:
#   https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46001.pdf

#   Args:
#     data: Input data. It is expected that data has either 2 dimensions for
#       national models and 3 for geo models.
#     ad_effect_retention_rate: Retention rate of the advertisement effect.
#       Default is 0.5.
#     peak_effect_delay: Delay of the peak effect in the carryover function.
#       Default is 1.
#     number_lags: Number of lags to include in the carryover calculation. Default
#       is 13.

#   Returns:
#     The carryover values for the given data with the given parameters.
#   """
#   lags_arange = jnp.expand_dims(jnp.arange(number_lags, dtype=jnp.float32),
#                                 axis=-1)
#   convolve_func = _carryover_convolve
#   if data.ndim == 3:
#     # Since _carryover_convolve is already vmaped in the decorator we only need
#     # to vmap it once here to handle the geo level data. We keep the windows bi
#     # dimensional also for three dims data and vmap over only the extra data
#     # dimension.
#     convolve_func = jax.vmap(
#         fun=_carryover_convolve, in_axes=(2, None, None), out_axes=2)
#   weights = ad_effect_retention_rate**((lags_arange - peak_effect_delay)**2)
#   return convolve_func(data, weights, number_lags)


#Staticed out as used in jnp.arange (fails with dynamic)
@functools.partial(jax.jit, static_argnames=("number_lags",))
def carryover_original(data: jnp.ndarray,
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
  lags_arange = jnp.expand_dims(jnp.arange(number_lags, dtype=jnp.float32),
                                axis=-1)
  weights = ad_effect_retention_rate**((lags_arange - peak_effect_delay)**2)

  zeros = jnp.zeros((number_lags - 1, data.shape[1]))
  window = jnp.concatenate([zeros, weights])
  #window = weights

  # return jax.numpy.apply_along_axis(
  #   lambda m: jnp.convolve(m[window.shape[0]:], m[:window.shape[0]], mode='same'),
  #   axis=0,
  #   arr=jnp.concatenate([window, data], axis=0)
  # ) / weights.sum(axis=0).reshape(1, -1)

  return jnp.concatenate([
      jax.numpy.convolve(
        data[:, i],
        window[:, i],
        mode='same',
      ).reshape(-1, 1)
    for i in range(data.shape[1])
  ], axis=1) / weights.sum(axis=0).reshape(1, -1)

@functools.partial(jax.jit, static_argnames=('number_lags',))
def carryover(
  data: jnp.ndarray,
  ad_effect_retention_rate: jnp.ndarray,
  peak_effect_delay: jnp.ndarray,
  number_lags: int = 100) -> jnp.ndarray:

  #ad_effect_retention_rate = jnp.clip(ad_effect_retention_rate, None, RETENTION_LIMIT)
  ad_effect_retention_rate = ad_effect_retention_rate * RETENTION_LIMIT

  lags_arange = jnp.expand_dims(jnp.arange(number_lags, dtype=jnp.float32),
                              axis=-1)
  weights = ad_effect_retention_rate**((lags_arange - peak_effect_delay) ** 2)
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

  #return jnp.

  # dn = jax.lax.conv_dimension_numbers(
  #   data.shape, weights.shape,
  #   ('NC', 'IW', 'NC')
  # )

  # return jax.lax.conv_general_dilated(
  #   d,   # lhs = image tensor
  #   weights, # rhs = conv kernel tensor
  #   (1,),   # window strides
  #   'SAME', # padding mode
  #   (1,),   # lhs/image dilation
  #   (1,),   # rhs/kernel dilation
  #   dn
  # )[0]         # dimension_numbers = lhs, rhs, out dimension permutation


  return jax.lax.conv_general_dilated(
    data,
    weights,

  )

  return jax.scipy.ndimage.convolve(
    data,
    weights,
    axis=0,
    mode='same',
    origin=1
  )

  # a = jax.scipy.ndimage.convolve1d()
  # a = jax.scipy.signal.convolve(data, window, mode="same") / weights.sum(axis=0).reshape(1, -1)


  # a = jnp.apply_along_axis(
  #   lambda m: jnp.convolve(m, filt, mode='same'),
  #   axis=0,
  #   arr=a
  # ) / weights.sum(axis=0).reshape(1, -1)

  # b = _carryover_convolve(data, weights, number_lags)

  # assert jnp.all(a==b)
  return jax.scipy.signal.fftconvolve(
    data,
    window,
    mode="same",
    axes=[0]
  ) / weights.sum(axis=0).reshape(1, -1)

#   return a
  # return jnp.concatenate([
  #   jnp.expand_dims(jax.scipy.signal.convolve(data[:, i], weights[:, i], mode="same") / weights.sum(), axis=1)
  #   for i in range(data.shape[1])
  # ], axis=1)

  # return convolve_func(data, weights, number_lags)
  #convolved_media = convolve_func(data, window)
  #convolved_media = convolve_func(data, weights, number_lags)
  #return convolved_media #jax.block_until_ready(convolve_func(data, weights, number_lags))
  return convolve_func(data, weights, number_lags)

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


# @jax.jit
# def hill(data: jnp.ndarray,
#          half_max_effective_concentration: jnp.ndarray,
#          slope: jnp.ndarray,
#          **_
#   ) -> jnp.ndarray:
#   """Calculates the hill function for a given array of values.

#   Refer to the following link for detailed information on this equation:
#     https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)

#   Args:
#     data: Input data.
#     half_max_effective_concentration: ec50 value for the hill function.
#     slope: Slope of the hill function.

#   Returns:
#     The hill values for the respective input data.
#   """
#   #exponent_safe = jnp.where(condition=(data == 0), x=1, y=(data / half_max_effective_concentration)) ** (-slope)
#   #save_transform = jnp.where(condition=(data == 0), x=0, y=exponent_safe)

#   save_transform = apply_exponent_safe(
#       data=data / half_max_effective_concentration, exponent=-slope)
#   hill_media = jnp.where(condition=(data == 0), x=0, y=1. / (1 + save_transform))

#   return hill_media