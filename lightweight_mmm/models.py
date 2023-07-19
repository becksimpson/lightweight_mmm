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

"""Module containing the different models available in the lightweightMMM lib.

Currently this file contains a main model with three possible options for
processing the media data. Which essentially grants the possibility of building
three different models.
  - Adstock
  - Hill-Adstock
  - Carryover
"""
import functools
from itertools import chain
import sys
#  pylint: disable=g-import-not-at-top
if sys.version_info >= (3, 8):
  from typing import Protocol
else:
  from typing_extensions import Protocol
from typing import List


from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Union

import immutabledict
import jax
import jax.numpy as jnp
from jax.scipy.stats import beta as jbeta
import numpyro
from numpyro import distributions as dist
from numpyro.contrib.control_flow import cond

from lightweight_mmm import media_transforms
#from lightweight_mmm.lightweight_mmm import _NAMES_TO_MODEL_TRANSFORMS


Prior = Union[
    dist.Distribution,
    Dict[str, float],
    Sequence[float],
    float
]

class TransformFunction(Protocol):

  def __call__(
      self,
      media_data: jnp.ndarray,
      custom_priors: MutableMapping[str, Prior],
      **kwargs: Any) -> jnp.ndarray:
    ...

_INTERCEPT = "intercept"
_COEF_TREND = "coef_trend"
_EXPO_TREND = "expo_trend"
_SIGMA = "sigma"
_GAMMA_SEASONALITY = "gamma_seasonality"
_WEEKDAY = "weekday"
_COEF_EXTRA_FEATURES = "coef_extra_features"
_COEF_SEASONALITY = "coef_seasonality"
_PARAM_DAY_OF_MONTH = 'param_dayofmonth'
_MULTIPLIER_DAY_OF_MONTH = 'multiplier_dayofmonth'

MODEL_PRIORS_NAMES = frozenset((
    _INTERCEPT,
    _COEF_TREND,
    _EXPO_TREND,
    _SIGMA,
    _GAMMA_SEASONALITY,
    _WEEKDAY,
    _PARAM_DAY_OF_MONTH,
    _MULTIPLIER_DAY_OF_MONTH,
    _COEF_EXTRA_FEATURES,
    _COEF_SEASONALITY))

_EXPONENT = "exponent"
_LAG_WEIGHT = "lag_weight"
_HALF_MAX_EFFECTIVE_CONCENTRATION = "half_max_effective_concentration"
_SLOPE = "slope"
_AD_EFFECT_RETENTION_RATE = "ad_effect_retention_rate"
_PEAK_EFFECT_DELAY = "peak_effect_delay"
_SATURATION = 'saturation'


GEO_ONLY_PRIORS = frozenset((_COEF_SEASONALITY,))

def _get_default_priors() -> Mapping[str, Prior]:
  # Since JAX cannot be called before absl.app.run in tests we get default
  # priors from a function.
  return immutabledict.immutabledict({
      _INTERCEPT: dist.HalfNormal(scale=0.5),
      _COEF_TREND: dist.Normal(loc=0., scale=1.),
      _EXPO_TREND: dist.Uniform(low=0.5, high=1.5),
      _SIGMA: dist.Gamma(concentration=1., rate=1.),
      _GAMMA_SEASONALITY: dist.Normal(loc=0., scale=1.),
      _WEEKDAY: dist.Normal(loc=0., scale=.5),
      _COEF_EXTRA_FEATURES: dist.Normal(loc=0., scale=1.),
      _COEF_SEASONALITY: dist.HalfNormal(scale=.5),
      _PARAM_DAY_OF_MONTH: dist.TruncatedNormal(loc=1.0, scale=0.5, low=1e-6),
      _MULTIPLIER_DAY_OF_MONTH: dist.HalfNormal(0.5),
  })


def _get_transform_hyperprior_distributions() -> Mapping[str, Mapping[str, Union[float, Prior]]]:
  return immutabledict.immutabledict({
    # For Beta Distribution
    _EXPONENT: immutabledict.immutabledict({
        'concentration': dist.TruncatedNormal(0., 3., low=0.0, high=8.0)
    }),
    # Adstock lag_weight (Beta), [0.0, 1.0], higher, more carryover
    _LAG_WEIGHT: immutabledict.immutabledict({
        'concentration': dist.Uniform(0., 8.),
    }),
    # Carryover delay to peak (halfnormal)
    _PEAK_EFFECT_DELAY: immutabledict.immutabledict({
        # Median 1.6, <1 27%, longtail
        'scale': dist.LogNormal(0.5, 0.8)
    }),
    # hill saturation (gamma) create range 0.1 -> 2/3ish, 0.1 -> 1.0 peak
    _SLOPE: immutabledict.immutabledict({
        'concentration': dist.Uniform(1.5, 3.),
        'rate': 2.0 # Fixed to contrain hyperparameter distribution to appropriate range
    }),
    # Half point most effective, gamma
    # Create range 0.5 -> 2 (Half --> double mean contribution)
    _HALF_MAX_EFFECTIVE_CONCENTRATION: immutabledict.immutabledict({
      'concentration': dist.Uniform(2., 5.),
      'rate': 2.0
    }),
    # Retention rate of advertisement Beta
    _AD_EFFECT_RETENTION_RATE: immutabledict.immutabledict({
        'concentration': dist.Uniform(0., 8.),
    }),
    # Saturation for exponential saturation
    _SATURATION: immutabledict.immutabledict({
        # 1.34 mean --> HalfNormal(1.34)
        'scale': dist.LogNormal(loc=0.3, scale=0.3)
    }),
  })

def _get_transform_prior_distributions() -> Mapping[str, Prior]:
  return immutabledict.immutabledict({
    # Strongest assumption lag effect
    # concentration1 is alpha
    _LAG_WEIGHT: dist.Beta(concentration1= 2., concentration0= 1.),
    _AD_EFFECT_RETENTION_RATE: dist.Beta(concentration1=1., concentration0= 1.),
    _PEAK_EFFECT_DELAY:dist.HalfNormal(scale= 2.),

    # Saturation effects
    _EXPONENT: dist.Beta(concentration1=9., concentration0=1.),
    _SATURATION: dist.HalfNormal(scale= 2.),
    _HALF_MAX_EFFECTIVE_CONCENTRATION: dist.Gamma(concentration= 1., rate= 1.),
    _SLOPE: dist.Gamma(concentration= 1., rate= 1.)
  })

def _get_transform_prior_hyperprior_distribution(
    prior_name: str
) -> Mapping[str, Mapping[str, jnp.ndarray]]:
  
  hyperprior_distributions = _get_transform_hyperprior_distributions()[prior_name]

  # Sample from hyperprior distributions
  hyperprior_samples = {
    hyperprior_name: numpyro.sample(
      name=prior_name + '_' + hyperprior_name,
      fn=distr
    ) if not isinstance(distr, float) else distr
    for hyperprior_name, distr in hyperprior_distributions.items()
  }

  prior_fn = _get_transform_prior_distributions()[prior_name].__class__

  if prior_fn != dist.Beta:
    return prior_fn(
      **hyperprior_samples
    )
  # Beta hyperprior specs are symmetric, parameterized around a concentration
  return prior_fn(
    concentration0= 1 + hyperprior_samples['concentration'],
    concentration1= 9 - hyperprior_samples['concentration']
  )

_ENSEMBLE_ADSTOCK_TRANSFORMS = immutabledict.immutabledict({
  (_LAG_WEIGHT,): media_transforms.adstock,
  (_AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY): media_transforms.carryover
})
_ENSEMBLE_SATURATION_TRANSFORMS = immutabledict.immutabledict({
  (_EXPONENT,): media_transforms.apply_exponent_safe,
  (_SATURATION,): media_transforms.exponential_saturation,
  (_HALF_MAX_EFFECTIVE_CONCENTRATION, _SLOPE): media_transforms.hill,
})

_ENSEMBLE_TRANSFORMS_PRIOR_NAMES = set([
  *list(chain.from_iterable(_ENSEMBLE_ADSTOCK_TRANSFORMS.keys())),
  *list(chain.from_iterable(_ENSEMBLE_SATURATION_TRANSFORMS.keys()))
])


def _get_transform_kwargs(fn):
  from inspect import signature, _empty
  lst = []
  sig = signature(fn)
  for param in sig.parameters.values():
      if (param._default !=  _empty):
        lst.append(param.name)
  return frozenset(lst)

_TRANSFORM_KWARGS = immutabledict.immutabledict({
  params: _get_transform_kwargs(fn)
  for params, fn in [
    *_ENSEMBLE_ADSTOCK_TRANSFORMS.items(),
    *_ENSEMBLE_SATURATION_TRANSFORMS.items()
  ]
})
# _TRANSFORM_KWARGS = immutabledict.immutabledict({
#   (_HALF_MAX_EFFECTIVE_CONCENTRATION, _SLOPE): frozenset(['hill_normalise', ]),
#   (_LAG_WEIGHT,): frozenset(['adstock_normalise', ]),
#   (_AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY): frozenset(['number_lags',]),
#   (_SATURATION,): frozenset([])
# }) 

def transform_exponential_adstock(
                    media_data: jnp.ndarray,
                    transform_samples: jnp.ndarray,
                    adstock_normalise: bool = True,
                    **kwargs
) -> jnp.ndarray:
  """Transforms the input data with exponetial saturation function and carryover

  Benefit: Requires only one parameter per media channel

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    transform_samples
  Returns:
    The transformed media data.
  """
  lag_weight = transform_samples[_LAG_WEIGHT]
  saturation = transform_samples[_SATURATION]

  adstock = media_transforms.adstock(
      data=media_data, lag_weight=lag_weight, adstock_normalise=adstock_normalise)

  return media_transforms.exponential_saturation(
    data=adstock, saturation=saturation
  )

def transform_exponential_carryover(
                    media_data: jnp.ndarray,
                    transform_samples,
                    number_lags: int = 13,
                    **kwargs
) -> jnp.ndarray:
  """Transforms the input data with exponetial saturation function and carryover

  Benefit: Requires only one parameter per media channel

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. The possible names of parameters for exponential_carryover
      are "slope".

  Returns:
    The transformed media data.
  """
  peak_effect_delay = transform_samples[_PEAK_EFFECT_DELAY]
  ad_effect_retention_rate = transform_samples[_AD_EFFECT_RETENTION_RATE]
  saturation = transform_samples[_SATURATION]

  # TODO Fix number_lags carrythrough
  carryover = media_transforms.carryover(
      data=media_data,
      ad_effect_retention_rate=ad_effect_retention_rate,
      peak_effect_delay=peak_effect_delay,
      number_lags=number_lags) # Max lags allowed is 2months

  # with numpyro.plate(name=f"{_SATURATION}_plate", size=media_data.shape[1]):
  #   slopes = numpyro.sample(
  #     name=_SATURATION,
  #     fn=custom_priors.get(_SATURATION, transform_default_priors[_SATURATION])
  #   )
  return media_transforms.exponential_saturation(
    data=carryover, saturation=saturation
  )


def transform_adstock(media_data: jnp.ndarray,
                      transform_samples,
                      adstock_normalise: bool = True,
                      **kwargs) -> jnp.ndarray:
  """Transforms the input data with the adstock function and exponent.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. The possible names of parameters for adstock and exponent
      are "lag_weight" and "exponent".
    adstock_normalise: Whether to normalise the output values.

  Returns:
    The transformed media data.
  """
  exponent = transform_samples[_EXPONENT]
  lag_weight = transform_samples[_LAG_WEIGHT]

  if media_data.ndim == 3:
    lag_weight = jnp.expand_dims(lag_weight, axis=-1)
    exponent = jnp.expand_dims(exponent, axis=-1)

  adstock = media_transforms.adstock(
      data=media_data, lag_weight=lag_weight, adstock_normalise=adstock_normalise)

  n = media_transforms.apply_exponent_safe(data=adstock, exponent=exponent)

  return n / n.sum(axis=0) * adstock.sum(axis=0)


def transform_hill_adstock(media_data: jnp.ndarray,
                           transform_samples,
                           hill_normalise: bool = True,
                           adstock_normalise: bool = True,
                           **_) -> jnp.ndarray:
  """Transforms the input data with the adstock and hill functions.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    adstock_normalise: Whether to normalise the output values, so adstock contribution sums to 1
    saturation_normalise: Whether to normalise so 0.5 input = 0.5 output, ensure 50% linearity.

  Returns:
    The transformed media data.
  """
  lag_weight = transform_samples[_LAG_WEIGHT]
  half_max_effective_concentration = transform_samples[_HALF_MAX_EFFECTIVE_CONCENTRATION]
  slope = transform_samples[_SLOPE]

  if media_data.ndim == 3:
    lag_weight = jnp.expand_dims(lag_weight, axis=-1)
    half_max_effective_concentration = jnp.expand_dims(
        half_max_effective_concentration, axis=-1)
    slope = jnp.expand_dims(slope, axis=-1)

  return media_transforms.hill(
      data=media_transforms.adstock(
          data=media_data, lag_weight=lag_weight, adstock_normalise=adstock_normalise),
      half_max_effective_concentration=half_max_effective_concentration,
      slope=slope,
      hill_normalise=hill_normalise
    )

def transform_carryover(media_data: jnp.ndarray,
                        transform_samples,
                        number_lags: int = 60,
                        **_
                        ) -> jnp.ndarray:
  """Transforms the input data with the carryover function and exponent.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    transform_samples:
    number_lags: Number of lags for the carryover function.

  Returns:
    The transformed media data.
  """
  carryover = media_transforms.carryover(
      data=media_data,
      ad_effect_retention_rate=transform_samples[_AD_EFFECT_RETENTION_RATE],
      peak_effect_delay=transform_samples[_PEAK_EFFECT_DELAY],
      number_lags=number_lags)

  if media_data.ndim == 3:
    exponent = jnp.expand_dims(exponent, axis=-1)
  return media_transforms.apply_exponent_safe(
    data=carryover,
    exponent=transform_samples[_EXPONENT]
  )

# def transform_carryover(media_data: jnp.ndarray,
#                         transform_samples,
#                         number_lags: int = 60,
#                         **_
#                         ) -> jnp.ndarray:
#   """Transforms the input data with the carryover function and exponent.

#   Args:
#     media_data: Media data to be transformed. It is expected to have 2 dims for
#       national models and 3 for geo models.
#     transform_samples:
#     number_lags: Number of lags for the carryover function.

#   Returns:
#     The transformed media data.
#   """
#   carryover = media_transforms.carryover(
#       data=media_data,
#       ad_effect_retention_rate=transform_samples[_AD_EFFECT_RETENTION_RATE],
#       peak_effect_delay=transform_samples[_PEAK_EFFECT_DELAY],
#       number_lags=number_lags)

# @functools.partial(jax.jit, static_argnames=(
#   'number_lags', 'hill_normalise', 'adstock_normalise'
# ))
def transform_ensemble_multi(
    media_data: jnp.ndarray,
    transform_samples,
    #number_lags: int = 60,
    #hill_normalise: bool = True,
    #adstock_normalise: bool = True,
    **kwargs
) -> jnp.ndarray:
  """Transforms the input data with the carryover function and exponent.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    transform_samples:
  Returns:
    The transformed media data.
  """
  # kwargs = immutabledict.immutabledict({
  #   #'number_lags': number_lags,
  #   'hill_normalise': hill_normalise,
  #   'adstock_normalise': adstock_normalise
  # })

  # transformed_media = media_transforms.carryover(
  #   data=media_data,
  #   ad_effect_retention_rate=transform_samples[_AD_EFFECT_RETENTION_RATE],
  #   peak_effect_delay=transform_samples[_PEAK_EFFECT_DELAY] 
  # )
  # transformed_media = media_transforms.exponential_saturation(
  #   data=media_data,
  #   saturation=transform_samples[_SATURATION]
  # )
  # return jnp.expand_dims(transformed_media, axis=0)


  # transformed_media = None
  # #[((_AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY), media_transforms.carryover)]:#[((_LAG_WEIGHT,), media_transforms.adstock)]:#
  # for ad_params, ad_f in _ENSEMBLE_ADSTOCK_TRANSFORMS.items():
  #   adstocked_media_data = ad_f(
  #     data=jnp.copy(media_data),
  #     **{p: transform_samples[p] for p in ad_params},
  #     #**{k: kwargs[k] for k in _TRANSFORM_KWARGS[ad_params]}
  #     #number_lags=number_lags,
  #     #adstock_normalise=adstock_normalise
  #   )
  #   for sat_params, sat_f in _ENSEMBLE_SATURATION_TRANSFORMS.items():
  #     saturated_media_data = sat_f(
  #       jnp.copy(adstocked_media_data),
  #       *[transform_samples[p] for p in sat_params],
  #       #**{k: kwargs[k] for k in _TRANSFORM_KWARGS[sat_params]}
  #     )
  #     if transformed_media is None:
  #       transformed_media = jnp.expand_dims(saturated_media_data, axis=0)
  #     else:
  #       transformed_media = jnp.concatenate(
  #         [
  #           transformed_media,
  #           jnp.expand_dims(saturated_media_data, axis=0)
  #         ], axis=0
  #       )
  # return transformed_media

  transformed_media = jnp.concatenate([
    jnp.expand_dims(sat_f(
      ad_f(
        media_data,
        *[transform_samples[p] for p in ad_params],
        **{k: kwargs.get(k, None) for k in _TRANSFORM_KWARGS[ad_params]}
        #number_lags=number_lags,
        #adstock_normalise=adstock_normalise
      ),
      *[transform_samples[p] for p in sat_params],
      #**{k:v for k,v in kwargs.items() if _HALF_MAX_EFFECTIVE_CONCENTRATION in sat_params}
      **{k: kwargs.get(k, None) for k in _TRANSFORM_KWARGS[sat_params]}
      #hill_normalise=hill_normalise
    ), 0)
    for sat_params, sat_f in _ENSEMBLE_SATURATION_TRANSFORMS.items()
    for ad_params, ad_f in _ENSEMBLE_ADSTOCK_TRANSFORMS.items()
  ], axis=0)

  return transformed_media

# @functools.partial(jax.jit, static_argnames=(
#   'number_lags', 'hill_normalise', 'adstock_normalise'
# ))
def transform_ensemble(media_data: jnp.ndarray,
                        transform_samples,
                        number_lags: int = 60,
                        adstock_normalise: bool = True,
                        hill_normalise: bool = True,
                        **_
                        ) -> jnp.ndarray:
  """Transforms the input data with the carryover function and exponent.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    transform_samples:
  Returns:
    The transformed media data.
  """
  transformed_media = transform_ensemble_multi(
    media_data,
    transform_samples,
    number_lags=number_lags,
    adstock_normalise=adstock_normalise,
    hill_normalise=hill_normalise
  )

  # for ad_params, ad_f in [((_AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY), media_transforms.carryover)]:#[((_LAG_WEIGHT,), media_transforms.adstock)]:#_ENSEMBLE_ADSTOCK_TRANSFORMS.items():
  #   # adstocked_media_data = ad_f(
  #   #   data=jnp.copy(media_data),
  #   #   **{p: transform_samples[p] for p in ad_params},
  #   #   #**{k: kwargs[k] for k in _TRANSFORM_KWARGS[ad_params]}
  #   #   number_lags=number_lags,
  #   #   #adstock_normalise=adstock_normalise
  #   # )

  # for sat_params, sat_f in _ENSEMBLE_SATURATION_TRANSFORMS.items():
  #   saturated_media_data = sat_f(
  #     jnp.copy(adstocked_media_data),
  #     *[transform_samples[p] for p in sat_params]
  #   )
  #   if transformed_media is None:
  #     transformed_media = jnp.expand_dims(saturated_media_data, axis=0)
  #   else:
  #     transformed_media = jnp.concatenate(
  #       [
  #         transformed_media,
  #         jnp.expand_dims(saturated_media_data, axis=0)
  #       ], axis=0
  #     )

  # transformed_media = media_transforms.exponential_saturation(
  #   data=transformed_media,
  #   saturation=transform_samples[_SATURATION],
  # )


  # transformed_media = None
  # transformed_media = media_transforms.carryover(
  #     data=media_data,
  #     ad_effect_retention_rate=transform_samples[_AD_EFFECT_RETENTION_RATE],
  #     peak_effect_delay=transform_samples[_PEAK_EFFECT_DELAY],
  #     number_lags=number_lags)
  # transformed_media = media_transforms.hill(
  #   data=transformed_media,
  #   half_max_effective_concentration=transform_samples[_HALF_MAX_EFFECTIVE_CONCENTRATION],
  #   slope=transform_samples[_SLOPE],
  #   hill_normalise=hill_normalise
  # )

  # yes
  # transformed_media = media_transforms.exponential_saturation(
  #     data=media_transforms.carryover(
  #         data=media_data,
  #         ad_effect_retention_rate=transform_samples[_AD_EFFECT_RETENTION_RATE],
  #         peak_effect_delay=transform_samples[_PEAK_EFFECT_DELAY],
  #         #number_lags=number_lags
  #     ),
  #     saturation=transform_samples[_SATURATION]
  # )

  # transformed_media = media_transforms.hill(
  #     data=media_data,
  #     half_max_effective_concentration=transform_samples[_HALF_MAX_EFFECTIVE_CONCENTRATION],
  #     slope=transform_samples[_SLOPE],
  #     #hill_normalise=hill_normalise
  # )

  # no
  # carried_data = media_transforms.carryover(
  #     data=media_data,
  #     ad_effect_retention_rate=transform_samples[_AD_EFFECT_RETENTION_RATE],
  #     peak_effect_delay=transform_samples[_PEAK_EFFECT_DELAY],
  #     #number_lags=number_lags
  # )
  # transformed_media = media_transforms.hill(
  #     data=carried_data,
  #     half_max_effective_concentration=transform_samples[_HALF_MAX_EFFECTIVE_CONCENTRATION],
  #     slope=transform_samples[_SLOPE],
  #     #hill_normalise=hill_normalise
  # )

  # yes
  # transformed_media = media_transforms.apply_exponent_safe(
  #   data=media_transforms.carryover(
  #         data=media_data,
  #         ad_effect_retention_rate=transform_samples[_AD_EFFECT_RETENTION_RATE],
  #         peak_effect_delay=transform_samples[_PEAK_EFFECT_DELAY],
  #         #number_lags=number_lags
  #     ),
  #   exponent=transform_samples[_EXPONENT]
  # )

  # transformed_media = media_transforms.carryover(
  #     data=media_data,
  #     ad_effect_retention_rate=transform_samples[_AD_EFFECT_RETENTION_RATE],
  #     peak_effect_delay=transform_samples[_PEAK_EFFECT_DELAY],
  #     #number_lags=number_lags
  # )

  # yes and quick
  # transformed_media = media_transforms.exponential_saturation(
  #     data=media_transforms.adstock(
  #         data=media_data,
  #         lag_weight=transform_samples[_LAG_WEIGHT]
  #     ), 
  #     saturation=transform_samples[_SATURATION]
  # )

  # yes and quick
  # transformed_media = media_transforms.hill(
  #     data=media_transforms.adstock(
  #         data=media_data,
  #         lag_weight=transform_samples[_LAG_WEIGHT]
  #     ), 
  #     half_max_effective_concentration=transform_samples[_HALF_MAX_EFFECTIVE_CONCENTRATION],
  #     slope=transform_samples[_SLOPE],
  #     #hill_normalise=hill_normalise
  # )

  #transformed_media = jnp.expand_dims(transformed_media, axis=0)

  # transformed_media = transform_ensemble_multi(
  #   media_data,
  #   transform_samples,
  #   number_lags=number_lags,
  #   adstock_normalise=adstock_normalise,
  #   hill_normalise=hill_normalise
  # )
  # transformed_media = media_transforms.hill(
  #     data=media_transforms.adstock(
  #         data=media_data,
  #         lag_weight=transform_samples[_LAG_WEIGHT],
  #         adstock_normalise=adstock_normalise),
  #     half_max_effective_concentration=transform_samples[_HALF_MAX_EFFECTIVE_CONCENTRATION],
  #     slope=transform_samples[_SLOPE],
  #     hill_normalise=hill_normalise
  # )
  # transformed_media = jnp.expand_dims(transformed_media, axis=0)
    

  n_models = transformed_media.shape[0]# len(_ENSEMBLE_ADSTOCK_TRANSFORMS) * len(_ENSEMBLE_SATURATION_TRANSFORMS)

  with numpyro.plate(
      name=f"transform_weights_plate",
      size=n_models):
    model_weights = numpyro.sample(
      name='transform_weights',
      fn=dist.Beta(concentration1=1.0, concentration0=1.0)
    )

  model_weights = model_weights / model_weights.sum()

  return (transformed_media.T @ model_weights).T


TRANSFORM_PRIORS_NAMES = immutabledict.immutabledict({
    "carryover":
        frozenset((_AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY, _EXPONENT)),
    "adstock":
        frozenset((_EXPONENT, _LAG_WEIGHT)),
    "hill_adstock":
        frozenset((_LAG_WEIGHT, _HALF_MAX_EFFECTIVE_CONCENTRATION, _SLOPE)),
    "exponential_carryover":
      frozenset((_SATURATION, _AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY)),
    "exponential_adstock":
        frozenset((_SATURATION, _LAG_WEIGHT)),
    "ensemble_avg": frozenset(tuple(_ENSEMBLE_TRANSFORMS_PRIOR_NAMES)),
    "ensemble": frozenset(tuple(_ENSEMBLE_TRANSFORMS_PRIOR_NAMES))
})

_NAMES_TO_MODEL_TRANSFORMS = immutabledict.immutabledict({
    "hill_adstock": transform_hill_adstock,
    "adstock": transform_adstock,
    "carryover": transform_carryover,
    "exponential_carryover": transform_exponential_carryover,
    "exponential_adstock": transform_exponential_adstock,
    "ensemble_avg": transform_ensemble,
    "ensemble": transform_ensemble_multi
})

_MODEL_TRANSFORMS_TO_PRIOR_NAMES = immutabledict.immutabledict({
  fn: TRANSFORM_PRIORS_NAMES[name]
  for name, fn in _NAMES_TO_MODEL_TRANSFORMS.items()
})


def _get_transform_function_prior_names(
    transform_function: TransformFunction
  ) -> List[str]:
 
  prior_list = _MODEL_TRANSFORMS_TO_PRIOR_NAMES.get(
    transform_function, _ENSEMBLE_TRANSFORMS_PRIOR_NAMES
  )

  return prior_list


def _get_transform_default_priors(
    transform_hyperprior,
    transform_function
  ) -> Mapping[str, Prior]:

  # # Prior distributions considered
  # transform_prior_lists = _get_transform_default_priors_lists()

  # Generate hyperprior distribution samples for all possible hyper-priors.
  prior_distributions = _get_transform_prior_distributions()

  def get_prior(prior_name):
    """ We don't insist on prior_defaults sharing the same distribution as hyperpriors"""
    if transform_hyperprior:
      return _get_transform_prior_hyperprior_distribution(prior_name)
    else:
      return prior_distributions[prior_name]
  
  prior_names = _get_transform_function_prior_names(transform_function)
  return immutabledict.immutabledict({
    prior_name: get_prior(prior_name)
    for prior_name in prior_names
  })

def _get_transform_param_samples(
    transform_function,
    transform_hyperprior,
    custom_priors,
    n_media_channels: int
  ) -> Mapping[str, jnp.ndarray]:
  transform_default_priors = _get_transform_default_priors(
    transform_hyperprior,
    transform_function
  )

  def get_sample(site_name, dist):
    with numpyro.plate(name=f"{site_name}_plate",
                      size=n_media_channels):
      return numpyro.sample(
          name=site_name,
          fn=custom_priors.get(site_name,               
              dist#transform_default_priors[site_name]
        )
      )
    
  return {
    prior_name: get_sample(prior_name, dist)
    for prior_name, dist in transform_default_priors.items()
  }

def apply_media_transform_function(
  transform_function: TransformFunction,
  media_data: jnp.ndarray,
  transform_hyperprior: bool,
  custom_priors: MutableMapping[str, Prior],
  transform_kwargs
) -> jnp.ndarray:
  transform_param_samples = _get_transform_param_samples(
    transform_function,
    transform_hyperprior,
    custom_priors,
    media_data.shape[1]
  )
  return transform_function(
    media_data,
    transform_param_samples,
    **transform_kwargs
  )


def calculate_seasonal_effects(
    media_data: jnp.ndarray,
    doms: jnp.ndarray,
    degrees_seasonality: int,
    frequency:int,
    weekday_seasonality: bool,
    custom_priors: MutableMapping[str, Prior],
):
  """ """
  data_size = media_data.shape[0]
  n_geos = media_data.shape[2] if media_data.ndim == 3 else 1

  default_priors = _get_default_priors()

  # Time of Year Seasonality
  with numpyro.plate(name=f"{_GAMMA_SEASONALITY}_sin_cos_plate", size=2):
    with numpyro.plate(name=f"{_GAMMA_SEASONALITY}_plate",
                       size=degrees_seasonality):
      gamma_seasonality = numpyro.sample(
          name=_GAMMA_SEASONALITY,
          fn=custom_priors.get(
              _GAMMA_SEASONALITY, default_priors[_GAMMA_SEASONALITY]))
      
  seasonality = media_transforms.calculate_seasonality(
      number_periods=data_size,
      degrees=degrees_seasonality,
      frequency=frequency,
      gamma_seasonality=gamma_seasonality)
  

  # Day of Month Seasonality
  if doms is not None:
    with numpyro.plate(name=f"{_PARAM_DAY_OF_MONTH}_plate", size=2):
      dom_param = numpyro.sample(
        name=_PARAM_DAY_OF_MONTH,
        fn=custom_priors.get(_PARAM_DAY_OF_MONTH, default_priors[_PARAM_DAY_OF_MONTH])
      )
    dom_multiplier = numpyro.sample(
      name=_MULTIPLIER_DAY_OF_MONTH,
      fn=custom_priors.get(_MULTIPLIER_DAY_OF_MONTH, default_priors[_MULTIPLIER_DAY_OF_MONTH])
    )
    dom_series = jbeta.pdf(doms / 32, *dom_param) * dom_multiplier# (1.0 + dom_multiplier)
    
  # Day of Week Seasonality
  if weekday_seasonality:
    with numpyro.plate(name=f"{_WEEKDAY}_plate", size=6):
      weekday = numpyro.sample(
          name=_WEEKDAY,
          fn=custom_priors.get(_WEEKDAY, default_priors[_WEEKDAY]))
    weekday = jnp.concatenate(arrays=[weekday, jnp.array([0])], axis=0)
    weekday_series = weekday[jnp.arange(data_size) % 7]

  # Extensions to account for n_geos.
  coef_seasonality = 1
  if media_data.ndim == 3:  # For geo model's case
    seasonality = jnp.expand_dims(seasonality, axis=-1)

    if weekday_seasonality:
      weekday_series = jnp.expand_dims(weekday_series, axis=-1)
      dom_series = jnp.expand_dims(dom_series, axis=-1)
    with numpyro.plate(name="seasonality_plate", size=n_geos):
      coef_seasonality = numpyro.sample(
          name=_COEF_SEASONALITY,
          fn=custom_priors.get(
              _COEF_SEASONALITY, default_priors[_COEF_SEASONALITY]))
  
  total_seasonality = (
    seasonality * coef_seasonality
  )
  if doms is not None:
    total_seasonality += dom_series
  if weekday_seasonality:
    total_seasonality += weekday_series

  # Total seasonality (to aid in re-distribution later)
  return numpyro.deterministic(
    name="total_seasonality",
    value=total_seasonality
  )

def calculate_trend_effects(
  media_data: jnp.ndarray,
  custom_priors
):
  data_size = media_data.shape[0]
  n_geos = media_data.shape[2] if media_data.ndim == 3 else 1

  default_priors = _get_default_priors()

  # TODO(): Force all geos to have the same trend sign.
  with numpyro.plate(name=f"{_COEF_TREND}_plate", size=n_geos):
    coef_trend = numpyro.sample(
        name=_COEF_TREND,
        fn=custom_priors.get(_COEF_TREND, default_priors[_COEF_TREND]))

  expo_trend = numpyro.sample(
      name=_EXPO_TREND,
      fn=custom_priors.get(
          _EXPO_TREND, default_priors[_EXPO_TREND]))

  # For national model's case
  trend = jnp.arange(data_size)
  if media_data.ndim == 3:  # For geo model's case
    trend = jnp.expand_dims(trend, axis=-1)

  total_trend = coef_trend * trend ** expo_trend

  # Total trend (to aid in re-distribution later)
  return numpyro.deterministic(
    name='total_trend',
    value=total_trend
  )

def calculate_extra_features_effects(
    extra_features: jnp.ndarray,
    custom_priors,
    geo_shape,
):
  default_priors = _get_default_priors()

  # Contribution of control factors
  if extra_features is not None:
    plate_prefixes = ("extra_feature",)
    extra_features_einsum = "tf, f -> t"  # t = time, f = feature
    extra_features_plates_shape = (extra_features.shape[1],)
    if extra_features.ndim == 3:
      plate_prefixes = ("extra_feature", "geo")
      extra_features_einsum = "tfg, fg -> tg"  # t = time, f = feature, g = geo
      extra_features_plates_shape = (extra_features.shape[1], *geo_shape)
    with numpyro.plate_stack(plate_prefixes,
                             sizes=extra_features_plates_shape):
      coef_extra_features = numpyro.sample(
          name=_COEF_EXTRA_FEATURES,
          fn=custom_priors.get(
              _COEF_EXTRA_FEATURES, default_priors[_COEF_EXTRA_FEATURES]))
    extra_features_effect = jnp.einsum(extra_features_einsum,
                                       extra_features,
                                       coef_extra_features)
    
  return extra_features_effect

#@functools.partial(jax.jit, static_argnames=('frequency', 'transform_kwargs', 'transform_function', 'transform_hyperprior'))
def calculate_media_effects(
    media_data: jnp.ndarray,
    media_prior: jnp.ndarray,
    transform_function: TransformFunction,
    transform_hyperprior: bool,
    custom_priors: MutableMapping[str, Prior],
    frequency: int,
    transform_kwargs
):

  n_channels = media_data.shape[1]
  n_geos = media_data.shape[2] if media_data.ndim == 3 else 1

  carryover_models = [
    transform_carryover, transform_exponential_carryover,
    transform_ensemble, transform_ensemble_multi
  ]

  # In case of daily data, number of lags should be 13*7.
  # transform_kwargs = transform_kwargs or {}
  # if transform_function in carryover_models:
  #   transform_kwargs['number_lags'] = 13 if frequency == 52 else 60

  media_transformed = apply_media_transform_function(
    transform_function,
    media_data,
    transform_hyperprior,
    custom_priors,
    transform_kwargs
  )

  if transform_function not in [transform_ensemble_multi]:
    media_transformed = numpyro.deterministic(
      name="media_transformed",
      value=media_transformed
    )
    with numpyro.plate(
        name="channel_media_plate",
        size=n_channels,
        dim=-2 if media_data.ndim == 3 else -1):
      coef_media = numpyro.sample(
          name="channel_coef_media" if media_data.ndim == 3 else "coef_media",
          #fn=dist.TruncatedNormal(loc=media_prior, scale=0.05, low=1e-6)
          fn=dist.HalfNormal(scale=media_prior)
        )#)
      if media_data.ndim == 3:
        with numpyro.plate(
            name="geo_media_plate",
            size=n_geos,
            dim=-1):
          # Corrects the mean to be the same as in the channel only case.
          normalisation_factor = jnp.sqrt(2.0 / jnp.pi)
          coef_media = numpyro.sample(
              name="coef_media",
              fn=dist.HalfNormal(scale=coef_media * normalisation_factor)
          )

    # For national model's case
    media_einsum = "tc, c -> t"  # t = time, c = channel
    if media_data.ndim == 3:  # For geo model's case
      media_einsum = "tcg, cg -> tg"  # t = time, c = channel, g = geo

    media_contribution = numpyro.deterministic(
      name = 'media_contribution',
      value = jnp.einsum(media_einsum, media_transformed, coef_media)
    )

    return media_contribution, media_transformed, coef_media

  else:

    n_models = len(_ENSEMBLE_ADSTOCK_TRANSFORMS) * len(_ENSEMBLE_SATURATION_TRANSFORMS)

    # # model_selection = numpyro.sample(
    # #   name='model_selection',
    # #   fn=dist.Categorical(
    # #     logits=jnp.ones(n_models) / n_models
    # #   )
    # # )
    # #with numpyro.plate("model_selection_plate", media_data.shape[0]):
    # model_selection = numpyro.sample(
    #   name='model_selection',
    #   fn=dist.Categorical(
    #     logits=jnp.ones(n_models) / n_models
    #   )
    # )
    #model_selection = 2

    # def model_select(arr, dims):
    #   return jnp.where(
    #     jnp.ones(arr.shape) * jnp.expand_dims((jnp.arange(n_models) == model_selection[0]), dims),
    #     arr,
    #     jnp.zeros(arr.shape)
    #   ).sum(axis=0)

    # media_transformed = numpyro.deterministic(
    #   name='media_transformed',
    #   value=model_select(
    #     media_transformed,
    #     (1, 2) if media_data.ndim == 2 else (1, 2, 3)
    #   )
    # )

    with numpyro.plate(
      name="model_media_plate",
      size=n_models,
      dim=-3 if media_data.ndim == 3 else -2
    ):
      with numpyro.plate(
          name="channel_media_plate",
          size=n_channels,
          dim=-2 if media_data.ndim == 3 else -1):
        coef_media = numpyro.sample(
            name="channel_coef_media_models" if media_data.ndim == 3 else "coef_media_models",
            #fn=dist.TruncatedNormal(loc=media_prior, scale=0.05, low=1e-6)
            fn=dist.HalfNormal(scale=media_prior)
          )#)
        if media_data.ndim == 3:
          with numpyro.plate(
              name="geo_media_plate",
              size=n_geos,
              dim=-1):
            # Corrects the mean to be the same as in the channel only case.
            normalisation_factor = jnp.sqrt(2.0 / jnp.pi)
            coef_media = numpyro.sample(
                name="coef_media_models",
                fn=dist.HalfNormal(scale=coef_media * normalisation_factor)
            )
    # coef_media = numpyro.deterministic(
    #   name='coef_media',
    #   value=model_select(
    #     coef_media,
    #     dims=(1,)
    #   )
    # )

    # For national model's case
    media_einsum = "mtc, mc -> mt"  # t = time, c = channel
    if media_data.ndim == 3:  # For geo model's case
      media_einsum = "mtcg, mcg -> mtg"  # t = time, c = channel, g = geo

    media_contribution = jnp.einsum(media_einsum, media_transformed, coef_media)

    return media_contribution, media_transformed, coef_media


def media_mix_model(
    media_data: jnp.ndarray,
    target_data: jnp.ndarray,
    media_prior: jnp.ndarray,
    degrees_seasonality: int,
    frequency: int,
    transform_function: TransformFunction,
    transform_hyperprior: bool,
    custom_priors: MutableMapping[str, Prior],
    transform_kwargs: Optional[MutableMapping[str, Any]] = None,
    doms: Optional[jnp.ndarray] = None,
    weekday_seasonality: bool = False,
    extra_features: Optional[jnp.array] = None
    ) -> None:
  """Media mix model.

  Args:
    media_data: Media data to be be used in the model.
    target_data: Target data for the model.
    media_prior: Cost prior for each of the media channels.
    degrees_seasonality: Number of degrees of seasonality to use.
    frequency: Frequency of the time span which was used to aggregate the data.
      Eg. if weekly data then frequency is 52.
    transform_function: Function to use to transform the media data in the
      model. Currently the following are supported: 'transform_adstock',
        'transform_carryover' and 'transform_hill_adstock'.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. See our custom_priors documentation for details about the
      API and possible options.
    transform_kwargs: Any extra keyword arguments to pass to the transform
      function. For example the adstock function can take a boolean to noramlise
      output or not.
    weekday_seasonality: In case of daily data you can estimate a weekday (6)
      parameter.
    extra_features: Extra features data to include in the model.
  """
  default_priors = _get_default_priors()
  transform_kwargs = transform_kwargs if transform_kwargs else {}

  geo_shape = (media_data.shape[2],) if media_data.ndim == 3 else ()
  n_geos = media_data.shape[2] if media_data.ndim == 3 else 1

  with numpyro.plate(name=f"{_INTERCEPT}_plate", size=n_geos):
    intercept = numpyro.sample(
        name=_INTERCEPT,
        fn=custom_priors.get(_INTERCEPT, default_priors[_INTERCEPT]))

  with numpyro.plate(name=f"{_SIGMA}_plate", size=n_geos):
    sigma = numpyro.sample(
        name=_SIGMA,
        fn=custom_priors.get(_SIGMA, default_priors[_SIGMA]))

  media_effects, _, _ = calculate_media_effects(
    media_data,
    media_prior,
    transform_function,
    transform_hyperprior,
    custom_priors,
    frequency,
    transform_kwargs
  )

  seasonal_effects = calculate_seasonal_effects(
    media_data,
    doms,
    degrees_seasonality,
    frequency,
    weekday_seasonality,
    custom_priors
  )
  trend_effects = calculate_trend_effects(
    media_data,
    custom_priors
  )
  extra_features_effects = calculate_extra_features_effects(
    extra_features,
    custom_priors,
    geo_shape
  )


  if transform_function in [transform_ensemble_multi]:

    n_models = len(_ENSEMBLE_ADSTOCK_TRANSFORMS) * len(_ENSEMBLE_SATURATION_TRANSFORMS)

    with numpyro.plate(name='model_weight_plate', size=n_models):
      weights = numpyro.sample(
        'model_weights',
        fn=dist.Beta(1, 1)
      )

    weights = weights / weights.sum()
    media_effects = numpyro.deterministic(
      name = 'media_contribution',
      value = media_effects.T @ weights
    )

    # prediction = (
    #   intercept + seasonal_effects + trend_effects + 
    #   avg_media_effects
    #   + extra_features_effects
    # )

    # with numpyro.plate("data", len(target_data)):
    #   model_selection = numpyro.sample(
    #     name='model_selection',
    #     fn=dist.Categorical(
    #       logits=jnp.ones(n_models) / n_models
    #     )
    #   )
    #   media_transformed = numpyro.deterministic(
    #     name='media_transformed',
    #     value=media_transformed[:, model_selection]
    #   )
    #   coef_media = numpyro.deterministic(
    #     name='coef_media',
    #     value=coef_media[model_selection]
    #   )

    #   prediction = (
    #     intercept + seasonal_effects + trend_effects + 
    #     media_effects[model_selection]
    #     + extra_features_effects
    #   )
    #   mu = numpyro.deterministic(name="mu", value=prediction)

    #   numpyro.sample(
    #     name="target", fn=dist.Normal(loc=mu, scale=sigma), obs=target_data)


  prediction = (
    intercept + seasonal_effects + trend_effects + 
    media_effects
    + extra_features_effects
  )

  mu = numpyro.deterministic(name="mu", value=prediction)

  numpyro.sample(
      name="target", fn=dist.Normal(loc=mu, scale=sigma), obs=target_data)


# # DEFUNCT
# def _get_transform_default_priors_lists():
#   """ First in list is default. """
#   return immutabledict.immutabledict({
#     _SATURATION: [
#       numpyro.distributions.HalfNormal(2.0),
#       numpyro.distributions.HalfNormal(2.5),
#       numpyro.distributions.HalfNormal(1.5)
#     ],
#     _LAG_WEIGHT: [
#       numpyro.distributions.Beta(concentration1=1., concentration0=3.),
#       numpyro.distributions.Beta(concentration1=1., concentration0=1.),
#       numpyro.distributions.Beta(concentration1=4., concentration0=6.),
#     ],
#     _HALF_MAX_EFFECTIVE_CONCENTRATION: [
#       dist.Gamma(concentration=1., rate=1.), # Original by MMM
#       numpyro.distributions.LogNormal(loc=0.1, scale=0.5),
#       numpyro.distributions.LogNormal(loc=0.5, scale=0.5),
#     ],
#     _AD_EFFECT_RETENTION_RATE: [
#       numpyro.distributions.Beta(concentration1=1., concentration0=1.),
#       numpyro.distributions.Beta(concentration1=1., concentration0=3.),
#       numpyro.distributions.Beta(concentration1=4., concentration0=6.),
#     ],
#     _PEAK_EFFECT_DELAY: [
#       numpyro.distributions.HalfNormal(scale=2.),
#       numpyro.distributions.HalfNormal(scale=1.),
#     ],
#     _EXPONENT: [
#       numpyro.distributions.Beta(concentration1=9., concentration0=1.),
#       numpyro.distributions.Beta(concentration1=5., concentration0=1.),
#       numpyro.distributions.Beta(concentration1=1., concentration0=1.),
#     ],
#     _SLOPE: [
#       numpyro.distributions.Gamma(concentration=1., rate=1.),
#       numpyro.distributions.Gamma(concentration=3., rate=1.),
#       numpyro.distributions.HalfNormal(scale=2.0),
#     ]
#   })