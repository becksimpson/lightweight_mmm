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
from typing import (
  Any,
  Callable,
  Dict,
  List,
  Mapping,
  MutableMapping,
  Optional,
  Sequence,
  Set,
  Tuple,
  Union
)

import immutabledict
import jax
import jax.numpy as jnp
from jax.scipy.stats import beta as jbeta
import numpyro
import numpy as np
from numpyro import distributions as dist
from numpyro.contrib.control_flow import cond

from lightweight_mmm import media_transforms

Prior = Union[
    dist.Distribution,
    Dict[str, float],
    Sequence[float],
    float
]
Bound = Union[
  Dict[int, Tuple[float]],
  Tuple[float]
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
_MODEL_SIGMA = 'model_sigma'
_GAMMA_SEASONALITY = "gamma_seasonality"
_WEEKDAY = "weekday"
_COEF_EXTRA_FEATURES = "coef_extra_features"
_COEF_SEASONALITY = "coef_seasonality"
_PARAM_DAY_OF_MONTH = 'param_dayofmonth'
_MULTIPLIER_DAY_OF_MONTH = 'multiplier_dayofmonth'
_MEDIA_TRANSFORM_WEIGHTS = 'media_transform_weights'
_MODEL_WEIGHTS = 'model_weights'
_DEGREES_FREEDOM = 'degrees_freedom'
_COEF_MEDIA = 'coef_media'

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
_HALF_MAX_EFFECTIVE_CONCENTRATION_CONSTRAINED = 'half_max_effective_concentraton_constrained'
_SLOPE_CONSTRAINED = "slope_constrained"

GEO_ONLY_PRIORS = frozenset((_COEF_SEASONALITY,))

def _get_default_priors() -> Mapping[str, Prior]:
  """ Get Default Priors for key model parameters (non-media transforms)"""
  # Since JAX cannot be called before absl.app.run in tests we get default
  # priors from a function.
  return immutabledict.immutabledict({
      _INTERCEPT: dist.HalfNormal(scale=0.2),
      _COEF_TREND: dist.HalfNormal(scale=0.005),# dist.Normal(loc=0., scale=0.01),
      _EXPO_TREND: dist.Uniform(low=0.5, high=1.5),
      _SIGMA: dist.Gamma(concentration=1., rate=1.0),
      _MODEL_SIGMA: dist.Gamma(concentration=1., rate=1.),
      _GAMMA_SEASONALITY: dist.Normal(loc=0., scale=.05),
      _WEEKDAY: dist.Normal(loc=0., scale=.5),
      _COEF_EXTRA_FEATURES: dist.Normal(loc=0., scale=.01),
      #_COEF_EXTRA_FEATURES: dist.HalfNormal(scale=.1),
      _COEF_SEASONALITY: dist.HalfNormal(scale=.1),
      _PARAM_DAY_OF_MONTH: dist.TruncatedNormal(loc=1.0, scale=0.5, low=0.1, high=10.0),
      _MULTIPLIER_DAY_OF_MONTH: dist.HalfNormal(0.05),
      _MODEL_WEIGHTS: dist.Beta(concentration1=1.0, concentration0=1.0),
      _MEDIA_TRANSFORM_WEIGHTS: dist.Beta(concentration1=1.0, concentration0=1.0),
      #_DEGREES_FREEDOM: dist.Gamma(concentration=25., rate=0.5),
  })


def _get_transform_hyperprior_distributions() -> Mapping[str, Mapping[str, Union[float, Prior]]]:
  """
  Return Hyperprior distributions for media transform parameter prior distributions.
  Used for adstock/saturation functions.
  """
  return immutabledict.immutabledict({
    # Exponent saturation (Beta) - high focus on minimal saturation
    _EXPONENT: immutabledict.immutabledict({
        'concentration': dist.TruncatedNormal(0., 3., low=0.0, high=8.0)
    }),
    # Adstock retention (Beta), higher more carryover
    _LAG_WEIGHT: immutabledict.immutabledict({
        'concentration': dist.TruncatedNormal(2.0, 2.0, low=0.0, high=8.0)
    }),
    # Carryover retention (Beta)
    _AD_EFFECT_RETENTION_RATE: immutabledict.immutabledict({
        'concentration': dist.TruncatedNormal(2.0, 2.0, low=0.0, high=8.0)
    }),
    # Carryover delay to peak (halfnormal)
    _PEAK_EFFECT_DELAY: immutabledict.immutabledict({
        # Median 1.6, <1 27%, longtail
        'scale': dist.TruncatedNormal(0.0, 1.0, low=0.5, high=3.0)
    }),
    # Hill saturation (gamma) create range 0.1 -> 2/3ish, 0.1 -> 1.0 peak
    _SLOPE: immutabledict.immutabledict({
        'concentration': dist.Uniform(low=6., high=12.0),
        'rate': 10.0
    }),
    # Constrained Hill Saturation - Slope (fixed hyperprior)
    _SLOPE_CONSTRAINED: immutabledict.immutabledict({
      # 'concentration1': dist.Uniform(low=.5, high=10.0),
      # 'concentration0': 5.0
      # Fixed Prob
      'concentration1': 2.0,
      'concentration0': 5.0
    }),
    # Constrained Hill Saturation - X when Y = 0.5 (half max)
    _HALF_MAX_EFFECTIVE_CONCENTRATION_CONSTRAINED: immutabledict.immutabledict({
      #'concentration': dist.Uniform(0.0, 8.0)
      # Uniform Prob
      'concentration1': 2.0,
      'concentration0': 5.0
    }),
    # Hill Saturation - X when Y = 0.5 (half max)
    _HALF_MAX_EFFECTIVE_CONCENTRATION: immutabledict.immutabledict({
      'concentration': dist.Uniform(low=6., high=16.0),
      'rate': 20.0
    }),
    # Logistic Saturation (higher, more saturation)
    _SATURATION: immutabledict.immutabledict({
        # More bias to lower saturation (c1 < c0)
        'concentration1': dist.Uniform(1.0, 5.0),
        'concentration0': 5.0,
    }),
  })

def _generate_media_prior_distribution(
    mean: float,
    alpha: float = 1.5,
    concentration: float = 2.0,
    mode: str = 'halfnormal'
) -> Prior:
  """
  Generate ROI media prior distributions.
  Industry standard is HalfNormal, but do we really believe 0-credit is the most likely outcome?.
  We support `beta`, `gamma`, `halfnormal`

  Args:
    mean: Distribution should be centered around this mean.
    alpha: (Optional) Determines shape of beta distribution
      Default 1.5 ensures, right skewed normal distribution
    concentration: (Optional) Determines shape of gamma distribution
      Default 2.0 ensures right skewed normal distribution
    mode: What prior distribution to use.
  Returns:
    Prior distribution to be used to generate ROI Prior.
  """
  if mode == 'beta':
    # Beta distributions can only model up to 1.
    if mean > 0.33:
      raise ValueError(
        'For a mean > 0.33, a beta distribution is unsuitable as a ROI prior. Try Gamma'
      )
    beta = alpha * (mean - 1) / mean * -1
    return dist.Beta(concentration1=alpha, concentration0=beta)
  elif mode == 'halfnormal':
    return dist.HalfNormal(scale=mean)
  elif mode == 'gamma':
    rate = concentration / mean
    return dist.Gamma(concentration=concentration, rate=rate)
  else:
    raise ValueError(
      "We support only 'beta' 'halfnormal' 'gamma' media prior distributions"
    )

def _get_transform_prior_distributions() -> Mapping[str, Prior]:
  """
  Get Prior distributions for 
  Used when transform_hyperprior=False, and dictates distribution used in both cases.

  Returns:
    Named pairs of Media (Adstock/Saturation) Transform Parameters to their prior distributions.
  """
  return immutabledict.immutabledict({
    # Adstock / Carryover Effects
    _LAG_WEIGHT: dist.Beta(concentration1= 1., concentration0= 1.),
    _AD_EFFECT_RETENTION_RATE: dist.Beta(concentration1=1., concentration0= 1.),
    _PEAK_EFFECT_DELAY:dist.HalfNormal(scale= 2.),

    # Saturation effects
    _EXPONENT: dist.Beta(concentration1=5., concentration0=1.),
    _SATURATION: dist.Beta(concentration1=5.0, concentration0=2.0),
    _HALF_MAX_EFFECTIVE_CONCENTRATION: dist.Gamma(concentration= 1.5, rate= 2.),
    _SLOPE: dist.Gamma(concentration=1.5, rate=2.),
    _HALF_MAX_EFFECTIVE_CONCENTRATION_CONSTRAINED: dist.Beta(concentration1=1., concentration0=1.),
    _SLOPE_CONSTRAINED: dist.Beta(concentration1=3., concentration0=5.),
  })


def _get_transform_prior_hyperprior_distribution(
    prior_name: str
) -> numpyro.distributions.Distribution:
  """
  Return prior distribution for media transform parameter,
  initialized using hyperprior sampling, for prior distribution parameters.
  
  Args:
    prior_name: Name of transform parameter to return
  """
  hyperprior_distributions = _get_transform_hyperprior_distributions()[prior_name]

  # Sample from hyperprior distributions
  hyperprior_samples = {
    hyperprior_name: numpyro.sample(
      name=prior_name + '_' + hyperprior_name,
      fn=distr
    ) if not isinstance(distr, float) else distr
    for hyperprior_name, distr in hyperprior_distributions.items()
  }

  # Initialize prior distribution, with hyperprior samples
  prior_fn = _get_transform_prior_distributions()[prior_name].__class__

  if (prior_fn != dist.Beta) or ('concentration' not in hyperprior_samples):
    return prior_fn(
      **hyperprior_samples
    )
  # Beta hyperpriors can be symmetric, parameterized around a concentration
  return prior_fn(
    concentration1= 9 - hyperprior_samples['concentration'],
    concentration0= 1 + hyperprior_samples['concentration'],
  )


# def create_bounds(
#   bounds: MutableMapping[str, Bound],
#   param_name: str,
#   n: int,
#   positive: bool = False
# ):
#   if bounds is None or param_name not in bounds:
#     return None, None
  
#   min_bound = 0 if positive else -1e6
  
#   return jnp.array([
#     bounds[param_name].get(i, min_bound)
#     for i in jnp.arange(n)
#   ]), jnp.array([
#     bounds[param_name].get(i, 1e6)
#     for i in jnp.arange(n)
#   ])

# Ensemble MMM - Adstock / Saturation Functions + Parameters used.
_ENSEMBLE_ADSTOCK_TRANSFORMS = immutabledict.immutabledict({
  (_LAG_WEIGHT,): media_transforms.adstock,
  (_AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY): media_transforms.carryover
})
_ENSEMBLE_SATURATION_TRANSFORMS = immutabledict.immutabledict({
  #(_EXPONENT,): media_transforms.apply_exponent_safe,
  (_SATURATION,): media_transforms.logistic_saturation,
  #(_HALF_MAX_EFFECTIVE_CONCENTRATION, _SLOPE): media_transforms.hill,
  (_HALF_MAX_EFFECTIVE_CONCENTRATION_CONSTRAINED, _SLOPE_CONSTRAINED): media_transforms.hill_constrained,
})
_ENSEMBLE_TRANSFORMS_PRIOR_NAMES = set([
  *list(chain.from_iterable(_ENSEMBLE_ADSTOCK_TRANSFORMS.keys())),
  *list(chain.from_iterable(_ENSEMBLE_SATURATION_TRANSFORMS.keys()))
])

def _get_transform_kwargs(fn: Callable) -> Set[str]:
  """ Get kwargs (with defaults) from fn"""
  from inspect import signature, _empty
  lst = []
  sig = signature(fn)
  for param in sig.parameters.values():
      if (param._default !=  _empty):
        lst.append(param.name)
  return frozenset(lst)

# Association of parameters to named kwargs, their func requires
_TRANSFORM_KWARGS = immutabledict.immutabledict({
  params: _get_transform_kwargs(fn)
  for params, fn in [
    *_ENSEMBLE_ADSTOCK_TRANSFORMS.items(),
    *_ENSEMBLE_SATURATION_TRANSFORMS.items()
  ]
})


def transform_logistic_adstock(
    media_data: jnp.ndarray,
    transform_samples: Dict[str, jnp.ndarray],
    adstock_normalise: bool = True,
    logistic_normalise: bool = True,
    **kwargs
) -> jnp.ndarray:
  """Transforms the input data with adstock then logistic saturation function

  Benefit: Requires only one parameter per media channel per transform (2)

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    transform_samples: Samples from media transform parameter distributions 
    adstock_normalise: Whether to normalise adstock transformed media so mean scale preserved
    logistic_normalise: Whether to normalise logistic transformed media so mean scale preserved
  Returns:
    The transformed media data.
  """
  lag_weight = transform_samples[_LAG_WEIGHT]
  saturation = transform_samples[_SATURATION]

  adstock = media_transforms.adstock(
    data=media_data,
    lag_weight=lag_weight,
    adstock_normalise=adstock_normalise
  )

  return media_transforms.logistic_saturation(
    data=adstock,
    saturation=saturation,
    logistic_normalise=logistic_normalise
  )

def transform_logistic_carryover(
    media_data: jnp.ndarray,
    transform_samples: Dict[str, jnp.ndarray],
    number_lags: int = 13,
    logistic_normalise: bool = True,
    **kwargs
) -> jnp.ndarray:
  """Transforms the input data with carryover fn, then logistic saturation

  Benefit: Requires only one parameter per media channel

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    transform_samples: Samples from media transform parameter distributions 
    custom_priors: The custom priors we want the model to take instead of the
      default ones. The possible names of parameters for exponential_carryover
      are "slope".

  Returns:
    The transformed media data.
  """
  peak_effect_delay = transform_samples[_PEAK_EFFECT_DELAY]
  ad_effect_retention_rate = transform_samples[_AD_EFFECT_RETENTION_RATE]
  saturation = transform_samples[_SATURATION]

  carryover = media_transforms.carryover(
      data=media_data,
      ad_effect_retention_rate=ad_effect_retention_rate,
      peak_effect_delay=peak_effect_delay,
      number_lags=number_lags)

  return media_transforms.logistic_saturation(
    data=carryover,
    saturation=saturation,
    logistic_normalise=logistic_normalise
  )

def transform_adstock(
    media_data: jnp.ndarray,
    transform_samples: Dict[str, jnp.ndarray],
    adstock_normalise: bool = True,
    **kwargs) -> jnp.ndarray:
  """Transforms the input data with the adstock function and exponent saturation.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    transform_samples: Samples from media transform parameter distributions 
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
    data=media_data,
    lag_weight=lag_weight,
    adstock_normalise=adstock_normalise
  )

  return media_transforms.apply_exponent_safe(data=adstock, exponent=exponent)

def transform_hill_constrained_adstock(
    media_data: jnp.ndarray,
    transform_samples: Dict[str, jnp.ndarray],
    hill_normalise: bool = False,
    adstock_normalise: bool = True,
    **kwargs) -> jnp.ndarray:
  """Transforms the input data with the adstock then constrained hill saturation fn.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    adstock_normalise: Whether to normalise the output values, so adstock contribution sums to 1
    saturation_normalise: Whether to normalise so 0.5 input = 0.5 output, ensure 50% linearity.

  Returns:
    The transformed media data.
  """
  half_max_effective_concentration_constrained = transform_samples[_HALF_MAX_EFFECTIVE_CONCENTRATION_CONSTRAINED]
  slope_constrained = transform_samples[_SLOPE_CONSTRAINED]
  lag_weight = transform_samples[_LAG_WEIGHT]

  if media_data.ndim == 3:
    lag_weight = jnp.expand_dims(lag_weight, axis=-1)
    half_max_effective_concentration_constrained = jnp.expand_dims(
        half_max_effective_concentration_constrained, axis=-1)
    slope_constrained = jnp.expand_dims(slope_constrained, axis=-1)

  adstock_media = media_transforms.adstock(
    data=media_data,
    lag_weight=lag_weight,
    adstock_normalise=adstock_normalise
  )

  return media_transforms.hill_constrained(
      data=adstock_media,
      half_max_effective_concentration_constrained=half_max_effective_concentration_constrained,
      slope_constrained=slope_constrained,
      hill_normalise=hill_normalise
    )

def transform_hill_constrained_carryover(
    media_data: jnp.ndarray,
    transform_samples: Dict[str, jnp.ndarray],
    hill_normalise: bool = False,
    number_lags:int = 60,
    **kwargs) -> jnp.ndarray:
  """Transforms the input data with the carryover then hill fn.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    transform_samples: Samples from media transform parameter distributions 
    adstock_normalise: Whether to normalise the output values, so adstock contribution sums to 1
    saturation_normalise: Whether to normalise so 0.5 input = 0.5 output, ensure 50% linearity.

  Returns:
    The transformed media data.
  """
  half_max_effective_concentration_constrained = transform_samples[_HALF_MAX_EFFECTIVE_CONCENTRATION_CONSTRAINED]
  slope_constrained = transform_samples[_SLOPE_CONSTRAINED]

  peak_effect_delay = transform_samples[_PEAK_EFFECT_DELAY]
  ad_effect_retention_rate = transform_samples[_AD_EFFECT_RETENTION_RATE]

  if media_data.ndim == 3:
    peak_effect_delay = jnp.expand_dims(peak_effect_delay, axis=-1)
    ad_effect_retention_rate = jnp.expand_dims(ad_effect_retention_rate, axis=-1)
    half_max_effective_concentration_constrained = jnp.expand_dims(
        half_max_effective_concentration_constrained, axis=-1)
    slope_constrained = jnp.expand_dims(slope_constrained, axis=-1)

  carryover = media_transforms.carryover(
      data=media_data,
      ad_effect_retention_rate=ad_effect_retention_rate,
      peak_effect_delay=peak_effect_delay,
      number_lags=number_lags)

  return media_transforms.hill_constrained(
      data=carryover,
      half_max_effective_concentration_constrained=half_max_effective_concentration_constrained,
      slope_constrained=slope_constrained,
      hill_normalise=hill_normalise
    )

def transform_hill_carryover(
    media_data: jnp.ndarray,
    transform_samples: Dict[str, jnp.ndarray],
    hill_normalise: bool = False,
    number_lags:int = 60,
    **kwargs) -> jnp.ndarray:
  """Transforms the input data with the carryover then hill fn.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    transform_samples: Samples from media transform parameter distributions
    adstock_normalise: Whether to normalise the output values, so adstock contribution sums to 1
    saturation_normalise: Whether to normalise so 0.5 input = 0.5 output, ensure 50% linearity.

  Returns:
    The transformed media data.
  """
  half_max_effective_concentration = transform_samples[_HALF_MAX_EFFECTIVE_CONCENTRATION]
  slope = transform_samples[_SLOPE]

  peak_effect_delay = transform_samples[_PEAK_EFFECT_DELAY]
  ad_effect_retention_rate = transform_samples[_AD_EFFECT_RETENTION_RATE]

  # Handle multi-geo model case
  if media_data.ndim == 3:
    peak_effect_delay = jnp.expand_dims(peak_effect_delay, axis=-1)
    ad_effect_retention_rate = jnp.expand_dims(ad_effect_retention_rate, axis=-1)
    half_max_effective_concentration = jnp.expand_dims(
        half_max_effective_concentration, axis=-1)
    slope = jnp.expand_dims(slope, axis=-1)

  carryover = media_transforms.carryover(
      data=media_data,
      ad_effect_retention_rate=ad_effect_retention_rate,
      peak_effect_delay=peak_effect_delay,
      number_lags=number_lags)

  return media_transforms.hill(
      data=carryover,
      half_max_effective_concentration=half_max_effective_concentration,
      slope=slope,
      hill_normalise=hill_normalise
    )


def transform_hill_adstock(
    media_data: jnp.ndarray,
    transform_samples: Dict[str, jnp.ndarray],
    hill_normalise: bool = False,
    adstock_normalise: bool = True,
    **kwargs) -> jnp.ndarray:
  """Transforms the input data with the adstock then hill fn.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    transform_samples: Samples from media transform parameter distributions
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

def transform_carryover(
    media_data: jnp.ndarray,
    transform_samples: Dict[str, jnp.ndarray],
    number_lags: int = 30,
    **kwargs
    ) -> jnp.ndarray:
  """Transforms the input data with the carryover fn then exponent saturation.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    transform_samples: Samples from media transform parameter distributions
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

def transform_ensemble_multi(
    media_data: jnp.ndarray,
    transform_samples: Dict[str, jnp.ndarray],
    adstock_normalise: bool = True,
    hill_normalise: bool = False,
    logistic_normalise: bool = False,
    **kwargs
) -> jnp.ndarray:
  """Transforms the input data with combinations of Adstock/Saturation Fn's

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models. (n, )
    transform_samples: Samples from media transform parameter distributions
  Returns:
    The transformed media data. ()
  """
  kwargs = immutabledict.immutabledict({
    **kwargs,
    'hill_normalise': hill_normalise,
    'adstock_normalise': adstock_normalise,
    'logistic_normalise': logistic_normalise
  })

  transformed_media = None
  for ad_params, ad_f in _ENSEMBLE_ADSTOCK_TRANSFORMS.items():
    adstocked_media_data = ad_f(
      media_data,
      *[transform_samples[p] for p in ad_params],
      **{k: kwargs[k] for k in _TRANSFORM_KWARGS[ad_params] if k in kwargs}
    )
    for sat_params, sat_f in _ENSEMBLE_SATURATION_TRANSFORMS.items():
      saturated_media_data = sat_f(
        adstocked_media_data,
        *[transform_samples[p] for p in sat_params],
        **{k: kwargs[k] for k in _TRANSFORM_KWARGS[sat_params] if k in kwargs}
      )

      if transformed_media is None:
        transformed_media = jnp.expand_dims(saturated_media_data, axis=0)
      else:
        transformed_media = jnp.concatenate(
          [
            transformed_media,
            jnp.expand_dims(saturated_media_data, axis=0)
          ], axis=0
        )
  return transformed_media

def transform_ensemble(
    media_data: jnp.ndarray,
    transform_samples: Dict[str, jnp.ndarray],
    **kwargs
) -> jnp.ndarray:
  """
  Transforms the input media with a weighted sum of all possible transforms.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models. ([g,] n, c)
    transform_samples: Samples from media transform parameter distributions
  Returns:
    The transformed media data ([g,] n, c).
  """
  transformed_media = transform_ensemble_multi(
    media_data,
    transform_samples,
    **kwargs
  )

  n_models = transformed_media.shape[0]
  default_priors = _get_default_priors()

  with numpyro.plate(
      name=f"{_MEDIA_TRANSFORM_WEIGHTS}_plate",
      size=n_models):
    model_weights = numpyro.sample(
      name=_MEDIA_TRANSFORM_WEIGHTS,
      fn=default_priors[_MEDIA_TRANSFORM_WEIGHTS]
    )

  model_weights = model_weights / model_weights.sum()

  return (transformed_media.T @ model_weights).T

# The different possible models (ensemble / single)
TRANSFORM_PRIORS_NAMES = immutabledict.immutabledict({
    "carryover":
        frozenset((_AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY, _EXPONENT)),
    "adstock":
        frozenset((_EXPONENT, _LAG_WEIGHT)),
    "hill_constrained_adstock":
        frozenset((_LAG_WEIGHT, _HALF_MAX_EFFECTIVE_CONCENTRATION_CONSTRAINED, _SLOPE_CONSTRAINED)),
    "hill_constrained_carryover":
        frozenset((_AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY, _HALF_MAX_EFFECTIVE_CONCENTRATION_CONSTRAINED, _SLOPE_CONSTRAINED)),
    "hill_carryover":
        frozenset((_AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY, _HALF_MAX_EFFECTIVE_CONCENTRATION, _SLOPE)),
    "hill_adstock":
        frozenset((_LAG_WEIGHT, _HALF_MAX_EFFECTIVE_CONCENTRATION, _SLOPE)),
    "logistic_carryover":
      frozenset((_SATURATION, _AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY)),
    "logistic_adstock":
        frozenset((_SATURATION, _LAG_WEIGHT)),
    "ensemble_avg": frozenset(tuple(_ENSEMBLE_TRANSFORMS_PRIOR_NAMES)),
    "ensemble": frozenset(tuple(_ENSEMBLE_TRANSFORMS_PRIOR_NAMES))
})

_NAMES_TO_MODEL_TRANSFORMS = immutabledict.immutabledict({
    "hill_constrained_carryover": transform_hill_constrained_carryover,
    "hill_constrained_adstock": transform_hill_constrained_adstock,
    "hill_carryover": transform_hill_carryover,
    "hill_adstock": transform_hill_adstock,
    "adstock": transform_adstock,
    "carryover": transform_carryover,
    "logistic_carryover": transform_logistic_carryover,
    "logistic_adstock": transform_logistic_adstock,
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
  """ Get Parameter Names required for transform function. """
  prior_list = _MODEL_TRANSFORMS_TO_PRIOR_NAMES.get(
    transform_function, _ENSEMBLE_TRANSFORMS_PRIOR_NAMES
  )

  return prior_list

def _generate_extra_features_custom_priors(
  default_priors: Dict,
  custom_priors: Dict,
  n_extra_features: int
) -> numpyro.distributions.Distribution:
  """
  Generate prior for extra features, using default and custom priors.
  All Extra features are assumed to have same distribution.

  Args:
    default_priors: the default priors for all model parameters
    custom_priors: the supplied custom priors to modelling
    n_extra_features: how many extra features we model
  Returns:
    Numpyro distribution with n_extra_features length specification
  """
  dp = default_priors[_COEF_EXTRA_FEATURES]
  cls = dp.__class__
  ef_cp = custom_priors.get(_COEF_EXTRA_FEATURES, {})

  # For scale, 
  dt = {}
  for p in cls.reparametrized_params:
    # Set to same as base distribution by default
    vals = jnp.ones(n_extra_features) * getattr(dp, p)
    for i, v in ef_cp.get(p, {}).items():
      vals = vals.at[i].set(v)
    dt[p] = vals

  # There are bounds
  if 'low' in ef_cp and 'high' in ef_cp:
    if cls == dist.HalfNormal or cls == dist.Normal:
      cls = dist.TruncatedNormal
    else:
      raise ValueError('Cannot make a bounded non-normal distribution')
    for p in ['low', 'high']:
      vals = jnp.ones(n_extra_features) * (-10 if p == 'low' else 10)
      for i, v in ef_cp.get(p, {}).items():
        vals = vals.at[i].set(v)
      dt[p] = vals

  return cls(**dt)


def _get_transform_default_priors(
    transform_hyperprior: bool,
    transform_function: TransformFunction
  ) -> Mapping[str, Prior]:
  """ Get default prior distributions for parameters used in transform_function.
  
  Args:
    transform_hyperprior: Whether prior distributions are using hyperpriors
    transform_function: Transform function to fetch parameter priors for

  Returns:
    Mapping of parameter names to their prior distributions
  """

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
    transform_function: TransformFunction,
    transform_hyperprior: bool,
    custom_priors: Dict,
    n_media_channels: int
  ) -> Mapping[str, jnp.ndarray]:
  """Sample for this transform_function, it's parameter samples.
  
  Args:
    transform_function: Function to fetch parameter samples for
    transform_hyperprior: Whether parameters use hyperpriors
    custom_priors: Manually specified channel-specific parameter priors that take precedence
    n_media_channels: Number of media channels
  """
  # Default priors that parameters are sampled from 
  transform_default_priors = _get_transform_default_priors(
    transform_hyperprior,
    transform_function
  )

  def get_sample(site_name: str, dist: numpyro.distributions.Distribution):
    # Assume matching distributions
    if site_name in custom_priors:
      # If specified custom prior is channel specific
      if isinstance(custom_priors[site_name], dict):
        cls = dist.__class__
        params = {}

        # For each hyperparameter in prior distribution
        for p in cls.reparametrized_params:
          # Set to same as base distribution by default
          vals = jnp.ones(n_media_channels) * getattr(dist, p)
          for ch_idx, prior in custom_priors[site_name].items():

            # Assert all custom priors have same distr, different params
            assert prior.__class__ == cls
            
            vals = vals.at[ch_idx].set(getattr(prior, p))
          params[p] = jnp.array(vals)
        dist = cls(**params)

      # If specified custom prior applies to all channels
      elif isinstance(custom_priors[site_name], numpyro.distributions.Distribution):
        dist = custom_priors[site_name]
      else: 
        raise ValueError(
          f'At site {site_name} Unrecognized Prior {custom_priors[site_name]} for default {dist}'
        )

    with numpyro.plate(
      name=f"{site_name}_plate",
      size=n_media_channels,
    ):
      return numpyro.sample(
          name=site_name,
          fn=dist
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
  transform_kwargs: Dict
) -> jnp.ndarray:
  """ Apply transform function to media data.
  
  Args:
    transform_function: Transform function to apply
    media_data: Media data used to train MMM (n, c)
    transform_hyperprior: Whether to sample hyperpriors for prior distributions
    custom_priors: Manually specified priors (channel specific)
    transform_kwargs: Other transform function configuration
  Returns:
    Transformed media data (n, c)
  """
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
) -> jnp.ndarray:
  """
  Calculate (s, n) target attributed to seasonal effects (DOW, DOM, DOY)
  Seasonal effects are define relative to worst possible day i.e. 0 min.

  Args:
    media_data: Media data used to train model (n, c)
    doms: (n, ) day of the month (1 .. 31) in data
    degrees_seasonality: how many cos/sin curves to combine to form  DOY seasonal effect
    frequency: daily (365), weekly (52) data
    custom_priors: Custom Prior Beliefs about model parameters
  Return:
    (s, n) target attributed to seasonal effects > 0 strictly positive
  """
  data_size = media_data.shape[0]
  n_geos = media_data.shape[2] if media_data.ndim == 3 else 1
  default_priors = _get_default_priors()

  # Time of Year Seasonality - Fourier series of `degrees_seasonality` curves
  # Cycles over `frequency` days
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
  

  # Day of Month Seasonality - Beta Distribution x Multiplier
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
    dom_series = jbeta.pdf(doms / 32, *dom_param) * dom_multiplier
    
    # DOM - min zero
    dom_series = numpyro.deterministic(
      name = 'dom_seasonality',
      value = dom_series - dom_series.min()
    )
  # Day of Week Seasonality
  if weekday_seasonality:
    with numpyro.plate(name=f"{_WEEKDAY}_plate", size=6):
      weekday = numpyro.sample(
          name=_WEEKDAY,
          fn=custom_priors.get(_WEEKDAY, default_priors[_WEEKDAY]))
    weekday = jnp.concatenate(arrays=[weekday, jnp.array([0])], axis=0)
    weekday = weekday - weekday.min()

    weekday_series = numpyro.deterministic(
      name='weekday_seasonality',
      value=weekday[jnp.arange(data_size) % 7]
    )

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

  # Total seasonality = doy + dom + dow
  total_seasonality = (
    seasonality * coef_seasonality
  )
  total_seasonality = numpyro.deterministic(
    name='year_seasonality',
    value=total_seasonality - total_seasonality.min()
  )
  if doms is not None:
    total_seasonality += dom_series
  if weekday_seasonality:
    total_seasonality += weekday_series

  return numpyro.deterministic(
    name="total_seasonality",
    value=total_seasonality
  )

def calculate_trend_effects(
  media_data: jnp.ndarray,
  custom_priors: Dict
) -> jnp.ndarray:
  """
  Calculate (s, n) target attributed to trend effects,
  learned linear and power component

  Args:
    media_data: Media data to be be used in the model.
    custom_priors: Custom Prior beliefs around model parameters
  Returns:
    (s, n) target attributed to trend effects
  """
  data_size = media_data.shape[0]
  n_geos = media_data.shape[2] if media_data.ndim == 3 else 1
  default_priors = _get_default_priors()

  # Trend for each National Model
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
    custom_priors: Dict,
    geo_shape: np.ndarray,
) -> jnp.ndarray:
  """ Calculate (s, n) target attributed to extra features.
  
  Args:
    extra_features: Extra features that partially explain target in MMM
    custom_priors: Manually specified priors for transform parameters
    geo_shape: Shape of national regions (nr,) or (,) if national model
  Returns:
    (s, n) target attributed to extra features.
  """
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
          fn = _generate_extra_features_custom_priors(
                  default_priors,
                  custom_priors,
                  extra_features.shape[-1]
          )
      )
    extra_features_effect = jnp.einsum(extra_features_einsum,
                                       extra_features,
                                       coef_extra_features)
    
  return extra_features_effect

def calculate_media_effects(
    media_data: jnp.ndarray,
    extra_features: jnp.ndarray,
    media_prior: jnp.ndarray,
    transform_function: TransformFunction,
    transform_hyperprior: bool,
    custom_priors: MutableMapping[str, Prior],
    #bounds: MutableMapping[str, Bound],
    frequency: int,
    transform_kwargs: Dict,
    target_data: jnp.ndarray,
    intercept: jnp.ndarray,
    seasonal_effects: jnp.ndarray,
    trend_effects: jnp.ndarray,
    extra_features_effects: jnp.ndarray
) -> jnp.ndarray:
  """ Calculate (s, n) target attributed to media.
  
  Args:
    media_data
    extra_features: Extra features that partially explain target in MMM
    media_prior: 
    transform_function: Transform function
    transform_hyperprior: Whether transform parameter distributions use hyperpriors
    custom_priors: Manually specified priors for media transform parameters
    frequency: 365/52 indicating daily/weekly data
    transform_kwargs: kwargs passed to media transform functions
    target_data: Observed target data
    intercept: Baseline target
    seasonal_effects: Target impact from seasonal effects (dow, dom, doy)
    trend_effects: Target impact from trend
    extra_feature_effects: Target impact from extra features
    
  Returns:
    (s, n) target attributed to extra features.
  """
  default_priors = _get_default_priors()

  n_channels = media_data.shape[1]
  n_geos = media_data.shape[2] if media_data.ndim == 3 else 1

  carryover_models = [
    transform_carryover,
    transform_logistic_carryover,
    transform_ensemble,
    transform_ensemble_multi
  ]

  # In case of daily data, number of lags should be 13*7.
  transform_kwargs = transform_kwargs or {}
  if transform_function in carryover_models and 'number_lags' not in transform_kwargs:
    transform_kwargs['number_lags'] = 13 if frequency == 52 else 180# 120

  media_transformed = apply_media_transform_function(
    transform_function,
    media_data,
    transform_hyperprior,
    custom_priors,
    transform_kwargs,
  )

  media_transformed = numpyro.deterministic(
    name="media_transformed",
    value=media_transformed
  )
  if transform_function.__name__ not in [transform_ensemble_multi.__name__]:

    with numpyro.plate(
        name="channel_media_plate",
        size=n_channels,
        dim=-2 if media_data.ndim == 3 else -1):
      coef_media = numpyro.sample(
          name="channel_coef_media" if media_data.ndim == 3 else "coef_media",
          #fn=dist.TruncatedNormal(loc=media_prior, scale=0.05, low=1e-6)
          #fn=dist.HalfNormal(scale=media_prior)
          fn=_generate_media_prior_distribution(media_prior)
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
              fn=_generate_media_prior_distribution(coef_media * normalisation_factor)
              #fn=dist.HalfNormal(scale=coef_media * normalisation_factor)
          )

    # Used in model evaluation plots to see channel contribution over time
    media_einsum = 'tc, c -> tc'
    if media_data.ndim == 3:  # For geo model's case
      media_einsum = "tcg, cg -> tcg"  # t = time, c = channel, g = geo
    channel_contribution = numpyro.deterministic(
      name = 'channel_contribution',
      value = jnp.einsum(media_einsum, media_transformed, coef_media)
    )


    # For national model's case
    media_einsum = "tc, c -> t"  # t = time, c = channel
    if media_data.ndim == 3:  # For geo model's case
      media_einsum = "tcg, cg -> tg"  # t = time, c = channel, g = geo

    media_contribution = numpyro.deterministic(
      name = 'media_contribution',
      value = jnp.einsum(media_einsum, media_transformed, coef_media)
    )

    return media_contribution

  else:

    n_models = len(_ENSEMBLE_ADSTOCK_TRANSFORMS) * len(_ENSEMBLE_SATURATION_TRANSFORMS)

    with numpyro.plate(
      name="model_media_plate",
      size=n_models,
      dim=-3 if media_data.ndim == 3 else -2
    ):
      with numpyro.plate(
          name="channel_media_plate",
          size=n_channels,
          dim=-2 if media_data.ndim == 3 else -1):
        
        # lower_bounds = jnp.ones(len(media_prior)) * 0.25
        # upper_bounds = jnp.ones(len(media_prior)) * 4.0
        # # I don't want bounds for visitor channels
        # # TODO Extra channels total may not always be 6, 
        # n_extra_channels = 6 - extra_features.shape[1]
        # upper_bounds = upper_bounds.at[-n_extra_channels:].set(10.0)
        # lower_bounds = lower_bounds.at[-n_extra_channels:].set(0.0)

        # lower_bounds, upper_bounds = create_bounds(
        #   bounds, 'coef_media', len(media_prior), positive=True
        # )
        if 'coef_media' in custom_priors:
          lower_bounds = custom_priors['coef_media']['low']
          upper_bounds = custom_priors['coef_media']['high']
        else:
          lower_bounds = upper_bounds =  None

        coef_media = numpyro.sample(
            name="channel_coef_media_models" if media_data.ndim == 3 else "coef_media_models",
            fn= dist.TruncatedNormal(
              loc=0.0,
              scale=media_prior,
              low=lower_bounds,
              high=upper_bounds
              #low=lower_bounds * media_prior,
              #high=upper_bounds * media_prior
            )
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
                fn=_generate_media_prior_distribution(coef_media * normalisation_factor)
                #fn=dist.HalfNormal(scale=coef_media * normalisation_factor)
            )

    # For national model's case
    media_einsum = "mtc, mc -> mtc"  # t = time, c = channel
    if media_data.ndim == 3:  # For geo model's case
      media_einsum = "mtcg, mcg -> mtcg"  # t = time, c = channel, g = geo

    media_contribution = jnp.einsum(media_einsum, media_transformed, coef_media)

    n_models = len(_ENSEMBLE_ADSTOCK_TRANSFORMS) * len(_ENSEMBLE_SATURATION_TRANSFORMS)

    with numpyro.plate(name=F'{_MODEL_WEIGHTS}_plate', size=n_models):
      weights = numpyro.sample(
        _MODEL_WEIGHTS + '_raw',
        fn=default_priors[_MODEL_WEIGHTS]
      )
      model_sigma = numpyro.sample(
          name=_MODEL_SIGMA,
          fn=custom_priors.get('model_sigma', default_priors[_MODEL_SIGMA])
      )
    # Credit must be split across 50% of models (min).
    weights = numpyro.deterministic(
      name=_MODEL_WEIGHTS,
      value=weights / (n_models / 2)
    )

    channel_einsum = "mtc, m -> tc"
    channel_contribution = numpyro.deterministic(
      name='channel_contribution',
      value=jnp.einsum(channel_einsum, media_contribution, weights)
    )

    model_predictions = numpyro.deterministic(
      name='submodel_mu',
      value = (
        media_contribution.sum(axis=-1) +
        intercept +
        seasonal_effects + trend_effects +
        extra_features_effects
      )
    )

    # Division by n_models to make weight of ensemble model 50%
    _ = numpyro.sample(
      name='submodel_target',
      fn=dist.Normal(
        loc=model_predictions, scale=model_sigma.reshape(-1, 1)
      ),
      obs=(jnp.repeat(
        jnp.expand_dims(
          target_data,
          axis=0
        ),
        n_models,
        axis=0
      )) if target_data is not None else None
    )

    media_effects = numpyro.deterministic(
      name = 'media_contribution',
      value = channel_contribution.sum(axis=-1)
    )

    return media_effects


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
    extra_features: Optional[jnp.array] = None,
    #bounds: Optional[Dict[str, Dict[int, Tuple[float]]]] = None
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
    transform_hyperprior: Whether media transform parameters use hyperpriors for prior distributions
    custom_priors: The custom priors we want the model to take instead of the
      default ones. See our custom_priors documentation for details about the
      API and possible options.
    transform_kwargs: Any extra keyword arguments to pass to the transform
      function. For example the adstock function can take a boolean to noramlise
      output or not.
    doms: (Optional) Day of month (1..31) of associated media data
    weekday_seasonality: In case of daily data you can estimate a weekday (6)
      parameter.
    extra_features: (Optional) Extra features data to include in the model.
  """
  default_priors = _get_default_priors()
  transform_kwargs = transform_kwargs if transform_kwargs else {}

  geo_shape = (media_data.shape[2],) if media_data.ndim == 3 else ()
  n_geos = media_data.shape[2] if media_data.ndim == 3 else 1

  # Get intercept and sigma of target
  with numpyro.plate(name=f"{_INTERCEPT}_plate", size=n_geos):
    intercept = numpyro.sample(
        name=_INTERCEPT,
        fn=custom_priors.get(_INTERCEPT, default_priors[_INTERCEPT]))

  with numpyro.plate(name=f"{_SIGMA}_plate", size=n_geos):
    sigma = numpyro.sample(
        name=_SIGMA,
        fn=custom_priors.get(_SIGMA, default_priors[_SIGMA]))

  # Get seasonal, trend, extra features, media effects
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
  if extra_features is None:
    extra_features_effects = 0
  else:
    extra_features_effects = calculate_extra_features_effects(
      extra_features,
      custom_priors,
      #bounds,
      geo_shape
    )

  media_effects = calculate_media_effects(
    media_data,
    extra_features,
    media_prior,
    transform_function,
    transform_hyperprior,
    custom_priors,
    #bounds,
    frequency,
    transform_kwargs,
    target_data,
    intercept,
    seasonal_effects,
    trend_effects,
    extra_features_effects,
  )

  prediction = (
    intercept + seasonal_effects + trend_effects + 
    media_effects
    + extra_features_effects
  )

  mu = numpyro.deterministic(name="mu", value=prediction)

  # with numpyro.plate(name=f"{_DEGREES_FREEDOM}_plate", size=n_geos):
  #   degrees_freedom = numpyro.sample(
  #     name=_DEGREES_FREEDOM,
  #     fn=custom_priors.get(_DEGREES_FREEDOM, default_priors[_DEGREES_FREEDOM])
  #   )

  # A studentT distribution is more resilient to outliers, than a normal distr
  numpyro.sample(
    name="target",
    #fn=dist.StudentT(degrees_freedom, loc=mu, scale=sigma),
    fn=dist.Normal(loc=mu, scale=sigma),
    obs=target_data
  )
