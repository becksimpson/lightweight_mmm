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
import sys
#  pylint: disable=g-import-not-at-top
if sys.version_info >= (3, 8):
  from typing import Protocol
else:
  from typing_extensions import Protocol
import functools

from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Union

import immutabledict
import jax
import jax.numpy as jnp
from jax.scipy.stats import beta as jbeta
import numpyro
from numpyro import distributions as dist
from numpyro.contrib.control_flow import cond



from lightweight_mmm import media_transforms


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
        frozenset((_SATURATION, _LAG_WEIGHT))
})

GEO_ONLY_PRIORS = frozenset((_COEF_SEASONALITY,))


def _get_default_priors() -> Mapping[str, Prior]:
  # Since JAX cannot be called before absl.app.run in tests we get default
  # priors from a function.
  return immutabledict.immutabledict({
      _INTERCEPT: dist.HalfNormal(scale=2.),
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


# def _get_transform_hyperprior_defaults() -> Mapping[str, jnp.array]:
#   return immutabledict.immutabledict({
#     _EXPONENT: jnp.array([]),
#     _LAG_WEIGHT: jnp.array([])
#   })

def _get_transform_hyperprior_distributions() -> Mapping[str, Prior]:
  return immutabledict.immutabledict({
    # For Beta Distribution
    _EXPONENT: immutabledict.immutabledict({
        'concentration1': dist.Uniform(1., 9.),
        'concentration0': dist.Uniform(1., 9.)
    }),
    # Adstock lag_weight (Beta), [0.0, 1.0], higher, more carryover
    _LAG_WEIGHT: immutabledict.immutabledict({
        'concentration1': dist.Uniform(1., 9.),
        'concentration0': dist.Uniform(1., 9.)
    }),
    # Carryover delay to peak (halfnormal)
    _PEAK_EFFECT_DELAY: immutabledict.immutabledict({
        'scale': dist.Uniform(1., 10.)
    }),
    # Exponential/hill saturation (gamma)
    _SLOPE: immutabledict.immutabledict({
        'concentration': dist.Uniform(1., 4.),
        'rate': dist.Uniform(0.1, 1.)
    }),
    # Half point most effective, gamma
    _HALF_MAX_EFFECTIVE_CONCENTRATION: immutabledict.immutabledict({
      'concentration': dist.Uniform(1., 5.),
      'rate': dist.Uniform(0.5, 5.0)
    }),
    # Retention rate of advertisement Beta
    _AD_EFFECT_RETENTION_RATE: immutabledict.immutabledict({
        'concentration1': dist.Uniform(1., 9.),
        'concentration0': dist.Uniform(1., 9.)
    }),
    # Saturation for exponential saturation
    _SATURATION: immutabledict.immutabledict({
        'concentration': dist.Uniform(1., 4.),
        'rate': dist.Uniform(0.1, 1.)
    }),
  })

#@jax.jit
# def _get_transform_default_hyperprior_priors(transform_func) -> Mapping[str, Prior]:
#   priors = _get_transform_default_priors(transform_func)
#   hyperpriors = _get_transform_default_hyperpriors()

#   def get_prior_hyperprior_sampled(prior, prior_name, transform_func):
#     # samples = {}
#     # for hyper_name, hyper_dist in hyperpriors[prior_name].items():
#     #   _ = numpyro.sample(
#     #       name=transform_func + '_' + prior_name + '_' + hyper_name,
#     #       fn=hyper_dist
#     #   )
#     #   samples[hyper_name] = _

#     # return prior.__class__(
#     #   **samples
#     # )
  
#     return prior.__class__(
#       **{
#         hyper_name: numpyro.sample(
#           name=transform_func + '_' + prior_name + '_' + hyper_name,
#           fn=hyper_dist
#         )
#         for hyper_name, hyper_dist in hyperpriors[prior_name].items()
#       }
#     )
#   return {
#     prior_name: get_prior_hyperprior_sampled(prior, prior_name, transform_func)
#     for prior_name, prior in priors.items()
#   }

#   return {
#     transform_func: {
#       prior_name: get_prior_hyperprior_sampled(prior, prior_name, transform_func)
#       for prior_name, prior in transform_priors.items()
#     }
#     for transform_func, transform_priors in priors.items()
#   }#[prior_name_chosen]

def get_hyperprior_values(prior):
  hyperprior = _get_transform_default_hyperpriors()[prior]
  #print(hyperprior)
  return jnp.array([
    numpyro.sample(
      name=prior + '_' + hyp_p,
      fn=dist
    )
    for hyp_p, dist in hyperprior.items()
  ])

def get_transform_hyperpriors(
    prior, transform_hyperprior, **default_hyperpriors
):
  return dict(zip(
    default_hyperpriors.keys(),
    jnp.where(
      transform_hyperprior,
      get_hyperprior_values(prior),
      jnp.array(list(default_hyperpriors.values()))
    )
  ))

  


def _get_transform_default_priors(transform_hyperprior) -> Mapping[str, Prior]:

  # Generate hyperprior distribution samples for all possible hyper-priors.
  hyperprior_distributions = _get_transform_hyperprior_distributions()
  hyperprior_samples = {
    prior_name: {
      hyperprior_name: numpyro.sample(
        name=prior_name + '_' + hyperprior_name,
        fn=distr
      )
      for hyperprior_name, distr in distrs.items()
    }
    for prior_name, distrs in hyperprior_distributions.items()
  }

  return (#immutabledict.immutabledict(
    {
      "carryover":
          #immutabledict.immutabledict
          ({
              _AD_EFFECT_RETENTION_RATE:
                  dist.Beta(concentration1=1., concentration0=1.),
              _PEAK_EFFECT_DELAY:
                  dist.HalfNormal(scale=2.),
              _EXPONENT:
                  dist.Beta(concentration1=9., concentration0=1.)
          }),
      "adstock":
          #immutabledict.immutabledict
          ({
              _EXPONENT: dist.Beta(concentration1=9., concentration0=1.),
              _LAG_WEIGHT: dist.Beta(concentration1=2., concentration0=1.)
          }),
      "hill_adstock":
          #immutabledict.immutabledict
          ({
              _LAG_WEIGHT:
                  dist.Beta(concentration1=2., concentration0=1.),
              _HALF_MAX_EFFECTIVE_CONCENTRATION:
                  dist.Gamma(concentration=1., rate=1.),
              _SLOPE:
                  dist.Gamma(concentration=1., rate=1.)
          }),
      "exponential_carryover": 
          #immutabledict.immutabledict(
          {
              _SATURATION:
                  dist.Gamma(concentration=1., rate=4.),
          _AD_EFFECT_RETENTION_RATE:
              dist.Beta(concentration1=1., concentration0=1.),
          _PEAK_EFFECT_DELAY:
              dist.HalfNormal(scale=2.),
          }
          #)
          ,
        "exponential_adstock":
          #immutabledict.immutabledict(
          {
            _SATURATION: cond(
              transform_hyperprior,
              lambda _: dist.Gamma(*jnp.array(list(hyperprior_samples[_SATURATION].values()))),
              lambda _: dist.Gamma(concentration=1., rate=1.),
              None
            ),
            # Strongest assumption no saturatio, linear, near 0
            # _SATURATION:dist.Gamma(
            #   #**dict(zip(
            #     #['concentration', 'rate'],
            #     *jnp.where(
            #       transform_hyperprior,
            #       jnp.array(list(hyperprior_samples[_SATURATION].values())),
            #       jnp.array([1., 1.]) #contentation1, concentration0
            #     #))
            #   )),
              #dist.Gamma(**get_transform_hyperpriors(_SLOPE, transform_hyperprior, concentration=1., rate=1.)),
            # Strongest assumption no lag effect, near 1.
            # concentration1 is alpha
            _LAG_WEIGHT:
              dist.Beta(
                # **dict(zip(
                # ['concentration1', 'concentration0'],
                *jnp.where(
                  transform_hyperprior,
                  jnp.array(list(hyperprior_samples[_LAG_WEIGHT].values())), #jnp.array([list(
                  jnp.array([1., 3.]) #contentation1, concentration0
                ))
                #**get_transform_hyperpriors(_LAG_WEIGHT, transform_hyperprior, concentration1=1.,concentration0=3.)
              # )
              
            }
            #)
  })

#@functools.partial(jax.jit, static_argnums=[0]
# #transform_hyperprior:bool)
#@jax.jit
# def _get_transform_default_priors() -> Mapping[str, Prior]:
#   # Since JAX cannot be called before absl.app.run in tests we get default
#   # priors from a function.
#   return (#immutabledict.immutabledict(
#     {
#       "carryover":
#           #immutabledict.immutabledict
#           ({
#               _AD_EFFECT_RETENTION_RATE:
#                   dist.Beta(concentration1=1., concentration0=1.),
#               _PEAK_EFFECT_DELAY:
#                   dist.HalfNormal(scale=2.),
#               _EXPONENT:
#                   dist.Beta(concentration1=9., concentration0=1.)
#           }),
#       "adstock":
#           #immutabledict.immutabledict
#           ({
#               _EXPONENT: dist.Beta(concentration1=9., concentration0=1.),
#               _LAG_WEIGHT: dist.Beta(concentration1=2., concentration0=1.)
#           }),
#       "hill_adstock":
#           #immutabledict.immutabledict
#           ({
#               _LAG_WEIGHT:
#                   dist.Beta(concentration1=2., concentration0=1.),
#               _HALF_MAX_EFFECTIVE_CONCENTRATION:
#                   dist.Gamma(concentration=1., rate=1.),
#               _SLOPE:
#                   dist.Gamma(concentration=1., rate=1.)
#           }),
#       "exponential_carryover": 
#           #immutabledict.immutabledict(
#           {
#               _SLOPE:
#                   dist.Gamma(concentration=1., rate=4.),
#           _AD_EFFECT_RETENTION_RATE:
#               dist.Beta(concentration1=1., concentration0=1.),
#           _PEAK_EFFECT_DELAY:
#               dist.HalfNormal(scale=2.),
#           }
#           #)
#           ,
#         "exponential_adstock": 
#           #immutabledict.immutabledict(
#           {
#             # Strongest assumption no saturatio, linear, near 0
#             _SLOPE:
#               dist.Gamma(concentration=1., rate=1.),
#               # numpyro.distributions.HalfNormal(
#               #   scale=1.0
#               # ),
#             # Strongest assumption no lag effect, near 1.
#             # concentration1 is alpha
#             _LAG_WEIGHT:
#               dist.Beta(
#                 concentration1=1.,
#                 concentration0=3.
#               )
#             }
#             #)
#   })


#@functools.partial(jax.jit, static_argnums=[1])
def transform_exponential_adstock(
                    media_data: jnp.ndarray,
                    transform_hyperprior: bool,
                    #default_priors: MutableMapping[str, Prior],
                    custom_priors: MutableMapping[str, Prior],
                    normalise: bool = True
) -> jnp.ndarray:
  """Transforms the input data with exponetial saturation function and carryover

  Benefit: Requires only one parameter per media channel

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. The possible names of parameters for exponential_adstock
      are "slope".

  Returns:
    The transformed media data.
  """
  # transform_default_priors = cond(
  #   transform_hyperprior,
  #   _get_transform_default_hyperprior_priors,
  #   _get_transform_default_priors,
  #   'exponential_adstock'
  # )
  #transform_default_priors = _get_transform_default_hyperprior_priors()["exponential_adstock"]
  #transform_default_priors = _get_transform_default_priors()["exponential_adstock"]
  # if transform_hyperprior:
  #   transform_default_priors = _get_transform_default_hyperprior_priors('exponential_adstock')#["exponential_adstock"]
  # else:
  #   transform_default_priors = _get_transform_default_priors('exponential_adstock')#["exponential_adstock"]

  transform_priors = _get_transform_default_priors(transform_hyperprior)['exponential_adstock']

  # transform_default_priors = jax.lax.cond(
  #   transform_hyperprior,
  #   lambda _: _get_transform_default_hyperprior_priors('exponential_adstock'),
  #   lambda _: _get_transform_default_priors('exponential_adstock'),
  #   None
  # )
  with numpyro.plate(name=f"{_LAG_WEIGHT}_plate",
                     size=media_data.shape[1]):
    lag_weight = numpyro.sample(
        name=_LAG_WEIGHT,
        fn=custom_priors.get(_LAG_WEIGHT,               
            transform_priors[_LAG_WEIGHT]
      )
    )

  #transform_default_priors = default_priors["exponential_adstock"]
  # with numpyro.plate(name=f"{_LAG_WEIGHT}_plate",
  #                    size=media_data.shape[1]):
  #   lag_weight = jnp.where(
  #     transform_hyperprior,
  #     numpyro.sample(
  #       name=_LAG_WEIGHT + 'hyper',
  #       fn=custom_priors.get(_LAG_WEIGHT,               
  #           hyperprior_priors[_LAG_WEIGHT]
  #     )),
  #     numpyro.sample(
  #       name=_LAG_WEIGHT,
  #       fn=custom_priors.get(_LAG_WEIGHT,
  #             transform_priors[_LAG_WEIGHT]
  #       )
  #     )
  #   )
    # lag_weight = numpyro.sample(
    #     name=_LAG_WEIGHT,
    #     fn=custom_priors.get(_LAG_WEIGHT,
    #         #transform_default_priors[_LAG_WEIGHT]                 
    #         jnp.where(
    #           transform_hyperprior,
    #           hyperprior_priors[_LAG_WEIGHT],
    #           priors[_LAG_WEIGHT]
    #         )),
    #     # Would need to be implemented via truncated distribution
    #     #constraint=numpyro.distributions.constraints.positive
    # )
  adstock = media_transforms.adstock(
      data=media_data, lag_weight=lag_weight, normalise=normalise)

  with numpyro.plate(name=f"{_SATURATION}_plate", size=media_data.shape[1]):

    slopes = numpyro.sample(
        name=_SATURATION,
        fn=custom_priors.get(_SATURATION,               
          transform_priors[_SATURATION]
      )
    )
    # slopes = jnp.where(
    #   transform_hyperprior,
    #   numpyro.sample(
    #     name=_SLOPE + 'hyper',
    #     fn=custom_priors.get(_SLOPE,               
    #         hyperprior_priors[_SLOPE]
    #   )),
    #   numpyro.sample(
    #     name=_SLOPE,
    #     fn=custom_priors.get(_SLOPE,
    #           transform_priors[_SLOPE]
    #     )
    #   )
    # )
    # slopes = numpyro.sample(
    #   name=_SLOPE,
    #   fn=custom_priors.get(_SLOPE,
    #       jnp.where(
    #         transform_hyperprior,
    #         hyperprior_priors[_SLOPE],
    #         priors[_SLOPE]
    #       ),
    #       #transform_default_priors[_SLOPE]
    #   )
    # )
  return media_transforms.exponential_saturation(
    data=adstock, slope=slopes
  )

def transform_exponential_carryover(
                    media_data: jnp.ndarray,
                    #default_priors: MutableMapping[str, Prior],
                    transform_hyperprior:bool,
                    custom_priors: MutableMapping[str, Prior],
                    #normalise: bool = True
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
  default_priors = (
    _get_transform_default_hyperprior_priors() 
    if transform_hyperprior
    else _get_transform_default_priors()
  )
  transform_default_priors = default_priors["exponential_carryover"]

  with numpyro.plate(name=f"{_AD_EFFECT_RETENTION_RATE}_plate",
                     size=media_data.shape[1]):
    ad_effect_retention_rate = numpyro.sample(
        name=_AD_EFFECT_RETENTION_RATE,
        fn=custom_priors.get(
            _AD_EFFECT_RETENTION_RATE,
            transform_default_priors[_AD_EFFECT_RETENTION_RATE]))

  with numpyro.plate(name=f"{_PEAK_EFFECT_DELAY}_plate",
                    size=media_data.shape[1]):
    peak_effect_delay = numpyro.sample(
        name=_PEAK_EFFECT_DELAY,
        fn=custom_priors.get(
            _PEAK_EFFECT_DELAY, transform_default_priors[_PEAK_EFFECT_DELAY]))

  carryover = media_transforms.carryover(
      data=media_data,
      ad_effect_retention_rate=ad_effect_retention_rate,
      peak_effect_delay=peak_effect_delay,
      number_lags=28)

  with numpyro.plate(name=f"{_SATURATION}_plate", size=media_data.shape[1]):
    slopes = numpyro.sample(
      name=_SATURATION,
      fn=custom_priors.get(_SATURATION, transform_default_priors[_SATURATION])
    )
  return media_transforms.exponential_saturation(
    data=carryover, slope=slopes
  )


def transform_adstock(media_data: jnp.ndarray,
                      transform_hyperprior:bool,
                      #default_priors: MutableMapping[str, Prior],
                      custom_priors: MutableMapping[str, Prior],
                      normalise: bool = True) -> jnp.ndarray:
  """Transforms the input data with the adstock function and exponent.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. The possible names of parameters for adstock and exponent
      are "lag_weight" and "exponent".
    normalise: Whether to normalise the output values.

  Returns:
    The transformed media data.
  """
  default_priors = (
    _get_transform_default_hyperprior_priors() 
    if transform_hyperprior
    else _get_transform_default_priors()
  )
  transform_default_priors = default_priors["adstock"]
  with numpyro.plate(name=f"{_LAG_WEIGHT}_plate",
                     size=media_data.shape[1]):
    lag_weight = numpyro.sample(
        name=_LAG_WEIGHT,
        fn=custom_priors.get(_LAG_WEIGHT,
                             transform_default_priors[_LAG_WEIGHT]))

  with numpyro.plate(name=f"{_EXPONENT}_plate",
                     size=media_data.shape[1]):
    exponent = numpyro.sample(
        name=_EXPONENT,
        fn=custom_priors.get(_EXPONENT,
                             transform_default_priors[_EXPONENT]))

  if media_data.ndim == 3:
    lag_weight = jnp.expand_dims(lag_weight, axis=-1)
    exponent = jnp.expand_dims(exponent, axis=-1)

  adstock = media_transforms.adstock(
      data=media_data, lag_weight=lag_weight, normalise=normalise)

  n = media_transforms.apply_exponent_safe(data=adstock, exponent=exponent)

  return n / n.sum(axis=0) * adstock.sum(axis=0)


def transform_hill_adstock(media_data: jnp.ndarray,
                           transform_hyperprior:bool, 
                           #default_priors: MutableMapping[str, Prior],
                           custom_priors: MutableMapping[str, Prior],
                           normalise: bool = True) -> jnp.ndarray:
  """Transforms the input data with the adstock and hill functions.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. The possible names of parameters for hill_adstock and
      exponent are "lag_weight", "half_max_effective_concentration" and "slope".
    normalise: Whether to normalise the output values.

  Returns:
    The transformed media data.
  """
  default_priors = (
    _get_transform_default_hyperprior_priors() 
    if transform_hyperprior
    else _get_transform_default_priors()
  )
  transform_default_priors = default_priors["hill_adstock"]
  with numpyro.plate(name=f"{_LAG_WEIGHT}_plate",
                     size=media_data.shape[1]):
    lag_weight = numpyro.sample(
        name=_LAG_WEIGHT,
        fn=custom_priors.get(_LAG_WEIGHT,
                             transform_default_priors[_LAG_WEIGHT]))

  with numpyro.plate(name=f"{_HALF_MAX_EFFECTIVE_CONCENTRATION}_plate",
                     size=media_data.shape[1]):
    half_max_effective_concentration = numpyro.sample(
        name=_HALF_MAX_EFFECTIVE_CONCENTRATION,
        fn=custom_priors.get(
            _HALF_MAX_EFFECTIVE_CONCENTRATION,
            transform_default_priors[_HALF_MAX_EFFECTIVE_CONCENTRATION]))

  with numpyro.plate(name=f"{_SLOPE}_plate",
                     size=media_data.shape[1]):
    slope = numpyro.sample(
        name=_SLOPE,
        fn=custom_priors.get(_SLOPE, transform_default_priors[_SLOPE]))

  if media_data.ndim == 3:
    lag_weight = jnp.expand_dims(lag_weight, axis=-1)
    half_max_effective_concentration = jnp.expand_dims(
        half_max_effective_concentration, axis=-1)
    slope = jnp.expand_dims(slope, axis=-1)

  return media_transforms.hill(
      data=media_transforms.adstock(
          data=media_data, lag_weight=lag_weight, normalise=normalise),
      half_max_effective_concentration=half_max_effective_concentration,
      slope=slope)



def transform_carryover(media_data: jnp.ndarray,
                        default_priors: MutableMapping[str, Prior],
                        custom_priors: MutableMapping[str, Prior],
                        number_lags: int = 13,
                        exponent: float = None,
                        peak_effect_delay:float= None
                        ) -> jnp.ndarray:
  """Transforms the input data with the carryover function and exponent.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. The possible names of parameters for carryover and exponent
      are "ad_effect_retention_rate_plate", "peak_effect_delay_plate" and
      "exponent".
    number_lags: Number of lags for the carryover function.

  Returns:
    The transformed media data.
  """
  transform_default_priors = default_priors["carryover"]
  with numpyro.plate(name=f"{_AD_EFFECT_RETENTION_RATE}_plate",
                     size=media_data.shape[1]):
    ad_effect_retention_rate = numpyro.sample(
        name=_AD_EFFECT_RETENTION_RATE,
        fn=custom_priors.get(
            _AD_EFFECT_RETENTION_RATE,
            transform_default_priors[_AD_EFFECT_RETENTION_RATE]))

  if peak_effect_delay is None:
    with numpyro.plate(name=f"{_PEAK_EFFECT_DELAY}_plate",
                      size=media_data.shape[1]):
      peak_effect_delay = numpyro.sample(
          name=_PEAK_EFFECT_DELAY,
          fn=custom_priors.get(
              _PEAK_EFFECT_DELAY, transform_default_priors[_PEAK_EFFECT_DELAY]))
  else:
    peak_effect_delay = jnp.ones(media_data.shape[1]) * peak_effect_delay

  if exponent is None:
    with numpyro.plate(name=f"{_EXPONENT}_plate",
                      size=media_data.shape[1]):
      exponent = numpyro.sample(
          name=_EXPONENT,
          fn=custom_priors.get(_EXPONENT,
                              transform_default_priors[_EXPONENT]))
  else:
    exponent = jnp.ones(media_data.shape[1]) * exponent
  
  carryover = media_transforms.carryover(
      data=media_data,
      ad_effect_retention_rate=ad_effect_retention_rate,
      peak_effect_delay=peak_effect_delay,
      number_lags=number_lags)

  if media_data.ndim == 3:
    exponent = jnp.expand_dims(exponent, axis=-1)
  return media_transforms.apply_exponent_safe(data=carryover, exponent=exponent)


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
  #default_transform_priors = transform_prior_function()

  # #@functools.partial(jax.jit, static_argnums=[0])
  # def get_default_transform_priors(transform_hyperprior):
  #   return cond(
  #     transform_hyperprior,
  #     _get_transform_default_hyperprior_priors,
  #     _get_transform_default_priors,
  #     None
  #   )
  #   # return (
  #   #   _get_transform_default_priors()
  #   #   if not transform_hyperprior
  #   #   else _get_transform_default_hyperprior_priors()
  #   # )
  # default_transform_priors = get_default_transform_priors(transform_hyperprior)

  data_size = media_data.shape[0]
  n_channels = media_data.shape[1]
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

  # TODO(): Force all geos to have the same trend sign.
  with numpyro.plate(name=f"{_COEF_TREND}_plate", size=n_geos):
    coef_trend = numpyro.sample(
        name=_COEF_TREND,
        fn=custom_priors.get(_COEF_TREND, default_priors[_COEF_TREND]))

  expo_trend = numpyro.sample(
      name=_EXPO_TREND,
      fn=custom_priors.get(
          _EXPO_TREND, default_priors[_EXPO_TREND]))

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

  with numpyro.plate(name=f"{_GAMMA_SEASONALITY}_sin_cos_plate", size=2):
    with numpyro.plate(name=f"{_GAMMA_SEASONALITY}_plate",
                       size=degrees_seasonality):
      gamma_seasonality = numpyro.sample(
          name=_GAMMA_SEASONALITY,
          fn=custom_priors.get(
              _GAMMA_SEASONALITY, default_priors[_GAMMA_SEASONALITY]))

  if weekday_seasonality:
    with numpyro.plate(name=f"{_WEEKDAY}_plate", size=6):
      weekday = numpyro.sample(
          name=_WEEKDAY,
          fn=custom_priors.get(_WEEKDAY, default_priors[_WEEKDAY]))
    weekday = jnp.concatenate(arrays=[weekday, jnp.array([0])], axis=0)
    weekday_series = weekday[jnp.arange(data_size) % 7]
    # In case of daily data, number of lags should be 13*7.
    if transform_function == "carryover" and transform_kwargs and "number_lags" not in transform_kwargs:
      transform_kwargs["number_lags"] = 13 * 7
    elif transform_function == "carryover" and not transform_kwargs:
      transform_kwargs = {"number_lags": 13 * 7}

  media_transformed = numpyro.deterministic(
      name="media_transformed",
      value=transform_function(media_data,
                               transform_hyperprior=transform_hyperprior,
                               custom_priors=custom_priors,
                               **transform_kwargs if transform_kwargs else {}))
  seasonality = media_transforms.calculate_seasonality(
      number_periods=data_size,
      degrees=degrees_seasonality,
      frequency=frequency,
      gamma_seasonality=gamma_seasonality)
  # For national model's case
  trend = jnp.arange(data_size)
  media_einsum = "tc, c -> t"  # t = time, c = channel
  coef_seasonality = 1

  # TODO(): Add conversion of prior for HalfNormal distribution.
  if media_data.ndim == 3:  # For geo model's case
    trend = jnp.expand_dims(trend, axis=-1)
    seasonality = jnp.expand_dims(seasonality, axis=-1)
    media_einsum = "tcg, cg -> tg"  # t = time, c = channel, g = geo
    if weekday_seasonality:
      weekday_series = jnp.expand_dims(weekday_series, axis=-1)
    with numpyro.plate(name="seasonality_plate", size=n_geos):
      coef_seasonality = numpyro.sample(
          name=_COEF_SEASONALITY,
          fn=custom_priors.get(
              _COEF_SEASONALITY, default_priors[_COEF_SEASONALITY]))
  # expo_trend is B(1, 1) so that the exponent on time is in [.5, 1.5].
  prediction = (
      intercept + coef_trend * trend ** expo_trend +
      seasonality * coef_seasonality +
      jnp.einsum(media_einsum, media_transformed, coef_media))
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
    prediction += extra_features_effect

  # Day of month effect
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
    dom_contribs = jbeta.pdf(doms / 32, *dom_param) * dom_multiplier# (1.0 + dom_multiplier)
    prediction += dom_contribs

  if weekday_seasonality:
    prediction += weekday_series
  mu = numpyro.deterministic(name="mu", value=prediction)

  numpyro.sample(
      name="target", fn=dist.Normal(loc=mu, scale=sigma), obs=target_data)
