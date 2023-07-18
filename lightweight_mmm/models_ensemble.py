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
from itertools import chain
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

_HYPERPRIOR_PRIOR_TRANSFORM_DISTRIBUTIONS = immutabledict.immutabledict({
  _SATURATION: dist.HalfNormal, #dist.Gamma,
  _AD_EFFECT_RETENTION_RATE: dist.Beta,
  _HALF_MAX_EFFECTIVE_CONCENTRATION: dist.Gamma,
  _SLOPE: dist.Gamma, 
  _PEAK_EFFECT_DELAY: dist.HalfNormal,
  _LAG_WEIGHT: dist.Beta,
  _EXPONENT: dist.Beta
})


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

#def _get_hyperparameter_hyperprior_distributions"
  # return immutabledict.immutabledict({
  #   # Range 1..5, highest prob 2
  #   _DEGREE_SEASONALITY: dist.Binomial(total_count=5, probs=0.4)
  # })

def _get_transform_hyperprior_distributions() -> Mapping[str, Mapping[str, Union[float, Prior]]]:
  return immutabledict.immutabledict({
    # For Beta Distribution
    _EXPONENT: immutabledict.immutabledict({
        #'concentration1': dist.Uniform(1., 9.),
        #'concentration0': dist.Uniform(1., 9.),
        'concentration': dist.TruncatedNormal(0., 3., low=0.0, high=8.0)
        #'concentration': dist.Uniform(0., 8.),
        #'concentration': dist.HalfNormal(3., constraint=dist.constraints.less_than(8.0))
        # TODO CONSTRAINED HALFNORMAL DISTRIBUTION
        # 'concentration': dist.TruncatedDistribution(
        #   dist.HalfNormal(3.),
        #   high=8.0
        # )
        
    }),
    # Adstock lag_weight (Beta), [0.0, 1.0], higher, more carryover
    _LAG_WEIGHT: immutabledict.immutabledict({
        #'concentration1': dist.Uniform(1., 9.),
        #'concentration0': dist.Uniform(1., 9.)
        'concentration': dist.Uniform(0., 8.),
    }),
    # Carryover delay to peak (halfnormal)
    _PEAK_EFFECT_DELAY: immutabledict.immutabledict({
        #'scale': dist.Uniform(1., 10.)
        # Median 1.6, <1 27%, longtail
        'scale': dist.LogNormal(0.5, 0.8)
    }),
    # hill saturation (gamma) create range 0.1 -> 2/3ish, 0.1 -> 1.0 peak
    _SLOPE: immutabledict.immutabledict({
        'concentration': dist.Uniform(1.5, 3.),
        #'rate': dist.Uniform(low=2.0, high=2.0)
        'rate': 2.0 # Fixed to contrain hyperparameter distribution to appropriate range
    }),
    # Half point most effective, gamma
    # Create range 0.5 -> 2 (Half --> double mean contribution)
    _HALF_MAX_EFFECTIVE_CONCENTRATION: immutabledict.immutabledict({
      'concentration': dist.Uniform(2., 5.),
      'rate': 2.0
      #'rate': dist.Uniform(low=2.0, high=2.0)
    }),
    # Retention rate of advertisement Beta
    _AD_EFFECT_RETENTION_RATE: immutabledict.immutabledict({
        #'concentration1': dist.Uniform(1., 9.),
        #'concentration0': dist.Uniform(1., 9.),
        'concentration': dist.Uniform(0., 8.),
    }),
    # Saturation for exponential saturation
    _SATURATION: immutabledict.immutabledict({
        #'concentration': dist.Uniform(1., 4.),
        #'rate': dist.Uniform(0.1, 1.)
        # 1.34 mean --> HalfNormal(1.34)
        'scale': dist.LogNormal(loc=0.3, scale=0.3)
    }),
  })

def _get_transform_prior_distributions() -> Mapping[str, Prior]:
  return immutabledict.immutabledict({
    # Strongest assumption no lag effect, near 1.
    # concentration1 is alpha
    _LAG_WEIGHT: dist.Beta(concentration1= 1., concentration0= 3.),
    _AD_EFFECT_RETENTION_RATE: dist.Beta(concentration1=1., concentration0= 1.),
    _PEAK_EFFECT_DELAY:dist.HalfNormal(scale= 2.),

    # Saturation effects
    _EXPONENT: dist.Beta(concentration1=9., concentration0=1.),
    _SATURATION: dist.HalfNormal(scale= 2.),
    _HALF_MAX_EFFECTIVE_CONCENTRATION: dist.Gamma(concentration= 1., rate= 1.),
    _SLOPE: dist.Gamma(concentration= 1., rate= 1.)
  })


def _get_transform_default_priors(transform_hyperprior) -> Mapping[str, Prior]:

  # # Prior distributions considered
  # transform_prior_lists = _get_transform_default_priors_lists()

  # Generate hyperprior distribution samples for all possible hyper-priors.
  prior_distributions = _get_transform_prior_distributions()
  hyperprior_distributions = _get_transform_hyperprior_distributions()
  hyperprior_samples = {
    prior_name: {
      # hyperprior_name: numpyro.sample(
      #   name=prior_name + '_' + hyperprior_name,
      #   fn=distr
      # )
      hyperprior_name: numpyro.sample(
        name=prior_name + '_' + hyperprior_name,
        fn=distr
      ) if not isinstance(distr, float) else distr
      # hyperprior_name: jax.lax.cond(
      #   isinstance(distr, float),
      #   lambda _: numpyro.deterministic(prior_name + '_' + hyperprior_name + '_' + 'fixed', distr),
      #   lambda _: numpyro.sample(
      #     name=prior_name + '_' + hyperprior_name,
      #     fn=distr
      #   ),
      #   None
      # )
      for hyperprior_name, distr in distrs.items()
    }
    for prior_name, distrs in hyperprior_distributions.items()
  }

  def get_prior(prior_name):
    """ We don't insist on prior_defaults sharing the same distribution as hyperpriors"""
    fn = _HYPERPRIOR_PRIOR_TRANSFORM_DISTRIBUTIONS[prior_name]
    return cond(
        transform_hyperprior,
        lambda _: (
          fn(**hyperprior_samples[prior_name])
          if fn != dist.Beta
          else fn(
            concentration1= 9 - hyperprior_samples[prior_name]['concentration'],
            concentration0= 1 + hyperprior_samples[prior_name]['concentration'],
          )
        ),
        # Use default prior
        lambda _: prior_distributions[prior_name],
        None
      )
  
  # return immutabledict.immutabledict({
  #     "carryover": immutabledict.immutabledict
  #       ({
  #         get_prior(param,
  #           transform_prior_lists[param][
  #             prior_indexes.get('carryover', {}).get(param, 0)
  #           ]
  #         )
  #         for param in [_AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY, _EXPONENT]
  #       })
  # })

  return immutabledict.immutabledict({
    prior_name: get_prior(prior_name)
    for prior_name in list(_get_transform_prior_distributions().keys())
  })

  return immutabledict.immutabledict({
    # Strongest assumption no lag effect, near 1.
    # concentration1 is alpha
    _LAG_WEIGHT: get_prior(_LAG_WEIGHT, 
        dist.Beta(concentration1= 1., concentration0= 3.)
    ),
    _AD_EFFECT_RETENTION_RATE: get_prior(_AD_EFFECT_RETENTION_RATE,
      dist.Beta(concentration1=1., concentration0= 1.)
    ),
    _PEAK_EFFECT_DELAY: get_prior(_PEAK_EFFECT_DELAY,
      dist.HalfNormal(scale= 2.)
    ),

    # Saturation effects
    _EXPONENT: get_prior(_EXPONENT,
      #transform_priors_lsits[_EXPONENT]
      dist.Beta(concentration1=9., concentration0=1.)
    ),
    # Strongest assumption no saturatio, linear, near 0
    _SATURATION: get_prior(_SATURATION, 
      #{'concentration': 1., 'rate': 1.}
      dist.HalfNormal(scale= 2.)
    ),
    _HALF_MAX_EFFECTIVE_CONCENTRATION:get_prior(_HALF_MAX_EFFECTIVE_CONCENTRATION,
      dist.Gamma(concentration= 1., rate= 1.)
    ),
    _SLOPE: get_prior(_SLOPE, dist.Gamma(concentration= 1., rate= 1.))
  })

  return immutabledict.immutabledict({
      "carryover": immutabledict.immutabledict
          ({
            _AD_EFFECT_RETENTION_RATE: get_prior(
              _AD_EFFECT_RETENTION_RATE,
              dist.Beta(concentration1=1., concentration0=1.)
              #transform_priors_lists[_AD_EFFECT_RETENTION_RATE]
            ),
            _PEAK_EFFECT_DELAY: get_prior(_PEAK_EFFECT_DELAY,
              dist.HalfNormal(scale=2.)
              #transform_priors_lists[_PEAK_EFFECT_DELAY][]
            ),
            _EXPONENT: get_prior(_EXPONENT,
              #transform_priors_lsits[_EXPONENT]
              dist.Beta(concentration1=9., concentration0=1.)
            )
          }),
      "adstock":immutabledict.immutabledict
        ({
          _EXPONENT: get_prior(_EXPONENT,
            dist.Beta(concentration1=9., concentration0=1.)),
          _LAG_WEIGHT: get_prior(_LAG_WEIGHT,
            dist.Beta(concentration1=2., concentration0=1.))
        }),
      "hill_adstock":
          immutabledict.immutabledict({
              _LAG_WEIGHT: get_prior(_LAG_WEIGHT,
                dist.Beta(concentration1= 2., concentration0= 1.)
              ),
              _HALF_MAX_EFFECTIVE_CONCENTRATION:get_prior(_HALF_MAX_EFFECTIVE_CONCENTRATION,
                dist.Gamma(concentration= 1., rate= 1.)
              ),
              _SLOPE: get_prior(_SLOPE, dist.Gamma(concentration= 1., rate= 1.))
          }),
      "exponential_carryover": 
          immutabledict.immutabledict({
            _SATURATION: get_prior(_SATURATION,
              dist.HalfNormal(scale=2.)
            ),
            _AD_EFFECT_RETENTION_RATE: get_prior(_AD_EFFECT_RETENTION_RATE,
              dist.Beta(concentration1=1., concentration0= 1.)
            ),
            _PEAK_EFFECT_DELAY: get_prior(_PEAK_EFFECT_DELAY,
              dist.HalfNormal(scale= 2.)
            )
      }),
      "exponential_adstock":
        immutabledict.immutabledict({
          # Strongest assumption no saturatio, linear, near 0
          _SATURATION: get_prior(_SATURATION, 
            #{'concentration': 1., 'rate': 1.}
            dist.HalfNormal(scale= 2.)
          ),
          # Strongest assumption no lag effect, near 1.
          # concentration1 is alpha
          _LAG_WEIGHT: get_prior(_LAG_WEIGHT, 
              dist.Beta(concentration1= 1., concentration0= 3.)
          )
        })
  })


# DEFUNCT
def _get_transform_default_priors_lists():
  """ First in list is default. """
  return immutabledict.immutabledict({
    _SATURATION: [
      numpyro.distributions.HalfNormal(2.0),
      numpyro.distributions.HalfNormal(2.5),
      numpyro.distributions.HalfNormal(1.5)
    ],
    _LAG_WEIGHT: [
      numpyro.distributions.Beta(concentration1=1., concentration0=3.),
      numpyro.distributions.Beta(concentration1=1., concentration0=1.),
      numpyro.distributions.Beta(concentration1=4., concentration0=6.),
    ],
    _HALF_MAX_EFFECTIVE_CONCENTRATION: [
      dist.Gamma(concentration=1., rate=1.), # Original by MMM
      numpyro.distributions.LogNormal(loc=0.1, scale=0.5),
      numpyro.distributions.LogNormal(loc=0.5, scale=0.5),
    ],
    _AD_EFFECT_RETENTION_RATE: [
      numpyro.distributions.Beta(concentration1=1., concentration0=1.),
      numpyro.distributions.Beta(concentration1=1., concentration0=3.),
      numpyro.distributions.Beta(concentration1=4., concentration0=6.),
    ],
    _PEAK_EFFECT_DELAY: [
      numpyro.distributions.HalfNormal(scale=2.),
      numpyro.distributions.HalfNormal(scale=1.),
    ],
    _EXPONENT: [
      numpyro.distributions.Beta(concentration1=9., concentration0=1.),
      numpyro.distributions.Beta(concentration1=5., concentration0=1.),
      numpyro.distributions.Beta(concentration1=1., concentration0=1.),
    ],
    _SLOPE: [
      numpyro.distributions.Gamma(concentration=1., rate=1.),
      numpyro.distributions.Gamma(concentration=3., rate=1.),
      numpyro.distributions.HalfNormal(scale=2.0),
    ]
  })

def transform_exponential_adstock_ensemble(
                    media_data: jnp.ndarray,
                    transform_samples,
                    #transform_default_priors,
                    #transform_hyperprior: bool,
                    #default_priors: MutableMapping[str, Prior],
                    #custom_priors: MutableMapping[str, Prior],
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
  #transform_default_priors = _get_transform_default_priors(transform_hyperprior)['exponential_adstock']

  # with numpyro.plate(name=f"{_LAG_WEIGHT}_plate",
  #                    size=media_data.shape[1]):
  #   lag_weight = numpyro.sample(
  #       name=_LAG_WEIGHT,
  #       fn=custom_priors.get(_LAG_WEIGHT,               
  #           transform_default_priors[_LAG_WEIGHT]
  #     )
  #   )
  lag_weight = transform_samples[_LAG_WEIGHT]
  slopes = transform_samples[_SATURATION]

  adstock = media_transforms.adstock(
      data=media_data, lag_weight=lag_weight, normalise=normalise)

  # with numpyro.plate(name=f"{_SATURATION}_plate", size=media_data.shape[1]):

  #   slopes = numpyro.sample(
  #       name=_SATURATION,
  #       fn=custom_priors.get(_SATURATION,               
  #         transform_default_priors[_SATURATION]
  #     )
  #   )

  return media_transforms.exponential_saturation(
    data=adstock, slope=slopes
  )

def transform_exponential_carryover_ensemble(
                    media_data: jnp.ndarray,
                    #default_priors: MutableMapping[str, Prior],
                    #transform_hyperprior:bool,
                    transform_samples,
                    #transform_default_priors,
                    #custom_priors: MutableMapping[str, Prior],
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
  #transform_default_priors = _get_transform_default_priors(transform_hyperprior)["exponential_carryover"]

  # with numpyro.plate(name=f"{_AD_EFFECT_RETENTION_RATE}_plate",
  #                    size=media_data.shape[1]):
  #   ad_effect_retention_rate = numpyro.sample(
  #       name=_AD_EFFECT_RETENTION_RATE,
  #       fn=custom_priors.get(
  #           _AD_EFFECT_RETENTION_RATE,
  #           transform_default_priors[_AD_EFFECT_RETENTION_RATE]))

  # with numpyro.plate(name=f"{_PEAK_EFFECT_DELAY}_plate",
  #                   size=media_data.shape[1]):
  #   peak_effect_delay = numpyro.sample(
  #       name=_PEAK_EFFECT_DELAY,
  #       fn=custom_priors.get(
  #           _PEAK_EFFECT_DELAY, transform_default_priors[_PEAK_EFFECT_DELAY]))

  peak_effect_delay = transform_samples[_PEAK_EFFECT_DELAY]
  ad_effect_retention_rate = transform_samples[_AD_EFFECT_RETENTION_RATE]
  slopes = transform_samples[_SATURATION]

  carryover = media_transforms.carryover(
      data=media_data,
      ad_effect_retention_rate=ad_effect_retention_rate,
      peak_effect_delay=peak_effect_delay,
      number_lags=60) # Max lags allowed is 2months

  # with numpyro.plate(name=f"{_SATURATION}_plate", size=media_data.shape[1]):
  #   slopes = numpyro.sample(
  #     name=_SATURATION,
  #     fn=custom_priors.get(_SATURATION, transform_default_priors[_SATURATION])
  #   )
  return media_transforms.exponential_saturation(
    data=carryover, slope=slopes
  )


def transform_adstock_ensemble(media_data: jnp.ndarray,
                      #transform_hyperprior:bool,
                      transform_samples,
                      #transform_default_priors,
                      #default_priors: MutableMapping[str, Prior],
                      #custom_priors: MutableMapping[str, Prior],
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
  #transform_default_priors = _get_transform_default_priors(transform_hyperprior)["adstock"]
  # with numpyro.plate(name=f"{_LAG_WEIGHT}_plate",
  #                    size=media_data.shape[1]):
  #   lag_weight = numpyro.sample(
  #       name=_LAG_WEIGHT,
  #       fn=custom_priors.get(_LAG_WEIGHT,
  #                            transform_default_priors[_LAG_WEIGHT]))

  # with numpyro.plate(name=f"{_EXPONENT}_plate",
  #                    size=media_data.shape[1]):
  #   exponent = numpyro.sample(
  #       name=_EXPONENT,
  #       fn=custom_priors.get(_EXPONENT,
  #                            transform_default_priors[_EXPONENT]))
    
  exponent = transform_samples[_EXPONENT]
  lag_weight = transform_samples[_LAG_WEIGHT]

  if media_data.ndim == 3:
    lag_weight = jnp.expand_dims(lag_weight, axis=-1)
    exponent = jnp.expand_dims(exponent, axis=-1)

  adstock = media_transforms.adstock(
      data=media_data, lag_weight=lag_weight, normalise=normalise)

  n = media_transforms.apply_exponent_safe(data=adstock, exponent=exponent)

  return n / n.sum(axis=0) * adstock.sum(axis=0)


def transform_hill_adstock_ensemble(media_data: jnp.ndarray,
                           #transform_hyperprior:bool, 
                           #transform_default_priors,
                           transform_samples,
                           #default_priors: MutableMapping[str, Prior],
                           #custom_priors: MutableMapping[str, Prior],
                           adstock_normalise: bool = True,
                           saturation_normalise: bool = True) -> jnp.ndarray:
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
  #transform_default_priors = _get_transform_default_priors(transform_hyperprior)["hill_adstock"]
  # with numpyro.plate(name=f"{_LAG_WEIGHT}_plate",
  #                    size=media_data.shape[1]):
  #   lag_weight = numpyro.sample(
  #       name=_LAG_WEIGHT,
  #       fn=custom_priors.get(_LAG_WEIGHT,
  #                            transform_default_priors[_LAG_WEIGHT]))

  # with numpyro.plate(name=f"{_HALF_MAX_EFFECTIVE_CONCENTRATION}_plate",
  #                    size=media_data.shape[1]):
  #   half_max_effective_concentration = numpyro.sample(
  #       name=_HALF_MAX_EFFECTIVE_CONCENTRATION,
  #       fn=custom_priors.get(
  #           _HALF_MAX_EFFECTIVE_CONCENTRATION,
  #           transform_default_priors[_HALF_MAX_EFFECTIVE_CONCENTRATION]))

  # with numpyro.plate(name=f"{_SLOPE}_plate",
  #                    size=media_data.shape[1]):
  #   slope = numpyro.sample(
  #       name=_SLOPE,
  #       fn=custom_priors.get(_SLOPE, transform_default_priors[_SLOPE]))
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
          data=media_data, lag_weight=lag_weight, normalise=adstock_normalise),
      half_max_effective_concentration=half_max_effective_concentration,
      slope=slope,
      normalise=saturation_normalise
    )
  # if saturation_normalise:
  #   return hill_adstock_media * (2 * half_max_effective_concentration)



def transform_carryover_ensemble(media_data: jnp.ndarray,
                        #transform_default_priors,
                        transform_samples,
                        #custom_priors: MutableMapping[str, Prior],
                        number_lags: int = 13
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
  #transform_default_priors = _get_transform_default_priors(transform_hyperprior)["carryover"]
  # with numpyro.plate(name=f"{_AD_EFFECT_RETENTION_RATE}_plate",
  #                    size=media_data.shape[1]):
  #   ad_effect_retention_rate = numpyro.sample(
  #       name=_AD_EFFECT_RETENTION_RATE,
  #       fn=custom_priors.get(
  #           _AD_EFFECT_RETENTION_RATE,
  #           transform_default_priors[_AD_EFFECT_RETENTION_RATE]))


  # with numpyro.plate(name=f"{_PEAK_EFFECT_DELAY}_plate",
  #                   size=media_data.shape[1]):
  #   peak_effect_delay = numpyro.sample(
  #       name=_PEAK_EFFECT_DELAY,
  #       fn=custom_priors.get(
  #           _PEAK_EFFECT_DELAY, transform_default_priors[_PEAK_EFFECT_DELAY]))


  # with numpyro.plate(name=f"{_EXPONENT}_plate",
  #                   size=media_data.shape[1]):
  #   exponent = numpyro.sample(
  #       name=_EXPONENT,
  #       fn=custom_priors.get(_EXPONENT,
  #                           transform_default_priors[_EXPONENT]))

  carryover = media_transforms.carryover(
      data=media_data,
      ad_effect_retention_rate=transform_samples[_AD_EFFECT_RETENTION_RATE],#ad_effect_retention_rate,
      peak_effect_delay=transform_samples[_PEAK_EFFECT_DELAY],#peak_effect_delay,
      number_lags=number_lags)

  if media_data.ndim == 3:
    exponent = jnp.expand_dims(exponent, axis=-1)
  return media_transforms.apply_exponent_safe(
    data=carryover,
    exponent=transform_samples[_EXPONENT]#exponent
  )


_NAMES_TO_MODEL_TRANSFORMS = immutabledict.immutabledict({
    "hill_adstock": transform_hill_adstock_ensemble,
    #"adstock": transform_adstock_ensemble,
    #"carryover": transform_carryover,
    #"exponential_carryover": transform_exponential_carryover_ensemble,
    "exponential_adstock": transform_exponential_adstock_ensemble,
})

_ADSTOCK_TRANSFORMS = immutabledict.immutabledict({
  (_LAG_WEIGHT,): media_transforms.adstock,
  (_AD_EFFECT_RETENTION_RATE, _PEAK_EFFECT_DELAY): media_transforms.carryover
})
_SATURATION_TRANSFORMS = immutabledict.immutabledict({
  #(_EXPONENT,): media_transforms.apply_exponent_safe,
  (_SATURATION,): media_transforms.exponential_saturation,
  (_HALF_MAX_EFFECTIVE_CONCENTRATION, _SLOPE): media_transforms.hill,
})

def ensemble_media_transform(
  media_data: jnp.ndarray,
  media_prior,
  transform_hyperprior: bool,
  custom_priors: MutableMapping[str, Prior],
  **transform_kwargs
):
  
  # Choose model and associated function
  #TODO: pass in list of potential model functions
  #model_names = list(TRANSFORM_PRIORS_NAMES.keys())
  #   'hill_adstock',
  #   'exponential_saturation',
  #   'exponential_carryover',
  # ]

  # model_selection = numpyro.sample(
  #   name='model_selection',
  #   fn=dist.DiscreteUniform(low=0, high=len(model_names) - 1)
  # )
  transform_default_priors = _get_transform_default_priors(transform_hyperprior)


  sample_list = set([
    *list(chain.from_iterable(_ADSTOCK_TRANSFORMS.keys())),
    *list(chain.from_iterable(_SATURATION_TRANSFORMS.keys()))
  ])

  def get_sample(site_name, dist):
    with numpyro.plate(name=f"{site_name}_plate",
                      size=media_data.shape[1]):
      return numpyro.sample(
          name=site_name,
          fn=custom_priors.get(site_name,               
              dist#transform_default_priors[site_name]
        )
      )

  transform_samples = {
    site_name: get_sample(site_name, dist)
    for site_name, dist in transform_default_priors.items()
    if site_name in sample_list
  }

  args = dict(
    media_data=media_data,
    transform_samples=transform_samples,
    #custom_priors=custom_priors
    #samples=transform_samples
  )

  # @functools.partial(jax.jit, static_argnums=(0,))
  # def apply_model(ms):
  #   if ms < 0:
  #     return jnp.array(_NAMES_TO_MODEL_TRANSFORMS['hill_adstock'](**args))
  #   return jnp.array(_NAMES_TO_MODEL_TRANSFORMS['exponential_adstock'](**args))

  transformed_media = jnp.concatenate([
    jnp.expand_dims(sat_f(
      ad_f(
        media_data,
        *[transform_samples[p] for p in ad_params]
      ),
      *[transform_samples[p] for p in sat_params],
      **(dict(normalise=True) if _HALF_MAX_EFFECTIVE_CONCENTRATION in sat_params else {})
    ), 0)
    for sat_params, sat_f in _SATURATION_TRANSFORMS.items()
    for ad_params, ad_f in _ADSTOCK_TRANSFORMS.items()
  ], axis=0)

  with numpyro.plate(name=f"model_weights_plate", size=transformed_media.shape[0]):
    model_weights = numpyro.sample(
      name='model_weights',
      fn=dist.Beta(concentration1=1.0, concentration0=1.0)#.to_event()
    )
  model_weights = model_weights / model_weights.sum()

  n_channels = media_data.shape[1]
  media_transformed_ensemble = numpyro.deterministic(
    'media_transformed',
    (transformed_media * model_weights.reshape(-1, 1, 1)).sum(axis=0)
  )

  with numpyro.plate(
    name="channel_media_plate",
    size=n_channels,
    #dim=1
  ):
    coef_media = numpyro.sample(
      name="channel_coef_media" if media_data.ndim == 3 else "coef_media",
      #fn=dist.TruncatedNormal(loc=media_prior, scale=0.05, low=1e-6)
      fn=dist.HalfNormal(scale=media_prior)
    )#)

  
  #tfg, fg -> tg
  media_einsum = "tc, c -> t"
  return jnp.einsum(media_einsum, media_transformed_ensemble, coef_media)
  #media_contribution = (media_contribution * model_weights.reshape(-1, 1)).sum(axis=0)
  


  # IF TREAT COEFFICIENTS FOR EACH DIFFERENT MODEL OPTION SEPARATELY
  
  # media_transformed = numpyro.deterministic(
  #   name="media_transformed",
  #   value=transformed_media
  # )

  # n_channels = media_data.shape[1]

  # with numpyro.plate(
  #     name="channel_media_plate",
  #     size=n_channels,
  #     #dim=1
  #   ):
  #   with numpyro.plate(
  #     name='model_media_plate',
  #     size=len(_SATURATION_TRANSFORMS.keys()),
  #     #dim=0
  #   ):
  #     coef_media = numpyro.sample(
  #       name="channel_coef_media" if media_data.ndim == 3 else "coef_media",
  #       #fn=dist.TruncatedNormal(loc=media_prior, scale=0.05, low=1e-6)
  #       fn=dist.HalfNormal(scale=media_prior.reshape(1, -1))
  #     )#)

  # coef_media = jnp.repeat(
  #   coef_media,
  #   len(_ADSTOCK_TRANSFORMS.keys()),
  #   axis=0
  # )
  
  # #tfg, fg -> tg
  # media_einsum = "mtc, mc -> mt"
  # media_contribution = jnp.einsum(media_einsum, media_transformed, coef_media)
  # media_contribution = (media_contribution * model_weights.reshape(-1, 1)).sum(axis=0)
  
  return media_contribution

  # Saturation specific coefficients !

  # # Transform via transforms as a whole
  # transformed_media = jnp.concatenate([
  #   jnp.expand_dims(f(
  #     #transform_default_priors=transform_default_priors[name],
      
  #     **args
  #   ), 0)
  #   for name, f in _NAMES_TO_MODEL_TRANSFORMS.items()
  # ], axis=0)

  # model_selection = numpyro.sample(
  #   name='model_selection',
  #   fn=dist.Beta(concentration1=1.0, concentration0=1.0)#.to_event()
  # )

  # n_models = len(_NAMES_TO_MODEL_TRANSFORMS.keys())
  # interval = 1.0 / n_models
  # test = jnp.linspace(0., 1., n_models).reshape(-1,1, 1)
  # test = (test <= model_selection) & ((test + interval) > model_selection)
  # model_cond = jnp.repeat(
  #   jnp.repeat(test, transformed_media.shape[1], axis=1),
  #   transformed_media.shape[2], axis=2
  # )

  # return jnp.where(
  #   model_cond,
  #   transformed_media,
  #   jnp.zeros(transformed_media.shape)
  # ).sum(axis=0)


  return apply_model(model_selection)

  if model_selection < 0:
    return jnp.array(_NAMES_TO_MODEL_TRANSFORMS['hill_adstock'](**args))
  return jnp.array(_NAMES_TO_MODEL_TRANSFORMS['exponential_adstock'](**args))

  media = cond(
    model_selection < 0,
    lambda _: jnp.array(_NAMES_TO_MODEL_TRANSFORMS['hill_adstock'](**args)),
    lambda _: jnp.array(_NAMES_TO_MODEL_TRANSFORMS['exponential_adstock'](**args)),
    None
  )
  return media

  #model_name = model_names[model_selection]

  #model_name = 'hill_adstock'
  model_transform = _NAMES_TO_MODEL_TRANSFORMS[model_name]

  # Transform hyperprior - bias towards using lists of priors
  # transform_hyperprior = numpyro.sample(
  #   name='transform_hyperprior',
  #   fn=dist.Binomial(total_count=1, probs=0.2)
  # )
  transform_hyperprior = True

  # Create custom priors



  # Call transform function & return
  transformed_media = model_transform(
    media_data=media_data,
    transform_hyperprior=transform_hyperprior,
    custom_priors=custom_priors
  )
  return transformed_media

def ensemble_media_mix_model(
    media_data: jnp.ndarray,
    target_data: jnp.ndarray,
    media_prior: jnp.ndarray,
    #doms:jnp.ndarray,
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

  # with numpyro.plate(
  #     name="channel_media_plate",
  #     size=n_channels,
  #     dim=-2 if media_data.ndim == 3 else -1):
  #   coef_media = numpyro.sample(
  #       name="channel_coef_media" if media_data.ndim == 3 else "coef_media",
  #       #fn=dist.TruncatedNormal(loc=media_prior, scale=0.05, low=1e-6)
  #       fn=dist.HalfNormal(scale=media_prior)
  #     )#)
  #   if media_data.ndim == 3:
  #     with numpyro.plate(
  #         name="geo_media_plate",
  #         size=n_geos,
  #         dim=-1):
  #       # Corrects the mean to be the same as in the channel only case.
  #       normalisation_factor = jnp.sqrt(2.0 / jnp.pi)
  #       coef_media = numpyro.sample(
  #           name="coef_media",
  #           fn=dist.HalfNormal(scale=coef_media * normalisation_factor)
  #       )
  # Optional weekday seasonality
  # weekday_seasonality = numpyro.sample(
  #   name=_WEEKDAY + '_present',
  #   fn=dist.Normal(0, 1.)
  # )
  if weekday_seasonality:
    with numpyro.plate(name=f"{_WEEKDAY}_plate", size=6):
      weekday = numpyro.sample(
          name=_WEEKDAY,
          fn=custom_priors.get(_WEEKDAY, default_priors[_WEEKDAY]))
    weekday = jnp.concatenate(arrays=[weekday, jnp.array([0])], axis=0)
    weekday_series = weekday[jnp.arange(data_size) % 7]

  # Yearly Seasonality

  # degrees_seasonality = int(numpyro.sample(
  #   name= 'degrees_seasonality',
  #   fn=dist.Binomial(total_count=MAX_SEASONALITY, probs=jnp.array([0.4]))
  # )[0])

  sample_degrees_seasonality = degrees_seasonality

  # ds = numpyro.sample(
  #   name='degrees_seasonality',
  #   fn=dist.Beta(1., 1.)
  # )
  # sample_degrees_seasonality = (ds.astype(int) * (degrees_seasonality - 1) + 1).astype(int)
  
  
  #degrees_seasonality=3
  # degrees_seasonality = cond(
  #   degrees_seasonality > 1,
  #   lambda _: 3,
  #   lambda _: 1,
  #   None
  # )

  with numpyro.plate(name=f"{_GAMMA_SEASONALITY}_sin_cos_plate", size=2):
    with numpyro.plate(name=f"{_GAMMA_SEASONALITY}_plate",
                       size=degrees_seasonality):
      gamma_seasonality = numpyro.sample(
          name=_GAMMA_SEASONALITY,
          fn=custom_priors.get(
              _GAMMA_SEASONALITY, default_priors[_GAMMA_SEASONALITY]))
  #gamma_seasonality = gamma_seasonality[:degrees_seasonality]
  
  seasonality = media_transforms.calculate_seasonality_ensemble(
    number_periods=data_size,
    degrees=sample_degrees_seasonality,
    frequency=frequency,
    gamma_seasonality=gamma_seasonality
  )

  # In case of daily data, number of lags should be 13*7.
  # if transform_function == "carryover" and transform_kwargs and "number_lags" not in transform_kwargs:
  #   transform_kwargs["number_lags"] = 13 * 7
  # elif transform_function == "carryover" and not transform_kwargs:
  #   transform_kwargs = {"number_lags": 13 * 7}

  media_contribution = numpyro.deterministic(
      name="media_contribution",
      value=transform_function(media_data,
                              media_prior=media_prior,
                               transform_hyperprior=transform_hyperprior,
                               custom_priors=custom_priors,
                               **transform_kwargs if transform_kwargs else {}))

  # For national model's case
  trend = jnp.arange(data_size)
  media_einsum = "tc, c -> t"  # t = time, c = channel
  coef_seasonality = 1

  # TODO(): Add conversion of prior for HalfNormal distribution.
  if media_data.ndim == 3:  # For geo model's case
    trend = jnp.expand_dims(trend, axis=-1)
    seasonality = jnp.expand_dims(seasonality, axis=-1)
    media_einsum = "tcg, cg -> tg"  # t = time, c = channel, g = geo
    weekday_series = jnp.expand_dims(weekday_series, axis=-1)
    with numpyro.plate(name="seasonality_plate", size=n_geos):
      coef_seasonality = numpyro.sample(
          name=_COEF_SEASONALITY,
          fn=custom_priors.get(
              _COEF_SEASONALITY, default_priors[_COEF_SEASONALITY]))
  
  total_trend = coef_trend * trend ** expo_trend
  total_seasonality = (
    seasonality * coef_seasonality
  )

  # Day of month effect
  # doms_present = numpyro.sample(
  #   name= 'doms_present',
  #   fn=dist.Normal(loc=0.0, scale=1.0)
  # )
  #doms_present = 1
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

    # dom_contribs = cond(
    #   doms_present > 0,
    #   lambda _: dom_contribs,
    #   lambda _: jnp.zeros(dom_contribs.shape),
    #   None
    # )
  
    total_seasonality += dom_contribs

  # Weekday Seasonality effect
  # if weekday_seasonality > 0:
  #   total_seasonality += weekday_series
  if weekday_seasonality:
    total_seasonality += weekday_series
  # total_seasonality += cond(
  #   weekday_seasonality > 0,
  #   lambda _: weekday_series,
  #   lambda _: jnp.zeros(weekday_series.shape),
  #   None
  # )

  # Total seasonality (to aid in re-distribution later)
  total_seasonality = numpyro.deterministic(
    name="total_seasonality",
    value=total_seasonality
  )

  # Total trend (to aid in re-distribution later)
  total_trend = numpyro.deterministic(
    name='total_trend',
    value=total_trend
  )

  # expo_trend is B(1, 1) so that the exponent on time is in [.5, 1.5].
  prediction = (
      intercept + total_seasonality + total_trend + 
      media_contribution
      #jnp.einsum(media_einsum, media_transformed, coef_media)
  )

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
    prediction += extra_features_effect


  mu = numpyro.deterministic(name="mu", value=prediction)

  numpyro.sample(
    name="target",
    fn=dist.Normal(loc=mu, scale=sigma),
    obs=target_data
  )
