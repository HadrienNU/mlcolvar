#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Variational Autoencoder collective variable.
"""

__all__ = ["FeedForward", "KANFeedForward", "get_feedforward"]


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Optional, Union

import torch
import lightning
from mlcolvar.core.nn.utils import get_activation, parse_nn_options
from better_kan import KAN, build_rbf_layers, build_splines_layers, build_chebyshev_layers


def get_feedforward(layers, options):
    if options.get("use_kan", False):
        return KANFeedForward(layers, **options)
    else:
        return FeedForward(layers, **options)


# =============================================================================
# STANDARD FEED FORWARD
# =============================================================================


class FeedForward(lightning.LightningModule):
    """Define a feedforward neural network given the list of layers.

    Optionally dropout and batchnorm can be applied (the order is activation -> dropout -> batchnorm).
    """

    def __init__(
        self,
        layers: list,
        activation: Union[str, list] = "relu",
        dropout: Optional[Union[float, list]] = None,
        batchnorm: Union[bool, list] = False,
        last_layer_activation: bool = False,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        layers : list
            Number of neurons per layer.
        activation : string or list[str], optional
            Add activation function (options: relu, tanh, elu, linear). If a
            ``list``, this must have length ``len(layers)-1``, and ``activation[i]``
            controls whether to add the activation to the ``i``-layer.
        dropout : float or list[float], optional
            Add dropout with this probability after each layer. If a ``list``,
            this must have length ``len(layers)-1``, and ``dropout[i]`` specifies
            the the dropout probability for the ``i``-th layer.
        batchnorm : bool or list[bool], optional
            Add batchnorm after each layer. If a ``list``, this must have
            length ``len(layers)-1``, and ``batchnorm[i]`` controls whether to
            add the batchnorm to the ``i``-th layer.
        last_layer_activation : bool, optional
            If ``True`` and activation, dropout, and batchnorm are added also to
            the output layer when ``activation``, ``dropout``, or ``batchnorm``
            (i.e., they are not lists). Otherwise, the output layer will be linear.
            This option is ignored for the arguments among ``activation``, ``dropout``,
            and ``batchnorm`` that are passed as lists.
        **kwargs:
            Optional arguments passed to torch.nn.Module
        """

        super().__init__(**kwargs)

        # Parse layers
        if not isinstance(layers[0], int):
            raise TypeError("layers should be a list-type of integers.")

        # Parse options per each hidden layer
        n_layers = len(layers) - 1
        # -- activation
        activation_list = parse_nn_options(activation, n_layers, last_layer_activation)
        # -- dropout
        dropout_list = parse_nn_options(dropout, n_layers, last_layer_activation)
        # -- batchnorm
        batchnorm_list = parse_nn_options(batchnorm, n_layers, last_layer_activation)

        # Create network
        modules = []
        for i in range(len(layers) - 1):
            modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
            activ, drop, norm = activation_list[i], dropout_list[i], batchnorm_list[i]

            if activ is not None:
                modules.append(get_activation(activ))

            if drop is not None:
                modules.append(torch.nn.Dropout(p=drop))

            if norm:
                modules.append(torch.nn.BatchNorm1d(layers[i + 1]))

        # store model and attributes
        self.nn = torch.nn.Sequential(*modules)
        self.in_features = layers[0]
        self.out_features = layers[-1]

    # def extra_repr(self) -> str:
    #    repr = f"in_features={self.in_features}, out_features={self.out_features}"
    #    return repr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


# =============================================================================
# FEED FORWARD WITH KAN
# =============================================================================


# define the LightningModule
class KANFeedForward(lightning.LightningModule):
    def __init__(
        self,
        layers,
        lamb=0.01,
        lamb_l1=1.0,
        lamb_entropy=1.0,
        update_grid=True,
        grid_update_num=10,
        stop_grid_update_step=50,
        kan_type="rbf",
        use_kan=True,
        **kwargs,
    ):
        super().__init__()
        if kan_type == "rbf":
            self.kan = KAN(build_rbf_layers(layers, **kwargs))
        elif kan_type == "splines":
            self.kan = KAN(build_splines_layers(layers, **kwargs))
        elif kan_type == "chebyshev":
            self.kan = KAN(build_chebyshev_layers(layers, **kwargs))
        self.lamb = lamb
        self.lamb_l1 = lamb_l1
        self.lamb_entropy = lamb_entropy

        # If training is used
        self.update_grid = update_grid
        self.stop_grid_update_step = stop_grid_update_step
        self.grid_update_freq = int(stop_grid_update_step / grid_update_num)

    def forward(self, x: torch.Tensor, update_grid=False) -> torch.Tensor:
        return self.kan.forward(x, update_grid=update_grid)

    def regularization(self):
        return self.lamb * self.kan.regularization_loss(self.lamb_l1, self.lamb_entropy)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        x, y = batch
        x = x.view(-1, self.kan.width[0])  # Assure input has the correct size

        pred = self.kan.forward(x, update_grid=(batch_idx % self.grid_update_freq == 0 and batch_idx < self.stop_grid_update_step and self.update_grid))
        train_loss = torch.mean((pred - y) ** 2)
        reg_ = self.regularization()
        loss = train_loss + reg_

        self.log("train_loss", loss)
        self.log("regularization", reg_)
        return loss
