import torch
from mlcolvar.core.transform import Transform


class BaseCV:
    """
    Base collective variable class.

    To inherit from this class, the class must define a BLOCKS class attribute.
    """

    def __init__(
        self,
        in_features,
        out_features,
        preprocessing: torch.nn.Module = None,
        postprocessing: torch.nn.Module = None,
        *args,
        **kwargs,
    ):
        """Base CV class options.

        Parameters
        ----------
        in_features : int
            Number of inputs of the CV model
        out_features : int
            Number of outputs of the CV model, should be the number of CVs
        preprocessing : torch.nn.Module, optional
            Preprocessing module, default None
        postprocessing : torch.nn.Module, optional
            Postprocessing module, default None

        """
        super().__init__(*args, **kwargs)

        # The parent class sets in_features and out_features based on their own
        # init arguments so we don't need to save them here (see #103).
        self.save_hyperparameters(ignore=["in_features", "out_features"])

        # MODEL
        self.initialize_blocks()
        self.in_features = in_features
        self.out_features = out_features

        # OPTIM
        self._optimizer_name = "Adam"
        self.optimizer_kwargs = {}
        self.lr_scheduler_kwargs = {}

        # PRE/POST
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    @property
    def n_cvs(self):
        """Number of CVs."""
        return self.out_features

    @property
    def example_input_array(self):
        return torch.randn((1, self.in_features) if self.preprocessing is None or not hasattr(self.preprocessing, "in_features") else self.preprocessing.in_features)

    def parse_options(self, options: dict = None):
        """
        Sanitize options and create defaults ({}) if not in options.
        Furthermore, it sets the optimizer kwargs, if given.

        Parameters
        ----------
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default None.
        """
        if options is None:
            options = {}

        for b in self.BLOCKS:
            options.setdefault(b, {})

        for o in options.keys():
            if o not in self.BLOCKS:
                if o == "optimizer":
                    self.optimizer_kwargs.update(options[o])
                elif o == "lr_scheduler":
                    self.lr_scheduler_kwargs.update(options[o])
                else:
                    raise ValueError(f'The key {o} is not available in this class. The available keys are: {", ".join(self.BLOCKS)}, optimizer and lr_scheduler.')

        return options

    def initialize_blocks(self):
        """
        Initialize the blocks as attributes of the CV class.
        """
        for b in self.BLOCKS:
            self.__setattr__(b, None)

    def setup(self, stage=None):
        if stage == "fit":
            self.initialize_transforms(self.trainer.datamodule)

    def initialize_transforms(self, datamodule):
        stats = datamodule.train_dataloader().get_stats(preprocessing=self.preprocessing)["data"]

        for b in self.BLOCKS:
            if isinstance(getattr(self, b), Transform):
                getattr(self, b).set_from_stats(stats, getattr(self, b).mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation of the CV

        - Apply preprocessing if any
        - Execute sequentially all the blocks in self.BLOCKS unless they are not initialized
        - Apply postprocessing if any

        Parameters
        ----------
        x : torch.Tensor
            Input of the forward operation of the model

        Returns
        -------
        torch.Tensor
            Output of the forward operation of the model
        """

        if self.preprocessing is not None:
            x = self.preprocessing(x)

        x = self.forward_cv(x)

        if self.postprocessing is not None:
            x = self.postprocessing(x)

        return x

    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute sequentially all the blocks in self.BLOCKS unless they are not initialized.

        No pre/post processing will be executed here. This is supposed to be called during training/validation and to be overloaded if necessary.

        Parameters
        ----------
        x : torch.Tensor
            Input of the forward operation of the model

        Returns
        -------
        torch.Tensor
            Output of the forward operation of the model
        """

        for b in self.BLOCKS:
            block = getattr(self, b)
            if block is not None:
                x = block(x)

        return x

    def validation_step(self, val_batch, batch_idx):
        """
        Equal to training step if not overridden. Different behaviors for train/valid step can be enforced in training_step() based on the self.training variable.
        """
        self.training_step(val_batch, batch_idx)

    def test_step(self, test_batch, batch_idx):
        """
        Equal to training step if not overridden. Different behaviors for train/valid step can be enforced in training_step() based on the self.training variable.
        """
        self.training_step(test_batch, batch_idx)

    @property
    def optimizer_name(self) -> str:
        """Optimizer name. Options can be set using optimizer_kwargs. Actual optimizer will be return during training from configure_optimizer function."""
        return self._optimizer_name

    @optimizer_name.setter
    def optimizer_name(self, optimizer_name: str):
        if not hasattr(torch.optim, optimizer_name):
            raise AttributeError(f"torch.optim does not have a {optimizer_name} optimizer.")
        self._optimizer_name = optimizer_name

    def configure_optimizers(self):
        """
        Initialize the optimizer based on self._optimizer_name and self.optimizer_kwargs.

        Returns
        -------
        torch.optim
            Torch optimizer
        """

        optimizer = getattr(torch.optim, self._optimizer_name)(self.parameters(), **self.optimizer_kwargs)

        if self.lr_scheduler_kwargs:
            scheduler_cls = self.lr_scheduler_kwargs["scheduler"]
            scheduler_kwargs = {k: v for k, v in self.lr_scheduler_kwargs.items() if k != "scheduler"}
            lr_scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def __setattr__(self, key, value):
        # PyTorch overrides __setattr__ to raise a TypeError when you try to assign
        # an attribute that is a Module to avoid substituting the model's component
        # by mistake. This means we can't simply assign to loss_fn a lambda function
        # after it's been assigned a Module, but we need to delete the Module first.
        #    https://github.com/pytorch/pytorch/issues/51896
        #    https://stackoverflow.com/questions/61116433/maybe-i-found-something-strange-on-pytorch-which-result-in-property-setter-not
        try:
            super().__setattr__(key, value)
        except TypeError as e:
            # We make an exception only for loss_fn.
            if (key == "loss_fn") and ("cannot assign" in str(e)):
                del self.loss_fn
                super().__setattr__(key, value)

    def load_state_dict(self, state_dict):
        missing_keys = {}
        unexpected_keys = {}
        if self.preprocessing is not None:
            mk, uk = self.preprocessing.load_state_dict(state_dict["preprocessing"])
            missing_keys["preprocessing"] = mk
            unexpected_keys["preprocessing"] = uk

        if self.postprocessing is not None:
            mk, uk = self.postprocessing.load_state_dict(state_dict["postprocessing"])
            missing_keys["postprocessing"] = mk
            unexpected_keys["postprocessing"] = uk
        for b in self.BLOCKS:
            block = getattr(self, b)
            if block is not None:
                mk, uk = block.load_state_dict(state_dict[b])
                missing_keys[b] = mk
                unexpected_keys[b] = uk
        # TODO: Add tests on missing keys and unexpected keys
        # self.in_features = dict["in_features"]
        # self.out_features = dict["out_features"]
        return missing_keys, unexpected_keys

    def state_dict(self):
        if self.preprocessing is not None:
            preprocessing_dict = self.preprocessing.state_dict()
        else:
            preprocessing_dict = None

        if self.postprocessing is not None:
            postprocessing_dict = self.postprocessing.state_dict()
        else:
            postprocessing_dict = None
        dict = {"preprocessing": preprocessing_dict, "postprocessing": postprocessing_dict}
        for b in self.BLOCKS:
            block = getattr(self, b)
            if block is not None:
                dict[b] = block.state_dict()
        dict["in_features"] = self.in_features
        dict["out_features"] = self.out_features
        return dict
