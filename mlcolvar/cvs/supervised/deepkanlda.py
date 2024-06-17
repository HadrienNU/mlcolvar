import torch
import lightning
from mlcolvar.cvs import BaseCV, DeepLDA
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.data import DictModule
from mlcolvar.core.stats import LDA
from mlcolvar.core.loss import ReduceEigenvaluesLoss
from mlcolvar.core.transform.utils import Statistics
from kan import KAN

__all__ = ["DeepKAN_LDA"]


class DeepKAN_LDA(DeepLDA):
    """Deep Linear Discriminant Analysis (Deep-LDA) CV.
    Non-linear generalization of LDA in which a feature map is learned by a neural network optimized
    as to maximize the classes separation. The method is described in [1]_.

    **Data**: for training it requires a DictDataset with the keys 'data' and 'labels'.

    **Loss**: maximize LDA eigenvalues (ReduceEigenvaluesLoss)

    References
    ----------
    .. [1] L. Bonati, V. Rizzi, and M. Parrinello, "Data-driven collective variables for enhanced
        sampling", JPCL 11, 2998–3004 (2020).

    See also
    --------
    mlcolvar.core.stats.LDA
        Linear Discriminant Analysis method
    mlcolvar.core.loss.ReduceEigenvalueLoss
        Eigenvalue reduction to a scalar quantity
    """

    BLOCKS = ["norm_in", "nn", "lda"]

    def __init__(self, layers: list, n_states: int, options: dict = None, **kwargs):
        """
        Define a Deep Linear Discriminant Analysis (Deep-LDA) CV composed by a
        neural network module and a LDA object.
        By default a module standardizing the inputs is also used.

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        n_states : int
            Number of states for the training
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['norm_in','nn','lda'] .
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs)

        # =======   LOSS  =======
        # Maximize the sum of all the LDA eigenvalues.
        self.loss_fn = ReduceEigenvaluesLoss(mode="sum")

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # Save n_states
        self.n_states = n_states

        # ======= BLOCKS =======

        # initialize norm_in
        o = "norm_in"
        if (options[o] is not False) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features, **options[o])

        # initialize nn
        o = "kan"
        self.nn = KAN(width=layers, **options[o])
        #  TODO:
        self.nn.lock(0, options["lock"])

        # initialize lda
        o = "lda"
        self.lda = LDA(layers[-1], n_states, **options[o])

        # regularization
        self.lorentzian_reg = 40  # == 2/sw_reg, see set_regularization
        self.set_regularization(sw_reg=0.05)

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        # =================get data===================
        x = train_batch["data"]
        y = train_batch["labels"]
        # =================forward====================
        h = self.forward_nn(x)
        # ===================lda======================
        eigvals, _ = self.lda.compute(h, y, save_params=True if self.training else False)
        # ===================loss=====================
        loss = self.loss_fn(eigvals)
        if self.lorentzian_reg > 0:
            s = self.lda(h)
            lorentzian_reg = self.regularization_lorentzian(s)
            loss += lorentzian_reg
        # ====================log=====================
        name = "train" if self.training else "valid"
        loss_dict = {f"{name}_loss": loss, f"{name}_lorentzian_reg": lorentzian_reg}
        eig_dict = {f"{name}_eigval_{i+1}": eigvals[i] for i in range(len(eigvals))}
        self.log_dict(dict(loss_dict, **eig_dict), on_step=True, on_epoch=True)
        return loss


def test_deepkanlda(n_states=2):
    from mlcolvar.data import DictDataset

    in_features, out_features = 2, n_states - 1
    layers = [in_features, 5, 3, out_features]

    # create dataset
    n_points = 500
    X, y = [], []
    for i in range(n_states):
        X.append(torch.randn(n_points, in_features) * (i + 1) + torch.Tensor([10 * i, (i - 1) * 10]))
        y.append(torch.ones(n_points) * i)

    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    dataset = DictDataset({"data": X, "labels": y})
    datamodule = DictModule(dataset, lengths=[0.8, 0.2], batch_size=n_states * n_points)

    # initialize CV
    opts = {
        "norm_in": {"mode": "mean_std"},
        "kan": {"grid": 5, "k": 3, "seed": 0},
        "lda": {},
    }
    model = DeepKAN_LDA(layers, n_states, options=opts)

    # create trainer and fit
    trainer = lightning.Trainer(max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False)
    trainer.fit(model, datamodule)

    # eval
    model.eval()
    with torch.no_grad():
        s = model(X).numpy()


if __name__ == "__main__":
    test_deepkanlda(n_states=2)
    test_deepkanlda(n_states=3)
