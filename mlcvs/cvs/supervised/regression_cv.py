import torch
import pytorch_lightning as pl

from mlcvs.core import FeedForward, Normalization
from mlcvs.utils.decorators import decorate_methods,call_submodules_hooks,allowed_hooks
from mlcvs.cvs.utils import BaseCV
from mlcvs.core.loss import MSE_loss

__all__ = ["Regression_CV"]

@decorate_methods(call_submodules_hooks,methods=allowed_hooks)
class Regression_CV(BaseCV, pl.LightningModule):
    """
    Example of collective variable obtained with a regression task.
    Combine the inputs with a neural-network and optimize it to match a target function.

    For the training it requires a DictionaryDataset with the keys 'data' and 'target' and optionally 'weights'.
    MSE Loss is used to optimize it.
    """

    BLOCKS = ['normIn', 'nn']

    def __init__(self, 
                layers : list, 
                options : dict = None,
                **kwargs):
        """Example of collective variable obtained with a regression task.

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default None.
            Available blocks: ['normIn', 'nn'].
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs)

        # ===== BLOCKS =====

        # Members
        options = self.initialize_block_defaults(options=options)

        # Initialize normIn
        o = 'normIn'
        if ( not options[o] ) and (options[o] is not None):
            self.normIn = Normalization(self.in_features,**options[o])

        # initialize NN
        o = 'nn'
        self.nn = FeedForward(layers, **options[o])

        # ===== LOSS OPTIONS =====
        self.loss_options = {}   

    def loss_function(self, diff, options = {}):
        # Reconstruction (MSE) loss
        return MSE_loss(diff,options)

    def training_step(self, train_batch, batch_idx):
        options = self.loss_options
        # =================get data===================
        x = train_batch['data']
        labels = train_batch['target']
        if 'weights' in train_batch:
            options['weights'] = train_batch['weights'] 
        # =================forward====================
        y = self(x)
        # ===================loss=====================
        diff = y - labels
        loss = self.loss_function(diff, options)
        # ====================log===================== 
        name = 'train' if self.training else 'valid'       
        self.log(f'{name}_loss', loss, on_epoch=True)
        return loss

def test_regression_cv():
    """
    Create a synthetic dataset and test functionality of the Regression_CV class
    """
    from mlcvs.utils.data import DictionaryDataset, TensorDataModule

    in_features, out_features = 2,1 
    layers = [in_features, 5, 10, out_features]

    # initialize via dictionary
    options= { 'FeedForward' : { 'activation' : 'relu' } }

    model = Regression_CV( layers = layers,
                        options = options)
    print('----------')
    print(model)

    # create dataset
    X = torch.randn((100,2))
    y = X.square().sum(1)
    dataset = DictionaryDataset({'data':X,'target':y})
    datamodule = TensorDataModule(dataset,lengths=[0.75,0.2,0.05], batch_size=25)
    # train model
    model.set_optim_name('SGD')
    model.set_optim_options(lr=1e-2)
    trainer = pl.Trainer(accelerator='cpu',max_epochs=1,logger=None, enable_checkpointing=False)
    trainer.fit( model, datamodule )
    model.eval()
    # trace model
    traced_model = model.to_torchscript(file_path=None, method='trace', example_inputs=X[0])
    assert torch.allclose(model(X),traced_model(X))

    print('custom loss')
    # use custom loss
    trainer = pl.Trainer(accelerator='cpu',max_epochs=1,logger=None, enable_checkpointing=False)

    model = Regression_CV( layers = [2,10,10,1])
    model.set_loss_fn( lambda x, options: x.abs().mean() )
    trainer.fit( model, datamodule )

if __name__ == "__main__":
    test_regression_cv() 