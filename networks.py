import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear
from blitz.modules import BayesianConv2d
from blitz.utils import variational_estimator
from blitz.modules.base_bayesian_module import BayesianModule



'''CNN with Dropout'''
class CNN(nn.Module):
    def __init__(self, ):
        super(CNN, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fcs = nn.Sequential(
            nn.Linear(11 * 11 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        out = x
        out = self.convs(out)
        out = out.view(-1, 11 * 11 * 32)
        out = self.fcs(out)
        return out


@variational_estimator
class BayesianCNN(nn.Module):
    def __init__(self, conv_rho=-8, conv_sigma_1=1.2, conv_sigma_2=0.009, conv_pi=0.5, linear_rho=-8, linear_sigma_1=0.4, linear_sigma_2=0.003, linear_pi=0.5):
        super().__init__()

        self.conv_rho = conv_rho
        self.conv_sigma_1 = conv_sigma_1
        self.conv_sigma_2 = conv_sigma_2
        self.conv_pi = conv_pi
        self.linear_rho = linear_rho
        self.linear_sigma_1 = linear_sigma_1
        self.linear_sigma_2 = linear_sigma_2
        self.linear_pi = linear_pi

        self.convs = nn.Sequential(
            BayesianConv2d(
                1, 32, (4, 4),
                prior_sigma_1 = self.conv_sigma_1,
                prior_sigma_2 = self.conv_sigma_2,
                prior_pi = self.conv_pi,
                posterior_rho_init = self.conv_rho
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            BayesianConv2d(
                32, 32, (4, 4),
                prior_sigma_1 = self.conv_sigma_1,
                prior_sigma_2 = self.conv_sigma_2,
                prior_pi = self.conv_pi,
                posterior_rho_init = self.conv_rho
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fcs = nn.Sequential(
            BayesianLinear(
                11*11*32, 128,
                prior_sigma_1 = self.linear_sigma_1,
                prior_sigma_2 = self.linear_sigma_2,
                prior_pi = self.linear_pi,
                posterior_rho_init = self.linear_rho
            ),
            nn.ReLU(),
            BayesianLinear(
                128, 10,
                prior_sigma_1 = self.linear_sigma_1,
                prior_sigma_2 = self.linear_sigma_2,
                prior_pi = self.linear_pi,
                posterior_rho_init = self.linear_rho
            ),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = x
        out = self.convs(out)
        out = out.view(-1,11*11*32)
        out = self.fcs(out)
        return out


def kl_divergence_from_nn(model):
    """
    Gathers the KL Divergence from a nn.Module object
    Works by gathering each Bayesian layer kl divergence and summing it, doing nothing with the non Bayesian ones
    """
    kl_divergence = 0
    for module in model.modules():
        if isinstance(module, (BayesianModule)):
            kl_divergence += module.log_variational_posterior - module.log_prior
    return kl_divergence


class sample_elbo_class(nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', complexity_cost_weight=None) -> None:
        super(sample_elbo_class, self).__init__(size_average, reduce, reduction)

        self.complexity_cost_weight = complexity_cost_weight
        self.model = None

    def forward(self, input: torch.Tensor, target: torch.Tensor, model) -> torch.Tensor:
        self.model = model
        return self.sample_elbo(input, target, model=model, complexity_cost_weight=self.complexity_cost_weight)

    def nn_kl_divergence(self, model):
        return kl_divergence_from_nn(model)

    def sample_elbo(self,
                    inputs,
                    labels,
                    model=None,
                    sample_nbr=20,
                    criterion=nn.CrossEntropyLoss(),
                    complexity_cost_weight=1):

        loss = 0
        for _ in range(sample_nbr):
            loss += F.cross_entropy(inputs, labels)
            loss += self.nn_kl_divergence(model) * complexity_cost_weight
        return loss / sample_nbr