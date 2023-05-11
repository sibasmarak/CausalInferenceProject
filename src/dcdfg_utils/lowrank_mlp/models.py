import torch
import numpy as np
import torch.nn as nn
import os
import pickle

from src.dcdfg_utils.lowrank_mlp.module import MLPModularGaussianModule

class MLPModuleGaussianModel(nn.Module):
    """
    Lightning module that runs augmented lagrangian
    """

    def __init__(
        self,
        num_vars,
        num_layers,
        num_modules,
        hid_dim,
        nonlin="leaky_relu",
        lr_init=1e-3,
        reg_coeff=0.1,
        constraint_mode="exp",
    ):
        super().__init__()
        self.module = MLPModularGaussianModule(
            num_vars,
            num_layers,
            num_modules,
            hid_dim,
            nonlin=nonlin,
            constraint_mode=constraint_mode,
        )
        # augmented lagrangian params
        # mu: penalty
        # gamma: multiplier
        self.mu_init = 1e-8
        self.gamma_init = 0.0
        self.omega_gamma = 1e-4
        self.omega_mu = 0.9
        self.h_threshold = 1e-8
        self.mu_mult_factor = 2
        # opt params
        # self.save_hyperparameters() 
        # self.hparams["name"] = self.__class__.__name__
        # self.hparams["module_name"] = self.module.__class__.__name__

        self.lr_init = lr_init
        self.reg_coeff = reg_coeff
        self.constraint_mode = constraint_mode

        # initialize stuff for learning loop
        self.aug_lagrangians = []
        self.not_nlls = []  # Augmented Lagrangrian minus (pseudo) NLL
        self.nlls = []  # NLL on train
        self.nlls_val = []  # NLL on validation
        self.best_nlls_val = np.inf
        self.regs = []

        # Augmented Lagrangian stuff
        self.mu = self.mu_init
        self.gamma = self.gamma_init

        # Optimization stuff
        self.satisfied = False
        self.model_freeze = False
        self.patience = 5
        self.frozen_patience = 5


        # bookkeeping for training
        self.acyclic = 0.0
        self.aug_lagrangians_val = []
        self.best_aug_lagrangians_val = np.inf
        self.not_nlls_val = []
        self.constraint_value = 0.0
        self.constraints_at_stat = []
        self.reg_value = 0.0
        self.internal_checkups = 0.0
        self.stationary_points = 0.0

        
    
    def dump(self, obj, exp_path, name, txt=False):
        """
        Save object either as a pickle or text file
        :param obj: object to save
        :param str exp_path: path where to save
        :param str name: name of the saved file
        :param boolean txt: if True, save as a text file
        """
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        if not txt:
            with open(os.path.join(exp_path, name + ".pkl"), "wb") as f:
                pickle.dump(obj, f)
        else:
            with open(os.path.join(exp_path, name + ".txt"), "w") as f:
                f.write(str(obj))
        print("Stuff dumped ! ")
    
    
    def forward(self, data):
        x, masks, regimes = data
        log_likelihood = torch.sum(
            self.module.log_likelihood(x) * masks, dim=0
        ) / masks.size(0)
        return -torch.mean(log_likelihood)

    def get_augmented_lagrangian(self, nll, constraint_violation, reg):
        # compute augmented langrangian
        return (
            nll
            + self.reg_coeff * reg
            + self.gamma * constraint_violation
            + 0.5 * self.mu * constraint_violation**2
        )

    def train(self, train_data, test_data, opt):
        # initialize optimizer
        self.optimizer = self.configure_optimizers()

        # start training
        for iter in range(opt.num_train_iter):
            # get data
            x, masks, regimes = train_data.sample(opt.train_batch_size)

            # compute loss
            nll, constraint_violation, reg = self.module.losses(x, masks)
            aug_lagrangian = self.get_augmented_lagrangian(nll, constraint_violation, reg)

            # update parameters
            self.optimizer.zero_grad()
            aug_lagrangian.backward()
            self.optimizer.step()

            # logging
            self.nlls.append(nll.item())
            self.aug_lagrangians.append(aug_lagrangian.item())
            self.not_nlls.append(aug_lagrangian.item() - nll.item())

            if not self.model_freeze and self.satisfied and self.patience == 0:
                self.model_freeze = True 
                self.freeze_model()

            if iter % 100 == 0:
                validation_metrics = self.validation(test_data, opt)
                self.aug_lagrangians_val += [validation_metrics["aug_lagrangian"]]
                self.constraint_value = validation_metrics["constraint"]
                self.reg_value = validation_metrics["reg"]
                self.not_nlls_val += [validation_metrics["aug_lagrangian"] - validation_metrics["nll"]]
                self.nlls_val += [validation_metrics["nll"]]
                self.regs += [self.reg_value]
                print(f'Validation NLL {validation_metrics["nll"]}')
                # self.acyclic = self.module.check_acyclicity()

            
            if not self.satisfied:
                self.update_lagrangians()

            if iter % 1000 == 0 and self.satisfied:  #NOTE: Does it make sense to check this every 1k iterations?
                if self.patience > 0 :
                    self.update_lagrangians()
                    self.early_stop_callback1()
                elif self.patience == 0 and self.frozen_patience > 0:
                    self.early_stop_callback2()

            if self.satisfied and self.patience == 0 and self.frozen_patience == 0:
                break

            
    def freeze_model(self):
        # freeze and prune adjacency
        self.module.threshold()
        # WE NEED THIS BECAUSE IF it's exactly a DAG THE POWER ITERATIONS DOESN'T CONVERGE
        # TODO Just refactor and remove constraint at validation time
        self.module.constraint_mode = "exp"
        # remove dag constraints: we have a prediction problem now!
        self.gamma = 0.0
        self.mu = 0.0

    def early_stop_callback1(self):
        if self.aug_lagrangians_val[-1] < self.best_aug_lagrangians_val:
            self.patience = 5
            self.best_aug_lagrangians_val = self.aug_lagrangians_val[-1]
        else:
            self.patience -=1 

    def early_stop_callback2(self):
        if self.nlls_val[-1] < self.best_nlls_val:
            self.frozen_patience = 5
            self.best_nlls_val = self.nlls_val[-1]
        else:
            self.frozen_patience -=1 
            
    def validation(self, test_data, opt):
        # NOTE: once we remove partitions, this might fail
        # NOTE (contd.): need to re-write to include iterations (and then take mean of losses) rather than on whole test_data
        # compute validation loss
         with torch.no_grad():
            x, masks, regime = test_data.sample(test_data.num_samples)
            nll, constraint_violation, reg = self.module.losses(x, masks)
            aug_lagrangian = self.get_augmented_lagrangian(nll, constraint_violation, reg)

            return {
                "aug_lagrangian": aug_lagrangian,
                "nll": nll,
                "constraint": constraint_violation,
                "reg": reg,
            }

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.module.parameters(), lr=self.lr_init)

    def update_lagrangians(self):
        self.internal_checkups += 1
        # self.log("Monitor/checkup", self.internal_checkups)
        # compute delta for gamma to check convergence status
        delta_gamma = -np.inf
        if len(self.aug_lagrangians_val) >= 3:
            t0, t_half, t1 = (
                self.aug_lagrangians_val[-3],
                self.aug_lagrangians_val[-2],
                self.aug_lagrangians_val[-1],
            )
            # if the validation loss went up and down, do not update lagrangian and penalty coefficients.
            if min(t0, t1) < t_half < max(t0, t1):
                delta_gamma = -np.inf
            else:
                delta_gamma = (t1 - t0) / 100

        # if we found a stationary point, but that is not satisfying the acyclicity constraints
        if (
            self.constraint_value > self.h_threshold
            and not self.acyclic
            and self.mu < 1e15
            or self.stationary_points < 10
        ):
            if abs(delta_gamma) < self.omega_gamma or delta_gamma > 0:
                self.stationary_points += 1
                # self.log("Monitor/stationary", self.stationary_points)
                self.gamma += self.mu * self.constraint_value
                print('Updated gamma to {}'.format(self.gamma))

                # Did the constraint improve sufficiently?
                if len(self.constraints_at_stat) > 1:
                    if (
                        self.constraint_value
                        > self.constraints_at_stat[-1] * self.omega_mu
                    ):
                        self.mu *= self.mu_mult_factor
                        print('Updated mu to {}'.format(self.mu))
                self.constraints_at_stat.append(self.constraint_value)

                # little hack to make sure the moving average is going down.
                gap_in_not_nll = (
                    self.get_augmented_lagrangian(
                        0.0, self.constraint_value, self.reg_value
                    )
                    - self.not_nlls_val[-1]
                )
                assert gap_in_not_nll > -1e-2
                self.aug_lagrangians_val[-1] += gap_in_not_nll

                # reset optimizer
                self.optimizer = self.configure_optimizers()

        # if we found a stationary point, that satisfies the acyclicity constraints, raise this flag, it will activate patience and terminate training soon
        else:
            self.satisfied = True


            


            
            
