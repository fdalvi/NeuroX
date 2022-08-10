"""Module for Gaussian Method to rank neurons

This module implements the Gaussian Method to rank the neuron importance

.. seealso::
        Lucas Torroba Hennigen, Adina Williams, and Ryan Cotterell. Intrinsic probing through dimension
selection. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP), pp. 197–216, Online, 2020. Association for Computational Linguistics. doi:
10.18653/v1/2020.emnlp-main.15.a

"""
import numpy as np

import torch

import torch.distributions.multivariate_normal as mn

class GaussianProbe():
    def __init__(self, X,y):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.train_features = torch.tensor(X).to(self.device)
        
        self.train_labels = torch.tensor(y).to(self.device).long()
        self.labels_dim = len(set(self.train_labels.tolist()))
  
        # self._get_categorical()
        self._get_mean_and_cov()

        self.feature_sets = {'train': self.train_features}
        self.label_sets = {'train': self.train_labels}
        self.categorical_sets = {}
        self.train_categorical = self._get_categorical('train')
        self.categorical_sets['train'] = self.train_categorical
    def _get_categorical(self, set_name):
        
        counts = torch.histc(self.label_sets[set_name].float(), bins=self.labels_dim)
        categorical = (counts / self.label_sets[set_name].size()[0]).to(self.device)
        return categorical
    def _get_mean_and_cov(self):
        self.features_by_label = [self.train_features[(self.train_labels == label).nonzero(as_tuple=False)].squeeze(1)
                                  for label in range(self.labels_dim)]
        empirical_means = torch.stack([features.mean(dim=0) for features in self.features_by_label])
        empirical_covs = [torch.tensor(np.cov(features.cpu(), rowvar=False)) for features in self.features_by_label]
        mu_0 = empirical_means  # [label_dim,feature_dim]
    
        lambda_0 = torch.stack([torch.diag(torch.diagonal(cov)) for cov in empirical_covs]).to(self.device)
        v_0 = torch.tensor(self.train_features.shape[1] + 2).to(self.device)  # int
        k_0 = torch.tensor(0.01).to(self.device)
        N_v = torch.tensor([features.shape[0] for features in self.features_by_label]).to(self.device)  # [label_dim]
        k_n = k_0 + N_v  # [label_dim]
        v_n = v_0 + N_v  # [label_dim]
        mu_n = (k_0 * mu_0 + N_v.unsqueeze(1) * empirical_means) / k_n.unsqueeze(1)  # [label_dim,feature_dim]
        S = []
        for label in range(self.labels_dim):
            features_minus_mean = self.features_by_label[label] - empirical_means[label]
            S.append(features_minus_mean.T @ features_minus_mean)
        S = torch.stack(S).to(self.device)
        lambda_n = lambda_0 + S
        self.mu_star = mu_n
        sigma_star = lambda_n / (v_n + self.train_features.shape[1] + 2).view(self.labels_dim, 1, 1)
        min_eig = []
        for sigma in sigma_star:
            eigs = torch.eig(sigma).eigenvalues[:, 0]
            min_eig.append(eigs.min())
        min_eig = torch.tensor(min_eig).to(self.device)
        sigma_star[min_eig < 0] -= min_eig.view(sigma_star.shape[0], 1, 1)[min_eig < 0] * torch.eye(
            sigma_star.shape[1]).to(self.device) * torch.tensor(10).to(self.device)
        self.sigma_star = sigma_star

    def _get_distributions(self, selected_features):
        self.distributions = []
        for label in range(self.labels_dim):
            if self.train_categorical[label].item() == 0:
                self.distributions.append(torch.distributions.normal.Normal(0., 0.))
            else:
                self.distributions.append(mn.MultivariateNormal(
                    self.mu_star[label, selected_features].double(),
                    self.sigma_star[label, selected_features][:, selected_features]
                ))

    def _compute_probs(self, selected_features, set_name):
        features = self.feature_sets[set_name]
        with torch.no_grad():
            log_probs = []
            for i in range(self.labels_dim):
                if self.train_categorical[i] == 0:
                    log_probs.append(torch.zeros(features.shape[0], dtype=torch.double))
                else:
                    log_probs.append(self.distributions[i].log_prob(features[:, selected_features]))
            log_probs = torch.stack(log_probs, dim=1)
            log_prob_times_cat = log_probs + self.train_categorical.log()
            self.not_normalized_probs = log_prob_times_cat
            self.normalizer = log_prob_times_cat.logsumexp(dim=1)
        self.probs = log_prob_times_cat - self.normalizer.unsqueeze(1)
    def _predict(self, set_name: str):
        preds = self.probs.argmax(dim=1)
        labels = self.label_sets[set_name]
        accuracy = ((preds == labels)).nonzero(as_tuple=False).shape[0] / labels.shape[0]
        categorical = self.categorical_sets[set_name]
        entropy = torch.distributions.Categorical(categorical).entropy()
        with torch.no_grad():
            conditional_entropy = -self.probs[list(range(self.probs.shape[0])), labels].mean()
        mutual_inf = (entropy - conditional_entropy) / torch.tensor(2.0).log()
        return accuracy, mutual_inf.item(), (mutual_inf / entropy).item()

    def evaluate_probe(self, X_test, y_test,selected_neurons):
        self.test_features = torch.tensor(X_test).to(self.device)
        
        self.test_labels = torch.tensor(y_test).to(self.device).long()
        self.feature_sets['test'] = self.test_features
        self.label_sets['test'] = self.test_labels
        self.test_categorical = self._get_categorical('test')

        self.categorical_sets['test'] = self.test_categorical
        
        self._get_distributions(selected_neurons)

        self._compute_probs(selected_neurons, 'test')
        test_acc, test_mi, test_nmi = self._predict('test')
        return test_acc
    def get_neuron_ordering(self, num_of_neurons):
        selected_neurons = []
        # for most configs it crashes even before 400, but anyway we don't need such a large number
        for num_of_neuron in range(len(selected_neurons), num_of_neurons):
            # for num_of_neurons in range(len(selected_neurons), constants.SUBSET_SIZE):
            best_neuron = -1
            best_acc = 0.
            best_mi, best_nmi = float('-inf'), float('-inf')
            
            acc_on_best_mi = 0.
            mi_on_best_acc, nmi_on_best_acc = 0., 0.
            for neuron in range(768):
                if neuron in selected_neurons:
                    continue
                self._get_distributions(selected_neurons + [neuron])
                self._compute_probs(selected_neurons + [neuron], 'train')
                acc, mi, nmi = self._predict('train')
                
                if mi > best_mi:
                    best_mi = mi
                    best_nmi = nmi
                    best_neuron = neuron
                    acc_on_best_mi = acc
                    
            selected_neurons.append(best_neuron)
            print('added neuron ', best_neuron)
            self._get_distributions(selected_neurons)
            self._compute_probs(selected_neurons, 'train')
            train_acc, train_mi, train_nmi = self._predict('train')
        return selected_neurons
