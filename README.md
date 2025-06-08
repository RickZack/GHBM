<header align='center'>
    <img align="left" width="100" height="100" src="data/readme/logo.png" alt="">
    <h1> 
    Communication-Efficient Federated Learning with Generalized Heavy-Ball Momentum 
    </h1>
  <p  align='center' class="tagline">
        <b>Authors</b>: Riccardo Zaccone, Sai Praneeth Karimireddy, Carlo Masone, Marco Ciccone.
  </p>
</header>

🚀 Welcome to the official repository for  _"[Communication-Efficient Federated Learning with Generalized Heavy-Ball Momentum](https://openreview.net/forum?id=LNoFjcLywb)"_.

In this work, we propose FL algorithms based on a novel _Generalized Heavy-Ball Momentum_ (**GHBM**) formulation designed to overcome the limitation of classical momentum in FL w.r.t. heterogeneous local distributions and partial participation.  

💪 **GHBM is theoretically unaffected by client heterogeneity:​** it is proven to converge in (cyclic) partial participation as other momentum-based FL algorithms do in _full participation_.  
💡 **GHBM is easy to implement:**  it is based on the key modification to make momentum effective in heterogeneous FL with partial participation. Indeed, GHBM with $\tau=1$ is equivalent to classical momentum   
🧠 **GHBM is very flexible:** in GHBM clients are _stateless_, enabling its use in cross-device scenarios, at the expense of $1.5\times$ overhead w.r.t FedAvg. In cases when the participation is not critical (e.g. $\geq10\\%$), we provide even more communication-effients variants that exploit periodic participation and local state to recover the same communication complexity as FedAvg!  
🏆 **GHBM substantially improves the state-of-art:** extensive experimentation on **large-scale settings** with high data heterogeneity and low client participation shows that **GHBM and its variants reach much better final model quality** and  **much higher convergence speed**.  
 

📄 **Read our paper on:** [[OpenReview]](https://openreview.net/forum?id=LNoFjcLywb) [[ArXiv]](https://arxiv.org/abs/2311.18578) <br>
🌍 **[Demo & Project Page](https://rickzack.github.io/GHBM)**


## Implementation of other FL algorithms

This software additionally implements the code we used in the paper to simulate the following SOTA algorithms:

- [X] FedAvg - from [McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- [X] FedProx - from [Li at al., Federated Optimization in Heterogeneous Networks
](https://arxiv.org/abs/1812.06127)
- [X] SCAFFOLD - from [Karimireddi et al., SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378)
- [X] FedDyn - from [Acar et al., Federated Learning Based on Dynamic Regularization](https://arxiv.org/abs/2111.04263)
- [X] AdaBest - from [Varno et al., AdaBest: Minimizing Client Drift in Federated Learning via Adaptive Bias Estimation](https://arxiv.org/abs/2204.13170)
- [x] Mime - from [Karimireddy et al., Breaking the centralized barrier for cross-device federated learning](https://openreview.net/forum?id=FMPuzXV1fR)
- [X] FedCM - from [Xu et al., FedCM: Federated Learning with Client-level Momentum](https://arxiv.org/abs/2106.10874)

## Installation

### Requirements
To install the requirements, you can use the provided requirement file and use conda:
```shell
$ conda env create --file requirements/environment.yaml
$ conda activate ghbm
```

## Reproducing our experiments

### Perform a single run
If you just want to run a configuration, simply run the ```train.py``` specifying the command line arguments. Please note that default arguments are specified in ```./config``` folder and all is configured such that the parameters are the ones reported in the paper. For example, to run our GHBM on CIFAR-10 with ResNet-20, just issue:
```shell
# runs GHBM on CIFAR-10 with ResNet-20, using default parameters specified in config files (K=100, C=0.1)
$ python train.py model=resnet \
                  dataset=cifar10 \
                  algo=ghbm \
                  algo.params.common.alpha=0 \
                  algo.params.center_server.args.tau=10 \
                  algo.params.client.args.optim.args.lr=0.01 \
                  algo.params.client.args.optim.args.weight_decay=1e-5 
```
This software uses Hydra to configure experiments, for more information on how to provide command
line arguments, please refer to the [official documentation](https://hydra.cc/docs/advanced/override_grammar/basic/).

## Paper

**Communication-Efficient Federated Learning with Generalized Heavy-Ball Momentum**
_Riccardo Zaccone, Sai Praneeth Karimireddy, Carlo Masone, Marco Ciccone_ <br>
[[Paper]](https://openreview.net/forum?id=LNoFjcLywb)


## How to cite us

```
@article{zaccone2025communicationefficient,
      title={Communication-Efficient Heterogeneous Federated Learning with Generalized Heavy-Ball Momentum}, 
      author={Riccardo Zaccone and Sai Praneeth Karimireddy and Carlo Masone and Marco Ciccone},
      year={2025},
      journal={Transactions on Machine Learning Research},
      url={https://openreview.net/forum?id=LNoFjcLywb},
}
```