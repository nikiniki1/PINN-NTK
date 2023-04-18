PINN NTK Pytorch implementation
-------------------------------
I'm not owner of the idea and paper, but original code use tensorflow and a I just wanted to rewrite it using Pytorch.
Some code I copy from original repository (plotting, Sampler), but solver is mine.

Problems and features
---------------------
In general, the solution is obtained, but lambdas is not equal. This may be a problem, but now I'm working on it to understand why there is such a difference. 

Original paper
--------------

S. Wang, X. Yu, and P. Perdikaris, “When and why PINNs fail to train: A neural tangent kernel perspective,” arXiv:2007.14527 [cs, math, stat], Jul. 2020, Accessed: Apr. 18, 2023. [Online]. Available: https://arxiv.org/abs/2007.14527
