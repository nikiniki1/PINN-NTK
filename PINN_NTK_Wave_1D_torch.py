import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import time


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = 'cuda'


class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name = None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name
    def sample(self, N):
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*np.random.rand(N, self.dim)
        y = self.func(x)
        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)
        return x, y

def flatten_list(list_of_lists, flat_list=[]):
    if not list_of_lists:
        return flat_list
    else:
        for item in list_of_lists:
            if type(item) == list:
                flatten_list(item, flat_list)
            else:
                flat_list.append(item)

    return flat_list

class Solver():
    def __init__(self, model, ics_sampler, bcs_sampler, res_sampler, c):

        self.model = model.to(device)
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler
        self.c = c

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        X, _ = self.res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)

        self.loss_bcs_log = []
        self.loss_res_log = []
        self.loss_ut_ics_log = []

        self.lam_u_log = []
        self.lam_ut_log = []
        self.lam_r_log = []

    def nn_autograd_simple(self, points, order,axis=0):
        points.requires_grad=True
        f = self.model(points).sum().to(device)
        for i in range(order):
            grads, = torch.autograd.grad(f, points, create_graph=True)
            f = grads[:,axis].sum()
        return grads[:,axis]

    def wave_op(self, points):
        u_tt = self.nn_autograd_simple(points, order = 2, axis=0)
        u_xx = self.nn_autograd_simple(points, order = 2, axis=1)
        operator = u_tt - c**2 * u_xx
        return operator

    def fetch_minibatch(self, sampler, N):
        X, y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X[:N], y[:N]

    def list_to_vector(self, list_):
        return torch.cat([x.reshape(-1) for x in list_])

    def jacobian(self, f):
        jac = []
        for i in range(len(f)):
            f[i].backward(retain_graph=True)
            deriv = [w.grad.reshape(-1).to(device) if w.grad is not None else torch.tensor([0]).to(device) for w in self.model.parameters()]
            jac.append(self.list_to_vector(deriv))
        jac = torch.vstack(jac)
        return jac

    def compute_ntk(self, J1, J2):
        return J1 @ J2.T

    def loss_bnd_op(self, X_bcs_1, X_bcs_2, X_ics, y_ics):
        bnd1 = self.model(X_bcs_1)
        bnd2 = self.model(X_bcs_2)
        ics = self.model(X_ics)
        ics_u_t = self.nn_autograd_simple(X_ics, order=1, axis=0)

        loss_bnd1 = torch.mean(torch.square(bnd1))
        loss_bnd2 = torch.mean(torch.square(bnd2))
        loss_ics = torch.mean(torch.square(ics - y_ics))

        loss_bnd = loss_bnd1 + loss_bnd2 + loss_ics
        loss_ics_u_t = torch.mean(torch.square(ics_u_t))
        return loss_bnd, loss_ics_u_t

    def train(self, epochs=10000, batch_size=128, update_lam=False):
        lb, lr, li = 1,1,1
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9)
        for iter in range(epochs):
            self.optimizer.zero_grad()
            X_ics, y_ics = self.fetch_minibatch(ics_sampler, batch_size // 3)
            X_bcs_1, y_bcs_1 = self.fetch_minibatch(bcs_sampler[0], batch_size // 3)
            X_bcs_2, y_bcs_2 = self.fetch_minibatch(bcs_sampler[1], batch_size // 3)
            X_res, y_res = self.fetch_minibatch(res_sampler, batch_size)

            op = self.wave_op(X_res)
            loss_op = torch.mean(torch.square(op))
            loss_bnd, loss_ics = self.loss_bnd_op(X_bcs_1, X_bcs_2, X_ics, y_ics)
            loss = lr * loss_op + lb * loss_bnd + li * loss_ics
            loss.backward()

            self.optimizer.step()
            if iter % 1000 == 0:
                scheduler.step()
                print(scheduler.get_last_lr())

            if iter % 100 == 0:
                self.loss_bcs_log.append(loss_bnd)
                self.loss_res_log.append(loss_op)
                self.loss_ut_ics_log.append(loss_ics)

                print('It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_bcs: %.3e, Loss_ut_ics: %.3e,' %
                      (iter, loss, loss_op.item(), loss_bnd, loss_ics.item()))

                print('lambda_u: {:.3e}'.format(lb))
                print('lambda_ut: {:.3e}'.format(li))
                print('lambda_r: {:.3e}'.format(lr))

                if update_lam:
                    bcs = torch.vstack([X_bcs_1,X_bcs_2,X_ics])
                    X_ics, y_ics = self.fetch_minibatch(ics_sampler, batch_size)

                    u_ntk = self.model(bcs)
                    ut_ntk = self.nn_autograd_simple(X_ics, order=1, axis=0)
                    r_ntk = self.wave_op(X_res)

                    J_u = self.jacobian(u_ntk)
                    J_ut = self.jacobian(ut_ntk)
                    J_r = self.jacobian(r_ntk)

                    K_u = self.compute_ntk(J_u,J_u)
                    K_ut = self.compute_ntk(J_ut,J_ut)
                    K_r = self.compute_ntk(J_r, J_r)

                    trace_K = torch.trace(K_u) + torch.trace(K_ut) + \
                                           torch.trace(K_r)

                    lb = trace_K / torch.trace(K_u)
                    li = trace_K /torch.trace(K_ut)
                    lr = trace_K / torch.trace(K_r)

                    # # Store NTK weights
                    self.lam_u_log.append(lb)
                    self.lam_ut_log.append(li)
                    self.lam_r_log.append(lr)

    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        u_star = self.model(X_star).to('cpu')
        return u_star


    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        r_star = self.wave_op(X_star).to('cpu')
        return r_star

    def model(self):
        return self.model


# Define the exact solution and its derivatives
def u(x, a, c):
    """
    :param x: x = (t, x)
    """
    t = x[:,0:1]
    x = x[:,1:2]
    return np.sin(np.pi * x) * np.cos(c * np.pi * t) + \
            a * np.sin(2 * c * np.pi* x) * np.cos(4 * c  * np.pi * t)

def u_t(x,a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_t = -  c * np.pi * np.sin(np.pi * x) * np.sin(c * np.pi * t) - \
            a * 4 * c * np.pi * np.sin(2 * c * np.pi* x) * np.sin(4 * c * np.pi * t)
    return u_t

def u_tt(x, a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_tt = -(c * np.pi)**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
            a * (4 * c * np.pi)**2 *  np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return u_tt

def u_xx(x, a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_xx = - np.pi**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
              a * (2 * c * np.pi)** 2 * np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return  u_xx


def r(x, a, c):
    return u_tt(x, a, c) - c**2 * u_xx(x, a, c)


a = 0.5
c = 2

ics_coords = np.array([[0.0, 0.0],
                        [0.0, 1.0]])
bc1_coords = np.array([[0.0, 0.0],
                        [1.0, 0.0]])
bc2_coords = np.array([[0.0, 1.0],
                        [1.0, 1.0]])
dom_coords = np.array([[0.0, 0.0],
                        [1.0, 1.0]])

ics_sampler = Sampler(2, ics_coords, lambda x: u(x, a, c), name='Initial Condition 1')
bc1 = Sampler(2, bc1_coords, lambda x: u(x, a, c), name='Dirichlet BC1')
bc2 = Sampler(2, bc2_coords, lambda x: u(x, a, c), name='Dirichlet BC2')
bcs_sampler = [bc1, bc2]
res_sampler = Sampler(2, dom_coords, lambda x: r(x, a, c), name='Forcing')

model = torch.nn.Sequential(
    torch.nn.Linear(2, 500),
    torch.nn.Tanh(),
    torch.nn.Linear(500, 500),
    torch.nn.Tanh(),
    torch.nn.Linear(500, 500),
    torch.nn.Tanh(),
    torch.nn.Linear(500, 500),
    torch.nn.Tanh(),
    torch.nn.Linear(500, 1)
)
model = Solver(model, ics_sampler, bcs_sampler, res_sampler,c=c)

itertaions = 80001
update_lam = True

start = time.time()
model.train(epochs=itertaions, update_lam=update_lam)
end = time.time()

print(end-start)

loss_res = model.loss_res_log
loss_bcs = model.loss_bcs_log
loss_u_t_ics = model.loss_ut_ics_log

loss_res = list(map(lambda x: x.item(),loss_res))
loss_bcs = list(map(lambda x: x.item(),loss_bcs))
loss_u_t_ics = list(map(lambda x: x.item(),loss_u_t_ics))

fig = plt.figure(figsize=(6, 5))
plt.plot(loss_res, label='$\mathcal{L}_{r}$')
plt.plot(loss_bcs, label='$\mathcal{L}_{u}$')
plt.plot(loss_u_t_ics, label='$\mathcal{L}_{u_t}$')
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('torch_upd_lam=True.png',dpi=100)
plt.show()


nn = 200
t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
t, x = np.meshgrid(t, x)
X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))
X_star = torch.from_numpy(X_star).float().to(device)

u_star = u(X_star.cpu(), a,c)
R_star = r(X_star.cpu(), a, c)

# Predictions
u_pred = model.predict_u(X_star)
r_pred = model.predict_r(X_star)
error_u = np.linalg.norm(u_star.numpy() - u_pred.detach().numpy(), 2) / np.linalg.norm(u_star.numpy(), 2)

print('Relative L2 error_u: %e' % (error_u))



U_star = griddata(X_star.cpu(), u_star.flatten(), (t, x), method='cubic')
R_star = griddata(X_star.cpu(), R_star.flatten(), (t, x), method='cubic')
U_pred = griddata(X_star.cpu(), u_pred.detach().numpy().flatten(), (t, x), method='cubic')
R_pred = griddata(X_star.cpu(), r_pred.detach().numpy().flatten(), (t, x), method='cubic')


plt.figure(figsize=(16, 4))
plt.subplot(1, 3, 1)
plt.pcolor(t, x, U_star, cmap='jet')
plt.colorbar()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Exact u(t, x)')
plt.tight_layout()

plt.subplot(1, 3, 2)
plt.pcolor(t, x, U_pred, cmap='jet')
plt.colorbar()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title('Predicted u(t, x)')
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.pcolor(t, x, np.abs(U_star - U_pred), cmap='jet')
plt.colorbar()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title('Absolute error')
plt.tight_layout()

plt.savefig('result_torch_upd_lam=True.png',dpi=100)
plt.show()


lam_u_log = model.lam_u_log
lam_ut_log = model.lam_ut_log
lam_r_log = model.lam_r_log

lam_u_log = list(map(lambda x: x.item(),lam_u_log))
lam_ut_log = list(map(lambda x: x.item(),lam_ut_log))
lam_r_log = list(map(lambda x: x.item(),lam_r_log))

plt.figure(figsize=(6, 5))
plt.plot(lam_u_log, label='$\lambda_b$')
plt.plot(lam_ut_log, label='$\lambda_{b_t}$')
plt.plot(lam_r_log, label='$\lambda_{u}$')
plt.xlabel('iterations')
plt.ylabel('$\lambda$')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('torch_lambda.png', dpi=100)
plt.show()