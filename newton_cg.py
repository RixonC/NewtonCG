import numpy as np
import torch


def _cg(A, b, rtol, maxit, M_inv=None):
    """The conjugate gradient method to approximately solve ``Ax = b`` for 
    ``x``, when ``A`` is symmetric and positive-definite.
    
    Args:
        A (callable): given vector ``x``, returns matrix-vector product ``Ax``.
            Assumes input and output of ``A`` are each stored as a flat 
            ``Tensor``, for performance. Matrix must be symmetric and 
            positive-definite.
        b (torch.Tensor): vector ``b`` in ``Ax = b``. Assumes ``b`` is stored as
            a flat ``Tensor``, for performance.
        rtol (float): termination tolerance for relative residual.
        maxit (int): maximum iterations.
        M_inv (callable, optional): given vector ``v``, return the inverse of 
            ``M`` times ``v``, when ``M`` is a preconditioner. Assumes input 
            and output of ``M_inv`` are each stored as a flat ``Tensor``, for 
            performance. Defaults to ``None``.
    """
    if rtol < 0.0:
        raise ValueError("Invalid termination tolerance: {}."
                         " It must be non-negative.".format(rtol))
    if maxit < 0:
        raise ValueError("Invalid maximum iterations: {}."
                         " It must be non-negative.".format(maxit))
    
    n = len(b)
    device = b.device
    iters = 0
    x = torch.zeros(n, device=device)
    bb = torch.dot(b,b).item()
    if bb == 0:
        return x
    r = b.clone()
    if M_inv is None:
        z = r
        assert(id(z) == id(r))
    else:
        z = M_inv(r)
    p = z.clone()
    rz_old = torch.dot(r,z).item()

    while iters < maxit:
        iters += 1
        Ap = A(p)
        pAp = torch.dot(p,Ap).item()
        assert pAp > 0, "A is not positive-definite."
        alpha = rz_old / pAp
        x.add_(p, alpha=alpha)
        r.sub_(Ap, alpha=alpha)
        rr = torch.dot(r,r).item()
        if np.sqrt(rr/bb) <= rtol:
            break
        if M_inv is None:
            assert(id(z) == id(r))
            rz_new = rr
        else:
            z = M_inv(r)
            rz_new = torch.dot(r,z).item()
        beta = rz_new/rz_old
        p.mul_(beta).add_(z)
        rz_old = rz_new
    
    return x


class NewtonCG(torch.optim.Optimizer):
    """Implements Newton-CG algorithm with backtracking line-search. CG refers 
    to the conjugate gradient method, which is the sub-problem solver.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float, optional): learning rate, if ``line_search_max_iter = 0``. 
            Otherwise, initial step-size tried in line-search. Defaults to 
            ``1``.
        cg_tol (float, optional): termination tolerance for relative 
            residual in CG. Defaults to ``1e-4``.
        cg_max_iter (int, optional): maximum CG iterations. Defaults to ``10``.
        line_search_max_iter (int, optional): maximum line-search iterations. 
            Defaults to ``100``.
        line_search_param (float, optional): backtracking line-search parameter. 
            Defaults to ``1e-4``.
    """
    def __init__(self, params, lr=1, cg_tol=1e-4, cg_max_iter=10, 
                 line_search_max_iter=100, line_search_param=1e-4):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}."
                             " It must be non-negative.".format(lr))
        if cg_tol < 0.0:
            raise ValueError("Invalid termination tolerance for CG: {}."
                             " It must be non-negative.".format(cg_tol))
        if cg_max_iter < 0:
            raise ValueError("Invalid maximum CG iterations: {}"
                             ". It must be non-negative.".format(cg_max_iter))
        if line_search_max_iter < 0:
            raise ValueError("Invalid maximum line-search iterations: {}."
                             " It must be non-negative.".format(
                                line_search_max_iter))
        if line_search_param < 0.0:
            raise ValueError("Invalid backtracking line-search parameter: {}."
                             " It must be non-negative.".format(
                                line_search_param))
        defaults = dict(lr=lr, cg_tol=cg_tol, cg_max_iter=cg_max_iter, 
                        line_search_max_iter=line_search_max_iter,
                        line_search_param=line_search_param)
        super(NewtonCG, self).__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("NewtonCG doesn't support per-parameter options "
                             "(parameter groups)")
        self._params = self.param_groups[0]['params']

    def _get_flat_grad(self):
        """Return stored gradients from parameters, as a flat ``Tensor``."""
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().reshape(-1)
            else:
                view = p.grad.reshape(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _get_flat_hess_vect(self, vect, closure):
        """Return Hessian-vector product, as a flat ``Tensor``.
        
        Args:
            vect (torch.Tensor): vector used in Hessian-vector product, as a 
                flat ``Tensor``.
            closure (callable): a closure that reevaluates the model and returns
                the loss.
        """
        with torch.enable_grad():
            loss = closure()
            assert loss.requires_grad == True
            for p in self._params: # zero gradients
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            loss.backward(create_graph=True)
            tmp_flat_grad = self._get_flat_grad()
            for p in self._params: # zero gradients
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            dot_prod = torch.dot(tmp_flat_grad, vect)
            dot_prod.backward()
        return self._get_flat_grad()

    def _get_flat_hess_inv_vect(self, vect, cg_tol, cg_max_iter, closure):
        """Return approximate solution, as a flat ``Tensor``, to Newton method's
        sub-problem using CG.
        
        Args:
            vect (torch.Tensor): vector in sub-problem, as a flat ``Tensor``.
            cg_tol (float): termination tolerance for relative residual in CG.
            cg_max_iter (int): maximum CG iterations.
            closure (callable): a closure that reevaluates the model and returns
                the loss.
        """
        hess_vect = lambda v : self._get_flat_hess_vect(v, closure)
        return _cg(hess_vect, vect, cg_tol, cg_max_iter)

    def _get_update_direction(self, flat_grad, cg_tol, cg_max_iter, closure):
        """Return Newton method's update direction, as a flat ``Tensor``.
        
        Args:
            flat_grad (torch.Tensor): the gradient, as a flat ``Tensor``.
            cg_tol (float): termination tolerance for relative residual in CG.
            cg_max_iter (int): maximum CG iterations.
            closure (callable): a closure that reevaluates the model and returns
                the loss.
        """
        return -self._get_flat_hess_inv_vect(flat_grad, cg_tol, cg_max_iter, 
                                             closure)

    def _line_search(self, current_loss, current_flat_grad, update_direction, 
                     init_step_size, line_search_max_iter, line_search_param, 
                     closure):
        """Perform backtracking line-search on the loss.
        
        Args:
            current_loss (float): the current loss.
            current_flat_grad (torch.Tensor): the current gradient, as a flat 
                ``Tensor``.
            update_direction (torch.Tensor): update direction, as a flat 
                ``Tensor``.
            init_step_size (float): initial step-size to try.
            line_search_max_iter (int): maximum line-search iterations.
            line_search_param (float): backtracking line-search parameter.
            closure (callable): a closure that reevaluates the model and returns
                the loss.
        """
        step_size = init_step_size
        ls_iter = 0
        while ls_iter < line_search_max_iter:
            directional_derivative = torch.dot(
                current_flat_grad, update_direction)
            condition = (current_loss 
                + step_size * line_search_param * directional_derivative)
            new_loss = self._directional_evaluate(
                update_direction, step_size, closure)
            if new_loss <= condition:
                break
            step_size *= 0.5
            ls_iter += 1
        return step_size, ls_iter

    def _update_params(self, update_direction, step_size):
        """Update parameters given an update direction and step-size.
        
        Args:
            update_direction (torch.Tensor): update direction, as a flat 
                ``Tensor``.
            step_size (float): step-size.
        """
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(update_direction[offset:offset+numel].reshape_as(p), 
                alpha=step_size)
            offset += numel

    def _directional_evaluate(self, update_direction, step_size, closure):
        """Return the loss after the parameters are updated given an update 
        direction and step-size. Changes made to the parameters, for this
        evaluation, are undone after this evaluation.
        
        Args:
            update_direction (torch.Tensor): update direction, as a flat 
                ``Tensor``.
            step_size (float): step-size.
            closure (callable): a closure that reevaluates the model and returns
                the loss.
        """
        self._update_params(update_direction, step_size)
        ret = closure()
        self._update_params(update_direction, -step_size)
        return ret

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.
        
        Args:
            closure (callable): A closure that reevaluates the model and returns
                the loss.
        """
        assert len(self.param_groups) == 1
        closure = torch.enable_grad()(closure)

        lr = self.param_groups[0]['lr']
        cg_tol = self.param_groups[0]['cg_tol']
        cg_max_iter = self.param_groups[0]['cg_max_iter']
        line_search_max_iter = self.param_groups[0]['line_search_max_iter']
        line_search_param = self.param_groups[0]['line_search_param']

        loss = closure()
        flat_grad = self._get_flat_grad()
        update_direction = self._get_update_direction(
            flat_grad, cg_tol, cg_max_iter, closure)
        step_size, ls_iter = self._line_search(
            loss, flat_grad, update_direction, lr, line_search_max_iter,
            line_search_param, closure)
        self._update_params(update_direction, step_size)

        return loss