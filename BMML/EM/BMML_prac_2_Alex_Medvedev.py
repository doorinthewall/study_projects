import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from scipy.signal import fftconvolve

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H, W, K = X.shape
    h, w = F.shape
    d_h = np.arange(H-h+1)[:,np.newaxis] + np.arange(h)[np.newaxis,:]
    d_w = np.arange(W-w+1)[:,np.newaxis] + np.arange(w)[np.newaxis,:]
    res = []
    to_derive = B[d_h[:,np.newaxis,:,np.newaxis],d_w[np.newaxis,:,np.newaxis,:]]
    for k in range(K):
        submatrices = X[d_h[:,np.newaxis,:,np.newaxis],d_w[np.newaxis,:,np.newaxis,:],k]
        
        density = -(((submatrices - F[np.newaxis, np.newaxis, :, :])**2)/(2*s**2)).sum(axis = (2,3))\
        + (((submatrices - to_derive)**2)/(2*s**2)).sum(axis = (2,3))
        #density = ((submatrices*((2*(to_derive - F[np.newaxis, np.newaxis, :, :]))[:,:,:,:,np.newaxis]) \
        #+ ((F**2)[np.newaxis, np.newaxis, :, :]-to_derive**2)[:,:,:,:,np.newaxis])/(2*s**2)).sum(axis = (2,3))
        to_add = (-np.log(s) - 0.5*np.log(2*np.pi) - \
        ((X[:,:,k] - B[:,:])**2)/(2*s**2))[np.newaxis, np.newaxis, :,:].sum(axis = (2,3))
        res += [(density + to_add)]
        
    return np.stack(res, axis=-1)

    # submatrices_F - подматрицы X, соответствующие F для d_h, d_w (H-h+1, W-w+1, h, w)
    #d_h = np.arange(H-h+1)[:,np.newaxis] + np.arange(h)[np.newaxis,:]
    #d_w = np.arange(W-w+1)[:,np.newaxis] + np.arange(w)[np.newaxis,:]
    #submatrices = X[d_h[:,np.newaxis,:,np.newaxis],d_w[np.newaxis,:,np.newaxis,:],:]
    #density = np.log(norm.pdf(submatrices, F[np.newaxis, np.newaxis, :, :, np.newaxis], s)).sum(axis=(2,3))
    #density = norm.pdf(X, 0, s)
    #d_h = np.arange(H-h+1)[:,np.newaxis] + np.arange(h)[np.newaxis,:]
    #d_w = np.arange(W-w+1)[:,np.newaxis] + np.arange(w)[np.newaxis,:]
    #ans = np.log(density[d_h[:,np.newaxis,:,np.newaxis],d_w[np.newaxis,:,np.newaxis,:],:] +\
    #            F[np.newaxis, np.newaxis, :, :, np.newaxis]).sum(axis=(2,3))
    # submatrices_B - подматрицы B, которые надо вычесть для каждого d_h, d_w
    #to_derive = B[d_h[:,np.newaxis,:,np.newaxis],d_w[np.newaxis,:,np.newaxis,:]]
    #to_derive = np.log(norm.pdf(submatrices, to_derive[:,:,:,:,np.newaxis], s)).sum(axis = (2,3))
    #to_derive = np.log(density[d_h[:,np.newaxis,:,np.newaxis],d_w[np.newaxis,:,np.newaxis,:],:]\
    #                  + to_derive[:,:,:,:,np.newaxis]).sum(axis = (2,3))
    #to_add = np.log(norm.pdf(X, B[:,:, np.newaxis], s))[np.newaxis, np.newaxis, :,:,:].sum(axis = (2,3))
    #to_add = np.log(density + B[:,:,np.newaxis])[np.newaxis, np.newaxis, :,:,:].sum(axis = (2,3))
    #density += to_add - to_derive
    #ans += to_add - to_derive
    #return density
    #return ans


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    #L = calculate_log_probability(X, F, B, s) + np.log(A)[:,:,np.newaxis]
    H, W, K = X.shape
    L = calculate_log_probability(X, F, B, s)
    aid = A[:,:,np.newaxis] + np.zeros(L.shape)
    if use_MAP:
        L = np.sum(L[q[0,:],q[1,:], np.arange(K)] + np.log(aid[q[0,:],q[1,:], np.arange(K)]))
    else:
        L[aid!=0] += np.log(aid[aid!=0])
        L = (L * q).sum() - (q[q!=0]*np.log(q[q!=0])).sum()
    return L


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    H,W,K = X.shape
    q = calculate_log_probability(X, F, B, s)
    if use_MAP:
        aid = A[:,:,np.newaxis] + np.zeros(q.shape)
        q[aid == 0] = -np.inf
        q[aid!=0] += np.log(aid[aid!=0])
        q = np.vstack(np.unravel_index(q.reshape(-1,K).argmax(0), q.shape[0:2]))
        return q
    A[A==0] += 1e-5
    q += np.log(A[:,:,np.newaxis])
    to_derive = q.max(axis=(0,1))
    q -= to_derive[np.newaxis, np.newaxis, :]
    normalizer = logsumexp(q,axis=(1,0))[np.newaxis, np.newaxis, :]
    q -= normalizer
    q = np.exp(q)
    
    return q


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, K = X.shape
    q_new = q
    if use_MAP:
        q_new = np.zeros((H-h+1,W-w+1,K))
        q_new[q[0,:],q[1,:],np.arange(K)] = 1
        
    A_new = q_new.sum(axis=-1)/K
    
    d_h = np.arange(H-h+1)[:,np.newaxis] + np.arange(h)[np.newaxis,:]
    d_w = np.arange(W-w+1)[:,np.newaxis] + np.arange(w)[np.newaxis,:]
    F_new = np.zeros((h,w))
    for k in range(K):
        submatrices = X[d_h[:,np.newaxis,:,np.newaxis],d_w[np.newaxis,:,np.newaxis,:],k]
        F_new += (submatrices * q_new[:,:,np.newaxis, np.newaxis, k]).sum(axis = (0,1))/K
        
    
    mask = np.ones((h,w,1))
    to_derive = fftconvolve(q_new, mask)
    
    to_mul = (1-to_derive)
    to_mul[to_mul < 0] = 0
    B_new = (X*to_mul).sum(axis=-1)
    B_new[B_new!=0] = B_new[B_new!=0]/((to_mul.sum(axis=-1))[B_new!=0])

    to_mul = (1-to_derive)
    to_mul[to_mul < 0] = 0
    arg1 = (((X-B_new[:,:,np.newaxis])**2)*to_mul).sum()
    arg2 = 0
    for k in range(K):
        submatrices = X[d_h[:,np.newaxis,:,np.newaxis],d_w[np.newaxis,:,np.newaxis,:],k]
        arg2 += (((submatrices - F_new[np.newaxis,np.newaxis,:,:])**2).sum(axis=(2,3))*q_new[:,:,k]).sum()
    s_new = (arg1 + arg2)/(K*H*W)
    s_new = float(s_new**0.5)

    return F_new,B_new,s_new,A_new
    


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters + 2,)
        L(q,F,B,s,A) at initial guess, after each EM iteration and after
        final estimate of posteriors;
        number_of_iters is actual number of iterations that was done.
    """  
    H, W, K = X.shape
    
    if F is None:
        F = np.random.rand(h, w)
    if B is None:
        B = np.random.rand(H, W)
    if A is None:
        A = np.random.rand(H - h + 1, W - w + 1)
        A /= A.sum()
    if s is None:
        s = 1
    LL = []
    for _ in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)
        LL_curr = calculate_lower_bound(X, F, B, s, A, q, use_MAP)
        if LL and LL_curr - LL[-1] < tolerance:
            break
        LL += [LL_curr]
    return F, B, s, A, LL


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    F_res, B_res, s_res, A_res, LL_res = 0,0,0,0,[]
    for _ in range(n_restarts):
        F, B, s, A, LL = run_EM(X, h, w, tolerance=tolerance, max_iter=max_iter, use_MAP=use_MAP)
        if not LL_res or LL[-1] > LL_res[-1]:
            F_res, B_res, s_res, A_res, LL_res = F, B, s, A, LL

    return F_res, B_res, s_res, A_res, LL_res