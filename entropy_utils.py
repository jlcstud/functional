from logging import root

import torch
import numpy as np
import yaml
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

def simulate_probs(num_classes = 2,
                   num_samples = 10000,
                   num_ensemble = 5,
                   logit_std = 20,
                   network_corr = 0.5,
                   class_corr = 0.5,
                   apply_softmax = True):
    # logits = r(N, n, C)*(1-rho_N)*(1-rho_C)+rho_N*r(1, n, C)*(1-rho_C)+rho_C*r(N, n, 1)
    # p = softmax(s*logits)
    logits = torch.randn(num_ensemble, num_samples, num_classes)*logit_std
    if network_corr>0:
        logits = (torch.randn(1, num_samples, num_classes)*logit_std*network_corr + 
                logits*(1-network_corr))
    if class_corr>0:
        logits = (torch.randn(num_ensemble, num_samples, 1)*logit_std*class_corr + 
                logits*(1-class_corr))
    if apply_softmax:
        probs = torch.softmax(logits, dim=2)
    else:
        probs = logits
    return probs

def load_ens_logits(file_path, only_id_gts=True, apply_temp=False):
    """loads ensemble logits from multiple CSV files"""
    if isinstance(file_path,str):
        file_paths = list(glob.glob(file_path))
        assert len(file_paths)>0, f"No files found for {file_path}"
    else:
        file_paths = file_path
    ens_out = []
    for file_path in file_paths:
        ens_out.append(load_logits(file_path, only_id_gts=only_id_gts, apply_temp=apply_temp))
    ens_out = {k: [d[k] for d in ens_out] for k in ens_out[0]}
    assert all(torch.equal(ens_out["id_classes"][0], d) for d in ens_out["id_classes"])
    for k in ens_out.keys():
        if k in ["logits", "probs"]:
            ens_out[k] = torch.stack(ens_out[k], dim=0)
        elif k in ["gt", "id_classes"]:
            ens_out[k] = ens_out[k][0]
    ens_out["paths"] = [str(Path(f)) for f in file_paths]
    return ens_out

def load_logits(file_path, only_id_gts=True, apply_temp=False):
    """loads logits from a CSV file, reads id_classes from the associated config.
    The config is in p.parent.parent/".hydra" / "config.yaml"
    if the parent name is "logits", otherwise just p.parent
    """
    out = {"logits": None, 
           "probs": None,
           "temp": None,
           "id_classes": None,
           "gt": None}
    raw_logits = np.loadtxt(file_path, delimiter=",", dtype=str)
    out["logits"] = torch.tensor(raw_logits[1:,:-1].astype(float))
    out["gt"] = torch.tensor(raw_logits[1:,-1].astype(int))
    # load config
    #root path is cifar + 3 children if relative path (to /home/jloch/Desktop/diff/functional/repo/)
    # e.g. usual input : cifar/logs/2026-09-11/04-35-57-C7-cifar01-ENetb6-scr-dropout
    file_path = file_path[file_path.index("cifar"):]
    root_path = Path("/".join(file_path.split("/")[:4]))
    config_path = root_path / ".hydra" / "config.yaml"
    temp_path = root_path / "optimal_temperature.txt"
    if temp_path.exists():
        out["temp"] = float(np.loadtxt(temp_path, dtype=str)[-1])
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    try:
        out["id_classes"]  = torch.tensor([int(x) for x in config["data"]["id_classes"]])
    except:
        out["id_classes"] = torch.arange(10)
    if only_id_gts:
        mask = np.isin(out["gt"], out["id_classes"])
        out["logits"] = out["logits"][mask]
        out["gt"] = out["gt"][mask]
    if apply_temp:
        if out["temp"] is not None:
            out["logits"] = out["logits"] / out["temp"]
    id_class_mask = [i in out["id_classes"] for i in range(out["logits"].shape[-1])]
    out["probs"] = torch.zeros_like(out["logits"])
    out["probs"][:,id_class_mask] = torch.softmax(out["logits"][:,id_class_mask], dim=-1)
    return out

def uncertainties(probs):
    AU = entropy(probs, dim=2).mean(0)
    mean_probs = probs.mean(0)
    TU = entropy(mean_probs, dim=1)
    EU = TU - AU
    return AU, EU, TU

def mxlogx(x):
    if torch.is_tensor(x):
        logx = torch.zeros_like(x, dtype=x.dtype)
        logx[x > 0] = torch.log(x[x > 0])
    else:
        logx = np.zeros_like(x, dtype=x.dtype)
        logx[x > 0] = np.log(x[x > 0])
    return -x * logx

def entropy(probs,dim=0):
    if torch.is_tensor(probs):
        return torch.sum(mxlogx(probs), dim=dim)
    else:
        return np.sum(mxlogx(probs), axis=dim)

def binary_entropy(p):
    return entropy(np.stack([p, 1-p], axis=0), dim=0)

def theoretical_boundaries(num_points,c=1):
    S = np.linspace(0,1,num_points)
    TU = binary_entropy(S/c)
    AU = binary_entropy(S)/c
    EU = TU - AU
    return AU, EU, TU

def meshgrid_nd(num_dims,vec):
    """ creates a meshgrid for an arbitrary number of dimensions, using the same
    vector of values for each dimension. Returns a list of flattened arrays"""
    grids = np.meshgrid(*[vec]*num_dims)
    return [g.flatten() for g in grids]

def get_nd_probs(num_classes = 2, num_samples = 10000, num_ensemble = 2,
                 filter=True,power_transform=None):
    if not num_classes==2:
        raise NotImplementedError("Only num_classes=2 is implemented")
    num_samples_per_dim = int(num_samples**(1/num_ensemble))
    axis_probs = np.linspace(0,1,num_samples_per_dim)
    grids = meshgrid_nd(num_ensemble, axis_probs)
    p0 = np.stack(grids, axis=1)
    p1 = 1-p0
    probs = np.stack([p0,p1], axis=2).transpose((1,0,2)).astype(np.float32)
    if filter:
        mask = 0
        p00 = probs[0,:,0]
        n_mask_conds = (num_ensemble-1)*num_classes
        for i in range(1,num_ensemble):
            for j in range(num_classes):
                mask += ((p00+probs[i,:,j])<=1).astype(int)
        mask = mask==n_mask_conds
    else:
        mask = slice(None)
    probs = torch.tensor(probs[:,mask], dtype=torch.float32)
    if power_transform is not None:
        assert isinstance(power_transform, (int,float))
        probs = torch.sign(probs - 0.5) * (torch.abs(probs - 0.5) ** power_transform) + 0.5
        probs = probs / probs.sum(dim=2, keepdim=True)  # re-normalize
    return probs

def theoretical_boundaries_binary(num_points,num_ensemble=2):
    k = num_ensemble
    # n1+n2=c-1
    S = np.linspace(0,1,num_points)
    AUs, EUs, TUs = [],[],[]
    for n1 in range(1,num_ensemble):
        TU = binary_entropy((n1-1+S)/k)
        AU = binary_entropy(S)/k
        EU = TU - AU
        AUs += AU.tolist()
        EUs += EU.tolist()
        TUs += TU.tolist()
    return AUs, EUs, TUs

def unique_compositions(k, c, max_val=None):
    """
    Generate all tuples (x1,...,xc) of non-negative ints summing to k,
    in non-increasing order (x1 >= x2 >= ... >= xc), i.e. unordered weak compositions.
    """
    if c == 1:
        # Only yield if the last part respects the non-increasing cap
        if max_val is None or k <= max_val:
            yield (k,)
        return

    cap = k if max_val is None else min(max_val, k)
    for first in range(cap, -1, -1):
        for rest in unique_compositions(k - first, c - 1, first):
            yield (first,) + rest

def pairs_less_than_c(c):
    return [(a, b) for a in range(c) for b in range(a + 1, c)]

def get_comp_S(comp,include_eq_pairs=False):
    pair_vals = set()
    pairs = []
    n = len(comp)
    for i in range(n):
        for j in range(i+1,n):
            pair_vals_ij = (comp[i],comp[j])
            if pair_vals_ij not in pair_vals:
                pair_vals.add(pair_vals_ij)
                if comp[i]==comp[j]:
                    if include_eq_pairs:
                        pairs.append((i,j))
                else:
                    pairs.append((i,j))
    comps_S = [[0 for _ in range(n)] for _ in range(len(pairs))]
    for i, (a, b) in enumerate(pairs):
        comps_S[i][a] = 1
        comps_S[i][b] = -1
    return comps_S

def entropy_from_comp(comp,comp_S,S):
    comp = np.array(comp)
    k = sum(comp)+1
    c = len(comp)
    prob = comp/k
    fixed_mask = [s == 0 for s in comp_S]
    entropy_fixed = mxlogx(prob[fixed_mask]).sum()
    i0 = comp_S.index(1)
    i1 = comp_S.index(-1)
    p0 = prob[i0]+S/k
    p1 = prob[i1]+(1-S)/k
    ent = mxlogx(p0)+mxlogx(p1)+entropy_fixed
    return ent

def theoretical_boundaries_all(num_points=200,num_ensemble=2,num_classes=2,include_eq_pairs=False,
                              detailed=False):
    k = num_ensemble
    c = num_classes
    # n1+n2=c-1
    if detailed:
        # sample with increasing frequency close to 0 and 1 usin sigmoid
        T = 30
        S = 1/(1+np.exp(-np.linspace(-T,T,num_points)))
    else:
        S = np.linspace(0,1,num_points)
    AUs, EUs, TUs = [],[],[]
    comps = list(unique_compositions(k-1, c))
    comps_with_S = [[(comp, comp_S) for comp_S in get_comp_S(comp,include_eq_pairs=include_eq_pairs)] for comp in comps]
    comps_with_S = sum(comps_with_S, [])
    for comp, comp_S in comps_with_S:
        TU = entropy_from_comp(comp, comp_S, S)
        AU = binary_entropy(S)/k
        EU = TU - AU
        AUs.append(AU)
        EUs.append(EU)
        TUs.append(TU)
    return AUs, EUs, TUs, comps_with_S

def get_theoretical_min_boundary(num_points_per_comp=200, num_final_points=1000, num_ensemble=2, num_classes=2,
                                detailed=False, interp=True):
    """Returns the minimum AU for a given TU as a collection of points.
    
    Args:
        num_points_per_comp: Number of points per composition curve
        num_final_points: Final number of points (only used if interp=True)
        num_ensemble: Number of ensemble members
        num_classes: Number of classes
        detailed: Use detailed sampling (more points near 0 and 1)
        interp: If True, use linspace interpolation. If False, use original curve points only.
                When False, a point is minimum if it's below all other curves at its TU value.
    """
    AUs, _, TUs, _ = theoretical_boundaries_all(num_points=num_points_per_comp,
                                                num_ensemble=num_ensemble,
                                                num_classes=num_classes,
                                                include_eq_pairs=True,
                                                detailed=detailed)
    
    # Sort all curves by TU
    for i in range(len(AUs)):
        sort_idx = np.argsort(TUs[i])
        TUs[i] = np.array(TUs[i])[sort_idx]
        AUs[i] = np.array(AUs[i])[sort_idx]
    
    if interp:
        # Original behavior: use linspace interpolation
        max_TU = np.concatenate(TUs).max()
        TU = np.linspace(0, max_TU, num_final_points)
        AU = np.zeros_like(TU) + np.inf
        
        for i in range(len(AUs)):
            mask = (TU >= TUs[i].min()) & (TU <= TUs[i].max())
            if not np.any(mask):
                continue
            AU_interp = np.interp(TU[mask], TUs[i], AUs[i])
            AU[mask] = np.minimum(AU[mask], AU_interp)
        
        assert not np.any(AU == np.inf), (f"Some TU values were not covered by any composition. Total fails {np.sum(AU == np.inf)}/{len(AU)}")
        EU = TU - AU
        return AU, EU, TU
    
    else:
        # New behavior: use original curve points only (vectorized for speed)
        # Collect all points from all curves and check if each is below all covering curves
        
        # Pre-compute min/max for each curve to avoid repeated calls
        curve_mins = [TUs[i].min() for i in range(len(TUs))]
        curve_maxs = [TUs[i].max() for i in range(len(TUs))]
        
        AU_mins = []
        TU_mins = []
        
        for i in range(len(AUs)):
            # Vectorized check: for each point on curve i, check against all other curves at once
            tu_vals = np.array(TUs[i])
            au_vals = np.array(AUs[i])
            
            # For each point, check if it's below all covering curves
            for tu_val, au_val in zip(tu_vals, au_vals):
                # Find which curves cover this TU value
                covering_curves = [k for k in range(len(AUs)) if k != i 
                                  and curve_mins[k] <= tu_val <= curve_maxs[k]]
                
                # If no other curves cover this point, it's automatically minimal
                if not covering_curves:
                    AU_mins.append(au_val)
                    TU_mins.append(tu_val)
                    continue
                
                # Check if below all covering curves
                is_minimum = True
                for k in covering_curves:
                    au_interp = np.interp(tu_val, TUs[k], AUs[k])
                    if au_val > au_interp:
                        is_minimum = False
                        break
                
                if is_minimum:
                    AU_mins.append(au_val)
                    TU_mins.append(tu_val)
        
        # Sort by TU for output
        if AU_mins:
            sort_idx = np.argsort(TU_mins)
            AU = np.array(AU_mins)[sort_idx]
            TU = np.array(TU_mins)[sort_idx]
        else:
            AU = np.array([])
            TU = np.array([])
        
        EU = TU - AU
        return AU, EU, TU


def unbubble(AU, TU, C, N):
    AU_min, EU_min, TU_min = get_theoretical_min_boundary(num_classes=C, num_ensemble=N,
                                                          detailed=False,
                                                          interp=True)
    
    if torch.is_tensor(AU):
        AU = AU.numpy()
    if torch.is_tensor(TU):
        TU = TU.numpy()
    
    AU_min_at_TU = np.interp(TU, TU_min, AU_min, left=AU_min[0], right=AU_min[-1])
    AU_new = AU - AU_min_at_TU
    #multiply with interval ratio
    old_interval = TU - AU_min_at_TU
    new_interval = TU
    interval_ratio = new_interval / old_interval
    AU_new = AU_new * interval_ratio
    EU_new = TU - AU_new
    return AU_new, EU_new, TU

def ensemble_plot(ens_logits, 
                  plot_boundary=1,
                  plot_xy=False,
                  gt_colormap=False,
                  use_label=False,
                  alpha=0.7,
                  limits=None, 
                  lw=2,
                  log=False,
                  interp=False):

    C = len(ens_logits["id_classes"])
    N = len(ens_logits["logits"])

    alpha = 0.7
    lw = 2
    max_tu = np.log(C)
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
    AU, EU, TU = uncertainties(ens_logits["probs"])
    #linearly interpolate colors between C0 and C1 for each point based on the correct ratio
    #conver to numpy if not already
    if torch.is_tensor(AU):
        AU = AU.numpy()
    if torch.is_tensor(EU):
        EU = EU.numpy()
    if torch.is_tensor(TU):
        TU = TU.numpy()


    ax = axs[0]

    ax.plot([0,max_tu],[max_tu,0], 'k-', label=r"TU$\leq\log(C)$", alpha=alpha, linewidth=lw)
    ax.plot([0,max_tu],[0,0], 'b', label=r"TU$\geq$AU", alpha=alpha, linewidth=lw)
    ax.set_xlabel("AU")
    ax.set_ylabel("EU")
    if gt_colormap:
        viridis = plt.get_cmap("viridis")
        cmap = [viridis(x) for x in np.linspace(0, 1, N+1)]

        # Number of ensemble members predicting the correct class
        preds = torch.argmax(ens_logits["logits"], dim=-1)
        correct = (preds == torch.tensor(ens_logits["gt"], dtype=torch.int64)).sum(dim=0)

        # Map correct-count to one of the 10 viridis colors
        idx = np.round(correct.numpy()).astype(int)
        cmap2 = [cmap[i] for i in idx]

        ax.scatter(
            AU, EU,
            c=cmap2, s=5, alpha=0.3
        )
        cmap = plt.get_cmap("viridis", N + 1)
        norm = BoundaryNorm(np.arange(-0.5, N + 1.5), N + 1)

        plt.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            ticks=np.arange(N + 1),
            label="Number correct"
        )
    else:
        ax.plot(AU, EU, '.',  markersize=1, alpha=alpha, linewidth=lw)

    if plot_xy:
        x_min,x_max = ax.get_xlim()
        y_min,y_max = ax.get_ylim()
        ax.plot([min(x_min,y_min), max(x_max,y_max)], [min(x_min,y_min), max(x_max,y_max)], 'k--', label=r"$\text{AU}=\text{EU}$")
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)


    ax = axs[1]
    ax.plot([0,max_tu],[max_tu,max_tu], 'k-', alpha=alpha, linewidth=lw)#, label=r"TU$\leq\log(C)$")
    ax.plot([0,max_tu],[0,max_tu], 'b', alpha=alpha, linewidth=lw)#, label=r"TU$\geq$AU")
    ax.set_xlabel("AU")
    ax.set_ylabel("TU")
    if gt_colormap:
        viridis = plt.get_cmap("viridis")
        cmap = [viridis(x) for x in np.linspace(0, 1, N+1)]

        # Number of ensemble members predicting the correct class
        preds = torch.argmax(ens_logits["logits"], dim=-1)
        correct = (preds == torch.tensor(ens_logits["gt"], dtype=torch.int64)).sum(dim=0)

        # Map correct-count to one of the 10 viridis colors
        idx = np.round(correct.numpy()).astype(int)
        cmap2 = [cmap[i] for i in idx]

        ax.scatter(
            AU, TU,
            c=cmap2, s=5, alpha=0.3
        )
        cmap = plt.get_cmap("viridis", N + 1)
        norm = BoundaryNorm(np.arange(-0.5, N + 1.5), N + 1)

        plt.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            ticks=np.arange(N + 1),
            label="Number correct"
        )
    else:
        ax.plot(AU, TU, '.',  markersize=1, alpha=alpha, linewidth=lw)

    if plot_boundary==1:
        AUb,EUb,TUb = get_theoretical_min_boundary(num_classes=C, num_ensemble=N,
                                                        interp=interp,
                                                        detailed=log)
        axs[0].plot(AUb, EUb, "m-", linewidth=2, label=f"Infeasible Boundary (N={N}, C={C})")
        axs[1].plot(AUb, TUb, "m-", linewidth=2, label=f"Infeasible Boundary (N={N}, C={C})")
    elif plot_boundary>=2:
        AUs, EUs, TUs, comps_with_S = theoretical_boundaries_all(200, num_ensemble=N, num_classes=C,
                                                                 detailed=log)
        #get tab10 colors
        colors = plt.get_cmap('tab10').colors[1:]
        for AUb, EUb, TUb, comp_S,i in zip(AUs, EUs, TUs, comps_with_S, range(len(AUs))):
            label = "$("
            for n, is_S in zip(*comp_S):
                if is_S!=0:
                    l = f"{n}\u0332,"
                else:
                    l = f"{n},"
                label += l
                
            label = label[:-1]+")$" if plot_boundary==3 else None
            axs[0].plot(AUb, EUb, label=label, color=colors[i%len(colors)])
            axs[1].plot(AUb, TUb, label=label, color=colors[i%len(colors)])
    if limits is not None:
        if len(limits)==2:
            axs[0].set_xlim(limits[0])
            axs[0].set_ylim(limits[1])
            axs[1].set_xlim(limits[0])
            axs[1].set_ylim(limits[1])
        elif len(limits)==3:
            axs[0].set_xlim(limits[0])
            axs[0].set_ylim(limits[1])
            axs[1].set_xlim(limits[0])
            axs[1].set_ylim(limits[2])
        else:
            raise ValueError("limits must be a list of 2 or 3 tuples")
    if log:
        axs[0].set_xscale('log'), axs[0].set_yscale('log')
        axs[1].set_xscale('log'), axs[1].set_yscale('log')
    axs[0].grid()
    axs[1].grid()
    axs[0].legend(ncol=1)
    axs[1].legend(ncol=1)
    return fig, axs


def ensemble_plot_log(ens_logits, 
                      plot_boundary=1,
                      plot_xy=False,
                      gt_colormap=False,
                      use_label=False,
                      alpha=0.7,
                      limits=None, 
                      lw=2,
                      TU_instead=False,
                      interp=False,
                      C=None, N=None):
    """Plot uncertainty space with linear and log scale side by side.
    
    Args:
        ens_logits: Ensemble logits dictionary
        plot_boundary: 1=min boundary, 2=all boundaries, 3=labeled boundaries
        plot_xy: Plot diagonal reference line
        gt_colormap: Color points by number correct
        use_label: Show labels on boundary curves
        alpha: Transparency
        limits: Axis limits [(x_min, x_max), (y_min, y_max)] or [(x_min, x_max), (y_min_left, y_max_left), (y_min_right, y_max_right)]
        lw: Line width
        TU_instead: If False, plot (AU, EU); if True, plot (AU, TU)
        interp: Use interpolation for boundary
    """
    if C is None:
        C = len(ens_logits["id_classes"])
    else:
        assert C is not None and C > 0, "C must be a positive integer"
    if N is None:
        N = len(ens_logits["logits"])
    else:
        assert N is not None and N > 0, "N must be a positive integer"

    max_tu = np.log(C)
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
    if ens_logits is not None:
        AU, EU, TU = uncertainties(ens_logits["probs"])
        
        # Convert to numpy if needed
        if torch.is_tensor(AU):
            AU = AU.numpy()
        if torch.is_tensor(EU):
            EU = EU.numpy()
        if torch.is_tensor(TU):
            TU = TU.numpy()
    
        # Select y-axis domain
        y_vals = TU if TU_instead else EU
    y_label = "TU" if TU_instead else "EU"
    
    # Plot both subplots with same domain
    for idx, ax in enumerate(axs):
        # Reference lines
        if TU_instead:
            ax.plot([0,max_tu],[max_tu,max_tu], 'k-', alpha=alpha, linewidth=lw)
            ax.plot([0,max_tu],[0,max_tu], 'b', alpha=alpha, linewidth=lw)
        else:
            if idx==0:
                ax.plot([0,max_tu],[max_tu,0], 'k-', label=r"TU=AU+EU$\leq\log(C)$", alpha=alpha, linewidth=lw)
            else:
                #use sigmoid to generate alot of points close to 0 and close to 1
                t = 1/(1+np.exp(-np.linspace(-10,10,1000)))
                t = (t-min(t))/(max(t)-min(t))
                x_interp = t*max_tu
                y_interp = max_tu - x_interp
                ax.plot(x_interp,y_interp, 'k-', alpha=alpha, linewidth=lw)
            #ax.plot([0,max_tu],[0,0], 'b', label=r"TU$\geq$AU", alpha=alpha, linewidth=lw)
        
        ax.set_xlabel("AU")
        ax.set_ylabel(y_label)
        if ens_logits is not None:
            # Plot data points
            if gt_colormap:
                viridis = plt.get_cmap("viridis")
                cmap = [viridis(x) for x in np.linspace(0, 1, N+1)]
                preds = torch.argmax(ens_logits["logits"], dim=-1)
                correct = (preds == torch.tensor(ens_logits["gt"], dtype=torch.int64)).sum(dim=0)
                idx_colors = np.round(correct.numpy()).astype(int)
                cmap2 = [cmap[i] for i in idx_colors]
                
                ax.scatter(AU, y_vals, c=cmap2, s=5, alpha=0.3)
                if idx == 0:
                    cmap_plot = plt.get_cmap("viridis", N + 1)
                    norm = BoundaryNorm(np.arange(-0.5, N + 1.5), N + 1)
                    plt.colorbar(
                        ScalarMappable(norm=norm, cmap=cmap_plot),
                        ax=ax,
                        ticks=np.arange(N + 1),
                        label="Number correct"
                    )
            else:
                ax.plot(AU, y_vals, '.', markersize=1, alpha=alpha, linewidth=lw)
        
        if plot_xy:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            ax.plot([0,max_tu/2], [0,max_tu/2], 
                   'k--', label=r"$\text{AU}="+y_label+"$")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        # Plot boundary
        if plot_boundary == 1:
            AUb, EUb, TUb = get_theoretical_min_boundary(num_classes=C, num_ensemble=N,
                                                          interp=interp, detailed=(idx==1))
            y_boundary = TUb if TU_instead else EUb
            ax.plot(AUb, y_boundary, "m-", label=f"Infeasible Boundary (N={N}, C={C})", linewidth=lw)
        elif plot_boundary >= 2:
            AUs, EUs, TUs, comps_with_S = theoretical_boundaries_all(200, num_ensemble=N, num_classes=C,
                                                                     detailed=False)
            colors = plt.get_cmap('tab10').colors[1:]
            for AUb, EUb, TUb, comp_S, i in zip(AUs, EUs, TUs, comps_with_S, range(len(AUs))):
                y_boundary = TUb if TU_instead else EUb
                label = "$("
                for n, is_S in zip(*comp_S):
                    l = (f"{n}\u0332," if is_S != 0 else f"{n},")
                    label += l
                label = label[:-1] + ")$" if plot_boundary == 3 else None
                ax.plot(AUb, y_boundary, label=label, color=colors[i%len(colors)], linewidth=lw)
        
        # Set limits
        if limits is not None:
            if len(limits) == 2:
                ax.set_xlim(limits[0])
                ax.set_ylim(limits[1])
            elif len(limits) == 3:
                ax.set_xlim(limits[0])
                ax.set_ylim(limits[1] if idx == 0 else limits[2])
            else:
                raise ValueError("limits must be a list of 2 or 3 tuples")
        
        # Log scale only for second subplot
        if idx == 1:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.grid()
        ax.legend(ncol=1)
    
    return fig, axs

def compute_optimal_temperature_nll(logits, labels):
    """Compute optimal temperature to minimize NLL on labeled data.
    
    Args:
        logits: (n_samples, n_classes) array
        labels: (n_samples,) array with class indices
    
    Returns:
        Optimal temperature (scalar)
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    
    logits_np = logits.cpu().numpy() if torch.is_tensor(logits) else logits
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    
    def nll_loss(temp):
        if temp <= 0:
            return 1e10
        scaled_logits = logits_np / temp
        # Use logsumexp for numerical stability
        log_probs = scaled_logits - logsumexp(scaled_logits, axis=1, keepdims=True)
        nll = -np.mean(log_probs[np.arange(len(labels_np)), labels_np])
        # Return large value if NLL is NaN or Inf
        if not np.isfinite(nll):
            return 1e10
        return nll
    
    result = minimize_scalar(nll_loss, bounds=(0.01, 10), method='bounded')
    return result.x

def model_paths(formatter,pretrained=False,enet_b=0):
    pre = "-pre" if pretrained else ""
    file_paths = []
    for i in range(10):
        file_path = formatter.format(enet_b=enet_b,pre=pre,i=i)
        # add if only 1 match is found, else error
        matches = list(Path(".").glob(file_path))
        if len(matches) == 1:
            file_paths.append(str(matches[0]))
        elif len(matches) > 1:
            #raise error but print paths
            raise ValueError(f"Expected exactly one match for {file_path}, but found {len(matches)}:\n" + "\n".join(f"  {match}" for match in matches))
        else:
            raise ValueError(f"Expected exactly one match for {file_path}, but found {len(matches)}")
    return file_paths

def epistemic_collapse_plot(logit_file_lists, x=None, xlabel=None):
    """Plot epistemic collapse (EU/TU ratio) across ensembles.
    
    Args:
        logit_file_lists: List of lists, where each inner list represents an ensemble
                         of logit files to load and analyze together
        x: Optional array of x-axis values. If None, uses range(len(logit_file_lists))
        xlabel: Optional label for x-axis. If None, defaults to "Index" or "X"
    
    Returns:
        fig, ax, metrics: Figure, axis, and dictionary with computed metrics
    """
    if x is not None:
        if len(x) != len(logit_file_lists):
            raise ValueError(f"Length of x ({len(x)}) must match length of logit_file_lists ({len(logit_file_lists)})")
        x_vals = np.asarray(x)
    else:
        x_vals = np.arange(len(logit_file_lists))
    
    mean_eu_tu = []
    median_eu_tu = []
    
    for file_list in logit_file_lists:
        # Load ensemble logits
        ens_logits = load_ens_logits(file_list, only_id_gts=False, apply_temp=False)
        
        # Get logits and compute uncertainties
        logits = ens_logits["logits"].float()
        probs = torch.softmax(logits, dim=-1)
        AU, EU, TU = uncertainties(probs)
        
        # Compute EU/TU ratio
        eu_tu_ratio = EU.numpy() / (TU.numpy() + 1e-8)
        
        mean_eu_tu.append(np.mean(eu_tu_ratio))
        median_eu_tu.append(np.median(eu_tu_ratio))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(x_vals, mean_eu_tu, 'o-', linewidth=2, markersize=6, label='Mean EU/TU')
    ax.plot(x_vals, median_eu_tu, 's-', linewidth=2, markersize=6, label='Median EU/TU')
    
    # Determine x-axis label
    if xlabel is None:
        xlabel = 'Index' if x is None else 'X'
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_xticks(x_vals)
    ax.set_ylabel('EU/TU Ratio', fontsize=11)
    ax.set_title('Epistemic Collapse Across Ensembles', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    metrics = {
        'x': x_vals.tolist(),
        'mean_eu_tu': mean_eu_tu,
        'median_eu_tu': median_eu_tu,
    }
    
    return fig, ax, metrics

def progress_plot(logit_file_lists, x=None, xlabel=None, apply_temp=False):
    """Plot evolution of validation metrics across ensembles (first 4 plots from ensemble_evolution_plot).
    
    Args:
        logit_file_lists: List of lists, where each inner list represents an ensemble
                         of logit files to load and analyze together
        x: Optional array of x-axis values. If None, uses range(len(logit_file_lists))
        xlabel: Optional label for x-axis. If None, defaults to "Index" or "X"
        apply_temp: If True, compute and apply optimal temperature scaling from ID data
    
    Returns:
        fig, axs, metrics: Figure, axes, and dictionary with computed metrics
    
    Plots:
        1. Validation accuracy (ID only)
        2. OOD detection performance (AUROC for AU and EU)
        3. Epistemic collapse (mean and median EU/TU ratio for ID samples)
        4. AU and EU evolution (mean and median)
    """
    from sklearn.metrics import roc_auc_score
    
    if x is not None:
        if len(x) != len(logit_file_lists):
            raise ValueError(f"Length of x ({len(x)}) must match length of logit_file_lists ({len(logit_file_lists)})")
        x_vals = np.asarray(x)
    else:
        x_vals = np.arange(len(logit_file_lists))
    
    # Determine x-axis label
    if xlabel is None:
        xlabel = 'Index' if x is None else 'X'
    
    # Initialize metrics storage
    metrics = {
        'x': x_vals.tolist(),
        'val_acc_id': [],
        'ood_auroc_au': [],
        'ood_auroc_eu': [],
        'epistemic_collapse_mean': [],
        'epistemic_collapse_median': [],
        'au_mean': [],
        'au_median': [],
        'eu_mean': [],
        'eu_median': [],
    }
    
    # Process each ensemble
    for file_list in logit_file_lists:
        # Load ensemble logits
        ens_logits = load_ens_logits(file_list, only_id_gts=False, apply_temp=False)
        
        # Get ID and OOD masks
        id_classes = ens_logits["id_classes"].numpy()
        gt = ens_logits["gt"].numpy()
        id_mask = np.isin(gt, id_classes)
        ood_mask = ~id_mask
        
        # Get logits and compute temperature if requested
        logits = ens_logits["logits"].float()
        
        if apply_temp:
            # Compute optimal temperature from ID data using ensemble mean
            mean_logits = logits.mean(dim=0)
            id_logits = mean_logits[id_mask]
            id_labels = torch.from_numpy(gt[id_mask]).long()
            
            temp = compute_optimal_temperature_nll(id_logits, id_labels)
            logits = logits / temp
        
        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Compute uncertainties
        AU, EU, TU = uncertainties(probs)
        AU, EU, TU = AU.numpy(), EU.numpy(), TU.numpy()
        
        # Validation accuracy (ID only)
        mean_logits_np = logits.mean(dim=0).numpy()
        id_preds = np.argmax(mean_logits_np[id_mask], axis=-1)
        id_labels_true = gt[id_mask]
        val_acc = np.mean(id_preds == id_labels_true)
        metrics['val_acc_id'].append(val_acc)
        
        # OOD detection (AUROC) - only if OOD samples exist
        if np.sum(ood_mask) > 0:
            # AU-based OOD detection
            ood_labels = np.concatenate([np.ones(np.sum(ood_mask)), np.zeros(np.sum(id_mask))])
            au_scores = np.concatenate([AU[ood_mask], AU[id_mask]])
            auroc_au = roc_auc_score(ood_labels, au_scores)
            metrics['ood_auroc_au'].append(auroc_au)
            
            # EU-based OOD detection
            eu_scores = np.concatenate([EU[ood_mask], EU[id_mask]])
            auroc_eu = roc_auc_score(ood_labels, eu_scores)
            metrics['ood_auroc_eu'].append(auroc_eu)
        else:
            metrics['ood_auroc_au'].append(np.nan)
            metrics['ood_auroc_eu'].append(np.nan)
        
        # Epistemic collapse (EU/TU ratio for ID samples)
        eu_tu_ratio = EU[id_mask] / (TU[id_mask] + 1e-8)
        metrics['epistemic_collapse_mean'].append(np.mean(eu_tu_ratio))
        metrics['epistemic_collapse_median'].append(np.median(eu_tu_ratio))
        
        # AU and EU statistics (ID samples only)
        metrics['au_mean'].append(np.mean(AU[id_mask]))
        metrics['au_median'].append(np.median(AU[id_mask]))
        metrics['eu_mean'].append(np.mean(EU[id_mask]))
        metrics['eu_median'].append(np.median(EU[id_mask]))
    
    # Create figure with 4 subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    
    # Plot 1: Validation accuracy
    axs[0].plot(x_vals, metrics['val_acc_id'], 'o-', linewidth=2, markersize=6)
    axs[0].set_xlabel(xlabel, fontsize=11)
    axs[0].set_xticks(x_vals)
    axs[0].set_ylabel('Accuracy', fontsize=11)
    axs[0].set_title('Validation Accuracy (ID Samples)', fontsize=12)
    axs[0].grid(alpha=0.3)
    
    # Plot 2: OOD performance
    valid_auroc = ~np.isnan(metrics['ood_auroc_au'])
    if np.any(valid_auroc):
        x_valid = x_vals[valid_auroc]
        auroc_au_valid = np.array(metrics['ood_auroc_au'])[valid_auroc]
        auroc_eu_valid = np.array(metrics['ood_auroc_eu'])[valid_auroc]
        
        axs[1].plot(x_valid, auroc_au_valid, 'o-', label='AU AUROC', linewidth=2, markersize=6)
        axs[1].plot(x_valid, auroc_eu_valid, 's-', label='EU AUROC', linewidth=2, markersize=6)
    axs[1].set_xlabel(xlabel, fontsize=11)
    axs[1].set_xticks(x_vals)
    axs[1].set_ylabel('AUROC', fontsize=11)
    axs[1].set_title('OOD Detection Performance', fontsize=12)
    axs[1].grid(alpha=0.3)
    axs[1].legend()
    
    # Plot 3: Epistemic collapse
    axs[2].plot(x_vals, metrics['epistemic_collapse_mean'], 'o-', 
               label='Mean EU/TU', linewidth=2, markersize=6)
    axs[2].plot(x_vals, metrics['epistemic_collapse_median'], 's-', 
               label='Median EU/TU', linewidth=2, markersize=6)
    axs[2].set_xlabel(xlabel, fontsize=11)
    axs[2].set_xticks(x_vals)
    axs[2].set_ylabel('EU/TU Ratio', fontsize=11)
    axs[2].set_title('Epistemic Collapse (ID Samples)', fontsize=12)
    axs[2].grid(alpha=0.3)
    axs[2].legend()
    
    # Plot 4: AU and EU evolution
    axs[3].plot(x_vals, metrics['au_mean'], linewidth=2, 
               color='C2', label='AU mean')
    axs[3].plot(x_vals, metrics['au_median'], linewidth=2, 
               color='C2', linestyle='--', label='AU median')
    axs[3].plot(x_vals, metrics['eu_mean'], linewidth=2, 
               color='C3', label='EU mean')
    axs[3].plot(x_vals, metrics['eu_median'], linewidth=2, 
               color='C3', linestyle='--', label='EU median')
    axs[3].set_xlabel(xlabel, fontsize=11)
    axs[3].set_xticks(x_vals)
    axs[3].set_ylabel('Uncertainty', fontsize=11)
    axs[3].set_title('AU and EU Evolution (ID Samples)', fontsize=12)
    axs[3].grid(alpha=0.3)
    axs[3].legend()
    
    plt.tight_layout()
    return fig, axs, metrics

    """Plot evolution of validation metrics across epochs.
    
    Args:
        path_patterns: List of path patterns, e.g., 
                      ["./cifar/logs/2026-09-0*/*-C7-save10-ENetb4-0", ...]
        apply_temp: If True, compute and apply optimal temperature scaling from ID data
        axs: Optional array of 6 matplotlib axes to plot on. If None, creates new figure.
             Must be array-like with shape (6,) or (1, 6).
    
    Returns:
        fig, axs, metrics: Figure (None if axs provided), axes, and dictionary with computed metrics
    
    Plots:
        1. Validation accuracy (ID only)
        2. OOD detection performance (AUROC for AU and EU)
        3. Epistemic collapse (mean and median EU/TU ratio for ID samples)
        4. AU and EU evolution (mean and median)
        5. AU vs EU scatter (Linear scale, last epoch) with theoretical minimum boundary
        6. AU vs EU scatter (Log scale, last epoch) with theoretical minimum boundary
    """
    from sklearn.metrics import roc_auc_score
    
    # Find all matching paths
    all_paths = []
    for pattern in path_patterns:
        matches = sorted(Path(".").glob(pattern))
        all_paths.extend(matches)
    
    if not all_paths:
        raise ValueError(f"No matches found for patterns {path_patterns}")
    
    all_paths = [Path(p) for p in all_paths]
    
    # Find available epochs from first model's logits folder
    first_logits_dir = all_paths[0] / "logits"
    if not first_logits_dir.exists():
        raise ValueError(f"No logits folder found at {first_logits_dir}")
    
    epoch_files = sorted(first_logits_dir.glob("val_logits_epoch_*.csv"))
    epochs = []
    for f in epoch_files:
        # Extract epoch number from filename like "val_logits_epoch_080.csv"
        epoch_str = f.stem.replace("val_logits_epoch_", "")
        try:
            epochs.append(int(epoch_str))
        except ValueError:
            continue
    
    if not epochs:
        raise ValueError(f"No epoch files found in {first_logits_dir}")
    
    epochs = sorted(epochs)
    
    # Initialize metrics storage
    metrics = {
        'epochs': epochs,
        'val_acc_id': [],
        'ood_auroc_au': [],
        'ood_auroc_eu': [],
        'epistemic_collapse_mean': [],
        'epistemic_collapse_median': [],
        'au_mean': [],
        'au_median': [],
        'eu_mean': [],
        'eu_median': [],
    }
    
    # Process each epoch
    for epoch in epochs:
        # Load logits for all models at this epoch
        file_paths = []
        for base_path in all_paths:
            logits_file = base_path / f"logits/val_logits_epoch_{epoch:03d}.csv"
            if logits_file.exists():
                file_paths.append(str(logits_file))
        
        if not file_paths:
            print(f"Warning: No files found for epoch {epoch}, skipping")
            continue
        
        # Load ensemble logits
        ens_logits = load_ens_logits(file_paths, only_id_gts=False, apply_temp=False)
        
        # Get ID and OOD masks
        id_classes = ens_logits["id_classes"].numpy()
        gt = ens_logits["gt"].numpy()
        id_mask = np.isin(gt, id_classes)
        ood_mask = ~id_mask
        
        # Get logits and compute temperature if requested
        logits = ens_logits["logits"].float()  # Already a tensor from load_ens_logits()
        
        if apply_temp:
            # Compute optimal temperature from ID data using ensemble mean
            mean_logits = logits.mean(dim=0)  # Average across ensemble
            id_logits = mean_logits[id_mask]
            id_labels = torch.from_numpy(gt[id_mask]).long()
            
            temp = compute_optimal_temperature_nll(id_logits, id_labels)
            logits = logits / temp
        
        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Compute uncertainties
        AU, EU, TU = uncertainties(probs)
        AU, EU, TU = AU.numpy(), EU.numpy(), TU.numpy()
        
        # Validation accuracy (ID only)
        mean_logits_np = logits.mean(dim=0).numpy()
        id_preds = np.argmax(mean_logits_np[id_mask], axis=-1)
        id_labels_true = gt[id_mask]
        val_acc = np.mean(id_preds == id_labels_true)
        metrics['val_acc_id'].append(val_acc)
        
        # OOD detection (AUROC) - only if OOD samples exist
        if np.sum(ood_mask) > 0:
            # AU-based OOD detection
            ood_labels = np.concatenate([np.ones(np.sum(ood_mask)), np.zeros(np.sum(id_mask))])
            au_scores = np.concatenate([AU[ood_mask], AU[id_mask]])
            auroc_au = roc_auc_score(ood_labels, au_scores)
            metrics['ood_auroc_au'].append(auroc_au)
            
            # EU-based OOD detection
            eu_scores = np.concatenate([EU[ood_mask], EU[id_mask]])
            auroc_eu = roc_auc_score(ood_labels, eu_scores)
            metrics['ood_auroc_eu'].append(auroc_eu)
        else:
            metrics['ood_auroc_au'].append(np.nan)
            metrics['ood_auroc_eu'].append(np.nan)
        
        # Epistemic collapse (EU/TU ratio for ID samples)
        eu_tu_ratio = EU[id_mask] / (TU[id_mask] + 1e-8)
        metrics['epistemic_collapse_mean'].append(np.mean(eu_tu_ratio))
        metrics['epistemic_collapse_median'].append(np.median(eu_tu_ratio))
        
        # AU and EU statistics (ID samples only)
        metrics['au_mean'].append(np.mean(AU[id_mask]))
        metrics['au_median'].append(np.median(AU[id_mask]))
        metrics['eu_mean'].append(np.mean(EU[id_mask]))
        metrics['eu_median'].append(np.median(EU[id_mask]))
    
    # Validate and create axes if needed
    if axs is None:
        fig, axs = plt.subplots(1, 6, figsize=(28, 4))
    else:
        # Validate supplied axes
        try:
            axs_array = np.asarray(axs)
            axs_shape = axs_array.shape
            # Handle both (6,) and (1, 6) shapes
            if axs_shape == (6,):
                pass  # Valid
            elif axs_shape == (1, 6):
                axs = axs[0]  # Flatten to 1D array
            else:
                raise ValueError(
                    f"axes array must have shape (6,) or (1, 6), got {axs_shape}"
                )
        except (TypeError, ValueError) as e:
            if "could not be converted to array" in str(e) or "object cannot be converted" in str(e):
                raise TypeError(f"axs parameter must be a numpy array or array-like, got {type(axs)}")
            else:
                raise ValueError(str(e))
        fig = None
    
    # Plot 1: Validation accuracy
    axs[0].plot(metrics['epochs'], metrics['val_acc_id'], 'o-', linewidth=2, markersize=6)
    axs[0].set_xlabel('Epoch', fontsize=11)
    axs[0].set_ylabel('Accuracy', fontsize=11)
    axs[0].set_title('Validation Accuracy (ID Samples)', fontsize=12)
    axs[0].grid(alpha=0.3)
    axs[0].set_ylim([0.8, 1.0])
    
    # Plot 2: OOD performance
    valid_auroc = ~np.isnan(metrics['ood_auroc_au'])
    if np.any(valid_auroc):
        epochs_valid = np.array(metrics['epochs'])[valid_auroc]
        auroc_au_valid = np.array(metrics['ood_auroc_au'])[valid_auroc]
        auroc_eu_valid = np.array(metrics['ood_auroc_eu'])[valid_auroc]
        
        axs[1].plot(epochs_valid, auroc_au_valid, 'o-', label='AU AUROC', linewidth=2, markersize=6)
        axs[1].plot(epochs_valid, auroc_eu_valid, 's-', label='EU AUROC', linewidth=2, markersize=6)
        axs[1].set_xlabel('Epoch', fontsize=11)
        axs[1].set_ylabel('AUROC', fontsize=11)
        axs[1].set_title('OOD Detection Performance', fontsize=12)
        axs[1].set_ylim([0.5, 0.85])
        axs[1].grid(alpha=0.3)
        axs[1].legend()
    
    # Plot 3: Epistemic collapse
    axs[2].plot(metrics['epochs'], metrics['epistemic_collapse_mean'], 'o-', 
               label='Mean EU/TU', linewidth=2, markersize=6)
    axs[2].plot(metrics['epochs'], metrics['epistemic_collapse_median'], 's-', 
               label='Median EU/TU', linewidth=2, markersize=6)
    axs[2].set_xlabel('Epoch', fontsize=11)
    axs[2].set_ylabel('EU/TU Ratio', fontsize=11)
    axs[2].set_title('Epistemic Collapse (ID Samples)', fontsize=12)
    axs[2].grid(alpha=0.3)
    axs[2].legend()
    
    # Plot 4: AU and EU evolution (no markers)
    axs[3].plot(metrics['epochs'], metrics['au_mean'], linewidth=2, 
               color='C2', label='AU mean')
    axs[3].plot(metrics['epochs'], metrics['au_median'], linewidth=2, 
               color='C2', linestyle='--', label='AU median')
    axs[3].plot(metrics['epochs'], metrics['eu_mean'], linewidth=2, 
               color='C3', label='EU mean')
    axs[3].plot(metrics['epochs'], metrics['eu_median'], linewidth=2, 
               color='C3', linestyle='--', label='EU median')
    axs[3].set_xlabel('Epoch', fontsize=11)
    axs[3].set_ylabel('Uncertainty', fontsize=11)
    axs[3].set_title('AU and EU Evolution (ID Samples)', fontsize=12)
    axs[3].grid(alpha=0.3)
    axs[3].legend()
    
    # Plot 5 & 6: Scatter plots of AU vs EU for last epoch (linear and log scale)
    # Load logits for the last epoch
    last_epoch = metrics['epochs'][-1]
    file_paths_last = []
    for base_path in all_paths:
        logits_file = base_path / f"logits/val_logits_epoch_{last_epoch:03d}.csv"
        if logits_file.exists():
            file_paths_last.append(str(logits_file))
    
    if file_paths_last:
        ens_logits_last = load_ens_logits(file_paths_last, only_id_gts=False, apply_temp=False)
        id_classes = ens_logits_last["id_classes"].numpy()
        gt = ens_logits_last["gt"].numpy()
        id_mask = np.isin(gt, id_classes)
        
        logits = ens_logits_last["logits"].float()
        
        # Apply temperature scaling if requested
        if apply_temp:
            mean_logits = logits.mean(dim=0)
            id_logits = mean_logits[id_mask]
            id_labels = torch.from_numpy(gt[id_mask]).long()
            temp = compute_optimal_temperature_nll(id_logits, id_labels)
            logits = logits / temp
        
        probs = torch.softmax(logits, dim=-1)
        AU, EU, TU = uncertainties(probs)
        
        # Convert to numpy
        AU_np = AU.numpy() if torch.is_tensor(AU) else AU
        EU_np = EU.numpy() if torch.is_tensor(EU) else EU
        TU_np = TU.numpy() if torch.is_tensor(TU) else TU
        
        C = len(id_classes)
        N = len(logits)
        max_tu = np.log(C)
        
        # Scatter plots for both linear and log scale
        for plot_idx, (ax, is_log) in enumerate([(axs[4], False), (axs[5], True)]):
            # Plot scatter points (ID samples only)
            ax.plot(AU_np[id_mask], EU_np[id_mask], '.', markersize=3, alpha=0.5, color='C0')
            
            # Plot theoretical minimum boundary
            AUb, EUb, TUb = get_theoretical_min_boundary(num_classes=C, num_ensemble=N,
                                                          interp=False, detailed=True)
            ax.plot(AUb, EUb, "m-", linewidth=2, label=f"Min Boundary (N={N}, C={C})")
            
            # Reference lines
            ax.plot([0, max_tu], [max_tu, 0], 'k-', alpha=0.5, linewidth=1.5, label=r"$TU \leq \log(C)$")
            ax.plot([0, max_tu], [0, 0], 'b-', alpha=0.5, linewidth=1.5, label=r"$TU \geq AU$")
            
            ax.set_xlabel('AU', fontsize=11)
            ax.set_ylabel('EU', fontsize=11)
            
            if is_log:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim([1e-8, max_tu])
                ax.set_ylim([1e-8, max_tu])
                ax.set_title(f'AU vs EU (Log Scale) - Epoch {last_epoch}', fontsize=12)
            else:
                ax.set_xlim([0, max_tu / 2])
                ax.set_ylim([0, max_tu / 2])
                ax.set_title(f'AU vs EU (Linear Scale) - Epoch {last_epoch}', fontsize=12)
            
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, axs, metrics