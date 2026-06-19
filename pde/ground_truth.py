"""Ground-truth Lie algebras and evaluation metrics."""
import numpy as np
from scipy.optimize import linear_sum_assignment


class GroundTruthAlgebra:
    @staticmethod
    def heat_generators(x, t, u, alpha=0.01):
        """
        Heat equation u_t = α u_xx: 6-dimensional finite algebra.
        
        Derived from canonical u_τ = u_{yy} via τ = αt, y = x.
        """
        zeros, ones = np.zeros_like(u), np.ones_like(u)
        X, T = np.meshgrid(x, t, indexing='ij')
        a = alpha
        return [
            (ones, zeros, zeros),                                    # v1: ∂_x
            (zeros, ones, zeros),                                    # v2: ∂_t
            (zeros, zeros, u),                                       # v3: u∂_u
            (2 * a * T, zeros, -X * u),                              # v4: Galilean
            (X, 2 * T, zeros),                                       # v5: scaling
            (4*a*T*X, 4*a*T**2, -(X**2 + 2*a*T) * u),              # v6: projective
        ]

    @staticmethod
    def burgers_generators(x, t, u):
        zeros, ones = np.zeros_like(u), np.ones_like(u)
        X, T = np.meshgrid(x, t, indexing='ij')
        return [(ones, zeros, zeros), (zeros, ones, zeros), (T, zeros, ones),
                (X, 2 * T, -u), (X * T, T ** 2, -(X + T * u))]

    @staticmethod
    def kdv_generators(x, t, u):
        zeros, ones = np.zeros_like(u), np.ones_like(u)
        X, T = np.meshgrid(x, t, indexing='ij')
        return [(ones, zeros, zeros), (zeros, ones, zeros),
                (T, zeros, (1. / 6.) * ones), (X, 3 * T, -2 * u)]


def get_gt_dim(pde): return {'heat': 6, 'burgers': 5, 'kdv': 4}[pde]


def get_gt_generators(pde, x, t, u):
    return {'heat': GroundTruthAlgebra.heat_generators,
            'burgers': GroundTruthAlgebra.burgers_generators,
            'kdv': GroundTruthAlgebra.kdv_generators}[pde](x, t, u)


def get_gt_names(pde):
    """Human-readable names for each GT generator."""
    return {
        'heat': ['∂_x', '∂_t', 'u∂_u', 'Galilean', 'scaling', 'projective'],
        'burgers': ['∂_x', '∂_t', 'Galilean', 'scaling', 'projective'],
        'kdv': ['∂_x', '∂_t', 'Galilean', 'scaling'],
    }[pde]


class Metrics:
    @staticmethod
    # def grassmann_distance(A, B):
    #     Ao, _ = np.linalg.qr(A)
    #     Bo, _ = np.linalg.qr(B)
    #     # Compute principal angles using all basis vectors (natively handles matrices of unequal rank)
    #     s = np.linalg.svd(Ao.T @ Bo, compute_uv=False)
    #     return float(np.sqrt(np.sum(np.arccos(np.clip(s, -1.0, 1.0)) ** 2)))
    def grassmann_distance(A, B):
        """
        Subspace distance between col-spans of A and B.
        
        For subspaces of equal dimension: standard Grassmann distance
        (sqrt of sum of squared principal angles).
        
        For subspaces of unequal dimension: principal angles computed
        between subspaces of min dim, plus |k_A - k_B| additional angles
        of pi/2 (missing/extra dimensions treated as maximally orthogonal).
        
        This is the standard generalization of Grassmann distance to
        subspaces of unequal rank (see Wang, Sun, Wang, 2006).
        It penalizes both under-discovery (k_A < k_B, missed GT generators)
        and over-discovery (k_A > k_B, spurious generators).
        """
        Ao, _ = np.linalg.qr(A)
        Bo, _ = np.linalg.qr(B)
        k_A = Ao.shape[1]
        k_B = Bo.shape[1]
    
        # Principal angles on min-dimensional intersection
        s = np.linalg.svd(Ao.T @ Bo, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        thetas_matched = np.arccos(s)
        
        # Missing/extra dimensions contribute pi/2 each
        n_missing = abs(k_A - k_B)
        thetas_missing = np.full(n_missing, np.pi / 2)
        
        thetas = np.concatenate([thetas_matched, thetas_missing])
        return float(np.sqrt(np.sum(thetas ** 2)))    

    @staticmethod
    def best_cosine_similarity(disc, gt):
        sim = Metrics.cosine_similarity_matrix(disc, gt)
        # Use full similarity matrix for Hungarian algorithm avoiding arbitrary slicing order-dependence
        ri, ci = linear_sum_assignment(-sim)
        return float(np.mean(sim[ri, ci]))

    @staticmethod
    def spectral_rank(norms, gap=3.0):
        s = np.sort(norms)[::-1]
        for k in range(len(s) - 1):
            if s[k + 1] < 1e-10: return k + 1
            if s[k] / s[k + 1] > gap: return k + 1
        return len(norms)

    @staticmethod
    def generators_to_matrix(gens):
        return np.array([np.concatenate([g.flatten() for g in gen]) for gen in gens]).T

    @staticmethod
    def cosine_similarity_matrix(discovered, ground_truth):
        """
        Full cosine similarity matrix [n_disc, n_gt].
        Each entry = |cos(angle)| between flattened generators.
        """
        nd, ng = len(discovered), len(ground_truth)
        sim = np.zeros((nd, ng))
        for i, d in enumerate(discovered):
            df = np.concatenate([g.flatten() for g in d])
            dn = np.linalg.norm(df) + 1e-12
            for j, g in enumerate(ground_truth):
                gf = np.concatenate([c.flatten() for c in g])
                gn = np.linalg.norm(gf) + 1e-12
                sim[i, j] = np.abs(np.dot(df, gf)) / (dn * gn)
        return sim

    @staticmethod
    def per_generator_matching(discovered, ground_truth, gt_names, norms=None):
        """
        Hungarian matching + per-generator report.
        Returns dict with 'matrix', 'matching', 'report_lines'.
        """
        sim = Metrics.cosine_similarity_matrix(discovered, ground_truth)
        nd, ng = sim.shape
        ri, ci = linear_sum_assignment(-sim)

        lines = []
        matching = {}
        for idx in range(nd):
            norm_str = f"  ‖v‖={norms[idx]:.4f}" if norms is not None else ""
            if idx in ri:
                pos = list(ri).index(idx)
                gt_idx = ci[pos]
                score = sim[idx, gt_idx]
                gt_name = gt_names[gt_idx] if gt_idx < len(gt_names) else f"GT_{gt_idx}"
                quality = "★" if score > 0.8 else "●" if score > 0.5 else "○"
                lines.append(f"    {quality} v_{idx + 1}{norm_str} → {gt_name} (cos={score:.3f})")
                matching[idx] = (gt_idx, gt_name, score)
            else:
                lines.append(f"    ○ v_{idx + 1}{norm_str} → no match")
                matching[idx] = (None, None, 0.0)

        return {'matrix': sim, 'matching': matching, 'report_lines': lines}

    @staticmethod
    def bracket_closure_error(generators, dx, dt):
        """
        Algebraic Closure Error (ACE).
        ACE = mean of ||[vi, vj]|| over all pairs.
        Small ACE = generators form a (near-)closed Lie algebra.

        Also returns relative projection error for reference.
        """
        n = len(generators)
        if n <= 1:
            return 0.0

        G = Metrics.generators_to_matrix(generators)
        bracket_norms = []
        for i in range(n):
            for j in range(i + 1, n):
                b = Metrics._lie_bracket(generators[i], generators[j], dx, dt)
                b_flat = np.concatenate([comp.flatten() for comp in b])
                
                # ACE evaluates closure: distance from bracket to the subspace of generators
                c, _, _, _ = np.linalg.lstsq(G, b_flat, rcond=None)
                proj = G @ c
                residual = b_flat - proj
                
                norm = np.sqrt(np.mean(residual ** 2))
                bracket_norms.append(norm)

        return float(np.mean(bracket_norms))

    @staticmethod
    def _lie_bracket(vi, vj, dx, dt):
        def dd(xi, eta, f, dx, dt):
            df_dx = np.zeros_like(f)
            df_dx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
            df_dx[0, :] = (f[1, :] - f[-1, :]) / (2 * dx)
            df_dx[-1, :] = (f[0, :] - f[-2, :]) / (2 * dx)
            df_dt = np.zeros_like(f)
            df_dt[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dt)
            df_dt[:, 0] = (f[:, 1] - f[:, 0]) / dt
            df_dt[:, -1] = (f[:, -1] - f[:, -2]) / dt
            return xi * df_dx + eta * df_dt

        xi_i, eta_i, phi_i = vi;
        xi_j, eta_j, phi_j = vj
        return (dd(xi_i, eta_i, xi_j, dx, dt) - dd(xi_j, eta_j, xi_i, dx, dt),
                dd(xi_i, eta_i, eta_j, dx, dt) - dd(xi_j, eta_j, eta_i, dx, dt),
                dd(xi_i, eta_i, phi_j, dx, dt) - dd(xi_j, eta_j, phi_i, dx, dt))