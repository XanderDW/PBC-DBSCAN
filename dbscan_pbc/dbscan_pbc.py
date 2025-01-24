from sklearn.cluster import DBSCAN
import numpy as np

class DBSCAN_PBC(DBSCAN):
    """
    DBSCAN with support for Periodic Boundary Conditions (PBC).
    Inherits from sklearn.cluster.DBSCAN.
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, 
                 algorithm='auto', leaf_size=30, p=None, n_jobs=None):
        """
        Initialize DBSCAN_PBC.
        
        Parameters:
        - Same parameters as for the plain DBSCAN constructor.
        """
        super().__init__(eps=eps, min_samples=min_samples, metric=metric,
                         metric_params=metric_params, algorithm=algorithm,
                         leaf_size=leaf_size, p=p, n_jobs=n_jobs)

    def fit(self, X, pbc_lower=0, pbc_upper=1, return_padded_dbs=False):
        """
        Fit the DBSCAN model while applying periodic boundary conditions to the data.
        
        Parameters:
        - X: array-like of shape (n_samples, n_features). Training instances to cluster.
        - pbc_lower, pbc_upper: scalar or array-like of shape (n_features). Define the periodic boundary limits. If None, no boundary condition is applied. If scalar the same boundary is applied to all features.
        - return_padded_dbs: boolean, whether to return the padded DBSCAN properties.
        
        Returns:
        - db: the fitted DBSCAN model.
        """
        if len(X) == 0:
            return None
        
        if pbc_lower is None and pbc_upper is None:
            return super().fit(X)
        
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Input must be of shape (n_samples, n_features), but got {X.ndim} dimensions.")

        D = X.shape[1]

        pbc_lower = np.asarray(pbc_lower)
        if pbc_lower.ndim == 0:
            pbc_lower = np.full(D, pbc_lower)
        elif pbc_lower.ndim == 1:
            if len(pbc_lower) != D:
                raise ValueError(f"pbc_lower must be of shape (n_features), but got ({pbc_lower.shape}).")
        else:
            raise ValueError(f"pbc_lower must be scalar or of shape (n_features), but got {pbc_lower.ndim} dimensions.")
        pbc_upper = np.asarray(pbc_upper)
        if pbc_upper.ndim == 0:
            pbc_upper = np.full(D, pbc_upper)
        elif pbc_upper.ndim == 1:
            if len(pbc_upper) != D:
                raise ValueError(f"pbc_upper must be of shape (n_features), but got ({pbc_upper.shape}).")
        else:
            raise ValueError(f"pbc_upper must be scalar or of shape (n_features), but got {pbc_upper.ndim} dimensions.")
        for d in range(D):
            if pbc_lower[d] is None or pbc_upper[d] is None:
                continue
            if pbc_upper[d] <= pbc_lower[d]:
                raise ValueError(f"pbc_upper must be larger than pbc_lower.")
        
        for d in range(D):
            if pbc_lower[d] is None or pbc_upper[d] is None:
                continue
            if np.min(X[:,d]) < pbc_lower[d]:
                raise ValueError(f"The minimum datapoint is smaller than the pbc_lower in dimension {d}.")
            if np.max(X[:,d]) > pbc_upper[d]:
                raise ValueError(f"The maximum datapoint is larger than the pbc_upper in dimension {d}.")
        
        if not isinstance(return_padded_dbs, bool):
            raise ValueError(f"return_padded_dbs must be a boolean value, but got {type(return_padded_dbs).__name__}.")

        
        # Canonicalize the domain onto [0, L]^D
        L = np.zeros(D)
        for d in range(D):
            if pbc_lower[d] is None or pbc_upper[d] is None:
                L[d] = -1  # Non-periodic direction
            else:
                L[d] = pbc_upper[d] - pbc_lower[d]
                X[:, d] -= pbc_lower[d]  # Set lower boundary to zero
        
        # Pad the boundary points
        padded_points, source_idx = self._pad_boundary_points(X, self.eps, L)
        
        # Apply DBSCAN on the padded points
        db = super().fit(padded_points)
        labels = db.labels_

        # Merge clusters after periodic boundary conditions
        new_labels = self._pbc_cluster_merger(labels, source_idx, X.shape[0], padded_points.shape[0])

        # Renumber labels in consecutive order
        for j, label in enumerate(np.unique(new_labels[new_labels >= 0])):
            if j != label:
                new_labels[new_labels == label] = j

        # Decanonicalize
        X += pbc_lower[np.newaxis, :]

        # Return padded DBSCAN object if requested
        if return_padded_dbs:
            db.padded_components_ = padded_points + pbc_lower[np.newaxis, :]  # Decanonicalize
            db.padded_labels_     = labels
            db.padded_core_sample_indices_ = db.core_sample_indices_

        # Save results into the original DBSCAN object
        db.labels_ = new_labels
        db.components_ = X
        db.core_sample_indices_ = db.core_sample_indices_[db.core_sample_indices_ < len(X)]

        return db

    def _pad_boundary_points(self, points, eps, L):
        """
        Pad points by extending into the domain [-eps, L+eps] ^ D.
        
        Parameters:
        - points: array-like, shape (n_samples, n_features)
          The input data points.
        - eps: float
          The neighborhood radius used for periodic boundary checks.
        - L: array-like, shape (n_features,)
          The size of the periodic box in each dimension.
        
        Returns:
        - padded_points: array-like, shape (n_padded_samples, n_features)
          The padded data points.
        - source_idx: list of int
          Indices of original points corresponding to the padded points.
        """
        padded_points = points.copy() #copy all original points
        source_idx = [] #list to keep track of index of source point of padded point

        D = points.shape[1] #dimensionality of the space

        #default list of shift vectors
        to_pad_init = []
        for d in range(D):
            to_pad_init.append(np.array([0]))

        #loop over all (unpadded) points
        for idx, point in enumerate(points):

            #check if particle is close to periodic boundary
            to_pad = to_pad_init.copy()
            boundary_point = False
            for d in range(D):
                if L[d] < 0: #-1 is the spurious value for open boundary
                    continue #skip this open boundary
                if point[d] < eps:
                    to_pad[d] = np.array([-1,0])
                    boundary_point = True
                elif point[d] > L[d] - eps:
                    to_pad[d] = np.array([1,0])
                    boundary_point = True

            #go to the next point if this is not a boundary point
            if not boundary_point:
                continue

            #compute the all needed shifts, including corners
            shifts = np.meshgrid(*to_pad, indexing='ij') #NOTE: requires Numpy version >= 1.9
            shift_vectors = np.stack(shifts, axis=-1).reshape(-1, D)

            #carry out the needed shifts
            for shift_vector in shift_vectors:
                if np.all(shift_vector == 0):
                    continue #skip the identity shift
                new_point = point - shift_vector * L
                padded_points = np.append(padded_points,new_point[np.newaxis,:],axis=0)
                source_idx.append(idx)

        return np.array(padded_points), np.array(source_idx)

    def _pbc_cluster_merger(self, labels, source_idx, Npoints, Npadded_points):
        """
        Merge clusters based on periodic boundary conditions.
        
        Parameters:
        - labels: array-like, shape (n_samples,)
          The labels assigned to the data points.
        - source_idx: list of int
          Indices of original points corresponding to the padded points.
        - Npoints: int
          The number of original points.
        - Npadded_points: int
          The number of padded points.
        
        Returns:
        - merged_labels: array-like, shape (n_samples,)
          The merged labels after applying periodic boundary conditions.
        """
        merged_cluster = labels[:Npoints].copy() #copy labels of src points

        matches = [] #the list to keep track of all equivalent clusters

        #loop over all padded points that are not src points
        for i in range(Npadded_points - Npoints):
            
            IDX = source_idx[i] #index of matching src point

            #get labels of padded point and src point
            label_pp = labels[Npoints +i]
            label_src = merged_cluster[ IDX ]

            if not label_pp == label_src: #check if labels are non-matching

                #if src point is a noise point, but padded point is not, then color the source point and go to the next padded point
                if label_src == -1:
                    merged_cluster[ IDX ] = label_pp
                    continue

                #if padded point is a noise point, just go to the next
                if label_pp == -1:
                    continue
                
                #now we obtained a match of two clusters, check if (one of) the labels is already in our list, and if not, then add them
                found = False
                for match in matches:
                    if label_pp in match and label_src in match:
                        found = True
                    else:
                        if label_pp in match:
                            match.append(label_src)
                            found = True
                        elif label_src in match:
                            match.append(label_pp)
                            found = True
                if not found:
                    matches.append([label_src, label_pp])
        

        #replace the matching cluster labels, using the smallest label
        for match in matches:
            correct_label = min(match)
            for wrong_label in match:
                if not wrong_label == correct_label:
                    merged_cluster[ merged_cluster == wrong_label ] = correct_label

        return merged_cluster