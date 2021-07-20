import numpy as np

class LDA:

    def __init__(self,n_components):
        self.n_components=n_components
        self.linear_discriminant=None

    def fit(self,X,y):
        n_features=X.shape[1] #150 4
        class_labels=np.unique(y)
        
        # S_W ,S_B
        mean_overall=np.mean(X,axis=0)
        S_W=np.zeros((n_features,n_features)) #4,4
        S_B=np.zeros((n_features,n_features)) #4,4
        for c in class_labels:
            X_c=X[y==c]
            mean_c=np.mean(X_c,axis=0)
            
            S_W+=(X_c-mean_c).T.dot(X_c-mean_c)
            
            n_c=X_c.shape[0]
            #(4,1)*(4,1)T=(4,4)
            mean_diff=(mean_c-mean_overall).reshape(n_features,1)
            S_B+=n_c*(mean_diff).dot(mean_diff.T)
        
        A=np.linalg.inv(S_W).dot(S_B)
        eigenvalues,eigenvectors=np.linalg.eig(A)
        eigenvectors=eigenvectors.T
        idx=np.argsort(abs(eigenvalues))[::-1]
        eigenvalues=eigenvalues[idx]
        eigenvector=eigenvectors[idx]
        self.linear_discriminant=eigenvectors[0:self.n_components]



    def transform(self,X):
        
        return np.dot(X,self.linear_discriminant.T)


