# -*- coding: utf-8 -*-

###############################################################################

# GMM

###############################################################################


# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv('CC GENERAL.csv', encoding='utf-8')
X = dataset[["BALANCE", "PURCHASES"]].values

### GMM
"""
- covariance_type => parámetro para definir el tipo de covarianza que se usa; por defecto, 'full',
hace que cada componente/cluster tenga su propia matriz de covarianzas

- n_init => numero de inicializaciones que se hacen (de cara a encontrar la mejor posible)

- init_params => método para inicializar los valores de los parametros. Por defecto usa primero un kmeans para 
encontrar dichos valores iniciales

- max_iter  => numero maximo de iteraciones que se llevan a cabo con EM.

"""
from sklearn.mixture import GaussianMixture

def gmm_n_selection(epsilon, figure=False):
    """
    Esta función prueba combinaciones de GMM para distinto numero de clusters
    y calcula su BIC/AIC. Cuando el BIC se estabilice y varie poco se detienen
    las iteraciones y se elige ese numero de clusters.
    """
    n_components = []
    bic = []
    aic = []
    
    diff = np.inf # Valor de diferencia entre dos iteraciones
    i = 1
    
    while diff > epsilon:
        print("Iteracion Nº Clusters: k: {k}".format(k=i))
        n_components.append(i)
        gmm = GaussianMixture(n_components=i, covariance_type='full', 
                              n_init=1, init_params='kmeans', 
                              random_state=0).fit(X)
        bic_t = gmm.bic(X)
        aic.append(gmm.aic(X))
        
        # Primera iteracion
        if diff == np.inf:
            diff = bic_t
        # Si ya fuese 0
        elif bic_t == 0:
            break
        # Resto de iteraciones
        else:
            diff = (bic[-1] - bic_t)/bic[-1]
        bic.append(bic_t)
        i += 1
    
    if figure:
        plt.plot(n_components, bic, label='BIC')
        plt.plot(n_components, aic, label='AIC')
        plt.title('BIC/AIC para GMM segun el nº de clusters')
        plt.legend(loc='best')
        plt.xlabel('n_components')
        plt.show()
    
    # Clusters finales
    k = i-1
    return bic, k


# Visualizacion
epsilon = 0.0001 # Valor de convergencia
bic, _ = gmm_n_selection(epsilon, figure=True)

# Seleccion
epsilon = 0.01 # Valor de convergencia
_, k = gmm_n_selection(epsilon, figure=False)

gmm = GaussianMixture(n_components=k, covariance_type='full', 
                      n_init=1, init_params='kmeans', 
                      max_iter=100, random_state=0)
gmm.fit(X) # Model fit

# Probabilidad de pertenecer a cada cluster
probs = gmm.predict_proba(X)

# Matrices con los parametros ajustados
mat_means = gmm.means_ # [n_components x n_features]
mat_covariances = gmm.covariances_ # [n_components x n_features x n_features]
mat_weights = gmm.weights_ # [n_components,]


### Visualizar clusters [1]
y_gmm = gmm.predict(X)

plt.scatter(X[y_gmm == 0, 0], X[y_gmm == 0, 1], s = 10, c = 'blue', label = 'C1') 
plt.scatter(X[y_gmm == 1, 0], X[y_gmm == 1, 1], s = 10, c = 'red', label = 'C2')
plt.scatter(X[y_gmm == 2, 0], X[y_gmm == 2, 1], s = 10, c = 'green', label = 'C3')
plt.scatter(X[y_gmm == 3, 0], X[y_gmm == 3, 1], s = 10, c = 'yellow', label = 'C4')
plt.scatter(X[y_gmm == 4, 0], X[y_gmm == 4, 1], s = 10, c = 'black', label = 'C5')
plt.title('Clusters de Clientes')
plt.xlabel('X1: Balance en la Cuenta ($)')
plt.ylabel('X2: Gastos en Compras ($)')
plt.legend()
plt.show()

### Visualizar clusters [2]
# https://plot.ly/scikit-learn/plot-gmm/
# !conda install -c plotly plotly
import plotly
import plotly.graph_objs as go
import itertools
from scipy import linalg
import math

color_iter = itertools.cycle(['navy', 'cyan', 'cornflowerblue', 'gold',
                              'orange'])

def plot_results(X, Y_, means, covariances,  title):
    data = []
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        trace = go.Scatter(x=X[Y_ == i, 0], y=X[Y_ == i, 1], 
                           mode='markers',
                           marker=dict(color=color))
        data.append(trace)
        # Plot an ellipse to show the Gaussian component
        a =  v[1]
        b =  v[0]
        x_origin = mean[0]
        y_origin = mean[1]
        x_ = [ ]
        y_ = [ ]
    
        for t in range(0,361,10):
            x = a*(math.cos(math.radians(t))) + x_origin
            x_.append(x)
            y = b*(math.sin(math.radians(t))) + y_origin
            y_.append(y)
    
        elle = go.Scatter(x=x_ , y=y_, mode='lines',
                          showlegend=False,
                          line=dict(color=color,
                                   width=2))
        data.append(elle)
       
    layout = go.Layout(title=title, showlegend=False,
                       xaxis=dict(zeroline=False, showgrid=False),
                       yaxis=dict(zeroline=False, showgrid=False),)
    fig = go.Figure(data=data, layout=layout)
    
    return fig

fig = plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_,
                   'Gaussian Mixture')

plotly.offline.plot(fig)