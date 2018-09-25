import numpy as np

# Input: two floating point numpy arrays of the same shape.
# The inputs will be interpreted as probability mass functions and the KL divergence is returned.
def KLD(P, G):
	if P.shape != G.shape:
		raise ValueError('The shape of P: {} must match the shape of G: {}'.format(P.shape, G.shape))
	if np.any(P<0):
		raise ValueError('P has some negative values')
	if np.any(G<0):
		raise ValueError('G has some negative values')

	# Normalize P and G so that they sum to 1
	p_n = P / np.sum(P)
	g_n = G / np.sum(G)

	EPS = 1e-16 # small regularization constant for numerical stability
	kl = np.sum(g_n * np.log2( EPS + (g_n / (EPS + p_n) ) ))

	return kl



# Example usage
P = np.random.rand(224, 224)
G = np.random.rand(224, 224)
k = KLD(P,G)

# Final evaluation criterion will be KLD averaged over all images in the test set.