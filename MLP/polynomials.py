# Madeleine Lindström, madeli@kth.se

# Polynomials used to generate data for testing networks
def polynomial_d1(x, a, b):
    """Polynomial function of degree 1. Outputs y = ax + b."""
    y = a*x + b
    return y

def polynomial_d2(x, a, b, c):
    """Polynomial function of degree 2. Outputs y = ax^2 + bx + c."""
    y = a*(x**2) + b*x + c
    return y

def polynomial_d3(x, a, b, c, d):
    """Polynomial function of degree 3. Outputs y = ax^3 + bx^2 + cx + d."""
    y = a*x**3 + b*x**2 + c*x + d
    return y
