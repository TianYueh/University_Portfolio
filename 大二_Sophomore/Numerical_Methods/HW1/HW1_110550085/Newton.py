import numpy as np

def jacobian(xyz):
    x, y, z=xyz
    return[[1, -3, -2*z],
           [6*x*x, 1, -10*z],
           [8*x, 1, 1]]

def f(xyz):
    x, y, z=xyz
    return [x-3*y-z*z+3, 2*x*x*x+y-5*z*z+2, 4*x*x+y+z-7]

def newton(f, x_init, jacobian):

    max_iter=1000
    tol=0.00001

    x_last=x_init
    for i in range(max_iter):
        #print(x_last)
        J=np.array(jacobian(x_last))
        F=np.array(f(x_last))

        diff=np.linalg.solve(J, -F)
        x_last=x_last+diff

        if(np.linalg.norm(diff)<tol):
            print('Num of Iter: ', i, '\n')
            return x_last
    
    print('Diverge\n')
    return x_last

a=complex(input())
b=complex(input())
c=complex(input())
x_sol=newton(f, [a, b, c], jacobian)
print(x_sol)





