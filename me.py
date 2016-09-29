
# coding: utf-8

# In[461]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5.05, 0.1)
mfx = fuzz.trapmf(x, [2, 2.5, 3, 4.5])

plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'm')
plt.title('Funcion Trapezoide')
plt.ylabel("Grado de pertencia")
plt.xlabel("Elemento")
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()




# In[450]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5.05, 0.1)
mfx = fuzz.trapmf(x, [2, 2.5, 3, 4.5])
mfx2 = fuzz.trimf(x, [2, 3, 4.5])

plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'm')
plt.plot(x, mfx2, 'c')

plt.title('Funciones de Membresia')
plt.ylabel("Grado de pertencia")
plt.xlabel("Elemento")

plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()


# In[462]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5.05, 0.1)
mfx = fuzz.trimf(x, [2, 3, 4.5])


plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'y')

plt.title('Funcion Triangulo')
plt.ylabel("Grado de pertencia")
plt.xlabel("Elemento")


plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()


# In[456]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5.05, 0.1)
mfx = fuzz.trimf(x, [2, 3, 4.5])
mfx2 = fuzz.trimf(x, [1, 2, 3.5])

plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'g')
plt.plot(x, mfx2, 'r')

plt.title('Union Triangulos')
plt.ylabel("Grado de pertencia")
plt.xlabel("Elemento")

plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()


# In[463]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-1, 4.1, 0.1)
mfx = fuzz.sigmf(x, 2, 4)


plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'c')


plt.title('Funcion Sigmoidal')
plt.ylabel("Grado de pertencia")
plt.xlabel("Elemento")

plt.ylim(0.0, 1.05)

plt.show()


# In[458]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5.05, 0.1)
x2 = np.arange(-1, 4.1, 0.1)

mfx = fuzz.trapmf(x, [2, 2.5, 3, 4.5])
mfx2 = fuzz.trimf(x, [2, 3, 4.5])
mfx3 = fuzz.sigmf(x2, 2, 4)

plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'm')
plt.plot(x, mfx2, 'c')
plt.plot(x, mfx3, 'g')

plt.title('Funciones de Membresia')
plt.ylabel("Grado de pertencia")
plt.xlabel("Elemento")
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()


# In[467]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 4)
c = np.linspace(0, 2)
def Hombroiz(x,c):

    return np.piecewise(x, [x < 0, x > 0.2], [0, 1])
    
plt.figure(figsize=(8, 5))
plt.plot(x, Hombroiz(x,c), 'c')

plt.title('Funcion Hombro Izquierdo')
plt.ylabel("Grado de pertencia")
plt.xlabel("Elemento")
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)
plt.show()


# In[459]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5.05, 0.1)
x2 = np.arange(-1, 4.1, 0.1)
x3 = np.linspace(-2, 2, 4)
c = np.linspace(0, 2)
def Hombroiz(x3,c):

    return np.piecewise(x3, [x3 < 0, x3 > 0.2], [0, 1])

mfx = fuzz.trapmf(x, [2, 2.5, 3, 4.5])
mfx2 = fuzz.trimf(x, [2, 3, 4.5])
mfx3 = fuzz.sigmf(x2, 2, 4)

plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'm')
plt.plot(x, mfx2, 'c')
plt.plot(x, mfx3, 'g')
plt.plot(x3, Hombroiz(x3,c), 'r')


plt.title('Funciones de Membresia')
plt.ylabel("Grado de pertencia")
plt.xlabel("Elemento")
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()


# In[466]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 4)
c = np.linspace(0, 2)
def Hombrodr(x,c):

    return np.piecewise(x, [x > 0, x < 0.2], [0, 1])
    
plt.figure(figsize=(8, 5))
plt.plot(x, Hombrodr(x,c), 'c')

plt.title('Funcion Hombro Derecho')
plt.ylabel("Grado de pertencia")
plt.xlabel("Elemento")
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)
plt.show()


# In[220]:

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0, 1, 500)

triangle = signal.sawtooth(2* np.pi * 1 * t,0.5)

plt.figure(figsize=(8, 5))

plt.plot(t, triangle)


plt.ylabel("Triangle-Shaped")
plt.xlabel("Universe variable (arb)")
plt.ylim(-0.1, 1.2)
plt.legend(loc=2)


plt.show()


# In[151]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10, 10)
e = np.e
def f(x, e):
    return ((e ** x)-(e ** -x))/((e ** x)+(e ** -x))
plt.figure(figsize=(8, 5))
plt.plot(x, f(x, e), 'c')

plt.ylabel("Sigmoidal-Shaped")
plt.xlabel("Universe variable (arb)")
plt.ylim(-1.05, 1.05)


plt.show()


# In[292]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 4.1, 0.1)
mfx = fuzz.gbellmf(x, 1, 4, 2)


plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'c')


plt.ylabel("Sigmoidal-Shaped")
plt.xlabel("Universe variable (arb)")
plt.ylim(0.0, 1.2)
plt.xlim(0.0, 5)

plt.show()

