Once `EinsteinVariation.py` is downloaded in an appropriate directory, open a Python editor and import the following:

```Python
import sympy as sp
import EinsteinVariation as ev
```

`sympy` is Python's symbolic computation library. You can read the full documentation [here.](https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html) 
We want to begin by telling sympy which symbols to use as math variables (in contrast to regular Python variables, see the sympy documentation). The symbols we need for spherical coordinates are:

```
t,r,T,P=sp.symbols('t r T P', real=True) 
```
We'll use upper case T and P to represent the Greek letters theta and phi respectively. The `real=True` parameter is optional but helps Python simplify longer expressions. Any other symbols we want to use will also have to be initialized this way:

```
#Kerr parameters
M,a=sp.symbols('M a', real=True, positive=True)
```

We can now define (i) a list of coordinates and (ii) a covariant metric. For Minkowski spacetime in spherical coordinates, we would do:

```Python
Xu=sp.Array([t,r,T,P])  
gll=sp.Array([[-1,0,0,0],[0,1,0,0],[0,0,r**2,0],[0,0,0,(r*sp.sin(T))**2]])  #inner brackets represent rows
```

In general, it is convenient to have all tensors stored as `sp.Array()` objects so that sympy recognizes them as such. This also allows higher-ranked tensors to be printed nicely. 

We can raise the indices on the metric using `ev.Inverse(gll)`:

---

With our metric and coordinate system, a line element is obtained with `ev.LineltFromMetric(gll,Xu)`:

Of course, we can work the other way:

```Python
ds=ev.LineltFromMetric(gll,Xu)
ev.MetricFromDS(ds,Xu)
```

returns the original Minkowski metric in spherical coordinates.

---

Now for the main event. With a metric an list of coordinates, we can calculate the Christoffel symbol of the first kind using `GetCClll(gll,Xu)`:

---

The 'lll' indicates three lower indices. `ev.GetCCull(gll,Xu)` raises the first index to get the Christoffel symbol of the second kind. Individual entries can be obtained using `ev.GetCCull(gll,Xu)[1,2,2]` keeping in mind that Python indices start at 0:

---

`ev.GetRiemann(gll,Xu)` returns the fourth-rank Riemann tensor in up-low-low-low form. All entries are zero since spacetime is flat:

---

On the surface of a sphere, the metric and coordinates are given by

```Python
sphere_surface_gll = sp.Array([[r**2, 0], [0, (r*sp.sin(T))**2]])
sphere_surface_Xu = sp.Array([T, P])
```

For this manifold the `ev.GetRiemann(sphere_surface_gll,sphere_surface_Xu)` returns

---

`sympy` comes with in-built tensorproduct and tensorcontraction capability. We can thus lower an index on the Riemann tensor via

```Python
Rulll=ev.GetRiemann(sphere_surface_gll,sphere_surface_Xu)
prod=sp.tensorproduct(sphere_surface_gll,Rulll)
Rllll=sp.tensorcontraction(prod,(0,2))
Rlll
```
---

Here, `sp.tensorproduct(sphere_surface_gll,Rulll)` creates a rank-six tensor: two indices from the metric and four from the Riemann tensor. Calling `sp.tensorcontraction(prod,(0,2))` contracts the 0 and 2 indices of `prod`, resulting in a fully covariant form for the Riemann tensor.

The Ricci tensor is a contraction of the Riemann tensor, and can be obtained using `ev.GetRicci(sphere_surface_gll,sphere_surface_Xu)`:

---

And the Ricci scalar is a contraction of this object with the metric, `ev.GetRicciS((sphere_surface_gll,sphere_surface_Xu)`:

---

We can compute most of these expressions for the Kerr metric in Boyer-Lindquist coordinates. Use `GetKerrgll(M,a,r,T)` (noting that we initialized `M` and `a` as math variables earlier) to get

---

Unfortunately, Python is not equpped to simplify large expressions, so `ev.GetRiemmann()`, `ev.GetRicci()`, and `ev.GetRicciS()` will output incomprehensible results. They are all zero in disguise. You can make a special case assumption to get Python to output the right results:

```Python
kerrgll=ev.GetKerrgll(M,a,r,T).subs(a,0)
kerrgll
```

Now the Riemann and Ricci tensors should be zero, as expected.



