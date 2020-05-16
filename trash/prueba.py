def func(x,*d, **kwargs):
    print(d)
    print(kwargs)
    ff = kwargs.get("tau", 0)
    return x, ff
print(func(1,2,miau = 1, tau=12))
