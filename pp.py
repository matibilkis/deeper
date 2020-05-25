a  = 1
for name in dir():
    print(name)
    print(type(name))
    if not name.startswith('_'):
        del globals()[name]
