def inspect(obj):
    attrs = list(
        filter(lambda x: len(x) < 2 or (x[0] != "_" and x[1] != "_"),
               dir(obj))
    )
    print(attrs)
