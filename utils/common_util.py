def split_list(lst, n):
    """split lst into n part
    Example:
    input: lst[2,3,4,5], n=3
    output: [2,3], [4], [5]
    """
    n = min(n, len(lst))
    assert n > 0
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def nslice(lst, n, truncate=False, reverse=False):
    """Splits lst into n-sized chunks, optionally reversing the chunks."""
    assert n > 0
    while len(lst) >= n:
        if reverse:
            yield lst[:n][::-1]
        else:
            yield lst[:n]
        lst = lst[n:]
    if len(lst) and not truncate:
        yield lst
