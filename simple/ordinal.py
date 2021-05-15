def __ordinal_suffix(value):
    s = str(value)
    if s.endswith('11') or s.endswith('12') or s.endswith('13'):
        return 'th'
    elif s.endswith('1'):
        return 'st'
    elif s.endswith('2'):
        return 'nd'
    elif s.endswith('3'):
        return 'rd'

    return 'th'


def ordinal(value):
    return str(value) + __ordinal_suffix(value)