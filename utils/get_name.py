
def get_name(a):
    if a < 10:
        new_name = "000" + str(a)
    elif a < 100:
        new_name = "00" + str(a)
    elif a < 1000:
        new_name = "0" + str(a)
    elif a < 10000:
        new_name = "" + str(a)
    return new_name
