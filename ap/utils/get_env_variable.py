import os


def env(key: str) -> str:
    variable = os.environ.get(key, None)
    if variable == "True":
        variable = True
    elif variable == "False":
        variable = False

    return variable
