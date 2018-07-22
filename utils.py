import logging


logger = logging.getLogger(__name__)

def print_key_pairs(v, title="Parameters", print_function=None):
    """
    Dump key/value pairs to print_function
    :param v:
    :param title:
    :param print_function:
    :return:
    """
    items = v.items() if type(v) is dict else v
    print_function("=" * 40)
    print_function(title)
    print_function("=" * 40)
    for key,value in items:
        print_function("{:<15}: {:<10}".format(key, value if value is not None else "None"))
    print_function("-" * 40)
