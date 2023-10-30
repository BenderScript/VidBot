import sys


def print_exception_details(e):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback_details = {
                         'filename': exc_traceback.tb_frame.f_code.co_filename,
                         'lineno'  : exc_traceback.tb_lineno,
                         'name'    : exc_traceback.tb_frame.f_code.co_name,
                         'type'    : exc_type.__name__,
                         'message' : str(e)
                        }
    del(exc_type, exc_value, exc_traceback)
    print(f"An error occurred: {traceback_details}")