import sys
import logging
def _error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        script_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script: [{script_name}] at line number: [{line_number}] error message: [{str(error)}]"
    else:
        return f"Error message: [{str(error)}]"

class CustomException(Exception):
    def __init__(self, message: str, error_detail: sys):
        super().__init__(message)
        self.error_message = _error_message_detail(message, error_detail)

    def __str__(self):
        return f"CustomException: {self.args[0]} | Error Detail: {self.error_message}"

if __name__ == "__main__":
    try:
        # Example code that raises an exception
        x = 1 / 0
    except Exception as e:
        logging.error("An error occurred", exc_info=True)
        raise CustomException(e, sys)