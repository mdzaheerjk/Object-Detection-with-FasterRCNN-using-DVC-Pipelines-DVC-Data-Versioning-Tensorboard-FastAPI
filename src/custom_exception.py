import sys
import traceback

class CustomException(Exception):
    def __init__(self, message: str, error_detail: Exception = None):
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)
        
    @staticmethod
    def get_detailed_error_message(message, error_detail):
        file_name = "Unknown"
        line_number = "Unknown"
        error_msg = str(error_detail) if error_detail is not None else "No Underlying Exception"
        
        if error_detail is not None and getattr(error_detail, "__traceback__", None) is not None:
            tb = error_detail.__traceback__
            while tb and tb.tb_next:
                tb = tb.tb_next
            if tb:
                file_name = tb.tb_frame.f_code.co_filename
                line_number = tb.tb_lineno
        else:
            _, _, tb_sys = sys.exc_info()
            if tb_sys is not None:
                while tb_sys and tb_sys.tb_next:
                    tb_sys = tb_sys.tb_next
                if tb_sys:
                    file_name = tb_sys.tb_frame.f_code.co_filename
                    line_number = tb_sys.tb_lineno
        
        return (
            f"{message} | "
            f"Error : {error_msg} | "
            f"File Name : {file_name} | "
            f"Line Number : {line_number}"
        )
        
    def __str__(self):
        return self.error_message