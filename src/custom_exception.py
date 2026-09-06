import sys
import traceback

class CustomException(Exception):
    def __init__(self, message:str,error_detail:Exception=None):
        self.error_message=self.get_detailed_error_message(message,error_detail)
        super().__init__(self.error_message)
        
    @staticmethod
    def get_detailed_error_message(message,error_detail):
        if error_detail is not None:
            tb=error_detail.__traceback__
            while tb.tb_next:
                tb=tb.tb_next
                
            file_name=tb.tb_frame.f_code.co_filename
            line_numer=tb.tb_lineno
            error_msg=str(error_detail)
        else:
            file_name="Unkown"
            line_numer="Unkown"
            error_msg="No UnderLying Exception"
        
        return (
            f"{message} | "
            f"Error : {error_msg} | "
            f"File Name : {file_name} | "
            f"Line Number : {line_numer}"
            )
        
    def __str__(self):
        return self.error_message
        