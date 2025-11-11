import traceback  # For tracking the error encounted
import sys #system library


class CustomException(Exception):

    def __init__(self, error_message, error_detail:sys):  # constructor
        super().__init__(error_message) # Calls Exception's __init__
        self.error_message = self.get_detailed_error_message(error_message, error_detail)


    @staticmethod 
    def get_detailed_error_message(error_message, error_detail:sys):
        """
        @staticmethod >>> This function doesn't need self.anything
        It just takes inputs and returns a formatted string
        """
        _, _, exc_tb = traceback.sys.exc_info() # keep only the trackback
        file_name = exc_tb.tb_frame.f_code.co_filename # which file caused the error
        line_number = exc_tb.tb_lineno # Line number of code causing the error

        return f"Error in {file_name}, line {line_number}: {error_message}"
    
    def __str__(self):  # Magic function 
        """
        Define how an object should be represented as a string. 
        Without it, printing would return <__main__.CustomError object at 0x7f8b4c3d2a90>
        """
        return self.error_message