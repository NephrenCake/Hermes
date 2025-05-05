import ast

def boolean_fix(output):
    return output.replace("true", "True").replace("false", "False")

def type_check(output, expected_type):
    try:
        output_eval = ast.literal_eval(output)
        if not isinstance(output_eval, expected_type):
            return None
        return output_eval
    except:
        '''
        if(expected_type == List):
            valid_output = self.extract_list_from_string(output)
            output_eval = ast.literal_eval(valid_output)
            if not isinstance(output_eval, expected_type):
                return None
            return output_eval
        elif(expected_type == dict):
            valid_output = self.extract_dict_from_string(output)
            output_eval = ast.literal_eval(valid_output)
            if not isinstance(output_eval, expected_type):
                return None
            return output_eval
        '''
        return None