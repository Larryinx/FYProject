def file_name(path:str)->str:
    tokens=path.split('/')
    return tokens[-1]

def split_suffix(file_name):
    tokens=file_name.split('.')
    if len(tokens)==1:
        return tokens[0],None
    return '.'.join(tokens[:len(tokens)-1]),tokens[-1]

if __name__ =='__main__':
    pass