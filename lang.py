from enum import Enum, auto


def _report(line_number, where, message):
    print(f'[line {line_number}] Error{where}: {message}')


def error(line_number, message):
    _report(line_number, '', message)


class TokenType(Enum):
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    COMMA = auto()
    DOT = auto()
    MINUS = auto()
    PLUS = auto()
    EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    EOF = auto()


def scan_tokens(source):
    tokens = []

    token = ''
    for c in source:
        if c == '(':
            tokens.append({'type': TokenType.LEFT_PAREN, 'lexeme': '('})
        elif c == ')':
            if token:
                if token.isdigit():
                    tokens.append({'type': TokenType.NUMBER, 'lexeme': token})
                token = ''
            tokens.append({'type': TokenType.RIGHT_PAREN, 'lexeme': ')'})
        elif c == '+':
            tokens.append({'type': TokenType.PLUS, 'lexeme': c})
        elif c == '-':
            tokens.append({'type': TokenType.MINUS, 'lexeme': c})
        elif c.isdigit():
            token += c
        elif c == ' ':
            if token:
                if token.isdigit():
                    tokens.append({'type': TokenType.NUMBER, 'lexeme': token})
                token = ''
        else:
            print(f'unknown char "{c}"')

    return tokens


def parse(tokens):
    ast = []
    stack_of_lists = [ast]
    for token in tokens[1:-1]:
        if token['type'] == TokenType.LEFT_PAREN:
            #start new expression
            new_list = []
            stack_of_lists[-1].append(new_list)
            stack_of_lists.append(new_list)
        elif token['type'] == TokenType.RIGHT_PAREN:
            stack_of_lists.pop(-1)
        else:
            if token['type'] == TokenType.NUMBER:
                stack_of_lists[-1].append(int(token['lexeme']))
            else:
                stack_of_lists[-1].append(token)
    return ast


def run(source):
    tokens = scan_tokens(source)
    for t in tokens:
        print(t)


def _run_file(file_name):
    print(f'Running {file_name}')
    with open(file_name, 'rb') as f:
        run(f.read().decode('utf8'))


def _run_prompt():
    print('Running prompt')
    while True:
        code = input('> ')
        if not code.strip():
            break
        run(code)


def main(file_name):
    if file_name:
        _run_file(file_name)
    else:
        _run_prompt()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Random language')
    parser.add_argument('file', type=str, nargs='?', help='file to compile')

    args = parser.parse_args()

    main(args.file)
