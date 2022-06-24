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
    NEGATIVE = auto()
    PLUS = auto()
    ASTERISK = auto()
    SLASH = auto()
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
    index = 0
    while index < len(source):
        c = source[index]
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
            if source[index+1].isdigit():
                tokens.append({'type': TokenType.NEGATIVE, 'lexeme': c})
            else:
                tokens.append({'type': TokenType.MINUS, 'lexeme': c})
        elif c == '*':
            tokens.append({'type': TokenType.ASTERISK, 'lexeme': c})
        elif c == '/':
            tokens.append({'type': TokenType.SLASH, 'lexeme': c})
        elif c.isdigit():
            token += c
        elif c == ' ':
            if token:
                if token.isdigit():
                    tokens.append({'type': TokenType.NUMBER, 'lexeme': token})
                token = ''
        else:
            print(f'unknown char "{c}"')

        index += 1

    return tokens


def parse(tokens):
    ast = []
    stack_of_lists = [ast]
    index = 1
    while index < (len(tokens) - 1):
        token = tokens[index]
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
            elif token['type'] == TokenType.NEGATIVE:
                stack_of_lists[-1].append(int(tokens[index+1]['lexeme']) * -1)
                # increment an extra time, since we're consuming two tokens here
                index = index + 1
            else:
                stack_of_lists[-1].append(token)
        index = index + 1
    return ast


def evaluate(node):
    if isinstance(node, list):
        first = node[0]
        rest = node[1:]
        if first['type'] == TokenType.PLUS:
            return sum([evaluate(n) for n in rest])
        elif first['type'] == TokenType.MINUS:
            return evaluate(rest[0]) - sum([evaluate(n) for n in rest[1:]])
        elif first['type'] == TokenType.ASTERISK:
            result = rest[0]
            for n in rest[1:]:
                result = result * evaluate(n)
            return result
        elif first['type'] == TokenType.SLASH:
            result = rest[0]
            for n in rest[1:]:
                result = result / evaluate(n)
            return result
    return node


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
