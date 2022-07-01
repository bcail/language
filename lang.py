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
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    IF = auto()


def _get_token(token_buffer):
    if token_buffer.isdigit():
        return {'type': TokenType.NUMBER, 'lexeme': token_buffer}
    elif token_buffer == 'true':
        return {'type': TokenType.TRUE}
    elif token_buffer == 'false':
        return {'type': TokenType.FALSE}
    elif token_buffer == 'nil':
        return {'type': TokenType.NIL}
    elif token_buffer == 'if':
        return {'type': TokenType.IF}
    else:
        print(f'unrecognized token: {token_buffer}')


def scan_tokens(source):
    tokens = []

    token_buffer = ''
    index = 0
    while index < len(source):
        c = source[index]
        if c == '(':
            tokens.append({'type': TokenType.LEFT_PAREN})
        elif c == ')':
            if token_buffer:
                tokens.append(_get_token(token_buffer))
                token_buffer = ''
            tokens.append({'type': TokenType.RIGHT_PAREN})
        elif c == '+':
            tokens.append({'type': TokenType.PLUS})
        elif c == '-':
            if source[index+1].isdigit():
                tokens.append({'type': TokenType.NEGATIVE})
            else:
                tokens.append({'type': TokenType.MINUS})
        elif c == '*':
            tokens.append({'type': TokenType.ASTERISK})
        elif c == '/':
            tokens.append({'type': TokenType.SLASH})
        elif c == '=':
            tokens.append({'type': TokenType.EQUAL})
        elif c.isalnum():
            token_buffer += c
        elif c == ' ':
            if token_buffer:
                tokens.append(_get_token(token_buffer))
                token_buffer = ''
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
        elif token['type'] == TokenType.NIL:
            stack_of_lists[-1].append(None)
        elif token['type'] == TokenType.TRUE:
            stack_of_lists[-1].append(True)
        elif token['type'] == TokenType.FALSE:
            stack_of_lists[-1].append(False)
        elif token['type'] == TokenType.NEGATIVE:
            stack_of_lists[-1].append(int(tokens[index+1]['lexeme']) * -1)
            # increment an extra time, since we're consuming two tokens here
            index = index + 1
        elif token['type'] == TokenType.NUMBER:
            stack_of_lists[-1].append(int(token['lexeme']))
        else:
            stack_of_lists[-1].append(token['type'])
        index = index + 1
    return ast


def evaluate(node):
    if isinstance(node, list):
        first = node[0]
        rest = node[1:]
        if first == TokenType.PLUS:
            return sum([evaluate(n) for n in rest])
        elif first == TokenType.MINUS:
            return evaluate(rest[0]) - sum([evaluate(n) for n in rest[1:]])
        elif first == TokenType.ASTERISK:
            result = rest[0]
            for n in rest[1:]:
                result = result * evaluate(n)
            return result
        elif first == TokenType.SLASH:
            result = rest[0]
            for n in rest[1:]:
                result = result / evaluate(n)
            return result
        elif first == TokenType.EQUAL:
            for n in rest[1:]:
                if n != rest[0]:
                    return False
            return True
        elif first == TokenType.IF:
            test_val = evaluate(rest[0])
            true_val = evaluate(rest[1])
            if len(rest) > 2:
                false_val = evaluate(rest[2])
            else:
                false_val = None
            if test_val in [False, None]:
                return false_val
            else:
                return true_val
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
