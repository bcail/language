from enum import Enum, auto


class Symbol:

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        try:
            return self.name == other.name
        except AttributeError:
            return False

    def __str__(self):
        return f'<Symbol {self.name}>'

    def __repr__(self):
        return str(self)


class Var:

    def __init__(self, name, value=None):
        self.name = name
        self.value = value

    def __eq__(self, other):
        try:
            return self.name == other.name
        except AttributeError:
            return False

    def __str__(self):
        return f'<Var {self.name}: {self.value}>'

    def __repr__(self):
        return str(self)


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
    SYMBOL = auto()
    STRING = auto()
    NUMBER = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    IF = auto()
    DEF = auto()
    QUOTE = auto()


def _get_token(token_buffer):
    if token_buffer.isdigit():
        return {'type': TokenType.NUMBER, 'lexeme': token_buffer}
    elif token_buffer == 'true':
        return {'type': TokenType.TRUE}
    elif token_buffer == 'false':
        return {'type': TokenType.FALSE}
    elif token_buffer == 'nil':
        return {'type': TokenType.NIL}
    elif token_buffer == 'def':
        return {'type': TokenType.DEF}
    elif token_buffer == 'if':
        return {'type': TokenType.IF}
    elif token_buffer == 'quote':
        return {'type': TokenType.QUOTE}
    else:
        return {'type': TokenType.SYMBOL, 'lexeme': token_buffer}


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
        elif c == '[':
            tokens.append({'type': TokenType.LEFT_BRACKET})
        elif c == ']':
            if token_buffer:
                tokens.append(_get_token(token_buffer))
                token_buffer = ''
            tokens.append({'type': TokenType.RIGHT_BRACKET})
        elif c in ['+', '*', '/', '=']:
            token_buffer += c
        elif c == '-':
            if source[index+1].isdigit():
                tokens.append({'type': TokenType.NEGATIVE})
            else:
                tokens.append({'type': TokenType.MINUS})
        elif c.isalnum():
            token_buffer += c
        elif c == ' ':
            if token_buffer:
                tokens.append(_get_token(token_buffer))
                token_buffer = ''
        else:
            print(f'unknown char "{c}"')

        index += 1

    if token_buffer:
        tokens.append(_get_token(token_buffer))

    return tokens


def parse(tokens):
    ast = []
    stack_of_lists = None
    current_list = ast
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token['type'] in [TokenType.LEFT_PAREN, TokenType.LEFT_BRACKET]:
            #start new expression
            new_list = []
            if stack_of_lists is None:
                stack_of_lists = [ast]
            stack_of_lists[-1].append(new_list)
            stack_of_lists.append(new_list)
            current_list = stack_of_lists[-1]
        elif token['type'] in [TokenType.RIGHT_PAREN, TokenType.RIGHT_BRACKET]:
            #finish an expression
            stack_of_lists.pop(-1)
            current_list = stack_of_lists[-1]
        elif token['type'] == TokenType.NIL:
            current_list.append(None)
        elif token['type'] == TokenType.TRUE:
            current_list.append(True)
        elif token['type'] == TokenType.FALSE:
            current_list.append(False)
        elif token['type'] == TokenType.NEGATIVE:
            current_list.append(int(tokens[index+1]['lexeme']) * -1)
            # increment an extra time, since we're consuming two tokens here
            index = index + 1
        elif token['type'] == TokenType.NUMBER:
            current_list.append(int(token['lexeme']))
        elif token['type'] == TokenType.SYMBOL:
            current_list.append(Symbol(name=token['lexeme']))
        else:
            current_list.append(token['type'])
        index = index + 1

    if len(ast) == 1:
        ast = ast[0]
    return ast


environment = {}


def add(params):
    return sum([evaluate(p) for p in params])


def subtract(params):
    return evaluate(params[0]) - sum([evaluate(n) for n in params[1:]])


def multiply(params):
    result = params[0]
    for n in params[1:]:
        result = result * evaluate(n)
    return result


def divide(params):
    result = params[0]
    for n in params[1:]:
        result = result / evaluate(n)
    return result


def equal(params):
    for n in params[1:]:
        if n != params[0]:
            return False
    return True


def define(params):
    name = params[0].name
    var = Var(name=name)
    if len(params) > 1:
        var.value = params[1]
    environment[name] = var
    return var


def if_form(params):
    test_val = evaluate(params[0])
    true_val = evaluate(params[1])
    if len(params) > 2:
        false_val = evaluate(params[2])
    else:
        false_val = None
    if test_val in [False, None]:
        return false_val
    else:
        return true_val


def evaluate(node):
    if isinstance(node, list):
        first = node[0]
        rest = node[1:]
        if isinstance(first, list):
            return [evaluate(n) for n in node]
        elif isinstance(first, Symbol):
            if first.name == '+':
                return add(rest)
            elif first.name == '*':
                return multiply(rest)
            elif first.name == '/':
                return divide(rest)
            elif first.name == '=':
                return equal(rest)
        elif first == TokenType.QUOTE:
            return rest[0]
        elif first == TokenType.MINUS:
            return subtract(rest)
        elif first == TokenType.DEF:
            return define(rest)
        elif first == TokenType.IF:
            return if_form(rest)
    if isinstance(node, Symbol):
        return environment[node.name].value
    return node


def run(source):
    tokens = scan_tokens(source)
    ast = parse(tokens)
    evaluate(ast)


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
