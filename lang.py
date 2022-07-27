import copy
from enum import Enum, auto


class AST:

    def __init__(self):
        self.forms = []

    def add(self, form):
        self.forms.append(form)

    def evaluate(self):
        results = [evaluate(f) for f in self.forms]

        if len(results) == 1:
            return results[0]
        else:
            return results


class Keyword:

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        try:
            return self.name == other.name
        except AttributeError:
            return False


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


class Vector:

    def __init__(self, items):
        self.items = items

    def __eq__(self, other):
        return self.items == other.items


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
    GREATER = auto()
    GREATER_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    KEYWORD = auto()
    SYMBOL = auto()
    STRING = auto()
    NUMBER = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    IF = auto()


def _get_token(token_buffer):
    if token_buffer.isdigit() or (token_buffer.startswith('-') and token_buffer[1:].isdigit()):
        return {'type': TokenType.NUMBER, 'lexeme': token_buffer}
    elif token_buffer == 'true':
        return {'type': TokenType.TRUE}
    elif token_buffer == 'false':
        return {'type': TokenType.FALSE}
    elif token_buffer == 'nil':
        return {'type': TokenType.NIL}
    elif token_buffer == 'if':
        return {'type': TokenType.IF}
    elif token_buffer.startswith(':'):
        return {'type': TokenType.KEYWORD, 'lexeme': token_buffer}
    else:
        return {'type': TokenType.SYMBOL, 'lexeme': token_buffer}


def scan_tokens(source):
    tokens = []

    inside_string = False
    token_buffer = ''
    index = 0
    while index < len(source):
        c = source[index]

        if inside_string:
            if c == '"':
                tokens.append({'type': TokenType.STRING, 'lexeme': token_buffer})
                token_buffer = ''
                inside_string = False
            else:
                token_buffer += c
        elif c == '(':
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
        elif c == '{':
            tokens.append({'type': TokenType.LEFT_BRACE})
        elif c == '}':
            if token_buffer:
                tokens.append(_get_token(token_buffer))
                token_buffer = ''
            tokens.append({'type': TokenType.RIGHT_BRACE})
        elif c in ['+', '-', '*', '/', '=']:
            token_buffer += c
        elif c == ',':
            pass
        elif c == ':':
            if token_buffer:
                raise Exception('invalid ":" char')
            token_buffer += c
        elif c.isalnum():
            token_buffer += c
        elif c == '?':
            token_buffer += c
        elif c == ' ':
            if token_buffer:
                tokens.append(_get_token(token_buffer))
                token_buffer = ''
        elif c == '"':
            inside_string = True
        else:
            print(f'unknown char "{c}"')

        index += 1

    if token_buffer:
        tokens.append(_get_token(token_buffer))

    return tokens


def _get_node(token):
    if token['type'] == TokenType.NIL:
        return None
    elif token['type'] == TokenType.TRUE:
        return True
    elif token['type'] == TokenType.FALSE:
        return False
    elif token['type'] == TokenType.NUMBER:
        return int(token['lexeme'])
    elif token['type'] == TokenType.STRING:
        return token['lexeme']
    elif token['type'] == TokenType.SYMBOL:
        return Symbol(name=token['lexeme'])
    elif token['type'] == TokenType.KEYWORD:
        return Keyword(name=token['lexeme'])
    else:
        return token['type']


def parse(tokens):
    ast_obj = AST()
    ast = []
    stack_of_lists = None
    current_list = ast
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token['type'] == TokenType.LEFT_PAREN:
            #start new expression
            new_list = []
            if stack_of_lists is None:
                stack_of_lists = [ast]
            stack_of_lists[-1].append(new_list)
            stack_of_lists.append(new_list)
            current_list = stack_of_lists[-1]
        elif token['type'] == TokenType.LEFT_BRACKET:
            vector_tokens = []
            while token['type'] != TokenType.RIGHT_BRACKET:
                index += 1
                token = tokens[index]
                if token['type'] != TokenType.RIGHT_BRACKET:
                    vector_tokens.append(_get_node(token))
            current_list.append(Vector(vector_tokens))
        elif token['type'] == TokenType.RIGHT_PAREN:
            #finish an expression
            stack_of_lists.pop(-1)
            current_list = stack_of_lists[-1]
        elif token['type'] == TokenType.LEFT_BRACE:
            map_tokens = []
            while token['type'] != TokenType.RIGHT_BRACE:
                index += 1
                token = tokens[index]
                if token['type'] != TokenType.RIGHT_BRACE:
                    map_tokens.append(_get_node(token))
            m = dict(zip(map_tokens[::2], map_tokens[1::2]))
            current_list.append(m)
        else:
            current_list.append(_get_node(token))

        index = index + 1

    for a in ast:
        ast_obj.add(a)

    return ast_obj


def add(params, env):
    return sum([evaluate(p, env=env) for p in params])


def subtract(params, env):
    return evaluate(params[0]) - sum([evaluate(n, env=env) for n in params[1:]])


def multiply(params, env):
    result = params[0]
    for n in params[1:]:
        result = result * evaluate(n, env=env)
    return result


def divide(params, env):
    result = params[0]
    for n in params[1:]:
        result = result / evaluate(n, env=env)
    return result


def equal(params, env):
    for n in params[1:]:
        if n != params[0]:
            return False
    return True


def define(params, env):
    name = params[0].name
    var = Var(name=name)
    if len(params) > 1:
        var.value = evaluate(params[1], env=env)
    environment[name] = var
    return var


def if_form(params, env):
    test_val = evaluate(params[0], env=env)
    true_val = evaluate(params[1], env=env)
    if len(params) > 2:
        false_val = evaluate(params[2], env=env)
    else:
        false_val = None
    if test_val in [False, None]:
        return false_val
    else:
        return true_val


def let(params, env):
    bindings = params[0]
    body = params[1:]

    paired_bindings = []
    for i in range(0, len(bindings.items), 2):
        paired_bindings.append(bindings.items[i:i+2])

    local_env = copy.deepcopy(env)
    for binding in paired_bindings:
        local_env[binding[0].name] = evaluate(binding[1], env=local_env)

    return evaluate(*body, env=local_env)


def str_func(params, env):
    if not params:
        return ''
    if len(params) == 1:
        return str(params[0])
    else:
        return ''.join([p for p in params])


def str_split(params, env):
    return params[0].split()


def map_get(params, env):
    return params[0][params[1]]


def map_keys(params, env):
    return list(params[0].keys())


def map_vals(params, env):
    return list(params[0].values())


def map_contains(params, env):
    return params[1] in params[0]


def map_assoc(params, env):
    d = copy.deepcopy(params[0])
    d[params[1]] = params[2]
    return d


def map_dissoc(params, env):
    d = copy.deepcopy(params[0])
    d.pop(params[1], None)
    return d


def println(params, env):
    print(params[0])


class Function:

    def __init__(self, params, body):
        self.params = params
        self.body = body

    def __call__(self, args):
        local_env = copy.deepcopy(environment)
        bindings = zip(self.params.items, args)
        for binding in bindings:
            local_env[binding[0].name] = evaluate(binding[1], env=local_env)
        return evaluate(self.body, env=local_env)


def create_function(params, env):
    return Function(params=params[0], body=params[1])


environment = {
    '+': add,
    '-': subtract,
    '*': multiply,
    '/': divide,
    '=': equal,
    'def': define,
    'let': let,
    'fn': create_function,
    'str': str_func,
    'get': map_get,
    'keys': map_keys,
    'vals': map_vals,
    'contains?': map_contains,
    'assoc': map_assoc,
    'dissoc': map_dissoc,
    'str/split': str_split,
    'println': println,
}


def evaluate(node, env=environment):
    if isinstance(node, list):
        first = node[0]
        rest = node[1:]
        if isinstance(first, list):
            results = [evaluate(n, env=env) for n in node]
            if callable(results[0]):
                return evaluate(results, env=env)
            else:
                raise Exception('first element of list not callable: {results[0]}')
        elif isinstance(first, Symbol):
            if first.name in env:
                if isinstance(env[first.name], Var) and isinstance(env[first.name].value, Function):
                    f = env[first.name].value
                    return f(rest)
                if callable(env[first.name]):
                    return env[first.name](rest, env=env)
                else:
                    raise Exception(f'symbol first in list and not callable: {first.name} -- {env[first.name]}')
            elif first.name == 'quote':
                return rest[0]
            else:
                raise Exception(f'unhandled symbol: {first}')
        elif first == TokenType.IF:
            return if_form(rest, env=env)
        elif isinstance(first, Function):
            return first(rest)
        else:
            raise Exception(f'first element of list not callable: {first}')
    if isinstance(node, Symbol):
        symbol = env[node.name]
        return getattr(symbol, 'value', symbol)
    if isinstance(node, Function):
        return node()
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
