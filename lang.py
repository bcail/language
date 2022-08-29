import copy
from enum import Enum, auto
import os
from pathlib import Path
import re
import tempfile


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

    def __init__(self, items=None):
        self.items = items or []

    def __eq__(self, other):
        if not isinstance(other, Vector):
            raise Exception(f'{other} is not a Vector')
        return self.items == other.items

    def __str__(self):
        return str(self.items)

    def __repr__(self):
        return str(self)

    def append(self, item):
        self.items.append(item)


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
    FLOAT = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    IF = auto()


FLOAT_RE = re.compile('-?\d+\.?\d*')


def _get_token(token_buffer):
    if token_buffer.isdigit() or (token_buffer.startswith('-') and token_buffer[1:].isdigit()):
        return {'type': TokenType.NUMBER, 'lexeme': token_buffer}
    elif FLOAT_RE.match(token_buffer):
        return {'type': TokenType.FLOAT, 'lexeme': token_buffer}
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
        elif c in ['+', '-', '*', '/', '=', '>', '<']:
            token_buffer += c
        elif c in [',', '\n']:
            pass
        elif c == ':':
            if token_buffer:
                raise Exception('invalid ":" char')
            token_buffer += c
        elif c.isalnum() or c in ['?', '.']:
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
    elif token['type'] == TokenType.FLOAT:
        return float(token['lexeme'])
    elif token['type'] == TokenType.STRING:
        return token['lexeme']
    elif token['type'] == TokenType.SYMBOL:
        return Symbol(name=token['lexeme'])
    elif token['type'] == TokenType.KEYWORD:
        return Keyword(name=token['lexeme'])
    else:
        return token['type']


class DictBuilder:

    def __init__(self):
        self.items = []

    def append(self, item):
        self.items.append(item)


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
            #start new expression
            new_vector = Vector()
            if stack_of_lists is None:
                stack_of_lists = [ast]
            stack_of_lists[-1].append(new_vector)
            stack_of_lists.append(new_vector)
            current_list = stack_of_lists[-1]
        elif token['type'] in [TokenType.RIGHT_PAREN, TokenType.RIGHT_BRACKET, TokenType.RIGHT_BRACE]:
            #finish an expression
            stack_of_lists.pop(-1)
            current_list = stack_of_lists[-1]
        elif token['type'] == TokenType.LEFT_BRACE:
            new_dict = DictBuilder()
            if stack_of_lists is None:
                stack_of_lists = [ast]
            stack_of_lists[-1].append(new_dict)
            stack_of_lists.append(new_dict)
            current_list = stack_of_lists[-1]
        else:
            current_list.append(_get_node(token))

        index = index + 1

    for a in ast:
        ast_obj.add(a)

    return ast_obj


def add(params, env):
    return sum([evaluate(p, env=env) for p in params])


def add_c(params, env):
    params = [emit_c(p, env=env) for p in params]
    return f'add({params[0]}, {params[1]})'


def subtract(params, env):
    return evaluate(params[0], env=env) - sum([evaluate(n, env=env) for n in params[1:]])


def multiply(params, env):
    result = evaluate(params[0], env=env)
    for n in params[1:]:
        result = result * evaluate(n, env=env)
    return result


def divide(params, env):
    result = evaluate(params[0], env=env)
    for n in params[1:]:
        result = result / evaluate(n, env=env)
    return result


def equal(params, env):
    first_param = evaluate(params[0], env=env)
    for param in params[1:]:
        if evaluate(param, env=env) != first_param:
            return False
    return True


def greater(params, env):
    return bool(evaluate(params[0], env=env) > evaluate(params[1], env=env))


def greater_equal(params, env):
    return bool(evaluate(params[0], env=env) >= evaluate(params[1], env=env))


def less(params, env):
    return bool(evaluate(params[0], env=env) < evaluate(params[1], env=env))


def less_equal(params, env):
    return bool(evaluate(params[0], env=env) <= evaluate(params[1], env=env))


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


def loop(params, env):
    bindings = params[0]
    body = params[1:]

    loop_params = bindings.items[::2]
    initial_args = bindings.items[1::2]

    local_env = copy.deepcopy(env)

    for index, loop_param in enumerate(loop_params):
        local_env[loop_param.name] = evaluate(initial_args[index], env=local_env)


    while True:
        result = evaluate(*body, env=local_env)

        if isinstance(result, tuple) and isinstance(result[0], Symbol) and result[0].name == 'recur':
            new_args = result[1:]
            for index, loop_param in enumerate(loop_params):
                local_env[loop_param.name] = evaluate(new_args[index], env=local_env)
        else:
            return result


def str_func(params, env):
    if not params:
        return ''
    if len(params) == 1:
        return str(evaluate(params[0], env=env))
    else:
        return ''.join([str(evaluate(p, env=env)) for p in params])


def str_split(params, env):
    return params[0].split()


def str_subs(params, env):
    s = evaluate(params[0], env=env)
    start = evaluate(params[1], env=env)

    if len(params) > 2:
        end = evaluate(params[2], env=env)
        return s[start:end]
    else:
        return s[start:]


def conj(params, env):
    l = evaluate(params[0], env=env).items
    new_element = evaluate(params[1], env=env)
    new_l = l[:]
    new_l.append(new_element)
    return Vector(new_l)


def subvec(params, env):
    l = evaluate(params[0], env=env).items
    start = evaluate(params[1], env=env)

    if len(params) > 2:
        end = evaluate(params[2], env=env)
        return Vector(l[start:end])
    else:
        return Vector(l[start:])


def map_get(params, env):
    d = evaluate(params[0], env=env)
    key = evaluate(params[1], env=env)
    return d[key]


def map_keys(params, env):
    d = evaluate(params[0], env=env)
    return list(d.keys())


def map_vals(params, env):
    d = evaluate(params[0], env=env)
    return list(d.values())


def map_contains(params, env):
    d = evaluate(params[0], env=env)
    key = evaluate(params[1], env=env)
    return key in d


def map_assoc(params, env):
    d = copy.deepcopy(evaluate(params[0], env=env))
    key = evaluate(params[1], env=env)
    d[key] = evaluate(params[2], env=env)
    return d


def map_dissoc(params, env):
    d = copy.deepcopy(evaluate(params[0], env=env))
    key = evaluate(params[1], env=env)
    d.pop(key, None)
    return d


def print_func(params, env):
    print(evaluate(params[0], env=env), end='')


def println(params, env):
    print(evaluate(params[0], env=env))


def println_c(params, env):
    param = emit_c(params[0], env=env)
    return f'int result = {param};\nprintf("%d", result);'


def read_line(params, env):
    line = input()
    return line


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


def defn(params, env):
    name = params[0].name
    var = Var(name=name)
    var.value = Function(params=params[1], body=params[2])
    environment[name] = var
    return var


def file_open(params, env):
    file_name = evaluate(params[0], env=env)
    return open(file_name, 'rb')


def file_read(params, env):
    f = evaluate(params[0], env=env)
    return f.read()


def file_close(params, env):
    f = evaluate(params[0], env=env)
    f.close()


environment = {
    '+': add,
    '-': subtract,
    '*': multiply,
    '/': divide,
    '=': equal,
    '>': greater,
    '>=': greater_equal,
    '<': less,
    '<=': less_equal,
    'def': define,
    'defn': defn,
    'let': let,
    'loop': loop,
    'fn': create_function,
    'str': str_func,
    'subs': str_subs,
    'conj': conj,
    'subvec': subvec,
    'get': map_get,
    'keys': map_keys,
    'vals': map_vals,
    'contains?': map_contains,
    'assoc': map_assoc,
    'dissoc': map_dissoc,
    'str/split': str_split,
    'print': print_func,
    'println': println,
    'read-line': read_line,
    'file/open': file_open,
    'file/read': file_read,
    'file/close': file_close,
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
            elif first.name == 'recur':
                params = [evaluate(r, env) for r in rest]
                return (first, *params)
            elif first.name == 'do':
                results = [evaluate(n, env=env) for n in rest]
                return results[-1]
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
    if isinstance(node, Vector):
        return Vector([evaluate(n, env=env) for n in node.items])
    if isinstance(node, DictBuilder):
        keys = [evaluate(k, env=env) for k in node.items[::2]]
        values = [evaluate(v, env=env) for v in node.items[1::2]]
        d = dict(zip(keys, values))
        return d
    return node


compile_env = {
    '+': add_c,
    'println': println_c,
}


def emit_c(node, env=compile_env):
    if isinstance(node, list):
        first = node[0]
        rest = node[1:]
        if isinstance(first, Symbol):
            if first.name in env:
                if isinstance(env[first.name], Var) and isinstance(env[first.name].value, Function):
                    f = env[first.name].value
                    return f(rest)
                if callable(env[first.name]):
                    return env[first.name](rest, env=env)
                else:
                    raise Exception(f'symbol first in list and not callable: {first.name} -- {env[first.name]}')
            else:
                raise Exception(f'unhandled symbol: {first}')
    if isinstance(node, str):
        return f'"{node}"'
    if isinstance(node, int):
        return f'{node}'
    return str(node)


def run(source):
    tokens = scan_tokens(source)
    ast = parse(tokens)
    return ast.evaluate()


def _run_file(file_name):
    with open(file_name, 'rb') as f:
        run(f.read().decode('utf8').strip())


def _run_prompt():
    while True:
        try:
            code = input('> ')
        except EOFError:
            break
        if not code.strip():
            break
        result = run(code)
        print(result)


def main(file_name):
    if file_name:
        _run_file(file_name)
    else:
        _run_prompt()


def _compile(source):
    tokens = scan_tokens(source)
    ast = parse(tokens)

    start = '''#include <stdio.h>

int add(int x, int y)
{
    return x + y;
}

int main()
{
'''
    end = '''
return 0;
}'''

    compiled_code = '\n'.join([emit_c(f) for f in ast.forms])

    c_code = '\n'.join([start, compiled_code, end])

    return c_code


def compile_to_c(file_name):
    tmp = tempfile.mkdtemp(dir='.', prefix='tmp')
    c_file = Path(tmp) / Path(f'{file_name.stem}.c')
    print(f'compiling {file_name} to {c_file}...')

    with open(file_name, 'rb') as f:
        source = f.read().decode('utf8')

    c_program = _compile(source)

    with open(c_file, mode='wb') as f:
        f.write(c_program.encode('utf8'))
    executable = Path(tmp) / file_name.stem
    print(f'Compile with: gcc -o {executable} {c_file}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Random language')
    parser.add_argument('-c', action='store_true', dest='compile', help='compile the file to C')
    parser.add_argument('file', type=str, nargs='?', help='file to compile')

    args = parser.parse_args()

    if args.compile:
        if args.file:
            compile_to_c(Path(args.file))
        else:
            print('no file to compile')
    else:
        main(args.file)
