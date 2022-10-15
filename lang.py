import copy
from enum import Enum, auto
import os
from pathlib import Path
import re
import subprocess
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
        if isinstance(other, list):
            return self.items == other
        if not isinstance(other, Vector):
            raise Exception(f'{other} is not a Vector')
        return self.items == other.items

    def __str__(self):
        return str(self.items)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

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
    DOUBLE = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    IF = auto()


DOUBLE_RE = re.compile('-?\d+\.?\d*')


def _get_token(token_buffer):
    if token_buffer.isdigit() or (token_buffer.startswith('-') and token_buffer[1:].isdigit()):
        return {'type': TokenType.NUMBER, 'lexeme': token_buffer}
    elif DOUBLE_RE.match(token_buffer):
        return {'type': TokenType.DOUBLE, 'lexeme': token_buffer}
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
    inside_comment = False
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
        elif inside_comment:
            if c == '\n':
                inside_comment = False
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
        elif c == ';':
            inside_comment = True
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
    elif token['type'] == TokenType.DOUBLE:
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
        p = evaluate(param, env=env)
        if p != first_param:
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
    global_env[name] = var
    return var


def if_form(params, env):
    test_val = evaluate(params[0], env=env)

    if test_val:
        return evaluate(params[1], env=env)
    else:
        if len(params) > 2:
            false_val = evaluate(params[2], env=env)
        else:
            false_val = None
        return false_val


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
    p = evaluate(params[0], env=env)
    return p.split()


def str_trim(params, env):
    p = evaluate(params[0], env=env)
    return p.strip()


def str_subs(params, env):
    s = evaluate(params[0], env=env)
    start = evaluate(params[1], env=env)

    if len(params) > 2:
        end = evaluate(params[2], env=env)
        return s[start:end]
    else:
        return s[start:]


def conj(params, env):
    coll = evaluate(params[0], env=env)
    new_element = evaluate(params[1], env=env)
    coll.append(new_element)
    return coll


def subvec(params, env):
    l = evaluate(params[0], env=env).items
    start = evaluate(params[1], env=env)

    if len(params) > 2:
        end = evaluate(params[2], env=env)
        return Vector(l[start:end])
    else:
        return Vector(l[start:])


def nth(params, env):
    collection = evaluate(params[0], env=env)
    index = evaluate(params[1], env=env)
    return collection[index]


def count(params, env):
    p = evaluate(params[0], env=env)
    if p is None:
        return 0
    return len(p)


def sort(params, env):
    if len(params) == 1:
        collection = evaluate(params[0], env=env)
        return sorted(collection.items)
    else:
        f = evaluate(params[0], env=env)
        collection = evaluate(params[1], env=env)
        if f.__name__ == 'greater':
            return sorted(collection, reverse=True)


def sort_by(params, env):
    f = lambda l: l[1] #TODO - actually implement this...
    if len(params) == 2:
        collection = evaluate(params[1], env=env)
        if isinstance(collection, Vector):
            return sorted(collection.items, key=f)
        else:
            return sorted(collection, key=f)
    else:
        comparator = evaluate(params[1], env=env)
        collection = evaluate(params[2], env=env)
        if comparator.__name__ == 'greater':
            return sorted(collection, key=f, reverse=True)


def map_get(params, env):
    d = evaluate(params[0], env=env)
    key = evaluate(params[1], env=env)
    if key in d:
        return d[key]
    else:
        if len(params) > 2:
            default = evaluate(params[2], env=env)
            return default
    raise KeyError(key)


def map_keys(params, env):
    d = evaluate(params[0], env=env)
    return list(d.keys())


def map_vals(params, env):
    d = evaluate(params[0], env=env)
    return list(d.values())


def map_pairs(params, env):
    d = evaluate(params[0], env=env)
    return [list(i) for i in d.items()]


def map_contains(params, env):
    d = evaluate(params[0], env=env)
    key = evaluate(params[1], env=env)
    return key in d


def map_assoc(params, env):
    d = evaluate(params[0], env=env)
    key = evaluate(params[1], env=env)
    d[key] = evaluate(params[2], env=env)
    return d


def map_dissoc(params, env):
    d = evaluate(params[0], env=env)
    key = evaluate(params[1], env=env)
    d.pop(key, None)
    return d


def print_func(params, env):
    print(evaluate(params[0], env=env), end='')


def println(params, env):
    print(evaluate(params[0], env=env))


def read_line(params, env):
    try:
        return input()
    except EOFError:
        return None


class Function:

    def __init__(self, params, body):
        self.params = params
        self.body = body

    def __call__(self, args, env=None):
        env = env or {}
        local_env = copy.deepcopy(env)
        bindings = zip(self.params.items, args)
        for binding in bindings:
            local_env[binding[0].name] = evaluate(binding[1], env=local_env)
        return evaluate(self.body, env=local_env)

    def __str__(self):
        return f'<Function params={self.params}; body={self.body}'

    def __repr__(self):
        return str(self)


def create_function(params, env):
    return Function(params=params[0], body=params[1])


def defn(params, env):
    name = params[0].name
    var = Var(name=name)
    var.value = Function(params=params[1], body=params[2])
    global_env[name] = var
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


global_env = {
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
    'nth': nth,
    'count': count,
    'sort': sort,
    'sort-by': sort_by,
    'get': map_get,
    'keys': map_keys,
    'vals': map_vals,
    'pairs': map_pairs,
    'contains?': map_contains,
    'assoc': map_assoc,
    'dissoc': map_dissoc,
    'str/split': str_split,
    'str/trim': str_trim,
    'print': print_func,
    'println': println,
    'read-line': read_line,
    'file/open': file_open,
    'file/read': file_read,
    'file/close': file_close,
}


def evaluate(node, env=None):
    env = env or {}
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
            if first.name in env or first.name in global_env:
                if first.name in env:
                    value = env[first.name]
                else:
                    value = global_env[first.name]
                if isinstance(value, Var) and isinstance(value.value, Function):
                    f = value.value
                    return f(rest, env=env)
                if callable(value):
                    return value(rest, env=env)
                else:
                    raise Exception(f'symbol first in list and not callable: {first.name} -- {value}')
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
            raise Exception(f'first element of list not callable: {node}')
    if isinstance(node, Symbol):
        if node.name in env:
            symbol = env[node.name]
            return getattr(symbol, 'value', symbol)
        elif node.name in global_env:
            symbol = global_env[node.name]
            return getattr(symbol, 'value', symbol)
        else:
            raise Exception(f'unhandled symbol: {node}')
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


def add_c(params, env):
    c_params = [compile_form(p, env=env)['code'] for p in params]
    return {'code': f'add({c_params[0]}, {c_params[1]})'}


def subtract_c(params, env):
    c_params = [compile_form(p, env=env)['code'] for p in params]
    return {'code': f'subtract({c_params[0]}, {c_params[1]})'}


def multiply_c(params, env):
    c_params = [compile_form(p, env=env)['code'] for p in params]
    return {'code': f'multiply({c_params[0]}, {c_params[1]})'}


def divide_c(params, env):
    c_params = [compile_form(p, env=env)['code'] for p in params]
    return {'code': f'divide({c_params[0]}, {c_params[1]})'}


def equal_c(params, env):
    c_params = [compile_form(p, env=env)['code'] for p in params]
    return {'code': f'equal({c_params[0]}, {c_params[1]})'}


def greater_c(params, env):
    c_params = [compile_form(p, env=env)['code'] for p in params]
    return {'code': f'greater({c_params[0]}, {c_params[1]})'}


def greater_equal_c(params, env):
    c_params = [compile_form(p, env=env)['code'] for p in params]
    return {'code': f'greater_equal({c_params[0]}, {c_params[1]})'}


def less_c(params, env):
    c_params = [compile_form(p, env=env)['code'] for p in params]
    return {'code': f'less({c_params[0]}, {c_params[1]})'}


def less_equal_c(params, env):
    c_params = [compile_form(p, env=env)['code'] for p in params]
    return {'code': f'less_equal({c_params[0]}, {c_params[1]})'}


def def_c(params, env):
    name = params[0].name
    c_name = _get_generated_name(base=f'u_{name}', env=env)
    value = compile_form(params[1], env=env)['code']
    code = f'Value {c_name} = {value};'
    env['user_globals'][name] = {'c_name': c_name, 'code': code}
    return {'code': ''}


def if_form_c(params, env):
    f_name = _get_generated_name(base='if_form', env=env)

    test_code = compile_form(params[0], env=env)['code']
    true_result = compile_form(params[1], env=env)
    if isinstance(true_result, tuple) and isinstance(true_result[0], Symbol) and true_result[0].name == 'recur':
        # return a recur struct with the new params
        recur_name = _get_generated_name('recur', env=env)
        # initialize ObjRecur - don't free here because we need to return it & free it in calling function
        if 'local' in env:
            env['local']['temps'][recur_name] = f'ObjRecur {recur_name};'
            env['local']['pre'].append(f'recur_init(&{recur_name});')
        else:
            env['temps'][recur_name] = f'ObjRecur {recur_name};'
            env['main_pre'].append(f'recur_init(&{recur_name});')
        true_code = ''
        for r in true_result[1:]:
            true_code += f'\nrecur_add(&{recur_name}, {r["code"]});'
    else:
        true_code = true_result['code']

    false_code = ''
    if len(params) > 2:
        false_code += '  else {'
        false_result = compile_form(params[2], env=env)
        if isinstance(false_result, tuple) and isinstance(false_result[0], Symbol) and false_result[0].name == 'recur':
            # return a recur struct with the new params
            recur_name = _get_generated_name('recur', env=env)
            # initialize ObjRecur - don't free here because we need to return it & free it in calling function
            if 'local' in env:
                env['local']['temps'][recur_name] = f'ObjRecur {recur_name};'
                env['local']['pre'].append(f'recur_init(&{recur_name});')
            else:
                env['temps'][recur_name] = f'ObjRecur {recur_name};'
                env['main_pre'].append(f'recur_init(&{recur_name});')
            for r in false_result[1:]:
                false_code += f'\n    recur_add(&{recur_name}, {r["code"]});'
            false_code += f'\n  return OBJ_VAL(&{recur_name});'
            false_code += '\n  }'
        else:
            false_code += '\n    return %s;\n  }' % false_result['code']
    else:
        false_code += '\n  else {\n    return NIL_VAL;\n  }'

    f_params = 'void'
    f_args = ''
    f_code = ''

    if 'local' in env:
        keys = list(env['local']['bindings'].keys())
        f_params = f'Value {keys[0]}'
        f_args = keys[0]
        for key in keys[1:]:
            f_params += f', Value {key}'
            f_args += f', {key}'
        f_code += '\n'.join(env['local']['temps'].values())
        f_code += '\n' + '\n'.join(env['local']['pre'])

    f_code += '\n  if (AS_BOOL(%s)) {\n' % test_code
    f_code += '    return %s;\n  }' % true_code
    f_code += false_code

    env['functions'][f_name] = 'Value %s(%s) {\n%s\n}' % (f_name, f_params, f_code)

    return {'code': f'{f_name}({f_args})'}


def let_c(params, env):
    bindings = params[0]
    body = params[1:]

    paired_bindings = []
    for i in range(0, len(bindings.items), 2):
        paired_bindings.append(bindings.items[i:i+2])

    f_name = _get_generated_name(base='let', env=env)
    f_code = ''

    env['local'] = {'temps': {}, 'pre': [], 'post': [], 'bindings': {}}
    for binding in paired_bindings:
        result = compile_form(binding[1], env=env)
        env['local']['bindings'][binding[0].name] = result
        f_code += f'Value {binding[0].name} = {result["code"]};\n'

    result = compile_form(*body, env=env)

    f_code += f'  return {result["code"]};'

    env['functions'][f_name] = 'Value %s(void) {\n  %s\n}' % (f_name, f_code)

    del env['local']

    return {'code': f'{f_name}()'}


def loop_c(params, env):
    bindings = params[0]
    body = params[1:]

    loop_params = bindings.items[::2]
    initial_args = bindings.items[1::2]

    f_name = _get_generated_name(base='loop', env=env)
    c_loop_params = ', '.join([f'Value {p.name}' for p in loop_params])
    env['local'] = {'temps': {}, 'pre': [], 'post': [], 'bindings': {}}

    for index, loop_param in enumerate(loop_params):
        env['local']['bindings'][loop_param.name] = compile_form(initial_args[index], env=env)['code']

    f_code = '  bool continueFlag = false;'
    f_code += '\n  do {\n'
    for form in body:
        compiled = compile_form(form, env=env)
        f_code += f'\n  Value result = {compiled["code"]};'
        f_code +=  '\n  if (IS_RECUR(result)) {'
        f_code += f'\n    /* grab values from result and update  */'
        for index, loop_param in enumerate(list(env['local']['bindings'].keys())):
            f_code += f'\n      {loop_param} = recur_get(result, NUMBER_VAL({index}));'
        f_code += f'\n    continueFlag = true;'
        f_code += f'\n    recur_free(AS_RECUR(result));'
        f_code +=  '\n  }\n  else {'
        f_code +=  '\n    return result;\n  }'
    f_code += '\n  } while (continueFlag);'
    f_code += '\n  return NIL_VAL;'

    env['functions'][f_name] = 'Value %s(%s) {\n%s\n}' % (f_name, c_loop_params, f_code)

    c_initial_args = ','.join([compile_form(arg, env=env)['code'] for arg in initial_args])

    del env['local']

    return {'code': f'{f_name}({c_initial_args})'}


def str_c(params, env):
    if not params:
        return {'code': ''}
    if len(params) == 1:
        return {
            'code': '"%s"' % str(compile_form(params[0], env=env)['code'])
        }
    else:
        return {
            'code': 'strcat(%s, %s)' % (compile_form(params[0], env=env)['code'], compile_form(params[1], env=env)['code'])
        }


def nth_c(params, env):
    lst = compile_form(params[0], env=env)
    index = compile_form(params[1], env=env)['code']
    return {'code': f'list_get({lst["code"]}, {index})'}


def count_c(params, env):
    lst = compile_form(params[0], env=env)
    return {'code': f'list_count({lst["code"]})'}


def map_get_c(params, env):
    m = compile_form(params[0], env=env)['code']
    key = compile_form(params[1], env=env)['code']
    if len(params) > 2:
        default = compile_form(params[2], env=env)['code']
        return {'code': f'map_get({m}, {key}, {default})'}
    return {'code': f'map_get({m}, {key}, NIL_VAL)'}


def map_assoc_c(params, env):
    m = compile_form(params[0], env=env)['code']
    key = compile_form(params[1], env=env)['code']
    value = compile_form(params[2], env=env)['code']
    return {'code': f'map_set(AS_MAP({m}), {key}, {value})'}


def print_c(params, env):
    result = compile_form(params[0], env=env)
    param = result['code']
    c_code = f'print({param})'
    return {'code': c_code}


def println_c(params, env):
    result = compile_form(params[0], env=env)
    param = result['code']
    c_code = f'println({param})'
    return {'code': c_code}


global_compile_env = {
    '+': add_c,
    '-': subtract_c,
    '*': multiply_c,
    '/': divide_c,
    '=': equal_c,
    '>': greater_c,
    '>=': greater_equal_c,
    '<': less_c,
    '<=': less_equal_c,
    'print': print_c,
    'println': println_c,
    'count': count_c,
    'nth': nth_c,
    'get': map_get_c,
    'assoc': map_assoc_c,
    'def': def_c,
    'let': let_c,
    'loop': loop_c,
    'str': str_c,
}


def _get_generated_name(base, env):
    if base not in env['functions'] and base not in env['temps'] and base not in env['user_globals'] and base not in env.get('local', {}).get('temps', {}):
        return base
    i = 1
    while True:
        name = f'{base}_{i}'
        if name not in env['functions'] and name not in env['temps'] and base not in env['user_globals'] and name not in env.get('local', {}).get('temps', {}):
            return name
        i += 1


def new_string_c(s, env):
    name = _get_generated_name('str', env=env)
    c_code = f'string_init(&{name});'
    c_code += f'\nstring_set(&{name}, "{s}", (size_t) {len(s)});'

    if 'local' in env:
        env['local']['temps'][name] = f'ObjString {name};'
        env['local']['pre'].append(f'{c_code}\n')
        env['local']['post'].append(f'string_free(&{name});')
    else:
        env['temps'][name] = f'ObjString {name};'
        env['main_pre'].append(f'{c_code}\n')
        env['main_post'].append(f'string_free(&{name});')
    return name


def new_vector_c(v, env):
    name = _get_generated_name('lst', env=env)
    env['temps'][name] = 'ObjList %s;' % name
    c_code = 'list_init(&%s);' % name
    c_items = [compile_form(item, env=env)['code'] for item in v.items]
    for c_item in c_items:
        c_code += f'\nlist_add(&{name}, {c_item});'

    env['main_pre'].append(f'{c_code}\n')
    env['main_post'].append(f'list_free(&{name});')
    return name


def new_map_c(node, env):
    name = _get_generated_name('map', env=env)
    env['temps'][name] = f'ObjMap {name};'
    c_code = f'map_init(&{name});'
    keys = [compile_form(k, env=env)['code'] for k in node.items[::2]]
    values = [compile_form(v, env=env)['code'] for v in node.items[1::2]]
    c_items = zip(keys, values)

    for key, value in c_items:
        c_code += f'\nmap_set(&{name}, {key}, {value});'

    env['main_pre'].append(f'{c_code}\n')
    env['main_post'].append(f'map_free(&{name});')
    return name


def compile_form(node, env):
    if isinstance(node, list):
        first = node[0]
        rest = node[1:]
        if isinstance(first, Symbol):
            if first.name in env['global']:
                if isinstance(env['global'][first.name], Var) and isinstance(env['global'][first.name].value, Function):
                    f = env['global'][first.name].value
                    return f(rest)
                if callable(env['global'][first.name]):
                    return env['global'][first.name](rest, env=env)
                else:
                    raise Exception(f'symbol first in list and not callable: {first.name} -- {env[first.name]}')
            elif first.name == 'do':
                env['local'] = {'temps': {}, 'pre': [], 'post': [], 'bindings': {}}
                do_exprs = [compile_form(n, env=env) for n in rest]

                f_name = _get_generated_name('do_f', env)

                f_code = ''
                f_code += '\n'.join([f for f in env['local']['temps'].values()])
                f_code += '\n' + '\n'.join(env['local']['pre'])
                for d in do_exprs[:-1]:
                    f_code += f'  {d["code"]};\n'
                f_code += f'  Value result = {do_exprs[-1]["code"]};'
                f_code += '\n' + '\n'.join(env['local']['post'])
                f_code += '\n  return result;'

                env['functions'][f_name] = 'Value %s(void) {\n%s\n}' % (f_name, f_code)
                del env['local']
                return {'code': f'{f_name}()'}
            elif first.name == 'recur':
                params = [compile_form(r, env) for r in rest]
                return (first, *params)
            else:
                raise Exception(f'unhandled symbol: {first}')
        elif first == TokenType.IF:
            return if_form_c(rest, env=env)
        else:
            raise Exception(f'unhandled list: {node}')
    if isinstance(node, Symbol):
        if node.name in env['user_globals']:
            return {'code': env['user_globals'][node.name]['c_name']}
        elif node.name in env['local']['bindings']:
            return {'code': node.name}
        else:
            raise Exception(f'unhandled symbol: {node}')
    if isinstance(node, Vector):
        name = new_vector_c(node, env=env)
        return {'code': f'OBJ_VAL(&{name})'}
    if isinstance(node, str):
        name = new_string_c(node, env=env)
        return {'code': f'OBJ_VAL(&{name})'}
    if isinstance(node, bool):
        if node:
            val = 'true'
        else:
            val = 'false'
        return {'code': f'BOOL_VAL({val})'}
    if isinstance(node, int):
        return {'code': f'NUMBER_VAL({node})'}
    if isinstance(node, float):
        return {'code': f'NUMBER_VAL({node})'}
    if node is None:
        return {'code': 'NIL_VAL'}
    if isinstance(node, DictBuilder):
        name = new_map_c(node, env=env)
        return {'code': f'OBJ_VAL(&{name})'}
    raise Exception(f'unhandled node: {type(node)} -- {node}')


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


c_includes = [
    '<stdio.h>',
    '<stdint.h>',
    '<stdlib.h>',
    '<stdbool.h>',
    '<string.h>',
]


c_functions = {
    'add': 'Value add(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) + AS_NUMBER(y)); }',
    'subtract': 'Value subtract(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) - AS_NUMBER(y)); }',
    'multiply': 'Value multiply(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) * AS_NUMBER(y)); }',
    'divide': 'Value divide(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) / AS_NUMBER(y)); }',
    'greater': 'Value greater(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) > AS_NUMBER(y)); }',
    'greater_equal': 'Value greater_equal(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) >= AS_NUMBER(y)); }',
    'less': 'Value less(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) < AS_NUMBER(y)); }',
    'less_equal': 'Value less_equal(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) <= AS_NUMBER(y)); }',
}


c_types = '''
#define ALLOCATE(type, count) \
    (type*)reallocate(NULL, sizeof(type) * (count))

#define GROW_CAPACITY(capacity) \
            ((capacity) < 8 ? 8 : (capacity) * 2)

#define GROW_ARRAY(type, pointer, oldCount, newCount) \
            (type*)reallocate(pointer, sizeof(type) * (newCount))

#define FREE_ARRAY(type, pointer) \
            reallocate(pointer, (size_t)0)

#define NIL_VAL  ((Value){NIL, {.number = 0}})
#define BOOL_VAL(value)  ((Value){BOOL, {.boolean = value}})
#define NUMBER_VAL(value)  ((Value){NUMBER, {.number = value}})
#define OBJ_VAL(object)   ((Value){OBJ, {.obj = (Obj*)object}})
#define AS_BOOL(value)  ((value).data.boolean)
#define AS_NUMBER(value)  ((value).data.number)
#define AS_OBJ(value)  ((value).data.obj)
#define AS_RECUR(value)       ((ObjRecur*)AS_OBJ(value))
#define AS_STRING(value)       ((ObjString*)AS_OBJ(value))
#define AS_CSTRING(value)      (((ObjString*)AS_OBJ(value))->chars)
#define AS_LIST(value)       ((ObjList*)AS_OBJ(value))
#define AS_MAP(value)       ((ObjMap*)AS_OBJ(value))
#define IS_NIL(value)  ((value).type == NIL)
#define IS_BOOL(value)  ((value).type == BOOL)
#define IS_NUMBER(value)  ((value).type == NUMBER)
#define IS_OBJ(value)  ((value).type == OBJ)
#define IS_RECUR(value)  isObjType(value, OBJ_RECUR)
#define IS_STRING(value)  isObjType(value, OBJ_STRING)
#define IS_LIST(value)  isObjType(value, OBJ_LIST)
#define IS_MAP(value)  isObjType(value, OBJ_MAP)
#define MAP_MAX_LOAD 0.75

void* reallocate(void* pointer, size_t newSize) {
  if (newSize == 0) {
    free(pointer);
    return NULL;
  }

  void* result = realloc(pointer, newSize);
  return result;
}

typedef enum {
  OBJ_RECUR,
  OBJ_STRING,
  OBJ_LIST,
  OBJ_MAP,
} ObjType;

typedef struct {
  ObjType type;
} Obj;

typedef enum {
  NIL,
  BOOL,
  NUMBER,
  OBJ,
} ValueType;

typedef struct {
  ValueType type;
  union {
    bool boolean;
    double number;
    char* string;
    Obj* obj;
  } data;
} Value;

static inline bool isObjType(Value value, ObjType type) {
  return IS_OBJ(value) && AS_OBJ(value)->type == type;
}

typedef struct {
  Obj obj;
  size_t length;
  uint32_t hash;
  char* chars;
} ObjString;

typedef struct {
  Obj obj;
  size_t count;
  size_t capacity;
  Value* values;
} ObjRecur;

typedef struct {
  Obj obj;
  size_t count;
  size_t capacity;
  Value* values;
} ObjList;

typedef struct {
  Value key;
  Value value;
} MapEntry;

typedef struct {
  Obj obj;
  size_t count;
  size_t capacity;
  MapEntry* entries;
} ObjMap;

void string_init(ObjString* string) {
  string->obj = (Obj){.type = OBJ_STRING};
  string->length = 0;
  string->hash = 0;
  string->chars = NULL;
}

static uint32_t hashString(const char* key, size_t length) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < length; i++) {
    hash ^= (uint8_t)key[i];
    hash *= 16777619;
  }
  return hash;
}

void string_set(ObjString* string, const char* chars, size_t length) {
  char* heapChars = ALLOCATE(char, length + 1);
  memcpy(heapChars, chars, length);
  heapChars[length] = 0; /* terminate it w/ NULL, so we can pass c-string to functions that need it */
  string->length = length;
  string->hash = hashString(chars, length);
  string->chars = heapChars;
}

void string_free(ObjString* string) {
  FREE_ARRAY(char, string->chars);
  string_init(string);
}

void list_init(ObjList* list) {
  list->obj = (Obj){.type = OBJ_LIST};
  list->count = 0;
  list->capacity = 0;
  list->values = NULL;
}

void list_free(ObjList* list) {
  FREE_ARRAY(Value, list->values);
  list_init(list);
}

void list_add(ObjList* list, Value item) {
  if (list->capacity < list->count + 1) {
    size_t oldCapacity = list->capacity;
    list->capacity = GROW_CAPACITY(oldCapacity);
    list->values = GROW_ARRAY(Value, list->values, oldCapacity, list->capacity);
  }

  list->values[list->count] = item;
  list->count++;
}

Value list_get(Value list, Value index) {
  /* size_t is the unsigned integer type returned by the sizeof operator */
  size_t num_index = (size_t) AS_NUMBER(index);
  if (num_index < AS_LIST(list)->count) {
    return AS_LIST(list)->values[num_index];
  }
  else {
    return NIL_VAL;
  }
}

Value list_count(Value list) {
  return NUMBER_VAL((int) AS_LIST(list)->count);
}

void recur_init(ObjRecur* recur) {
  recur->obj = (Obj){.type = OBJ_RECUR};
  recur->count = 0;
  recur->capacity = 0;
  recur->values = NULL;
}

void recur_free(ObjRecur* recur) {
  FREE_ARRAY(Value, recur->values);
  recur_init(recur);
}

void recur_add(ObjRecur* recur, Value item) {
  if (recur->capacity < recur->count + 1) {
    size_t oldCapacity = recur->capacity;
    recur->capacity = GROW_CAPACITY(oldCapacity);
    recur->values = GROW_ARRAY(Value, recur->values, oldCapacity, recur->capacity);
  }

  recur->values[recur->count] = item;
  recur->count++;
}

Value recur_get(Value recur, Value index) {
  /* size_t is the unsigned integer type returned by the sizeof operator */
  size_t num_index = (size_t) AS_NUMBER(index);
  if (num_index < AS_RECUR(recur)->count) {
    return AS_RECUR(recur)->values[num_index];
  }
  else {
    return NIL_VAL;
  }
}

Value recur_count(Value recur) {
  return NUMBER_VAL((int) AS_RECUR(recur)->count);
}

void map_init(ObjMap* map) {
  map->obj = (Obj){.type = OBJ_MAP};
  map->count = 0;
  map->capacity = 0;
  map->entries = NULL;
}

void map_free(ObjMap* map) {
  FREE_ARRAY(MapEntry, map->entries);
  map_init(map);
}

Value map_count(Value map) {
  return NUMBER_VAL((int) AS_MAP(map)->count);
}

Value equal(Value x, Value y) {
  if (x.type != y.type) {
    return BOOL_VAL(false);
  }
  else if (IS_NIL(x)) {
    return BOOL_VAL(true);
  }
  else if (IS_BOOL(x)) {
    return BOOL_VAL(AS_BOOL(x) == AS_BOOL(y));
  }
  else if (IS_NUMBER(x)) {
    return BOOL_VAL(AS_NUMBER(x) == AS_NUMBER(y));
  }
  else if (IS_STRING(x)) {
    ObjString* xString = AS_STRING(x);
    ObjString* yString = AS_STRING(y);
    if ((xString->length == yString->length) &&
        (memcmp(xString->chars, yString->chars, xString->length) == 0)) {
      return BOOL_VAL(true);
    }
    return BOOL_VAL(false);
  }
  else if (IS_LIST(x)) {
    ObjList* xList = AS_LIST(x);
    ObjList* yList = AS_LIST(y);
    if (xList->count == yList->count) {
      Value num_items = list_count(x);
      for (int i = 0; i < AS_NUMBER(num_items); i++) {
        Value xItem = list_get(x, NUMBER_VAL(i));
        Value yItem = list_get(y, NUMBER_VAL(i));
        if (!AS_BOOL(equal(xItem, yItem))) {
          return BOOL_VAL(false);
        }
      }
      return BOOL_VAL(true);
    }
    return BOOL_VAL(false);
  }
  else if (IS_MAP(x)) {
    ObjMap* xMap = AS_MAP(x);
    ObjMap* yMap = AS_MAP(y);
    size_t x_num_items = xMap->count;
    size_t y_num_items = yMap->count;
    if (x_num_items != y_num_items) {
      return BOOL_VAL(false);
    }
    size_t x_num_entries = xMap->capacity;
    for (size_t i = 0; i < x_num_entries; i++) {
      MapEntry x_entry = xMap->entries[i];
      MapEntry y_entry = yMap->entries[i];
      if (!AS_BOOL(equal(x_entry.key, y_entry.key))) {
        return BOOL_VAL(false);
      }
      if (!AS_BOOL(equal(x_entry.value, y_entry.value))) {
        return BOOL_VAL(false);
      }
    }
    return BOOL_VAL(true);
  }
  else {
    return BOOL_VAL(false);
  }
}

static MapEntry* findEntry(MapEntry* entries, size_t capacity, Value key) {
  ObjString* keyString = AS_STRING(key);
  uint32_t index = keyString->hash % (uint32_t) capacity;
  for (;;) {
    MapEntry* entry = &entries[index];
    if (AS_BOOL(equal(entry->key, key)) || AS_BOOL(equal(entry->key, NIL_VAL))) {
      return entry;
    }

    index = (index + 1) % (uint32_t)capacity;
  }
}

static void adjustCapacity(ObjMap* map, size_t capacity) {
  MapEntry* entries = ALLOCATE(MapEntry, capacity);
  for (size_t i = 0; i < capacity; i++) {
    entries[i].key = NIL_VAL;
    entries[i].value = NIL_VAL;
  }

  map->entries = entries;
  map->capacity = capacity;
}

Value map_set(ObjMap* map, Value key, Value value) {
  if ((double)map->count + 1 > (double)map->capacity * MAP_MAX_LOAD) {
    size_t capacity = GROW_CAPACITY(map->capacity);
    adjustCapacity(map, capacity);
  }

  MapEntry* entry = findEntry(map->entries, map->capacity, key);
  bool isNewKey = AS_BOOL(equal(entry->key, NIL_VAL));
  if (isNewKey) map->count++;

  entry->key = key;
  entry->value = value;
  return OBJ_VAL(map);
}

Value map_get(Value map, Value key, Value defaultVal) {
  ObjMap* objMap = AS_MAP(map);
  MapEntry* entry = findEntry(objMap->entries, objMap->capacity, key);
  bool isNewKey = AS_BOOL(equal(entry->key, NIL_VAL));
  if (isNewKey) {
    return defaultVal;
  }
  else {
    return entry->value;
  }
}

Value print(Value value) {
  if IS_NIL(value) {
    printf("nil");
  }
  else if IS_BOOL(value) {
    if AS_BOOL(value) {
      printf("true");
    }
    else {
      printf("false");
    }
  }
  else if IS_NUMBER(value) {
    printf("%g", AS_NUMBER(value));
  }
  else if (IS_LIST(value)) {
    Value num_items = list_count(value);
    printf("[");
    print(list_get(value, NUMBER_VAL(0)));
    for (int i = 1; i < AS_NUMBER(num_items); i++) {
      printf(" ");
      print(list_get(value, NUMBER_VAL(i)));
    }
    printf("]");
  }
  else if (IS_MAP(value)) {
    size_t num_entries = AS_MAP(value)->capacity;
    printf("{");
    bool first_entry = true;
    for (size_t i = 0; i < num_entries; i++) {
      if (!AS_BOOL(equal(AS_MAP(value)->entries[i].key, NIL_VAL))) {
        if (!first_entry) {
          printf(", ");
        }
        print(AS_MAP(value)->entries[i].key);
        printf(" ");
        print(AS_MAP(value)->entries[i].value);
        first_entry = false;
      }
    }
    printf("}");
  }
  else {
    printf("%s", AS_CSTRING(value));
  }
  return NIL_VAL;
}

Value println(Value value) {
  print(value);
  printf("\\n");
  return NIL_VAL;
}
    '''


def _compile(source):
    tokens = scan_tokens(source)
    ast = parse(tokens)

    compiled_forms = []
    env = {
        'global': copy.deepcopy(global_compile_env),
        'functions': {},
        'user_globals': {},
        'temps': {},
        'main_pre': [],
        'main_post': [],
    }
    for f in ast.forms:
        result = compile_form(f, env=env)
        c = result['code']
        if not c.endswith(';'):
            c = f'{c};'
        compiled_forms.append(c)

    c_code = '\n'.join([f'#include {i}' for i in c_includes])
    c_code += '\n\n'
    c_code += c_types
    c_code += '\n'.join([f for f in c_functions.values()])
    c_code += '\n\n' + '\n\n'.join([f for f in env['functions'].values()])
    c_code += '\n\nint main(void)\n{'
    c_code += '\n' + '\n'.join([g['code'] for g in env['user_globals'].values()])
    c_code += '\n' + '\n'.join([f for f in env['temps'].values()])
    c_code += '\n' + '\n'.join(env['main_pre'])
    c_code += '\n' + '\n'.join(compiled_forms)
    c_code += '\n' + '\n'.join(env['main_post'])
    c_code += '\nreturn 0;\n}'

    return c_code


def compile_to_c(file_name, run=False):
    with open(file_name, 'rb') as f:
        source = f.read().decode('utf8')

    c_program = _compile(source)

    if run:
        with tempfile.TemporaryDirectory() as tmp:
            c_filename = os.path.join(tmp, 'code.c')
            with open(c_filename, 'wb') as f:
                f.write(c_program.encode('utf8'))
            compile_cmd = ['gcc', '-o', str(file_name.stem), c_filename]
            subprocess.run(compile_cmd, check=True)
            subprocess.run([f'./{str(file_name.stem)}'], check=True)
    else:
        tmp = tempfile.mkdtemp(dir='.', prefix='tmp')
        c_file = Path(tmp) / Path(f'{file_name.stem}.c')

        with open(c_file, mode='wb') as f:
            f.write(c_program.encode('utf8'))

        executable = Path(tmp) / file_name.stem
        print(f'Compile with: gcc -o {executable} {c_file}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Random language')
    parser.add_argument('-c', action='store_true', dest='compile', help='compile the file to C')
    parser.add_argument('-r', action='store_true', dest='run', help='compile to C & run')
    parser.add_argument('file', type=str, nargs='?', help='file to interpret')

    args = parser.parse_args()

    if args.compile:
        if args.file:
            compile_to_c(Path(args.file))
        else:
            print('no file to compile')
    elif args.run:
        if args.file:
            compile_to_c(Path(args.file), run=True)
        else:
            print('no file to compile')
    else:
        main(args.file)
