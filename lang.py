import copy
from enum import Enum, auto
import os
from pathlib import Path
import re
import subprocess
import sys
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


####### COMPILE TO C ##############

def nil_c(params, envs):
    param = compile_form(params[0], envs=envs)
    return {'code': f'nil_Q_({param["code"]})'}


def add_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return {'code': f'add({c_params[0]}, {c_params[1]})'}

    # TODO - probably need to use this style for all functions?
    #   even functions that could only return nil/true/false could still take time to run,
    #   and I don't want them to be run multiple times for conditional IS_OBJ ref-counting checks?
    # name = _get_generated_name('add_result', envs=envs)
    # envs[-1]['pre'].append(f'  Value {name} = add({c_params[0]}, {c_params[1]});')
    # envs[-1]['temps'].add(name)
    # return {'code': name}


def subtract_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return {'code': f'subtract({c_params[0]}, {c_params[1]})'}


def multiply_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return {'code': f'multiply({c_params[0]}, {c_params[1]})'}


def divide_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return {'code': f'divide({c_params[0]}, {c_params[1]})'}


def equal_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return {'code': f'equal({c_params[0]}, {c_params[1]})'}


def greater_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return {'code': f'greater(user_globals, {c_params[0]}, {c_params[1]})'}


def greater_equal_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return {'code': f'greater_equal({c_params[0]}, {c_params[1]})'}


def less_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return {'code': f'less(user_globals, {c_params[0]}, {c_params[1]})'}


def less_equal_c(params, envs):
    c_params = [compile_form(p, envs=envs)['code'] for p in params]
    return {'code': f'less_equal({c_params[0]}, {c_params[1]})'}


def def_c(params, envs):
    name = params[0].name
    local_env = {'temps': envs[-1]['temps'], 'pre': [], 'post': [], 'bindings': {}}
    envs.append(local_env)
    result = compile_form(params[1], envs=envs)
    c_name = _get_generated_name(base=f'u_{name}', envs=envs)
    envs[0]['user_globals'][name] = {'type': 'var', 'c_name': c_name, 'code': result['code']}
    if local_env['pre']:
        envs[0]['init'].extend(local_env['pre'])
    if local_env['post']:
        envs[0]['post'].extend(local_env['post'])
    envs.pop()
    return {'code': ''}


def _get_previous_bindings(envs):
    bindings = []
    for e in envs[1:]:
        for name, value in e.get('bindings', {}).items():
            if value and 'c_name' in value:
                bindings.append(value['c_name'])
            else:
                bindings.append(name)
    return list(set(bindings))


def if_form_c(params, envs):
    local_env = {'temps': envs[-1]['temps'], 'pre': [], 'post': [], 'bindings': {}}
    envs.append(local_env)

    result_name = _get_generated_name('if_result', envs=envs)
    local_env['temps'].add(result_name)

    test_code = compile_form(params[0], envs=envs)['code']
    true_env = {'temps': local_env['temps'], 'pre': [], 'post': [], 'bindings': {}}
    envs.append(true_env)
    true_result = compile_form(params[1], envs=envs)

    true_code = '\n  if (is_truthy(%s)) {\n  ' % test_code
    if true_env['pre']:
        true_code += '\n  '.join(true_env['pre']) + '\n  '
    if isinstance(true_result, tuple) and isinstance(true_result[0], Symbol) and true_result[0].name == 'recur':
        for e in envs:
            for b in e.get('bindings', {}).keys():
                if b.startswith('recur'):
                    recur_name = b
        for r in true_result[1:]:
            true_code += f'  recur_add(AS_RECUR({recur_name}), {r["code"]});\n'
        true_code += f'    {result_name} = {recur_name};'
    else:
        true_val = true_result['code']
        true_code += f'    {result_name} = {true_val};'
        # inc-ref result_name if needed, so object doesn't get freed
        if true_val not in ['BOOL_VAL(true)', 'BOOL_VAL(false)']:
            true_code += '\n    if (IS_OBJ(%s)) {\n      inc_ref(AS_OBJ(%s));\n    }' % (result_name, result_name)
    if true_env['post']:
        true_code += '\n  ' + '\n  '.join(true_env['post'])
    true_code += '\n  } // end true code\n'
    envs.pop()

    false_code = ''
    if len(params) > 2:
        false_env = {'temps': local_env['temps'], 'pre': [], 'post': [], 'bindings': {}}
        envs.append(false_env)
        false_code += '  else {'
        false_result = compile_form(params[2], envs=envs)
        if false_env['pre']:
            false_code += '\n' + '\n  '.join(false_env['pre'])
        if isinstance(false_result, tuple) and isinstance(false_result[0], Symbol) and false_result[0].name == 'recur':
            for e in envs:
                for b in e.get('bindings', {}).keys():
                    if b.startswith('recur'):
                        recur_name = b
            for r in false_result[1:]:
                false_code += f'\n    recur_add(AS_RECUR({recur_name}), {r["code"]});'
            false_code += f'\n    {result_name} = {recur_name};'
        else:
            false_val = false_result['code']
            false_code += f'\n  {result_name} = {false_val};'
            # inc-ref result_name if needed, so object doesn't get freed early
            if false_val not in ['BOOL_VAL(true)', 'BOOL_VAL(false)']:
                false_code += '\n    if (IS_OBJ(%s)) {\n      inc_ref(AS_OBJ(%s));\n    }' % (result_name, result_name)

        if false_env['post']:
            false_code += '\n' + '\n  '.join(false_env['post'])
        false_code += '\n  } // end false code'
        envs.pop()

    f_code = f'  Value {result_name} = NIL_VAL;'
    if local_env['pre']:
        f_code += '\n' + '\n  '.join(local_env['pre'])
    f_code += true_code
    f_code += false_code

    if local_env['post']:
        envs[-2]['post'].extend(local_env['post'])

    envs.pop()
    envs[-1]['pre'].append(f_code)
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))

    return {'code': result_name}


def let_c(params, envs):
    bindings = params[0]
    body = params[1:]

    paired_bindings = []
    for i in range(0, len(bindings.items), 2):
        paired_bindings.append(bindings.items[i:i+2])

    f_params = 'ObjMap* user_globals'
    f_args = 'user_globals'

    previous_bindings = _get_previous_bindings(envs)
    if previous_bindings:
        for previous_binding in previous_bindings:
            f_params += f', Value {previous_binding}'
            f_args += f', {previous_binding}'

    local_env = {'temps': envs[-1]['temps'], 'pre': [], 'post': [], 'bindings': {}}
    envs.append(local_env)
    for binding in paired_bindings:
        result = compile_form(binding[1], envs=envs)
        binding_name = _get_generated_name(base=binding[0].name, envs=envs)
        result['c_name'] = binding_name
        local_env['bindings'][binding[0].name] = result
        local_env['pre'].append(f'  Value {binding_name} = {result["code"]};\n')

    f_code = ''

    expr_results = [compile_form(form, envs=envs) for form in body]
    final_result = expr_results[-1]

    if local_env['pre']:
        f_code += '\n'.join(local_env['pre']) + '\n'

    return_val = ''
    if isinstance(final_result, tuple) and isinstance(final_result[0], Symbol) and final_result[0].name == 'recur':
        for e in envs:
            for b in e.get('bindings', {}).keys():
                if b.startswith('recur'):
                    recur_name = b
        for r in final_result[1:]:
            f_code += f'\n    recur_add(AS_RECUR({recur_name}), {r["code"]});'
        return_val = recur_name
    else:
        return_val = final_result['code']
        if return_val not in ['NIL_VAL', 'BOOL_VAL(true)', 'BOOL_VAL(false)']:
            f_code += '\n  if (IS_OBJ(%s)) {\n    inc_ref(AS_OBJ(%s));\n  }\n' % (return_val, return_val)

    if local_env['post']:
        f_code += '\n'.join(local_env['post']) + '\n'

    f_code += f'  return {return_val};'

    f_name = _get_generated_name(base='let', envs=envs)
    envs[0]['functions'][f_name] = 'Value %s(%s) {\n  %s\n}' % (f_name, f_params, f_code)

    result_name = _get_generated_name('let_result', envs=envs)
    envs[-1]['temps'].add(result_name)

    envs.pop()

    envs[-1]['pre'].append(f'  Value {result_name} = {f_name}({f_args});')
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))

    return {'code': result_name}


def _loop(envs, bindings, exprs):
    recur_name = _get_generated_name('recur', envs=envs)
    envs[-1]['temps'].add(recur_name)
    envs[-1]['bindings'][recur_name] = None

    loop_result = _get_generated_name('loop_result', envs=envs)
    envs[-1]['temps'].add(loop_result)

    loop_post = []

    loop_code = f'  Recur {recur_name}_1;'
    loop_code += f'\n  recur_init(&{recur_name}_1);'
    loop_code += f'\n  Value {recur_name} = RECUR_VAL(&{recur_name}_1);'

    loop_code += '\n  bool continueFlag = false;'
    loop_code += '\n  do {'
    for form in exprs[:-1]:
        form_env = {'temps': envs[-1]['temps'], 'pre': [], 'post': [], 'bindings': {}}
        envs.append(form_env)
        compiled = compile_form(form, envs=envs)
        if form_env['pre']:
            loop_code += '\n' + '\n  '.join(form_env['pre'])
        if form_env['post']:
            loop_post.extend(form_env['post'])
        envs.pop()

    form_env = {'temps': envs[-1]['temps'], 'pre': [], 'post': [], 'bindings': {}}
    envs.append(form_env)
    compiled = compile_form(exprs[-1], envs=envs)
    if form_env['pre']:
        loop_code += '\n' + '\n  '.join(form_env['pre'])

    loop_code += f'\n    Value {loop_result} = {compiled["code"]};'
    loop_code += '\n    if (IS_OBJ(%s)) {\n      inc_ref(AS_OBJ(%s));\n    }' % (loop_result, loop_result)

    if form_env['post']:
        loop_code += '\n' + '\n  '.join(form_env['post'])

    envs.pop()

    loop_code +=  '\n    if (IS_RECUR(%s)) {' % loop_result
    loop_code += f'\n      /* grab values from result and update  */'

    for index, var in enumerate(bindings):
        loop_code += '\n      if (IS_OBJ(%s)) {\n      dec_ref_and_free(AS_OBJ(%s));\n    }' % (var, var)
        loop_code += f'\n      {var} = recur_get({loop_result}, NUMBER_VAL({index}));'
        loop_code += '\n      if (IS_OBJ(%s)) {\n      inc_ref(AS_OBJ(%s));\n    }' % (var, var)
    if loop_post:
        loop_code += '\n' + '\n'.join(loop_post)

    loop_code += f'\n    continueFlag = true;'
    loop_code += f'\n    recur_free(&{recur_name}_1);'
    loop_code +=  '\n  }\n    else {\n'
    if loop_post:
        loop_code += '\n  '.join(loop_post)

    loop_code += f'\n      recur_free(&{recur_name}_1);'
    loop_code +=  '\n      return %s;\n    }' % loop_result

    loop_code += '\n  } while (continueFlag);'
    loop_code += '\n  return NIL_VAL;'

    return loop_code


def loop_c(params, envs):
    bindings = params[0]
    exprs = params[1:]

    loop_params = bindings.items[::2]
    initial_args = bindings.items[1::2]

    previous_bindings = _get_previous_bindings(envs)

    local_env = {'temps': set(previous_bindings).union(envs[-1]['temps']), 'pre': [], 'post': [], 'bindings': {}}
    envs.append(local_env)

    f_code =  ''

    for index, loop_param in enumerate(loop_params):
        param_env = {'temps': envs[-1]['temps'], 'pre': [], 'post': [], 'bindings': {}}
        envs.append(param_env)
        c_name = _get_generated_name(base=loop_param.name, envs=envs)
        result = compile_form(initial_args[index], envs=envs)
        local_env['bindings'][loop_param.name] = {
            'code': result['code'],
            'c_name': c_name,
        }
        if param_env['pre']:
            f_code += '\n' + '\n'.join(param_env['pre'])
        f_code += f'\n  Value {c_name} = {result["code"]};'
        f_code += '\n  if (IS_OBJ(%s)) {\n    inc_ref(AS_OBJ(%s));\n  }' % (c_name, c_name)
        if param_env['post']:
            f_code += '\n' + '\n  '.join(param_env['post'])
        envs.pop()
        envs[-1]['temps'] = param_env['temps']

    loop_code = _loop(envs, [v['c_name'] for v in list(local_env['bindings'].values())], exprs)

    f_code += '\n' + loop_code

    c_loop_params = [f'Value {pb}' for pb in previous_bindings]
    c_loop_params_str = ', '.join(c_loop_params)
    if c_loop_params_str:
        c_loop_params_str = f'ObjMap* user_globals, {c_loop_params_str}'
    else:
        c_loop_params_str = 'ObjMap* user_globals'

    c_initial_args = ','.join([pb for pb in previous_bindings])
    if c_initial_args:
        c_initial_args = f'user_globals, {c_initial_args}'
    else:
        c_initial_args = 'user_globals'

    f_name = _get_generated_name(base='loop', envs=envs)

    envs[0]['functions'][f_name] = 'Value %s(%s) {\n%s\n}' % (f_name, c_loop_params_str, f_code)

    result_name = _get_generated_name('loop_result', envs=envs)

    envs.pop()

    envs[-1]['pre'].append(f'  Value {result_name} = {f_name}({c_initial_args});')
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))

    return {'code': result_name}


def str_c(params, envs):
    if not params:
        arg_name = 'NIL_VAL'
    elif len(params) == 1:
        result = compile_form(params[0], envs=envs)
        arg_name = result['code']
    else:
        num_params = len(params)
        tmp_list_name = _get_generated_name('str_arg_tmp_list', envs=envs)
        name = _get_generated_name('str', envs=envs)
        envs[-1]['temps'].add(name)
        envs[-1]['pre'].append(f'  Value {tmp_list_name} = OBJ_VAL(allocate_list((size_t) {num_params}));\n  inc_ref(AS_OBJ({tmp_list_name}));')
        for param in params:
            result = compile_form(param, envs=envs)
            envs[-1]['pre'].append(f'  list_add(AS_LIST({tmp_list_name}), {result["code"]});')
        envs[-1]['pre'].append(f'  Value {name} = str_join({tmp_list_name});\n  inc_ref(AS_OBJ({name}));')
        envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({tmp_list_name}));')
        envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
        return {'code': name}

    name = _get_generated_name('str', envs=envs)
    envs[-1]['temps'].add(name)
    envs[-1]['pre'].append(f'  Value {name} = str_str({arg_name});')
    # envs[-1]['pre'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return {'code': name}


def str_lower_c(params, envs):
    result = compile_form(params[0], envs=envs)
    param_name = result['code']
    name = _get_generated_name('str_lower_', envs=envs)
    envs[-1]['pre'].append(f'  Value {name} = str_lower({param_name});')
    envs[-1]['pre'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return {'code': name}


def str_blank_c(params, envs):
    result = compile_form(params[0], envs=envs)
    param_name = result['code']
    name = _get_generated_name('str_blank_', envs=envs)
    envs[-1]['pre'].append(f'  Value {name} = str_blank({param_name});')
    return {'code': name}


def str_split_c(params, envs):
    result = compile_form(params[0], envs=envs)
    param_name = result['code']
    name = _get_generated_name('str_split_', envs=envs)
    envs[-1]['pre'].append(f'  Value {name} = str_split({param_name});')
    envs[-1]['pre'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return {'code': name}


def nth_c(params, envs):
    lst = compile_form(params[0], envs=envs)
    index = compile_form(params[1], envs=envs)['code']
    return {'code': f'list_get({lst["code"]}, {index})'}


def sort_c(params, envs):
    if len(params) == 1:
        lst = compile_form(params[0], envs=envs)
        return {'code': f'list_sort(user_globals, {lst["code"]}, *less)'}
    else:
        compare = compile_form(params[0], envs=envs)
        lst = compile_form(params[1], envs=envs)
        return {'code': f'list_sort(user_globals, {lst["code"]}, {compare["code"]})'}


def count_c(params, envs):
    lst = compile_form(params[0], envs=envs)
    return {'code': f'list_count({lst["code"]})'}


def map_get_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    key = compile_form(params[1], envs=envs)['code']
    default = 'NIL_VAL'
    if len(params) > 2:
        default = compile_form(params[2], envs=envs)['code']
    name = _get_generated_name('map_get_', envs=envs)
    envs[-1]['pre'].append(f'  Value {name} = map_get(AS_MAP({m}), {key}, {default});')
    envs[-1]['pre'].append('  if (IS_OBJ(%s)) {\n    inc_ref(AS_OBJ(%s));\n  }' % (name, name))
    envs[-1]['temps'].add(name)
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (name, name))
    return {'code': name}


def map_contains_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    key = compile_form(params[1], envs=envs)['code']
    return {'code': f'map_contains(AS_MAP({m}), {key})'}


def map_assoc_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    key = compile_form(params[1], envs=envs)['code']
    value = compile_form(params[2], envs=envs)['code']
    result_name = _get_generated_name('map_assoc', envs=envs)
    envs[-1]['pre'].append(f'  Value {result_name} = map_set(AS_MAP({m}), {key}, {value});')
    return {'code': result_name}


def map_keys_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    name = _get_generated_name('map_keys_', envs=envs)
    envs[-1]['pre'].append(f'  Value {name} = map_keys(AS_MAP({m}));')
    envs[-1]['pre'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return {'code': name}


def map_vals_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    name = _get_generated_name('map_vals_', envs=envs)
    envs[-1]['pre'].append(f'  Value {name} = map_vals(AS_MAP({m}));')
    envs[-1]['pre'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return {'code': name}


def map_pairs_c(params, envs):
    m = compile_form(params[0], envs=envs)['code']
    name = _get_generated_name('map_pairs_', envs=envs)
    envs[-1]['pre'].append(f'  Value {name} = map_pairs(AS_MAP({m}));')
    envs[-1]['pre'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return {'code': name}


def print_c(params, envs):
    result = compile_form(params[0], envs=envs)
    param_name = result['code']
    name = _get_generated_name('print_result', envs=envs)
    envs[-1]['temps'].add(name)
    envs[-1]['pre'].append(f'  Value {name} = print({param_name});')
    # print always returns NIL_VAL, so don't need to dec_ref_and_free at the end
    return {'code': name}


def println_c(params, envs):
    result = compile_form(params[0], envs=envs)
    param_name = result['code']
    name = _get_generated_name('println_result', envs=envs)
    envs[-1]['temps'].add(name)
    envs[-1]['pre'].append(f'  Value {name} = println({param_name});')
    return {'code': name}


def fn_c(params, envs, f_name=None):
    bindings = params[0]
    exprs = params[1:]

    local_env = {'temps': envs[-1]['temps'], 'pre': [], 'post': [], 'bindings': {}}
    envs.append(local_env)

    for binding in bindings:
        local_env['bindings'][binding.name] = None

    loop_code = _loop(envs, list(local_env['bindings'].keys()), exprs)

    f_params = ', '.join([f'Value {binding.name}' for binding in bindings])
    if f_params:
        f_params = f'ObjMap* user_globals, {f_params}'
    else:
        f_params = 'ObjMap* user_globals'

    if not f_name:
        f_name = _get_generated_name(base='fn', envs=envs)
    envs[0]['functions'][f_name] = 'Value %s(%s) {\n  %s\n}' % (f_name, f_params, loop_code)

    envs.pop()

    return {'code': f_name}


def defn_c(params, envs):
    name = params[0].name
    f_name = _get_generated_name(base=f'u_{name}', envs=envs)

    envs[0]['user_globals'][name] = {'type': 'function', 'c_name': f_name}

    fn_result = fn_c(params[1:], envs, f_name=f_name)

    return {'code': ''}


def readline_c(params, envs):
    result_name = _get_generated_name('readline_result', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['pre'].append(f'  Value {result_name} = readline();')
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))
    return {'code': result_name}


def file_open_c(params, envs):
    path = compile_form(params[0], envs=envs)['code']
    mode = 'r'
    if len(params) > 1:
        if params[1] == 'w':
            mode = 'w'
    result_name = _get_generated_name('file_obj', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['pre'].append(f'  Value {result_name} = file_open({path}, "{mode}");')
    return {'code': result_name}


def file_read_c(params, envs):
    file_obj = compile_form(params[0], envs=envs)['code']
    result_name = _get_generated_name('file_data', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['pre'].append(f'  Value {result_name} = file_read({file_obj});')
    envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))
    return {'code': result_name}


def file_write_c(params, envs):
    file_obj = compile_form(params[0], envs=envs)['code']
    data = compile_form(params[1], envs=envs)['code']
    result_name = _get_generated_name('file_write_result', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['pre'].append(f'  Value {result_name} = file_write({file_obj}, {data});')
    return {'code': result_name}


def file_close_c(params, envs):
    file_obj = compile_form(params[0], envs=envs)['code']
    result_name = _get_generated_name('file_close_result', envs=envs)
    envs[-1]['temps'].add(result_name)
    envs[-1]['pre'].append(f'  Value {result_name} = file_close({file_obj});')
    return {'code': result_name}


global_compile_env = {
    'nil?': {'function': nil_c},
    '+': {'function': add_c},
    '-': {'function': subtract_c},
    '*': {'function': multiply_c},
    '/': {'function': divide_c},
    '=': {'function': equal_c},
    '>': {'function': greater_c, 'c_name': '*greater'},
    '>=': {'function': greater_equal_c},
    '<': {'function': less_c},
    '<=': {'function': less_equal_c},
    'print': {'function': print_c},
    'println': {'function': println_c},
    'count': {'function': count_c},
    'nth': {'function': nth_c},
    'sort': {'function': sort_c},
    'get': {'function': map_get_c},
    'contains?': {'function': map_contains_c},
    'assoc': {'function': map_assoc_c},
    'keys': {'function': map_keys_c},
    'vals': {'function': map_vals_c},
    'pairs': {'function': map_pairs_c},
    'def': {'function': def_c},
    'let': {'function': let_c},
    'loop': {'function': loop_c},
    'fn': {'function': fn_c},
    'defn': {'function': defn_c},
    'read-line': {'function': readline_c},
    'str': {'function': str_c},
    'str/split': {'function': str_split_c},
    'str/lower': {'function': str_lower_c},
    'str/blank?': {'function': str_blank_c},
    'file/open': {'function': file_open_c},
    'file/read': {'function': file_read_c},
    'file/write': {'function': file_write_c},
    'file/close': {'function': file_close_c},
}


character_replacements = {
    '-': '_M_',
    '?': '_Q_',
    '!': '_E_',
}


def _get_generated_name(base, envs):
    for c, replacement in character_replacements.items():
        base = base.replace(c, replacement)

    env = envs[0]
    if base not in env['functions'] and base not in env['temps'] and base not in env['user_globals'] and base not in envs[-1].get('temps', set()):
        return base
    i = 1
    while True:
        name = f'{base}_{i}'
        if name not in env['functions'] and name not in env['temps'] and base not in env['user_globals'] and name not in envs[-1].get('temps', set()):
            return name
        i += 1


def new_string_c(s, envs):
    name = _get_generated_name('str', envs=envs)

    envs[-1]['temps'].add(name)
    envs[-1]['pre'].append(f'  Value {name} = OBJ_VAL(copyString("{s}", (size_t) {len(s)}));')
    envs[-1]['pre'].append(f'  inc_ref(AS_OBJ({name}));')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return name


def new_vector_c(v, envs):
    name = _get_generated_name('lst', envs=envs)
    envs[-1]['temps'].add(name)
    num_items = len(v.items)
    c_code = f'  Value {name} = OBJ_VAL(allocate_list((size_t) {num_items}));\n  inc_ref(AS_OBJ({name}));'
    c_items = [compile_form(item, envs=envs)['code'] for item in v.items]
    for c_item in c_items:
        c_code += f'\n  list_add(AS_LIST({name}), {c_item});'

    envs[-1]['pre'].append(f'{c_code}\n')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return name


def new_map_c(node, envs):
    name = _get_generated_name('map', envs=envs)
    envs[-1]['temps'].add(name)
    c_code = f'  Value {name} = OBJ_VAL(allocate_map());\n  inc_ref(AS_OBJ({name}));'
    keys = [compile_form(k, envs=envs)['code'] for k in node.items[::2]]
    values = [compile_form(v, envs=envs)['code'] for v in node.items[1::2]]
    c_items = zip(keys, values)

    for key, value in c_items:
        c_code += f'\n  map_set(AS_MAP({name}), {key}, {value});'

    envs[-1]['pre'].append(f'{c_code}\n')
    envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
    return name


def compile_form(node, envs):
    if isinstance(node, list):
        first = node[0]
        rest = node[1:]
        if isinstance(first, list):
            results = [compile_form(n, envs=envs) for n in node]
            args = 'user_globals'
            if len(results) > 1:
                args += ', ' + ', '.join([r['code'] for r in results[1:]])
            result_name = _get_generated_name('fn_result', envs=envs)
            envs[-1]['pre'].append(f'  Value {result_name} = {results[0]["code"]}({args});')
            envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))
            return {'code': result_name}
        elif isinstance(first, Symbol):
            if first.name in envs[0]['global']:
                if isinstance(envs[0]['global'][first.name]['function'], Var) and isinstance(envs[0]['global'][first.name]['function'].value, Function):
                    f = envs[0]['global'][first.name]['function'].value
                    return f(rest)
                if callable(envs[0]['global'][first.name]['function']):
                    return envs[0]['global'][first.name]['function'](rest, envs=envs)
                else:
                    raise Exception(f'symbol first in list and not callable: {first.name} -- {env[first.name]}')
            elif first.name in envs[0]['user_globals']:
                f_name = envs[0]['user_globals'][first.name]['c_name']
                results = [compile_form(n, envs=envs) for n in rest]
                args = 'user_globals'
                if results:
                    args += ', ' + ', '.join([r['code'] for r in results])
                result_name = _get_generated_name('u_f_result', envs=envs)
                envs[-1]['temps'].add(result_name)
                envs[-1]['pre'].append(f'  Value {result_name} = {f_name}({args});')
                envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n    dec_ref_and_free(AS_OBJ(%s));\n  }' % (result_name, result_name))
                return {'code': result_name}
            elif first.name == 'for':
                bindings = rest[0]
                binding_name = bindings[0].name
                c_name = _get_generated_name(f'u_{binding_name}', envs=envs)
                envs[-1]['bindings'][bindings[0].name] = {'c_name': c_name}
                lst = compile_form(bindings[1], envs=envs)
                lst_name = _get_generated_name('tmp_lst', envs=envs)
                lst_count = _get_generated_name('tmp_lst_count', envs=envs)
                envs[-1]['temps'].add(lst_name)
                envs[-1]['temps'].add(lst_count)
                envs[-1]['pre'].append(f'  ObjList* {lst_name} = AS_LIST({lst["code"]});')
                envs[-1]['pre'].append('  for(size_t i=0; i<%s->count; i++) {\n' % lst_name)
                envs[-1]['pre'].append(f'    Value {c_name} = {lst_name}->values[i];')
                local_env = {'temps': envs[-1]['temps'], 'pre': [], 'post': [], 'bindings': {}}
                envs.append(local_env)
                for expr in rest[1:]:
                    compile_form(expr, envs=envs)
                code_lines = []
                if local_env['pre']:
                    code_lines.extend(local_env['pre'])
                if local_env['post']:
                    code_lines.extend(local_env['post'])
                envs.pop()
                for code_line in code_lines:
                    envs[-1]['pre'].append(code_line)
                envs[-1]['pre'].append('  }')
                return {'code': 'NIL_VAL'}
            elif first.name == 'do':
                do_exprs = [compile_form(n, envs=envs) for n in rest]

                do_result = _get_generated_name('do_result', envs)

                f_code = f'  Value {do_result} = NIL_VAL;'
                if isinstance(do_exprs[-1], tuple) and isinstance(do_exprs[-1][0], Symbol) and do_exprs[-1][0].name == 'recur':
                    for e in envs:
                        for b in e.get('bindings', {}).keys():
                            if b.startswith('recur'):
                                recur_name = b
                    for r in do_exprs[-1][1:]:
                        f_code += f'\n  recur_add(AS_RECUR({recur_name}), {r["code"]});'
                    f_code += f'\n  {do_result} = {recur_name};'
                else:
                    f_code += f'\n  {do_result} = {do_exprs[-1]["code"]};'
                    f_code += '\n  if (IS_OBJ(%s)) {\n    inc_ref(AS_OBJ(%s));\n  }' % (do_result, do_result)

                envs[-1]['pre'].append(f_code)
                envs[-1]['post'].append('  if (IS_OBJ(%s)) {\n  dec_ref_and_free(AS_OBJ(%s));\n  }' % (do_result, do_result))
                return {'code': do_result}
            elif first.name == 'recur':
                params = [compile_form(r, envs=envs) for r in rest]
                return (first, *params)
            else:
                print(f'global: {envs[0]["global"]}')
                print(f'user globals: {envs[0]["user_globals"]}')
                raise Exception(f'unhandled symbol: {first}')
        elif first == TokenType.IF:
            return if_form_c(rest, envs=envs)
        else:
            raise Exception(f'unhandled list: {node}')
    if isinstance(node, Symbol):
        if node.name in envs[0]['user_globals']:
            if envs[0]['user_globals'][node.name]['type'] == 'var':
                name = _get_generated_name('user_global_lookup', envs=envs)
                envs[0]['temps'].add(name)
                code = f'  Value {name} = OBJ_VAL(copyString("{node.name}", (size_t) {len(node.name)}));'
                code += f'\n  inc_ref(AS_OBJ({name}));'
                envs[-1]['pre'].append(code)
                envs[-1]['post'].append(f'  dec_ref_and_free(AS_OBJ({name}));')
                return {'code': f'map_get(user_globals, {name}, NIL_VAL)'}
            else:
                return {'code': envs[0]['user_globals'][node.name]['c_name']}
        else:
            for env in envs:
                if node.name in env.get('bindings', {}):
                    if env['bindings'][node.name] and 'c_name' in env['bindings'][node.name]:
                        return {'code': env['bindings'][node.name]['c_name']}
                    else:
                        return {'code': node.name}
        if node.name in envs[0]['global']:
            return {'code': envs[0]['global'][node.name]['c_name']}
        print(f'{envs[-1]=}')
        raise Exception(f'unhandled symbol: {node}')
    if isinstance(node, Vector):
        name = new_vector_c(node, envs=envs)
        return {'code': name}
    if isinstance(node, str):
        name = new_string_c(node, envs=envs)
        return {'code': name}
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
        name = new_map_c(node, envs=envs)
        return {'code': name}
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
    '<ctype.h>',
    '<stdlib.h>',
    '<stdbool.h>',
    '<string.h>',
    '<math.h>',
]


c_types = '''
#define ALLOCATE(type, count) \
    (type*)reallocate(NULL, sizeof(type) * (count))

#define ALLOCATE_OBJ(type, objectType) \
    (type*)allocateObject(sizeof(type), objectType)

#define GROW_CAPACITY(capacity) \
            ((capacity) < 8 ? 8 : (capacity) * 2)

#define GROW_ARRAY(type, pointer, oldCount, newCount) \
            (type*)reallocate(pointer, sizeof(type) * (newCount))

#define FREE(type, pointer) reallocate(pointer, (size_t)0)

#define FREE_ARRAY(type, pointer) \
            reallocate(pointer, (size_t)0)

#define NIL_VAL  ((Value){NIL, {.number = 0}})
#define BOOL_VAL(value)  ((Value){BOOL, {.boolean = value}})
#define NUMBER_VAL(value)  ((Value){NUMBER, {.number = value}})
#define RECUR_VAL(value)  ((Value){RECUR, {.recur = value}})
#define FILE_VAL(value)   ((Value){FILE_HANDLE, {.file = (FILE*)value}})
#define OBJ_VAL(object)   ((Value){OBJ, {.obj = (Obj*)object}})
#define AS_BOOL(value)  ((value).data.boolean)
#define AS_NUMBER(value)  ((value).data.number)
#define AS_RECUR(value)       ((value).data.recur)
#define AS_FILE(value)       ((value).data.file)
#define AS_OBJ(value)  ((value).data.obj)
#define AS_STRING(value)       ((ObjString*)AS_OBJ(value))
#define AS_CSTRING(value)      (((ObjString*)AS_OBJ(value))->chars)
#define AS_LIST(value)       ((ObjList*)AS_OBJ(value))
#define AS_MAP(value)       ((ObjMap*)AS_OBJ(value))
#define IS_NIL(value)  ((value).type == NIL)
#define IS_BOOL(value)  ((value).type == BOOL)
#define IS_NUMBER(value)  ((value).type == NUMBER)
#define IS_RECUR(value)  ((value).type == RECUR)
#define IS_FILE(value)  ((value).type == FILE_HANDLE)
#define IS_OBJ(value)  ((value).type == OBJ)
#define IS_STRING(value)  isObjType(value, OBJ_STRING)
#define IS_LIST(value)  isObjType(value, OBJ_LIST)
#define IS_MAP(value)  isObjType(value, OBJ_MAP)
#define MAP_EMPTY (-1)
#define MAP_MAX_LOAD 0.75
#define MAX_LINE 1000

void* reallocate(void* pointer, size_t newSize) {
  if (newSize == 0) {
    free(pointer);
    return NULL;
  }

  void* result = realloc(pointer, newSize);
  return result;
}

typedef enum {
  OBJ_STRING,
  OBJ_LIST,
  OBJ_MAP,
} ObjType;

typedef struct {
  ObjType type;
  uint32_t ref_cnt;
} Obj;

typedef enum {
  NIL,
  BOOL,
  NUMBER,
  RECUR,
  FILE_HANDLE,
  OBJ,
} ValueType;

typedef struct Recur Recur;

typedef struct {
  ValueType type;
  union {
    bool boolean;
    double number;
    Obj* obj;
    Recur* recur;
    FILE* file;
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

struct Recur {
  size_t count;
  size_t capacity;
  Value* values;
};

typedef struct {
  Obj obj;
  size_t count;
  size_t capacity;
  Value* values;
} ObjList;

/* Maps
 * Ideas from:
 *   https://github.com/python/cpython/blob/main/Objects/dictobject.c
 *   https://github.com/python/cpython/blob/main/Include/internal/pycore_dict.h
 *   https://mail.python.org/pipermail/python-dev/2012-December/123028.html
 *   https://morepypy.blogspot.com/2015/01/faster-more-memory-efficient-and-more.html
 * Use a sparse list that contains indices into a compact list of MapEntries
 *
 * MAP_EMPTY (-1) - marks a slot as never used
 * 0 - 2147483648 - marks an index into the entries list
 *
 * MinSize - starting size of new dict - 8 might be good
 */

typedef struct {
  uint32_t hash;
  Value key;
  Value value;
} MapEntry;

typedef struct {
  Obj obj;
  size_t num_entries;
  size_t indices_capacity;
  size_t entries_capacity;
  int32_t* indices; /* start with always using int32 for now */
  MapEntry* entries;
} ObjMap;

static Obj* allocateObject(size_t size, ObjType type) {
  Obj* object = (Obj*)reallocate(NULL, size);
  object->type = type;
  object->ref_cnt = 0;
  return object;
}

ObjMap* interned_strings;
void free_object(Obj* object);

// http://www.toccata.io/2019/02/RefCounting.html
void inc_ref(Obj* object) {
  object->ref_cnt++;
}

void dec_ref_and_free(Obj* object) {
  object->ref_cnt--;
  if (object->ref_cnt == 0) {
    free_object(object);
  }
}

static uint32_t hashString(const char* key, size_t length) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < length; i++) {
    hash ^= (uint8_t)key[i];
    hash *= 16777619;
  }
  return hash;
}

Value map_set(ObjMap* map, Value key, Value value);

static ObjString* allocate_string(char* chars, size_t length, uint32_t hash) {
  ObjString* string = ALLOCATE_OBJ(ObjString, OBJ_STRING);
  string->length = length;
  string->hash = hash;
  string->chars = chars;
  if (length < 4) {
    map_set(interned_strings, OBJ_VAL(string), NIL_VAL);
  }
  return string;
}

ObjString* find_interned_string(const char* chars, size_t length, uint32_t hash) {
  if (interned_strings->num_entries == 0) { return NULL; }
  uint32_t index = hash % (uint32_t)interned_strings->indices_capacity;
  for (;;) {
    if (interned_strings->indices[index] == MAP_EMPTY) {
      return NULL;
    }
    MapEntry entry = interned_strings->entries[interned_strings->indices[index]];
    ObjString* key_string = AS_STRING(entry.key);
    if (key_string->length == length &&
        key_string->hash == hash &&
        memcmp(key_string->chars, chars, length) == 0) {
      // We found it.
      return key_string;
    }

    index = (index + 1) % (uint32_t)interned_strings->indices_capacity;
  }

  return NULL;
}

ObjString* copyString(const char* chars, size_t length) {
  uint32_t hash = hashString(chars, length);
  if (length < 4) {
    ObjString* interned = find_interned_string(chars, length, hash);
    if (interned != NULL) {
      return interned;
    }
  }
  char* heapChars = ALLOCATE(char, length + 1);
  memcpy(heapChars, chars, length);
  heapChars[length] = 0; /* terminate it w/ NULL, so we can pass c-string to functions that need it */
  return allocate_string(heapChars, length, hash);
}

ObjList* allocate_list(size_t initial_capacity) {
  ObjList* list = ALLOCATE_OBJ(ObjList, OBJ_LIST);
  list->count = 0;
  list->capacity = (size_t) initial_capacity;
  if (initial_capacity == 0) {
    list->values = NULL;
  } else {
    list->values = GROW_ARRAY(Value, NULL, 0, (size_t) initial_capacity);
  }
  return list;
}

void list_add(ObjList* list, Value item) {
  if (list->capacity < list->count + 1) {
    size_t oldCapacity = list->capacity;
    list->capacity = GROW_CAPACITY(oldCapacity);
    list->values = GROW_ARRAY(Value, list->values, oldCapacity, list->capacity);
  }

  list->values[list->count] = item;
  list->count++;
  if (IS_OBJ(item)) {
    inc_ref(AS_OBJ(item));
  }
}

Value list_get(Value list, Value index) {
  if (AS_NUMBER(index) < 0) {
    return NIL_VAL;
  }
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

void swap(Value v[], size_t i, size_t j) {
  if (i == j) {
    return;
  }
  Value temp = v[i];
  v[i] = v[j];
  v[j] = temp;
}

Value nil_Q_(Value value) {
  return BOOL_VAL(IS_NIL(value));
}

Value add(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) + AS_NUMBER(y)); }
Value subtract(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) - AS_NUMBER(y)); }
Value multiply(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) * AS_NUMBER(y)); }
Value divide(Value x, Value y) { return NUMBER_VAL(AS_NUMBER(x) / AS_NUMBER(y)); }
Value greater(ObjMap* user_globals, Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) > AS_NUMBER(y)); }
Value greater_equal(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) >= AS_NUMBER(y)); }
Value less_equal(Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) <= AS_NUMBER(y)); }
Value less(ObjMap* user_globals, Value x, Value y) { return BOOL_VAL(AS_NUMBER(x) < AS_NUMBER(y)); }

void quick_sort(ObjMap* user_globals, Value v[], size_t left, size_t right, Value (*compare) (ObjMap*, Value, Value)) {
  /* C Programming Language K&R p87*/
  size_t i, last;
  if (left >= right) {
    return;
  }
  if ((int) left < 0) {
    return;
  }
  if ((int) right < 0) {
    return;
  }
  swap(v, left, (left + right)/2);
  last = left;
  for (i = left+1; i <= right; i++) {
    if (AS_BOOL((*compare) (user_globals, v[i], v[left]))) {
      swap(v, ++last, i);
    }
  }
  swap(v, left, last);
  quick_sort(user_globals, v, left, last-1, *compare);
  quick_sort(user_globals, v, last+1, right, *compare);
}

Value list_sort(ObjMap* user_globals, Value list, Value (*compare) (ObjMap*, Value, Value)) {
  ObjList* lst = AS_LIST(list);
  quick_sort(user_globals, lst->values, (size_t)0, (lst->count)-1, *compare);
  return OBJ_VAL(lst);
}

void recur_init(Recur* recur) {
  recur->count = 0;
  recur->capacity = 0;
  recur->values = NULL;
}

void recur_free(Recur* recur) {
  for (size_t i = 0; i < recur->count; i++) {
    Value v = recur->values[i];
    if (IS_OBJ(v)) {
      dec_ref_and_free(AS_OBJ(v));
    }
  }
  FREE_ARRAY(Value, recur->values);
  recur_init(recur);
}

void recur_add(Recur* recur, Value item) {
  if (recur->capacity < recur->count + 1) {
    size_t oldCapacity = recur->capacity;
    recur->capacity = GROW_CAPACITY(oldCapacity);
    recur->values = GROW_ARRAY(Value, recur->values, oldCapacity, recur->capacity);
  }

  recur->values[recur->count] = item;
  recur->count++;
  if (IS_OBJ(item)) {
    inc_ref(AS_OBJ(item));
  }
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

ObjMap* allocate_map(void) {
  ObjMap* map = ALLOCATE_OBJ(ObjMap, OBJ_MAP);
  map->num_entries = 0;
  map->indices_capacity = 0;
  map->entries_capacity = 0;
  map->indices = NULL;
  map->entries = NULL;
  return map;
}

Value map_count(Value map) {
  return NUMBER_VAL((int) AS_MAP(map)->num_entries);
}

bool is_truthy(Value value) {
  if (IS_NIL(value)) {
    return false;
  }
  if (IS_BOOL(value)) {
    if (AS_BOOL(value) == false) {
      return false;
    }
  }
  return true;
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
    double x_double = AS_NUMBER(x);
    double y_double = AS_NUMBER(y);
    double diff = fabs(x_double - y_double);
    return BOOL_VAL(diff < 1e-7);
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
    size_t x_num_items = xMap->num_entries;
    size_t y_num_items = yMap->num_entries;
    if (x_num_items != y_num_items) {
      return BOOL_VAL(false);
    }
    size_t x_num_entries = xMap->num_entries;
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

static int32_t find_indices_index(int32_t* indices, MapEntry* entries, size_t capacity, Value key) {
  /* hash the key and get an index
   * - if indices[index] is empty, return it
   * - if indices[index] points to an entry in entries with a hash that matches our hash, return index
   * Otherwise, keep adding one till we get to the correct key or an empty slot. */

  ObjString* key_string = AS_STRING(key);

  uint32_t index = key_string->hash % (uint32_t) capacity;
  for (;;) {
    if (indices[index] == MAP_EMPTY) {
      return (int32_t) index;
    }
    ObjString* entry_key_string = AS_STRING(entries[indices[index]].key);
    if (key_string->length == entry_key_string->length &&
        key_string->hash == entry_key_string->hash &&
        memcmp(key_string->chars, entry_key_string->chars, key_string->length) == 0) {
      return (int32_t) index;
    }

    index = (index + 1) % (uint32_t)capacity;
  }
}

static void adjustCapacity(ObjMap* map, size_t capacity) {
  // allocate new space
  int32_t* indices = ALLOCATE(int32_t, capacity);
  MapEntry* entries = ALLOCATE(MapEntry, capacity);

  // initialize all indices to MAP_EMPTY
  for (size_t i = 0; i < capacity; i++) {
    indices[i] = MAP_EMPTY;
  }

  // copy entries over to new space, filling in indices slots as well
  size_t num_entries = map->num_entries;
  size_t entries_index = 0;
  for (; entries_index < num_entries; entries_index++) {
    // find new index
    int32_t indices_index = find_indices_index(indices, entries, capacity, map->entries[entries_index].key);
    indices[indices_index] = (int32_t) entries_index;
    entries[entries_index] = map->entries[entries_index];
  }

  // fill in remaining entries with nil values
  for (; entries_index < capacity; entries_index++) {
    entries[entries_index].hash = 0;
    entries[entries_index].key = NIL_VAL;
    entries[entries_index].value = NIL_VAL;
  }

  FREE_ARRAY(int32_t, map->indices);
  FREE_ARRAY(MapEntry, map->entries);

  map->indices_capacity = capacity;
  map->entries_capacity = capacity;
  map->indices = indices;
  map->entries = entries;
}

Value map_set(ObjMap* map, Value key, Value value) {
  /* keep indices & entries same number of entries for now */
  if ((double)map->num_entries + 1 > (double)map->indices_capacity * MAP_MAX_LOAD) {
    size_t capacity = GROW_CAPACITY(map->indices_capacity);
    adjustCapacity(map, capacity);
  }

  int32_t indices_index = find_indices_index(map->indices, map->entries, map->indices_capacity, key);
  int32_t entries_index = (int32_t) map->indices[indices_index];

  bool isNewKey = (entries_index == MAP_EMPTY);
  if (isNewKey) {
    entries_index = (int32_t) map->num_entries;
    map->num_entries++;
    map->indices[indices_index] = entries_index;
  }

  if (IS_OBJ(key)) {
    inc_ref(AS_OBJ(key));
  }
  if (IS_OBJ(value)) {
    inc_ref(AS_OBJ(value));
  }

  MapEntry* entry = &(map->entries[entries_index]);

  if (IS_OBJ(entry->key)) {
    dec_ref_and_free(AS_OBJ(entry->key));
  }
  if (IS_OBJ(entry->value)) {
    dec_ref_and_free(AS_OBJ(entry->value));
  }

  entry->hash = AS_STRING(key)->hash;
  entry->key = key;
  entry->value = value;
  return OBJ_VAL(map);
}

Value map_contains(ObjMap* map, Value key) {
  if (map->num_entries == 0) {
    return BOOL_VAL(false);
  }
  if (!IS_STRING(key)) {
    return BOOL_VAL(false);
  }
  int32_t indices_index = find_indices_index(map->indices, map->entries, map->indices_capacity, key);
  int32_t entries_index = map->indices[indices_index];
  bool isNewKey = (entries_index == MAP_EMPTY);
  if (isNewKey) {
    return BOOL_VAL(false);
  }
  else {
    return BOOL_VAL(true);
  }
}

Value map_get(ObjMap* map, Value key, Value defaultVal) {
  if (map->num_entries == 0) {
    return defaultVal;
  }

  int32_t indices_index = find_indices_index(map->indices, map->entries, map->indices_capacity, key);
  int32_t entries_index = map->indices[indices_index];

  bool isNewKey = (entries_index == MAP_EMPTY);
  if (isNewKey) {
    return defaultVal;
  }
  else {
    return map->entries[entries_index].value;
  }
}

Value map_keys(ObjMap* map) {
  size_t num_entries = map->num_entries;
  ObjList* keys = allocate_list(num_entries);
  for (size_t i = 0; i < num_entries; i++) {
    list_add(keys, map->entries[i].key);
  }
  return OBJ_VAL(keys);
}

Value map_vals(ObjMap* map) {
  size_t num_entries = map->num_entries;
  ObjList* vals = allocate_list(num_entries);
  for (size_t i = 0; i < num_entries; i++) {
    list_add(vals, map->entries[i].value);
  }
  return OBJ_VAL(vals);
}

Value map_pairs(ObjMap* map) {
  size_t num_entries = map->num_entries;
  ObjList* pairs = allocate_list(num_entries);
  for (size_t i = 0; i < num_entries; i++) {
    if (!AS_BOOL(equal(map->entries[i].key, NIL_VAL))) {
      ObjList* pair = allocate_list((size_t) 2);
      list_add(pair, map->entries[i].key);
      list_add(pair, map->entries[i].value);
      list_add(pairs, OBJ_VAL(pair));
    }
  }
  return OBJ_VAL(pairs);
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
    if (AS_NUMBER(num_items) > 0) {
      print(list_get(value, NUMBER_VAL(0)));
    }
    for (int i = 1; i < AS_NUMBER(num_items); i++) {
      printf(" ");
      print(list_get(value, NUMBER_VAL(i)));
    }
    printf("]");
  }
  else if (IS_MAP(value)) {
    size_t num_entries = AS_MAP(value)->num_entries;
    printf("{");
    bool first_entry = true;
    for (size_t i = 0; i < num_entries; i++) {
      if (!first_entry) {
        printf(", ");
      }
      print(AS_MAP(value)->entries[i].key);
      printf(" ");
      print(AS_MAP(value)->entries[i].value);
      first_entry = false;
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

Value readline(void) {
  /* K&R p29 */
  int ch = 0;
  char buffer[MAX_LINE];
  int num_chars;
  for (num_chars=0; num_chars<(MAX_LINE-1) && (ch=getchar()) != EOF && ch != '\\n'; num_chars++) {
    buffer[num_chars] = (char) ch;
  }
  if ((ch == EOF) && (num_chars == 0)) {
    return NIL_VAL;
  }
  Value result = OBJ_VAL(copyString(buffer, (size_t) num_chars));
  inc_ref(AS_OBJ(result));
  return result;
}

Value str_blank(Value string) {
  if (IS_NIL(string)) {
    return BOOL_VAL(true);
  }
  ObjString* s = AS_STRING(string);
  if (s->length == 0) {
    return BOOL_VAL(true);
  }
  for (int i = 0; s->chars[i] != '\\0'; i++) {
    if (!isspace(s->chars[i])) {
      return BOOL_VAL(false);
    }
  }
  return BOOL_VAL(true);
}

Value str_lower(Value string) {
  ObjString* s = AS_STRING(string);
  ObjString* s_lower = copyString(s->chars, s->length);
  for (int i=0; s_lower->chars[i] != '\\0'; i++) {
    s_lower->chars[i] = (char) tolower((int) s_lower->chars[i]);
  }
  return OBJ_VAL(s_lower);
}

Value str_split(Value string) {
  ObjString* s = AS_STRING(string);
  ObjList* splits = allocate_list((size_t) 0);
  size_t split_length = 0;
  int split_start_index = 0;
  for (int i=0; s->chars[i] != '\\0'; i++) {
    if (s->chars[i] == ' ') {
      ObjString* split = copyString(&(s->chars[split_start_index]), split_length);
      list_add(splits, OBJ_VAL(split));
      split_start_index = i + 1;
      split_length = 0;
    }
    else {
      split_length++;
    }
  }
  ObjString* split = copyString(&(s->chars[split_start_index]), split_length);
  list_add(splits, OBJ_VAL(split));
  return OBJ_VAL(splits);
}

Value str_str(Value v) {
  // if we got a string, no need to do any work, just return it
  if (IS_STRING(v)) {
    inc_ref(AS_OBJ(v));
    return v;
  }

  Value s;
  if (IS_BOOL(v)) {
    if (AS_BOOL(v)) {
      s = OBJ_VAL(copyString("true", (size_t)4));
    }
    else {
      s = OBJ_VAL(copyString("false", (size_t)5));
    }
  }
  else if (IS_NUMBER(v)) {
    char str[100];
    int num_chars = sprintf(str, "%g", AS_NUMBER(v));
    s = OBJ_VAL(copyString(str, (size_t)num_chars));
  }
  else if (IS_LIST(v)) {
    s = OBJ_VAL(copyString("[]", (size_t)2));
  }
  else if (IS_MAP(v)) {
    s = OBJ_VAL(copyString("{}", (size_t)2));
  }
  else {
    s = OBJ_VAL(copyString("", (size_t)0));
  }
  inc_ref(AS_OBJ(s));
  return s;
}

Value str_join(Value list_val) {
  ObjList* list = AS_LIST(list_val);

  size_t num_bytes = 0;

  for (size_t i = 0; i < list->count; i++) {
    Value v = list_get(list_val, NUMBER_VAL((double)i));
    Value v_str = str_str(v);
    num_bytes = num_bytes + AS_STRING(v_str)->length;
    dec_ref_and_free(AS_OBJ(v_str));
    if (IS_OBJ(v)) {
      dec_ref_and_free(AS_OBJ(v));
    }
  }

  char* heapChars = ALLOCATE(char, num_bytes+1);
  char* start_char = heapChars;

  for (size_t i = 0; i < list->count; i++) {
    Value v = list_get(list_val, NUMBER_VAL((double)i));
    ObjString* s = AS_STRING(str_str(v));
    memcpy(start_char, s->chars, s->length);
    start_char = start_char + s->length;
  }
  heapChars[num_bytes] = 0;
  uint32_t hash = hashString(heapChars, num_bytes);
  return OBJ_VAL(allocate_string(heapChars, num_bytes, hash));
}

Value file_open(Value path, const char* mode) {
  FILE* fp = fopen(AS_CSTRING(path), mode);
  return FILE_VAL(fp);
}

Value file_read(Value file) {
  int ch = 0;
  char buffer[MAX_LINE];
  int num_chars;
  FILE* fp = AS_FILE(file);
  for (num_chars=0; num_chars<(MAX_LINE-1) && (ch=getc(fp)) != EOF; num_chars++) {
    buffer[num_chars] = (char) ch;
  }
  if ((ch == EOF) && (num_chars == 0)) {
    return NIL_VAL;
  }
  Value result = OBJ_VAL(copyString(buffer, (size_t) num_chars));
  inc_ref(AS_OBJ(result));
  return result;
}

Value file_write(Value file, Value data) {
  FILE* fp = AS_FILE(file);
  fprintf(fp, "%s", AS_CSTRING(data));
  fflush(fp);
  return NIL_VAL;
}

Value file_close(Value file) {
  fclose(AS_FILE(file));
  return NIL_VAL;
}

void free_object(Obj* object) {
  switch (object->type) {
    case OBJ_STRING: {
      ObjString* string = (ObjString*)object;
      FREE_ARRAY(char, string->chars);
      FREE(ObjString, object);
      break;
    }
    case OBJ_LIST: {
      ObjList* list = (ObjList*)object;
      for (size_t i = 0; i < list->count; i++) {
        Value v = list_get(OBJ_VAL(object), NUMBER_VAL((double)i));
        if (IS_OBJ(v)) {
          dec_ref_and_free(AS_OBJ(v));
        }
      }
      FREE_ARRAY(Value, list->values);
      FREE(ObjList, object);
      break;
    }
    case OBJ_MAP: {
      ObjMap* map = (ObjMap*)object;
      for (size_t i = 0; i < map->num_entries; i++) {
        MapEntry entry = map->entries[i];
        if (IS_OBJ(entry.key)) {
          dec_ref_and_free(AS_OBJ(entry.key));
        }
        if (IS_OBJ(entry.value)) {
          dec_ref_and_free(AS_OBJ(entry.value));
        }
      }
      FREE_ARRAY(int32_t, map->indices);
      FREE_ARRAY(MapEntry, map->entries);
      FREE(ObjMap, object);
      break;
    }
    default: {
      break;
    }
  }
}'''


def _compile(source):
    tokens = scan_tokens(source)
    ast = parse(tokens)

    compiled_forms = []
    env = {
        'global': copy.deepcopy(global_compile_env),
        'functions': {},
        'user_globals': {},
        'temps': set(),
        'init': [],
        'pre': [],
        'post': [],
        'bindings': {},
    }
    for f in ast.forms:
        result = compile_form(f, envs=[env])
        c = result['code']
        # hack to check if there's a function call we need to put in
        if c and '(' in c and ')' in c:
            if not c.endswith(';'):
                c = f'{c};'
            compiled_forms.append(f'  {c}')

    c_code = '\n'.join([f'#include {i}' for i in c_includes])
    c_code += '\n\n'
    c_code += c_types

    c_code += '\n\n/* CUSTOM CODE */\n\n'

    if env['functions']:
        c_code += '\n\n'.join([f for f in env['functions'].values()]) + '\n\n'

    c_code += 'int main(void)\n{'
    c_code += '\n  interned_strings = allocate_map();'
    c_code += '\n  ObjMap* user_globals = allocate_map();\n'

    if env['init']:
        c_code += '\n'.join(env['init'])

    for name, value in env['user_globals'].items():
        if value['type'] == 'var':
            c_code += f'\n  map_set(user_globals, OBJ_VAL(copyString("{name}", (size_t) {len(name)})), {value["code"]});\n'

    c_code += '\n' + '\n'.join(env['pre'])

    if compiled_forms:
        c_code += '\n' + '\n'.join(compiled_forms)
    if env['post']:
        c_code += '\n' + '\n'.join(env['post'])
    c_code += '\n  free_object((Obj*)user_globals);'
    c_code += '\n  free_object((Obj*)interned_strings);'
    c_code += '\n  return 0;\n}'

    return c_code


GCC_CMD = 'gcc'
CLANG_CMD = 'clang'

# See https://github.com/airbus-seclab/c-compiler-security
GCC_CHECK_OPTIONS = [
    '-O2',
    '-std=c99',
    '-Werror',
    '-Wall',
    '-Wextra',
    '-Wno-error=unused-parameter',
    '-Wno-error=unused-variable',
    '-pedantic',
    '-Wpedantic',
    '-Wformat=2',
    '-Wformat-overflow=2',
    '-Wformat-truncation=2',
    '-Wformat-security',
    '-Wnull-dereference',
    '-Wstack-protector',
    '-Wtrampolines',
    '-Walloca',
    '-Wvla',
    '-Warray-bounds=2',
    '-Wimplicit-fallthrough=3',
    '-Wtraditional-conversion',
    '-Wshift-overflow=2',
    '-Wcast-qual',
    '-Wstringop-overflow=4',
    '-Wconversion',
    '-Warith-conversion',
    '-Wlogical-op',
    '-Wduplicated-cond',
    '-Wduplicated-branches',
    '-Wformat-signedness',
    '-Wshadow',
    '-Wstrict-overflow=4',
    '-Wundef',
    '-Wstrict-prototypes',
    '-Wswitch-default',
    '-Wswitch-enum',
    '-Wstack-usage=1000000',
    # '-Wcast-align=strict',
    '-D_FORTIFY_SOURCE=2',
    '-fstack-protector-strong',
    '-fstack-clash-protection',
    '-fPIE',
    '-Wl,-z,relro',
    '-Wl,-z,now',
    '-Wl,-z,noexecstack',
    '-Wl,-z,separate-code',
    '-fsanitize=address',
    '-fsanitize=pointer-compare',
    '-fsanitize=pointer-subtract',
    '-fsanitize=leak',
    '-fno-omit-frame-pointer',
    '-fsanitize=undefined',
    '-fsanitize=bounds-strict',
    '-fsanitize=float-divide-by-zero',
    '-fsanitize=float-cast-overflow',
]
GCC_CHECK_ENV = {
    'ASAN_OPTIONS': 'strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1:detect_invalid_pointer_pairs=2',
    'PATH': os.environ.get('PATH', ''),
}

CLANG_CHECK_OPTIONS = [
    '-O2',
    '-std=c99',
    '-Werror',
    '-Walloca',
    '-Wcast-qual',
    '-Wconversion',
    '-Wformat=2',
    '-Wformat-security',
    '-Wnull-dereference',
    '-Wstack-protector',
    '-Wvla',
    '-Warray-bounds',
    '-Warray-bounds-pointer-arithmetic',
    '-Wassign-enum',
    '-Wbad-function-cast',
    '-Wconditional-uninitialized',
    '-Wconversion',
    '-Wfloat-equal',
    '-Wformat-type-confusion',
    '-Widiomatic-parentheses',
    '-Wimplicit-fallthrough',
    '-Wloop-analysis',
    '-Wpointer-arith',
    '-Wshift-sign-overflow',
    '-Wshorten-64-to-32',
    '-Wswitch-enum',
    '-Wtautological-constant-in-range-compare',
    '-Wunreachable-code-aggressive',
    '-Wthread-safety',
    '-Wthread-safety-beta',
    '-Wcomma',
    '-D_FORTIFY_SOURCE=3',
    '-fstack-protector-strong',
    '-fPIE',
    '-fstack-clash-protection',
    '-fsanitize=bounds',
    '-fsanitize-undefined-trap-on-error',
    '-Wl,-z,relro',
    '-Wl,-z,now',
    '-Wl,-z,noexecstack',
    '-Wl,-z,separate-code',
    '-fsanitize=address',
    '-fsanitize=leak',
    '-fno-omit-frame-pointer',
    '-fsanitize=undefined',
    '-fsanitize=float-divide-by-zero',
    '-fsanitize=float-cast-overflow',
    '-fsanitize=integer',
]
CLANG_CHECK_ENV = {
    'ASAN_OPTIONS': 'strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1:detect_invalid_pointer_pairs=2',
    'PATH': os.environ.get('PATH', ''),
}


def build_executable(file_name, output_file_name, with_checks=False):
    if os.path.exists(output_file_name):
        print(f'{output_file_name} already exists')
        sys.exit(1)

    if os.environ.get('CC'):
        compiler = [os.environ['CC']]
    else:
        compiler = [CLANG_CMD]

    env = None

    if with_checks:
        if 'clang' in compiler[0]:
            compiler.extend(CLANG_CHECK_OPTIONS)
        elif 'gcc' in compiler[0]:
            compiler.extend(GCC_CHECK_OPTIONS)
        else:
            raise RuntimeError(f'no checks to use for building with {compiler[0]}')
    else:
        compiler.extend(['-O2'])

    compile_cmd = compiler + ['-o', output_file_name, file_name]
    try:
        subprocess.run(compile_cmd, check=True, env=env, capture_output=True)
    except subprocess.CalledProcessError as e:
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                data = f.read().decode('utf8')
                print(data)
        print(e)
        print(e.stderr.decode('utf8'))
        sys.exit(1)


def run_executable(file_name):
    try:
        subprocess.run([f'./{file_name}'], check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(1)


def compile_to_c(file_name, output_file_name):
    with open(file_name, 'rb') as f:
        source = f.read().decode('utf8')

    c_program = _compile(source)

    with open(output_file_name, mode='wb') as f:
        f.write(c_program.encode('utf8'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Random language')
    parser.add_argument('-c', action='store_true', dest='compile', help='compile the file to C')
    parser.add_argument('-b', action='store_true', dest='build', help='build executable')
    parser.add_argument('-r', action='store_true', dest='run', help='compile, build, & run')
    parser.add_argument('-o', action='store', dest='output', help='output file')
    parser.add_argument('--with-checks', action='store_true', dest='with_checks', help='build with compiler checks')
    parser.add_argument('file', type=str, nargs='?', help='file to interpret')

    args = parser.parse_args()

    if args.compile:
        if args.file:
            if args.output:
                c_file_name = args.output
            else:
                tmp = tempfile.mkdtemp(dir='.')
                c_file_name = os.path.join(tmp, 'code.c')
            compile_to_c(Path(args.file), c_file_name)
            print(f'Compiled to {c_file_name}')
        else:
            print('no file to compile')
    elif args.build:
        if args.file:
            if args.output:
                executable = args.output
            else:
                tmp = tempfile.mkdtemp(dir='.')
                executable = os.path.join(tmp, 'program')
            if not args.file.endswith('.c'):
                with tempfile.TemporaryDirectory() as c_tmp:
                    c_file_name = os.path.join(c_tmp, 'code.c')
                    compile_to_c(Path(args.file), c_file_name)
                    build_executable(c_file_name, output_file_name=executable, with_checks=args.with_checks)
            else:
                c_file_name = args.file
                build_executable(c_file_name, output_file_name=executable, with_checks=args.with_checks)
            print(f'Built executable at {executable}')
        else:
            print('no file to build')
    elif args.run:
        if args.file:
            with tempfile.TemporaryDirectory(dir='.') as tmp:
                executable = os.path.join(tmp, 'program')
                if args.file.endswith('.c'):
                    build_executable(args.file, output_file_name=executable, with_checks=args.with_checks)
                    run_executable(executable)
                else:
                    c_file_name = os.path.join(tmp, 'code.c')
                    compile_to_c(Path(args.file), c_file_name)
                    build_executable(c_file_name, output_file_name=executable, with_checks=args.with_checks)
                    run_executable(executable)
        else:
            print('no file to run')
    else:
        main(args.file)
