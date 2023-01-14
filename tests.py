import os
import platform
import subprocess
import tempfile
import unittest
import sys
from unittest.mock import patch
from lang import TokenType, scan_tokens, parse, evaluate, Keyword, Symbol, Var, Vector, run, _compile
from lang import (
        GCC_CMD, GCC_CHECK_OPTIONS, GCC_CHECK_ENV,
        CLANG_CMD, CLANG_CHECK_OPTIONS, CLANG_CHECK_ENV,
    )


SOURCE = '(+ 10 2 (- 15 (+ 4 4)) -5)'
EXPECTED_TOKENS = [
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.SYMBOL, 'lexeme': '+'},
    {'type': TokenType.NUMBER, 'lexeme': '10'},
    {'type': TokenType.NUMBER, 'lexeme': '2'},
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.SYMBOL, 'lexeme': '-'},
    {'type': TokenType.NUMBER, 'lexeme': '15'},
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.SYMBOL, 'lexeme': '+'},
    {'type': TokenType.NUMBER, 'lexeme': '4'},
    {'type': TokenType.NUMBER, 'lexeme': '4'},
    {'type': TokenType.RIGHT_PAREN},
    {'type': TokenType.RIGHT_PAREN},
    {'type': TokenType.NUMBER, 'lexeme': '-5'},
    {'type': TokenType.RIGHT_PAREN},
]

EXPECTED_AST_FORMS = [
    [
        Symbol('+'),
        10,
        2,
        [
            Symbol('-'),
            15,
            [
                Symbol('+'),
                4,
                4,
            ]
        ],
        -5,
    ]
]


class ScanTokenTests(unittest.TestCase):

    def test(self):
        self.maxDiff = None
        tokens = scan_tokens(SOURCE)
        self.assertEqual(tokens, EXPECTED_TOKENS)

    def test_nil(self):
        tokens = scan_tokens('(= 1 nil)')
        self.assertEqual(tokens,
                [
                    {'type': TokenType.LEFT_PAREN},
                    {'type': TokenType.SYMBOL, 'lexeme': '='},
                    {'type': TokenType.NUMBER, 'lexeme': '1'},
                    {'type': TokenType.NIL},
                    {'type': TokenType.RIGHT_PAREN},
                ]
            )

    def test_number(self):
        tokens = scan_tokens('2')
        self.assertEqual(tokens, [{'type': TokenType.NUMBER, 'lexeme': '2'}])


class ParseTests(unittest.TestCase):

    def test(self):
        self.maxDiff = None
        ast = parse(EXPECTED_TOKENS)
        self.assertEqual(ast.forms, EXPECTED_AST_FORMS)

    def test_2(self):
        tokens = [
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.SYMBOL, 'lexeme': '='},
            {'type': TokenType.TRUE},
            {'type': TokenType.NIL},
            {'type': TokenType.RIGHT_PAREN},
        ]
        ast = parse(tokens)
        self.assertEqual(ast.forms,
            [
                [
                    Symbol('='),
                    True,
                    None
                ]
            ]
        )

    def test_sequence_of_forms(self):
        tokens = [
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.SYMBOL, 'lexeme': 'def'},
            {'type': TokenType.SYMBOL, 'lexeme': 'a'},
            {'type': TokenType.NUMBER, 'lexeme': '1'},
            {'type': TokenType.RIGHT_PAREN},
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.SYMBOL, 'lexeme': '+'},
            {'type': TokenType.SYMBOL, 'lexeme': 'a'},
            {'type': TokenType.NUMBER, 'lexeme': '1'},
            {'type': TokenType.RIGHT_PAREN},
        ]
        ast = parse(tokens)
        self.assertEqual(ast.forms,
                [
                    [
                        Symbol('def'),
                        Symbol('a'),
                        1
                    ],
                    [
                        Symbol('+'),
                        Symbol('a'),
                        1
                    ]
                ]
            )


class EvalTests(unittest.TestCase):

    def test(self):
        tests = [
            {'src': '2', 'result': 2},
            {'src': '-2', 'result': -2},
            {'src': '2.3', 'result': 2.3},
            {'src': '-2.3', 'result': -2.3},
            {'src': 'nil', 'result': None},
            {'src': 'true', 'result': True},
            {'src': 'false', 'result': False},
            {'src': ':hello', 'result': Keyword(':hello')},
            {'src': '"hello"', 'result': 'hello'},
            {'src': '"hello "', 'result': 'hello '},
            {'src': '"hello " ;comment', 'result': 'hello '},
            {'src': '(quote (1 2))', 'result': [1, 2]},
            {'src': '[1 2]', 'result': Vector([1, 2])},
            {'src': '[1 (+ 1 1)]', 'result': Vector([1, 2])},
            {'src': '{1 5, 2 4}', 'result': {1: 5, 2: 4}},
            {'src': '{1 (+ 2 3), 2 4}', 'result': {1: 5, 2: 4}},
            {'src': '1 2 nil', 'result': [1, 2, None]}, #multiple forms, not a list
            {'src': '1 [1 2] nil', 'result': [1, Vector([1, 2]), None]}, #multiple forms, not a list
            {'src': '(+ 1 2)', 'result': 3},
            {'src': '(+ 1 (+ 1 1))', 'result': 3},
            {'src': '(+ 1 (- 4 2))', 'result': 3},
            {'src': '(* 2 (- 4 2) 3)', 'result': 12},
            {'src': '(/ 12 (- 4 2) 3)', 'result': 2},
            {'src': '(+ 10 -2)', 'result': 8},
            {'src': '(= 1 2)', 'result': False},
            {'src': '(= 1 1 1)', 'result': True},
            {'src': '(= true 2)', 'result': False},
            {'src': '(= true nil)', 'result': False},
            {'src': '(> 1 1)', 'result': False},
            {'src': '(>= 1 1)', 'result': True},
            {'src': '(> 2 1)', 'result': True},
            {'src': '(< 1 1)', 'result': False},
            {'src': '(<= 1 1)', 'result': True},
            {'src': '(< 1 2)', 'result': True},
            {'src': '(if true true false)', 'result': True},
            {'src': '(if false true false)', 'result': False},
            {'src': '(def a 1)', 'result': Var(name='a', value=1)},
            {'src': '(def a 1) (+ a 2)', 'result': [Var(name='a', value=1), 3]},
            {'src': '(let [x 1] x)', 'result': 1},
            {'src': '(let [x (+ 1 1)] x)', 'result': 2},
            {'src': '((fn [x] x) 1)', 'result': 1},
            {'src': '(let [x 1 y 2] (+ x y))', 'result': 3},
            {'src': '((fn [x y] (+ x y)) 1 2)', 'result': 3},
            {'src': '(str)', 'result': ''},
            {'src': '(str 1)', 'result': '1'},
            {'src': '(str "hello " "world")', 'result': 'hello world'},
            {'src': '(str "hello " 1 " world")', 'result': 'hello 1 world'},
            {'src': '(str/split "hello world")', 'result': ['hello', 'world']},
            {'src': '(str/trim "\n")', 'result': ''},
            {'src': '(str/trim " hello ")', 'result': 'hello'},
            {'src': '(subs "world" 0)', 'result': 'world'},
            {'src': '(subs "world" 0 2)', 'result': 'wo'},
            {'src': '(conj [1 2] 3)', 'result': Vector([1, 2, 3])},
            {'src': '(subvec [1 2 3] 0)', 'result': Vector([1, 2, 3])},
            {'src': '(subvec [1 2 3] 0 1)', 'result': Vector([1])},
            {'src': '(nth [1 2 3] 1)', 'result': 2},
            {'src': '(nth (quote (1 2 3)) 1)', 'result': 2},
            {'src': '(count nil)', 'result': 0},
            {'src': '(count [1 2 3])', 'result': 3},
            {'src': '(count (quote (1 2 3)))', 'result': 3},
            {'src': '(sort [3 1 2])', 'result': [1, 2, 3]},
            {'src': '(sort > [3 1 2])', 'result': [3, 2, 1]},
            {'src': '(sort-by (fn [l] (do (println l) (nth l 1))) [["b" 3] ["a" 1] ["c" 2]])', 'result': [['a', 1], ['c', 2], ['b', 3]]},
            {'src': '(sort-by (fn [l] (nth l 1)) > [["b" 3] ["a" 1] ["c" 2]])', 'result': [['b', 3], ['c', 2], ['a', 1]]},
            {'src': '(get {1 2} 1)', 'result': 2},
            {'src': '(get {1 2} 5 99)', 'result': 99},
            {'src': '(keys {1 2 "a" 3})', 'result': [1, 'a']},
            {'src': '(vals {1 2 "a" 3})', 'result': [2, 3]},
            {'src': '(pairs {1 2 "a" 3})', 'result': [[1, 2], ["a", 3]]},
            {'src': '(count {1 2 "a" 3})', 'result': 2},
            {'src': '(contains? {1 2 "a" 3} 1)', 'result': True},
            {'src': '(contains? {1 2 "a" 3} "not-found")', 'result': False},
            {'src': '(assoc {1 2 "a" 3} "new-key" "new-val")', 'result': {1: 2, 'a': 3, 'new-key': 'new-val'}},
            {'src': '(dissoc {1 2 "a" 3} 1)', 'result': {'a': 3}},
            {'src': '(def d {}) (assoc d 1 "a") (get d 1)', 'result': [Var(name='d', value={1: 'a'}), {1: 'a'}, 'a']},
            {'src': '(def d {1 "a"}) (dissoc d 1) d', 'result': [Var(name='d', value={1: 'a'}), {}, {}]},
            {'src': '((fn [n] (loop [cnt n acc 1] (if (= 0 cnt) acc (recur (- cnt 1) (* acc cnt))))) 3)', 'result': 6},
            {'src': '(def d {}) (loop [i 0 k 1] (if (> i 1) d (do (assoc d k "a") (recur (+ i 1) 2))))', 'result': [Var(name='d', value={}), {1: 'a', 2: 'a'}]},
            {'src': '(def i 0) ((fn [n] n) i)', 'result': [Var(name='i', value=0), 0]},
        ]

        for test in tests:
            with self.subTest(test=test):
                self.assertEqual(parse(scan_tokens(test['src'])).evaluate(), test['result'])

    def test_other(self):
        results = parse(scan_tokens('(def f1 (fn [x y] (+ x y))) (f1 1 2)')).evaluate()
        self.assertEqual(results[1], 3)

        results = parse(scan_tokens('(defn f1 [x y] (+ x y)) (f1 1 2)')).evaluate()
        self.assertEqual(results[1], 3)

        results = parse(scan_tokens('(defn hello [name] (str "Hello, " name)) (hello "Someone")')).evaluate()
        self.assertEqual(results[1], 'Hello, Someone')

        results = parse(scan_tokens('(def name "Someone") (defn hello [name] (str "Hello, " name)) (hello name)')).evaluate()
        self.assertEqual(results[2], 'Hello, Someone')

        results = parse(scan_tokens('(def name "Someone") (defn hello [name] (str "Hello, " name)) (do (hello name) name)')).evaluate()
        self.assertEqual(results[2], 'Someone')

        results = parse(scan_tokens('(defn hello [name] (str "Hello, " name)) (loop [name "Someone Else"] (if (= name "Someone") name (do (hello name) (str name name))))')).evaluate()
        self.assertEqual(results[1], 'Someone ElseSomeone Else')

        results = parse(scan_tokens('(def v "\n") (str/trim v)')).evaluate()
        self.assertEqual(results[1], '')

        with patch('builtins.print') as print_mock:
            result = parse(scan_tokens('(print "1")')).evaluate()

        print_mock.assert_called_with('1', end='')
        self.assertIsNone(result)

        with patch('builtins.print') as print_mock:
            result = parse(scan_tokens('(println "1")')).evaluate()

        print_mock.assert_called_with('1')
        self.assertIsNone(result)

        with patch('builtins.input') as input_mock:
            input_mock.return_value = 'line1'
            result = parse(scan_tokens('(read-line)')).evaluate()

        input_mock.assert_called_once()
        self.assertEqual(result, 'line1')

        with patch('builtins.input') as input_mock:
            input_mock.side_effect = EOFError()
            result = parse(scan_tokens('(read-line)')).evaluate()

        input_mock.assert_called_once()
        self.assertEqual(result, None)

        with patch('builtins.print') as print_mock:
            result = parse(scan_tokens('(do (println "1") 2)')).evaluate()

        print_mock.assert_called_with('1')
        self.assertEqual(result, 2)

        with tempfile.NamedTemporaryFile(mode='w+b') as f:
            f.write('asdf'.encode('utf8'))
            f.flush()
            result = parse(scan_tokens(f'(let [f (file/open "{f.name}"), data (file/read f)] (do (file/close f) data))')).evaluate()
        self.assertEqual(result, 'asdf'.encode('utf8'))

    def test_exceptions(self):
        with self.assertRaises(Exception) as cm:
            parse(scan_tokens('(1 2 3)')).evaluate()
        self.assertIn('not callable', str(cm.exception))


class RunTests(unittest.TestCase):

    def test(self):
        source = '''
(defn f [l] ;comment
  l)
;comment
(loop [line "line1"]
  (if (= line "")
    0
    (do ;comment
      (f line)
      (recur ""))))'''

        result = run(source)

        self.assertEqual(result[1], 0)

    def test_if(self):
        source = '''
  (if (= 1 2)
    (println "true")
    (println "false"))'''

        with patch('builtins.print') as print_mock:
            parse(scan_tokens(source)).evaluate()
        print_mock.assert_called_once_with('false')

        source = '''
  (if (= 1 1)
    (println "true")
    (println "false"))'''

        with patch('builtins.print') as print_mock:
            parse(scan_tokens(source)).evaluate()
        print_mock.assert_called_once_with('true')

    def test_function(self):
        source = '''
(def counts {})

(defn countwords [word]
  (assoc counts word 0))

(countwords "the")

counts'''

        result = run(source)

        self.assertEqual(result[3], {'the': 0})


gcc_cmd = os.environ.get('GCC', GCC_CMD)
clang_cmd = os.environ.get('CLANG', CLANG_CMD)
IS_WINDOWS = False

if platform.system() == 'Darwin':
    compilers = [
        ([clang_cmd], None, 'clang_regular'),
        ([gcc_cmd], None, 'gcc_regular'),
    ]
elif platform.system() == 'Windows':
    IS_WINDOWS = True
    vs_dir = os.environ['VSDIR']
    # print(f'{vs_dir=}')
    # print(os.listdir(vs_dir))
    # print(os.listdir(os.path.join(vs_dir, 'MSBuild')))
    # print(os.listdir(os.path.join(vs_dir, 'VC')))
    # print(os.listdir(os.path.join(vs_dir, 'VC', 'Tools')))
    # print(os.listdir(os.path.join(vs_dir, 'VC', 'Tools', 'MSVC')))
    # print(os.listdir(os.path.join(vs_dir, 'VC', 'Tools', 'MSVC', '14.34.31933')))
    cc_path = os.path.join(vs_dir, 'VC', 'Tools', 'MSVC', '14.34.31933', 'bin', 'Hostx64', 'x64', 'cl.exe')
    # print(os.listdir(os.path.join(vs_dir, 'SDK')))
    # print(os.listdir(os.path.join(vs_dir, 'VSSDK')))
    # sys.exit(0)
    compilers = [
        ([cc_path], None, 'vscc_regular'),
    ]
    # compile_cmd = [cc_path, '/h']
    # try:
    #     result = subprocess.run(compile_cmd, check=True, env=None, capture_output=True)
    #     print(f'err: {result.stderr.decode("utf8")}')
    #     print(f'out: {result.stdout.decode("utf8")}')
    #     sys.exit(0)
    # except subprocess.CalledProcessError as e:
    #     print(f'err: {e.stderr.decode("utf8")}')
    #     print(f'out: {e.stdout.decode("utf8")}')
    #     raise
else:
    compilers = [
        ([clang_cmd], None, 'clang_regular'),
        ([clang_cmd] + CLANG_CHECK_OPTIONS, CLANG_CHECK_ENV, 'clang_checks'),
        ([gcc_cmd] + GCC_CHECK_OPTIONS, GCC_CHECK_ENV, 'gcc_checks'),
    ]


def _run_test(test, assert_equal):
    print(f'*** c test: {test["src"]}')
    c_code = _compile(test['src'])

    with tempfile.TemporaryDirectory() as tmp:
        c_filename = os.path.join(tmp, 'code.c')
        with open(c_filename, 'wb') as f:
            f.write(c_code.encode('utf8'))

        for cc_cmd, env, env_name in compilers:
            print(f'  ({env_name})')
            program_filename = os.path.join(tmp, env_name)

            if IS_WINDOWS:
                compile_cmd = cc_cmd + [f'/OUT"{program_filename}"', c_filename]
                print(f'{compile_cmd=}')
            else:
                compile_cmd = cc_cmd + ['-o', program_filename, c_filename]
            try:
                subprocess.run(compile_cmd, check=True, env=env, capture_output=True)
            except subprocess.CalledProcessError as e:
                if not IS_WINDOWS:
                    print(f'bad c code:\n{c_code}')
                print(f'err: {e.stderr.decode("utf8")}')
                raise

            program_cmd = [program_filename]
            if 'input' in test:
                input_ = test['input'].encode('utf8')
            else:
                input_ = None

            try:
                result = subprocess.run(program_cmd, check=True, input=input_, capture_output=True)
            except subprocess.CalledProcessError as e:
                if not IS_WINDOWS:
                    print(f'bad c code:\n{c_code}')
                print(f'err: {e.stderr.decode("utf8")}')
                print(f'out: {e.stdout.decode("utf8")}')
                raise

            try:
                assert_equal(result.stdout.decode('utf8'), test['output'])
            except AssertionError:
                if not IS_WINDOWS:
                    print(f'bad c code:\n{c_code}')
                raise


class CompileTests(unittest.TestCase):
    def test_values(self):
        tests = [
            {'src': '(print nil)', 'output': 'nil'},
            {'src': '(print true)', 'output': 'true'},
            {'src': '(print false)', 'output': 'false'},
            {'src': '(print 1)', 'output': '1'},
            {'src': '(print 1.2)', 'output': '1.2'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_math_operations(self):
        tests = [
            {'src': '(print (+ 1 3))', 'output': '4'},
            {'src': '(print (+ 1.5 2.3))', 'output': '3.8'},
            {'src': '(print (- 3 2))', 'output': '1'},
            {'src': '(print (- 3.5 2.1))', 'output': '1.4'},
            {'src': '(print (* 3 2))', 'output': '6'},
            {'src': '(print (* 3.6 2.5))', 'output': '9'},
            {'src': '(print (/ 6 2))', 'output': '3'},
            {'src': '(print (/ 7.5 2.5))', 'output': '3'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_strings(self):
        tests = [
            {'src': '(print "abc")', 'output': 'abc'},
            {'src': '(print (str/lower "Hello World"))', 'output': 'hello world'},
            {'src': '(print (str/blank? "Hello World"))', 'output': 'false'},
            {'src': '(print (str/blank? ""))', 'output': 'true'},
            {'src': '(print (str/blank? nil))', 'output': 'true'},
            {'src': '(print (str/blank? "\\n"))', 'output': 'true'},
            {'src': '(print (str/split "hello world"))', 'output': '[hello world]'},
            # {'src': '(str)', 'output': ''},
            # {'src': '(print (str 1))', 'output': '1'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_lists(self):
        tests = [
            {'src': '(print [])', 'output': '[]'},
            {'src': '(print [1])', 'output': '[1]'},
            {'src': '(print [1 nil "hello" 2.34 true [1] {"a" 1}])', 'output': '[1 nil hello 2.34 true [1] {a 1}]'},
            {'src': '(print (nth [1 2] 0))', 'output': '1'},
            {'src': '(print (nth ["1" 2] 0))', 'output': '1'},
            {'src': '(print (nth [1 (+ 1 1)] 1))', 'output': '2'},
            {'src': '(print (nth [1 (nth [2 3] 0)] 1))', 'output': '2'},
            {'src': '(print (nth [1 nil 2] 1))', 'output': 'nil'},
            {'src': '(print (nth [1 2 3] 3))', 'output': 'nil'},
            {'src': '(print (nth [1 2 3] -1))', 'output': 'nil'},
            {'src': '(print (count [1 2 3]))', 'output': '3'},
            {'src': '(print (sort [1]))', 'output': '[1]'},
            {'src': '(print (sort [1 2]))', 'output': '[1 2]'},
            {'src': '(print (sort [1 3 2]))', 'output': '[1 2 3]'},
            {'src': '(print (sort > [1 3 2]))', 'output': '[3 2 1]'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_maps(self):
        tests = [
            {'src': '(print {})', 'output': '{}'},
            {'src': '(print {"a" 1})', 'output': '{a 1}'},
            {'src': '(print {"a" 1 "b" 2})', 'output': '{a 1, b 2}'},
            {'src': '(print {"a" [1] "b" [2]})', 'output': '{a [1], b [2]}'},
            {'src': '(print {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7})',
                 'output': '{1 1, 2 2, 3 3, 4 4, 5 5, 6 6, 7 7}'},
            {'src': '(print (keys {"a" 2 "b" 3}))', 'output': '[a b]'},
            {'src': '(print (vals {"a" 2 "b" 3}))', 'output': '[2 3]'},
            {'src': '(print (pairs {"a" 2 "b" 3}))', 'output': '[[a 2] [b 3]]'},
            {'src': '(print (assoc {} "new-key" "new-val"))', 'output': '{new-key new-val}'},
            {'src': '(print (assoc {"1" 2 "a" 3} "new-key" "new-val"))', 'output': '{1 2, a 3, new-key new-val}'},
            {'src': '(print (assoc {"1" 2 "a" 3} "1" "new-val"))', 'output': '{1 new-val, a 3}'},
            {'src': '(print (assoc {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7 "8" 8} "9" 9))',
                'output': '{1 1, 2 2, 3 3, 4 4, 5 5, 6 6, 7 7, 8 8, 9 9}'},
            {'src': '(print (get {} "a"))', 'output': 'nil'},
            {'src': '(print (get {} "a" 99))', 'output': '99'},
            {'src': '(print (get {"a" 1} "a"))', 'output': '1'},
            {'src': '(print (get {"a" 1} "b"))', 'output': 'nil'},
            {'src': '(print (get {"a" 1} "b" 99))', 'output': '99'},
            {'src': '(print (get {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5} "1"))', 'output': '1'},
            {'src': '(print (get {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6} "1"))', 'output': '1'},
            {'src': '(print (get {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7} "1"))', 'output': '1'},
            {'src': '(print (get {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7 "8" 8 "9" 9} "1"))', 'output': '1'},
            {'src': '(print (get {"1" 1 "2" 2 "3" 3 "4" 4 "5" 5 "6" 6 "7" 7 "8" 8 "9" 9} "a"))', 'output': 'nil'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_comparisons(self):
        tests = [
            {'src': '(print (= nil nil))', 'output': 'true'},
            {'src': '(print (= true true))', 'output': 'true'},
            {'src': '(print (= false false))', 'output': 'true'},
            {'src': '(print (= nil false))', 'output': 'false'},
            {'src': '(print (= nil true))', 'output': 'false'},
            {'src': '(print (= false true))', 'output': 'false'},
            {'src': '(print (= 1 1))', 'output': 'true'},
            {'src': '(print (= 1 2))', 'output': 'false'},
            {'src': '(print (= 1 "1"))', 'output': 'false'},
            {'src': '(print (= 1 1.0))', 'output': 'true'}, #different from clojure
            {'src': '(print (= "abc" "abc"))', 'output': 'true'},
            {'src': '(print (= "abc" "def"))', 'output': 'false'},
            {'src': '(print (= [] []))', 'output': 'true'},
            {'src': '(print (= [1] [2]))', 'output': 'false'},
            {'src': '(print (= [1] [1]))', 'output': 'true'},
            {'src': '(print (= ["a" 1] ["a" 1]))', 'output': 'true'},
            {'src': '(print (= {} {}))', 'output': 'true'},
            {'src': '(print (= {"a" 1} {"a" 1}))', 'output': 'true'},
            {'src': '(print (= {"a" 2} {"a" 1}))', 'output': 'false'},
            {'src': '(print (= {"a" 1} {"b" 1}))', 'output': 'false'},
            {'src': '(print (> 3 2))', 'output': 'true'},
            {'src': '(print (> 3 4))', 'output': 'false'},
            {'src': '(print (>= 3 3))', 'output': 'true'},
            {'src': '(print (>= 3 4))', 'output': 'false'},
            {'src': '(print (< 2 3))', 'output': 'true'},
            {'src': '(print (< 2 1))', 'output': 'false'},
            {'src': '(print (<= 2 2))', 'output': 'true'},
            {'src': '(print (<= 2 1))', 'output': 'false'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_if(self):
        tests = [
            {'src': '(print (if true true))', 'output': 'true'},
            {'src': '(print (if true true false))', 'output': 'true'},
            {'src': '(print (if true "True" false))', 'output': 'True'},
            {'src': '(print (if false true))', 'output': 'nil'},
            {'src': '(print (if false true false))', 'output': 'false'},
            {'src': '(print (if false true "False"))', 'output': 'False'},
            {'src': '(print (if true ["True"] false))', 'output': '[True]'},
            {'src': '(print (if true {"True" "1"} false))', 'output': '{True 1}'},
            {'src': '(print (if false true ["False"]))', 'output': '[False]'},
            {'src': '(print (if false true {"False" "1"}))', 'output': '{False 1}'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_do(self):
        tests = [
            {'src': '(do (print 1) (print 2))', 'output': '12'},
            {'src': '(do (print 1) (if (< 1 2) (print 1) (print 2)))', 'output': '11'},
            {'src': '(print (do (print 1) (if (< 1 2) (print 1) (print 2)) "3"))', 'output': '113'},
            {'src': '(do (println "line1") (println "line2"))', 'output': 'line1\nline2\n'},
            {'src': '(print (do (println "output") 2))', 'output': 'output\n2'},
            {'src': '(print (do (println "output") "return"))', 'output': 'output\nreturn'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_let(self):
        tests = [
            {'src': '(print (let [y 1] y))', 'output': '1'},
            {'src': '(print (let [y 1] [y]))', 'output': '[1]'},
            {'src': '(print (let [y 1] {"a" y}))', 'output': '{a 1}'},
            {'src': '(let [x 1] (print x))', 'output': '1'},
            {'src': '(let [x {}] (print x))', 'output': '{}'},
            {'src': '(let [x-y {}] (print x-y))', 'output': '{}'},
            {'src': '(let [x-y {}] (let [z-a 1] (print x-y)))', 'output': '{}'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_loops(self):
        tests = [
            {'src': '(print (loop [cnt 3 acc 0] (if (= 0 cnt) acc (recur (- cnt 1) (+ acc 1)))))', 'output': '3'},
            {'src': '(print (loop [cnt 3 acc 0] (if (= 0 cnt) "3" (recur (- cnt 1) (+ acc 1)))))', 'output': '3'},
            {'src': '(print (loop [cnt 3 acc 0] (if (= 0 cnt) acc (recur (- cnt 1) (+ acc cnt)))))', 'output': '6'},
            {'src': '(loop [n 0] (do (print n) (if (< n 2) (recur (+ n 1)))))', 'output': '012'},
            {'src': '(loop [n 0] (if (> n 2) (print n) (let [y 1] (recur (+ n y)))))', 'output': '3'},
            {'src': '(let [b 2] (loop [n 0] (if (> n b) (print n) (let [y 1] (recur (+ n y))))))', 'output': '3'},
            {'src': '(loop [n 0] (if (> n 2) (print "done") (do (print n) (recur (+ n 1)))))', 'output': '012done'},
            {'src': '(loop [n 0] (do (print n) (print "    ") (println (/ (* 5 (- n 32)) 9)) (if (< n 70) (recur (+ 20 n)))))', 'output': '0    -17.7778\n20    -6.66667\n40    4.44444\n60    15.5556\n80    26.6667\n'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_fn(self):
        tests = [
            {'src': '(print ((fn [x] x) 1))', 'output': '1'},
            {'src': '(print ((fn [x y] (+ x y)) 1 2))', 'output': '3'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_def(self):
        tests = [
            {'src': '(def a 1) (print a)', 'output': '1'},
            {'src': '(def some-thing 1) (print some-thing)', 'output': '1'},
            {'src': '(def some-thing "1") (print some-thing)', 'output': '1'},
            {'src': '(def some-thing {"a" "b"}) (print some-thing)', 'output': '{a b}'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_defn(self):
        tests = [
            {'src': '(defn f1 [x y] (+ x y)) (print (f1 1 2))', 'output': '3'},
            {'src': '(defn f-1 [x y] (+ x y)) (print (f-1 1 2))', 'output': '3'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_advanced(self):
        tests = [
            {'src': '(def a 1) (let [b 2] (print (+ a b)))', 'output': '3'},
            {'src': '(let [x 1] (if (= x 1) (print true) (print false)))', 'output': 'true'},
            {'src': '(let [x-y 1] (do (print x-y) (print "done")))', 'output': '1done'},
            {'src': '(let [x-y 1] (loop [n-p 0] (if (= n-p 1) (print x-y) (recur (+ n-p 1)))))', 'output': '1'},
            {'src': '(let [a 1] (let [b 2] (print (+ a b))))', 'output': '3'},
            {'src': '(def d {}) (assoc d "a" 1) (print (get d "a"))', 'output': '1'},
            {'src': '(def i 0) (print ((fn [n] n) i))', 'output': '0'},
            {'src': '(print ((fn [n] (loop [cnt n acc 1] (if (= 0 cnt) acc (recur (- cnt 1) (* acc cnt))))) 3))', 'output': '6'},
            {'src': '(defn f1 [x] (let [y (+ x 1)] y)) (print (f1 1))', 'output': '2'},
            {'src': '(defn f1 [x] (+ x 1)) (let [y 1] (print (f1 y)))', 'output': '2'},
            {'src': '(defn f1 [z] (let [x 1] (loop [y 2] (if (< y 5) (recur (+ y 1)) (+ x z))))) (print (f1 3))', 'output': '4'},
            {'src': '(defn f1 [] {"key" "value"}) (print (get (f1) "key"))', 'output': 'value'},
            {'src': '(defn compare [a b] (> (nth a 1) (nth b 1))) (print (sort compare [["a" 1] ["b" 2]]))', 'output': '[[b 2] [a 1]]'},
            {'src': '(print (loop [line (read-line)] (if (= nil line) "done" (recur (read-line)))))', 'input': 'line', 'output': 'done'},
            {'src': '(print (loop [line (read-line)] (if (= nil line) ["done"] (recur (read-line)))))', 'input': 'line', 'output': '[done]'},
            {'src': '(print (loop [line (read-line)] (if (= nil line) {"a" "b"} (recur (read-line)))))', 'input': 'line', 'output': '{a b}'},
            {'src': '(print (loop [line (read-line)] (if (= nil line) "done" (do (print line) (recur (read-line))))))', 'input': 'line1\nline2\n', 'output': 'line1line2done'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)

    def test_other(self):
        tests = [
            {'src': '(println "hello")', 'output': 'hello\n'},
            {'src': '(print (read-line))', 'input': 'line\n', 'output': 'line'},
            {'src': '(print (read-line))', 'input': '\n', 'output': ''},
            {'src': '(print (read-line))', 'input': '', 'output': 'nil'},
        ]
        for test in tests:
            with self.subTest(test=test):
                _run_test(test, self.assertEqual)


if __name__ == '__main__':
    unittest.main()
