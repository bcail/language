import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from lang import (TokenType, scan_tokens, parse, evaluate,
        Keyword, Symbol, Var, Vector, run, _compile)


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
                print(f'*** test: {test["src"]}')
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


# See https://github.com/airbus-seclab/c-compiler-security
GCC_CMD = [
    'gcc',
    '-O2',
    '-Werror',
    '-Wall',
    '-Wextra',
    '-std=c99',
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

GCC_ENV = {
    'ASAN_OPTIONS': 'strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1:detect_invalid_pointer_pairs=2',
    'PATH': os.environ.get('PATH', ''),
}


class CompileTests(unittest.TestCase):
    def test(self):
        tests = [
            {'src': '(print nil)', 'output': 'nil'},
            {'src': '(print true)', 'output': 'true'},
            {'src': '(print false)', 'output': 'false'},
            {'src': '(print (= nil nil))', 'output': 'true'},
            {'src': '(print (= true true))', 'output': 'true'},
            {'src': '(print (= false false))', 'output': 'true'},
            {'src': '(print (= nil false))', 'output': 'false'},
            {'src': '(print (= nil true))', 'output': 'false'},
            {'src': '(print (= false true))', 'output': 'false'},
            {'src': '(print (= 1 1))', 'output': 'true'},
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
            {'src': '(print (+ 1 3))', 'output': '4'},
            {'src': '(print (+ 1.5 2.3))', 'output': '3.8'},
            {'src': '(print (- 3 2))', 'output': '1'},
            {'src': '(print (- 3.5 2.1))', 'output': '1.4'},
            {'src': '(print (* 3 2))', 'output': '6'},
            {'src': '(print (* 3.6 2.5))', 'output': '9'},
            {'src': '(print (/ 6 2))', 'output': '3'},
            {'src': '(print (/ 7.5 2.5))', 'output': '3'},
            {'src': '(print "hello")', 'output': 'hello'},
            {'src': '(println "hello")', 'output': 'hello\n'},
            {'src': '(if (> 3 2) (print true))', 'output': 'true'},
            {'src': '(if (> 3 4) (print true) (print false)', 'output': 'false'},
            {'src': '(if (>= 3 3) (print true))', 'output': 'true'},
            {'src': '(if (>= 3 4) (print true) (print false)', 'output': 'false'},
            {'src': '(if (< 2 3) (print true))', 'output': 'true'},
            {'src': '(if (< 2 1) (print true) (print false))', 'output': 'false'},
            {'src': '(if (<= 2 2) (print true))', 'output': 'true'},
            {'src': '(if (<= 2 1) (print true) (print false))', 'output': 'false'},
            {'src': '(do (println "line1") (println "line2"))', 'output': 'line1\nline2\n'},
            {'src': '(print (do (println "output") 2))', 'output': 'output\n2'},
            {'src': '(print [1 nil "hello" 2.34 true])', 'output': '[1 nil hello 2.34 true]'},
            {'src': '(print {"a" 1 "b" 2})', 'output': '{a 1, b 2}'},
            {'src': '(print (assoc {"1" 2 "a" 3} "new-key" "new-val"))', 'output': '{new-key new-val, 1 2, a 3}'},
            {'src': '(print (get {"a" 1} "a"))', 'output': '1'},
            {'src': '(print (get {"a" 1} "b"))', 'output': 'nil'},
            {'src': '(print (get {"a" 1} "b" 99))', 'output': '99'},
            {'src': '(print (nth [1 2] 0))', 'output': '1'},
            {'src': '(print (nth [1 (+ 1 1)] 1))', 'output': '2'},
            {'src': '(print (nth [1 (nth [2 3] 0)] 1))', 'output': '2'},
            {'src': '(print (nth [1 nil 2] 1))', 'output': 'nil'},
            {'src': '(print (nth [1 2 3] 3))', 'output': 'nil'},
            {'src': '(print (nth [1 2 3] -1))', 'output': 'nil'},
            {'src': '(print (count [1 2 3]))', 'output': '3'},
            {'src': '(def a 1) (print a)', 'output': '1'},
            {'src': '(print (let [x 1] x))', 'output': '1'},
            {'src': '(let [x 1] (print x))', 'output': '1'},
            {'src': '(let [x 1] (if (= x 1) (print true) (print false)))', 'output': 'true'},
            {'src': '(print (loop [cnt 3 acc 0] (if (= 0 cnt) acc (recur (- cnt 1) (+ acc 1)))))', 'output': '3'},
            {'src': '(print (loop [cnt 3 acc 0] (if (= 0 cnt) acc (recur (- cnt 1) (+ acc cnt)))))', 'output': '6'},
            {'src': '(loop [n 0] (do (print n) (if (< n 2) (recur (+ n 1)))))', 'output': '012'},
            {'src': '(loop [n 0] (do (print n) (print "    ") (println (/ (* 5 (- n 32)) 9)) (if (< n 70) (recur (+ 20 n)))))', 'output': '0    -17.7778\n20    -6.66667\n40    4.44444\n60    15.5556\n80    26.6667\n'},
            # {'src': '(str)', 'output': ''},
            # {'src': '(print (str 1))', 'output': '1'},
        ]

        for test in tests:
            with self.subTest(test=test):
                print(f'*** c test: {test["src"]}')
                c_code = _compile(test['src'])

                with tempfile.TemporaryDirectory() as tmp:
                    c_filename = os.path.join(tmp, 'code.c')
                    with open(c_filename, 'wb') as f:
                        f.write(c_code.encode('utf8'))

                    program_filename = os.path.join(tmp, 'program')
                    compile_cmd = GCC_CMD + ['-o', program_filename, c_filename]
                    try:
                        subprocess.run(compile_cmd, check=True, env=GCC_ENV)
                    except subprocess.CalledProcessError:
                        print(f'bad c code:\n{c_code}')
                        raise

                    program_cmd = [program_filename]
                    try:
                        result = subprocess.run(program_cmd, check=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        print(f'bad c code:\n{c_code}')
                        print(f'err: {e.stderr.decode("utf8")}')
                        print(f'out: {e.stdout.decode("utf8")}')
                        raise

                    try:
                        self.assertEqual(result.stdout.decode('utf8'), test['output'])
                    except AssertionError:
                        print(f'bad c code:\n{c_code}')
                        raise


if __name__ == '__main__':
    unittest.main()
