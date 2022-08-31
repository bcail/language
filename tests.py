import tempfile
import unittest
from unittest.mock import patch
from lang import TokenType, scan_tokens, parse, evaluate, Keyword, Symbol, Var, Vector, run


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
            {'src': '(get {1 2} 1)', 'result': 2},
            {'src': '(keys {1 2 "a" 3})', 'result': [1, 'a']},
            {'src': '(vals {1 2 "a" 3})', 'result': [2, 3]},
            {'src': '(pairs {1 2 "a" 3})', 'result': [[1, 2], ["a", 3]]},
            {'src': '(count {1 2 "a" 3})', 'result': 2},
            {'src': '(contains? {1 2 "a" 3} 1)', 'result': True},
            {'src': '(contains? {1 2 "a" 3} "not-found")', 'result': False},
            {'src': '(assoc {1 2 "a" 3} "new-key" "new-val")', 'result': {1: 2, 'a': 3, 'new-key': 'new-val'}},
            {'src': '(dissoc {1 2 "a" 3} 1)', 'result': {'a': 3}},
            {'src': '((fn [n] (loop [cnt n acc 1] (if (= 0 cnt) acc (recur (- cnt 1) (* acc cnt))))) 3)', 'result': 6},
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

if __name__ == '__main__':
    unittest.main()
